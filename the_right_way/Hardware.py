from dataclasses import dataclass, field
from typing import Literal, Dict, Tuple, Optional

Mode = Literal["OFF", "ACQUIRE", "TRACK", "DUTY"]

@dataclass
class GPSConfig:
    # Electrical (typical for tiny GNSS at 3.3V — tune for your module)
    v_out: float = 3.3            # GPS rail voltage
    eta: float = 0.90             # regulator efficiency to GPS rail (buck/boost)
    I_acquire: float = 35e-3      # A, cold acquisition current
    I_track: float = 24e-3        # A, tracking current (continuous)
    I_idle: float = 0.8e-3        # A, backup/idle current when "off" (RTC, assist)
    fix_rate_hz: float = 1.0      # nominal fix rate when tracking

    # Acquisition timing (simplified)
    t_hot_s: float = 3.0          # hot-start time (recently tracking)
    t_cold_s: float = 25.0        # cold-start time (power-cycled a while)
    hot_window_s: float = 60.0    # if off less than this, assume hot start

    # Low-power policy thresholds
    soc_low: float = 0.25         # below this, prefer duty-cycling
    temp_cold_C: float = -10.0    # colder than this, prefer duty-cycling
    v_brown_hint: float = 3.45    # if battery terminal near this, be cautious

    # Duty-cycling parameters (average downlink/compute still OK for navigation)
    # recall duty_cycle = duty_on / duty_period
    duty_period_s_nom: float = 10.0
    duty_on_s_nom: float = 2.0

    # More aggressive duty at low SoC/very cold
    duty_period_s_low: float = 20.0
    duty_on_s_low: float = 2.0

    # Landing boost (tighten fixes below this altitude AGL)
    landing_alt_m: float = 1500.0
    duty_period_s_landing: float = 5.0
    duty_on_s_landing: float = 2.0


@dataclass
class GPSState:
    mode: Mode = "OFF"
    time_in_mode: float = 0.0
    time_since_fix: float = 1e9      # big means "no recent fix"
    have_fix: bool = False
    # For duty cycling
    duty_t_in_period: float = 0.0
    duty_period_s: float = 10.0
    duty_on_s: float = 2.0
    # Acquisition bookkeeping
    acq_time_target: float = 0.0     # how long we must stay in ACQUIRE to get a fix

@dataclass
class GPSHistory:
    time: list = field(default_factory=list)
    mode: list = field(default_factory=list)
    have_fix: list = field(default_factory=list)
    fix_available: list = field(default_factory=list)
    P_W: list = field(default_factory=list)
    I_A: list = field(default_factory=list)
    duty_period_s: list = field(default_factory=list)
    duty_on_s: list = field(default_factory=list)

class GPSPowerManager:
    """
    Simple GPS power governor:
      - ACQUIRE: high current until fix obtained (hot/cold start time)
      - TRACK: continuous tracking
      - DUTY: periodic ON windows (ACQUIRE → TRACK briefly), then OFF/IDLE
      - OFF: only backup current (optionally keep time/ephemeris)
    """

    def __init__(self, cfg: GPSConfig):
        self.cfg = cfg
        self.s = GPSState()
        # default duty parameters
        self.s.duty_period_s = cfg.duty_period_s_nom
        self.s.duty_on_s = cfg.duty_on_s_nom
        self.history = GPSHistory()

    def _select_duty_params(self, soc: float, T_C: float, alt_m: float) -> Tuple[float, float]:
        cfg = self.cfg
        # Landing boost takes priority
        if alt_m <= cfg.landing_alt_m:
            return cfg.duty_period_s_landing, cfg.duty_on_s_landing
        # Low SoC or very cold → conservative duty
        if (soc <= cfg.soc_low) or (T_C <= cfg.temp_cold_C):
            return cfg.duty_period_s_low, cfg.duty_on_s_low
        # Nominal cruise
        return cfg.duty_period_s_nom, cfg.duty_on_s_nom

    def _acq_time_needed(self) -> float:
        # Hot vs cold start based on time since last valid fix
        return self.cfg.t_hot_s if self.s.time_since_fix <= self.cfg.hot_window_s else self.cfg.t_cold_s

    def _enter(self, new_mode: Mode):
        self.s.mode = new_mode
        self.s.time_in_mode = 0.0
        if new_mode == "ACQUIRE":
            self.s.have_fix = False
            self.s.acq_time_target = self._acq_time_needed()

    def step(self,
             dt: float,
             t: float,
             soc: float,
             v_term: float,
             temp_C: float,
             alt_m: float,
             mission_phase: Literal["prelaunch","ascend","float","descent","landing"]) -> Dict[str, object]:
        """
        Advance the GPS policy by dt.
        Args:
            dt: time step (s)
            t: current time (s)
            soc: battery state of charge (0..1)
            v_term: battery terminal voltage (V)
            temp_C: battery temperature (C)
            alt_m: current altitude (m)
            mission_phase: one of ["prelaunch","ascend","float","descent","landing"]
            
        Returns:
            {
              "mode": str,
              "have_fix": bool,
              "P_W": float,         # electrical power into GPS (includes regulator efficiency)
              "I_A": float,         # GPS current at v_out
              "fix_available": bool # new fix this step (True once per second in TRACK)
            }
        """
        cfg, s = self.cfg, self.s
        s.time_in_mode += dt
        s.time_since_fix += dt
        fix_available = False

        # --- Policy: decide desired operating class ---
        want_continuous = (mission_phase in ["landing"])  # always track during final landing
        brownish = (v_term <= cfg.v_brown_hint) or (soc <= cfg.soc_low) or (temp_C <= cfg.temp_cold_C)

        # Mode transitions
        if s.mode == "OFF":
            # Wake periodically or if continuous needed
            if want_continuous:
                self._enter("ACQUIRE")
            else:
                # Duty-cycle scheduling
                s.duty_period_s, s.duty_on_s = self._select_duty_params(soc, temp_C, alt_m)
                s.duty_t_in_period += dt
                if s.duty_t_in_period >= s.duty_period_s:
                    # start ON window
                    s.duty_t_in_period = 0.0
                    self._enter("ACQUIRE")

        elif s.mode == "ACQUIRE":
            # Burn current until we reach acq target, then TRACK
            if s.time_in_mode >= s.acq_time_target:
                s.have_fix = True
                s.time_since_fix = 0.0
                # If we're duty-cycling, we TRACK only for the on-window remainder
                if want_continuous:
                    self._enter("TRACK")
                else:
                    self._enter("TRACK")  # will drop to OFF when on-window ends
            # If things get very tight, we could abort acquire early (optional)
            # if brownish and s.time_in_mode > cfg.t_hot_s: pass

        elif s.mode == "TRACK":
            # Generate 1 Hz fixes while tracking
            if s.have_fix:
                # Emit a fix at ~1 Hz
                # accumulate fractional timing via time_in_mode and fix_rate
                # Simple approach: tick at integer multiples of 1/fix_rate
                if (int((s.time_in_mode - dt) * cfg.fix_rate_hz) != int(s.time_in_mode * cfg.fix_rate_hz)):
                    fix_available = True
                    s.time_since_fix = 0.0

            if want_continuous:
                # stay in TRACK
                pass
            else:
                # duty-cycling window management
                s.duty_t_in_period += dt
                if s.duty_t_in_period >= s.duty_on_s:
                    # close ON window -> go OFF until next period boundary
                    # push duty_t_in_period to just before period end so OFF spans remainder
                    s.duty_t_in_period = s.duty_on_s
                    self._enter("OFF")

        elif s.mode == "DUTY":
            # not used (we embed duty behavior inside OFF/ACQUIRE/TRACK)
            pass

        # --- Compute electrical power for the chosen state ---
        if self.s.mode == "OFF":
            I = cfg.I_idle
        elif self.s.mode == "ACQUIRE":
            I = cfg.I_acquire
        else:  # TRACK
            I = cfg.I_track

        # Convert to input power: P_in = V_out*I / eta
        P_W = (cfg.v_out * I) / max(cfg.eta, 1e-3)

        # --- Save history (optional) ---
        self.history.time.append(t)
        self.history.mode.append(s.mode)
        self.history.have_fix.append(s.have_fix)
        self.history.fix_available.append(fix_available)
        self.history.P_W.append(P_W)
        self.history.I_A.append(I)
        self.history.duty_period_s.append(s.duty_period_s)
        self.history.duty_on_s.append(s.duty_on_s)

        return {
            "mode": self.s.mode,
            "have_fix": self.s.have_fix,
            "fix_available": fix_available,
            "P_W": P_W,
            "I_A": I,
            "duty_period_s": self.s.duty_period_s,
            "duty_on_s": self.s.duty_on_s,
        }


Phase = Literal["prelaunch","ascend","float","descent","landing"]

# ----- Tune these for your payload -----
@dataclass
class PayloadConfig:
    """
    Configuration container for payload electrical and actuation loads.

    This class defines nominal current draws, duty cycles, and thresholds for 
    each major payload subsystem. The values are expressed in amps (A) at 
    their respective rails unless otherwise noted. 

    Parameters
    ----------
    eta_33 : float
        Efficiency of the 3.3 V regulator supplying MCU, IMU, wind sensor,
        and radio. Typical buck/boost efficiency ≈ 90%.
    
    I_mcu_active : float
        Active-mode current draw of the MCU (STM32U5F7VJT6) at 3.3 V.
        Reference: ST datasheet (~12 mA).
    I_mcu_idle : float
        Idle/low-power current draw of the MCU (~0.6 mA).

    I_imu_active : float
        Active-mode current draw of the IMU at 3.3 V (~2.5 mA).
    I_imu_idle : float
        Idle-mode current draw of the IMU (~0.1 mA).

    I_wind_active : float
        Active-mode current draw of the wind sensor (~4 mA).
    I_wind_idle : float
        Idle-mode current draw of the wind sensor (~0.1 mA).
    mass_board_g : float
        Mass of the sensor/MCU board (grams). Currently 1 g placeholder.

    I_radio_tx : float
        Radio/telemetry transmit current at 3.3 V (set to 0 if unused).
        Example: small telemetry module ~15 mA TX.
    I_radio_idle : float
        Idle current of radio when not transmitting (~1 mA).
    radio_tx_duty : float
        Fraction of mission time radio spends transmitting. 
        Example: 0.10 → 10% duty cycle in TX.
    mass_radio_g : float
        Mass of the radio module (grams). Currently 5.0 g placeholder.

    I_act_stall : float
        Peak/moving current draw of the micro-servo actuator at battery rail.
        Approx. 150 mA (stall/commanded movement).
    duty_servo_nom : float
        Duty cycle fraction of servo operation during nominal control (~4%).
    duty_servo_landing : float
        Duty cycle fraction during landing/stronger control (~10%).
    duty_servo_idle : float
        Background/no-command duty (~0.2%).

    soc_shed : float
        State-of-charge threshold below which payload will shed non-critical loads.
        Expressed as a fraction of nominal capacity (0–1).
    v_shed : float
        Voltage threshold (V) below which payload will shed loads (e.g., 3.45 V).
    temp_cold_C : float
        Minimum operating temperature (°C) for safe operation. Below this,
        battery and payload loads may need thermal management or shutdown.
    """

    eta_33: float = 0.90

    # MCU @3.3V https://www.mouser.com/ProductDetail/STMicroelectronics/STM32U5F7VJT6?qs=HoCaDK9Nz5e1ZGsW1DPWFA%3D%3D&mgh=1
    I_mcu_active: float = 0.012
    I_mcu_idle:   float = 0.0006
    mass_board_g:float = 1 # g
    
    # IMU @3.3V
    I_imu_active: float = 0.0025
    I_imu_idle:   float = 0.0001

    # Wind sensor @3.3V
    I_wind_active: float = 0.004
    I_wind_idle:   float = 0.0001
    

    #Optional radio/telemetry @3.3V (set to 0 if none)
    # https://www.cdebyte.com/products/E22-900T33S/1
    I_radio_tx: float = 1.2 # A when transmitting
    I_radio_idle: float = 0.014 # A when not transmitting
    radio_tx_duty: float = 0.02  # fraction of time in TX during flight
    mass_radio_g: float = 5.0
    ## 1. LightTracker 1.1 — ~5.4 g
    
    # Actuator/servo on battery rail, micro-servo specs
    I_act_stall: float = 0.2         # A at V_batt (peak/moving)
    duty_servo_nom: float = 0.04        # when actively commanded
    duty_servo_landing: float = 0.5    # stronger control near ground
    duty_servo_idle: float = 0.002      # ~idle/background (no command)
    mass_servo_g: float = 4.0        # g, e.g. SG90

    # Load-shedding thresholds
    soc_shed: float = 0.18
    v_shed:   float = 3.0
    temp_cold_C: float = -10.0


@dataclass
class PayloadHistory:
    time: list = field(default_factory=list)
    P_total: list = field(default_factory=list)
    P_gps: list = field(default_factory=list)
    P_33: list = field(default_factory=list)
    P_act: list = field(default_factory=list)
    duty_servo: list = field(default_factory=list)
    gps_mode: list = field(default_factory=list)
    servo_active: list = field(default_factory=list)
    servo_effort: list = field(default_factory=list)
    shedding: list = field(default_factory=list)

class PayloadPowerController:
    """
    Power supervisor that computes per-step payload power draw and performs
    **load shedding** (turning off / throttling subsystems) when energy margins
    get tight.

    Overview
    --------
    This class aggregates three load groups each step:
      1) GPS module power (delegated to `gps_mgr.step`)
      2) 3.3V-rail loads (MCU, IMU, wind sensor, radio)
      3) Actuator (servo) power on the battery rail

    It also applies **shedding** when any of the following trip:
      - `soc <= cfg.soc_shed` (state-of-charge low)
      - `v_batt <= cfg.v_shed` (battery brownout margin low)
      - `temp_C <= cfg.temp_cold_C` (too cold; battery/loads derate)

    When shedding, non-critical loads (wind sensor, radio TX duty, servo duty)
    are reduced or held at idle to preserve critical compute and GPS.

    Key Concepts
    ------------
    • Mission phase gating:
        - MCU/IMU are kept on (critical).
        - Wind sensor enabled in ["float", "descent", "landing"] unless shedding.
        - Radio TX allowed in ["descent", "landing"] unless shedding; otherwise idle.

    • 3.3V rail power:
        P_33 = (V_33 * ΣI_33) / η_33
        where V_33 = 3.3 V and η_33 = buck/boost efficiency (0..1).

    • Actuator (servo) power:
        Duty is determined by phase and current command activity.
        I_act = I_act_stall * duty_servo
        P_act = v_batt * I_act

        If shedding, servo duty is additionally reduced (halved by default).

    • History:
        The controller records a compact time-series of the computed powers,
        modes, and duty so you can graph energy usage and verify shedding.

    Parameters
    ----------
    cfg : PayloadConfig
        Electrical and duty-cycle assumptions for each subsystem.
    gps_mgr : object
        Manager with a `step(dt, t, soc, v_term, temp_C, alt_m, mission_phase)`
        method returning a dict with at least {"P_W": float, "mode": str}.

    Returns (from `step`)
    ---------------------
    Dict[str, Any] with:
      - "P_total_W": total payload power (W)
      - "P_gps_W", "P_33_W", "P_act_W": component powers (W)
      - "I_act_A": actuator current (A)
      - "duty_servo": final servo duty used (0..1)
      - "modes": { "gps_mode", "servo_active", "servo_effort", "shedding" }

    Notes
    -----
    • Shedding policy is intentionally simple and conservative:
      it flips on if *any* trip condition is true. Tune thresholds and which
      loads to curtail based on your mission risk posture.

    • Radio TX current is time-averaged via `cfg.radio_tx_duty`. For bursty
      links, you can refine this by scheduling discrete TX windows instead.

    • Cold temperature cutoff (`temp_cold_C`) defends against sag / poor
      deliverable power in Li-ion at low temps; might gate re-enabling.

    • Consider tying the 3.3 V efficiency to `cfg.eta_33` (see inline comment).
    """
    def __init__(self, cfg: PayloadConfig, gps_mgr):
        self.cfg = cfg
        self.gps = gps_mgr
        self.payload_history = PayloadHistory()

    def step(self,
             dt: float,
             t: float,
             phase: Phase,
             soc: float,
             v_batt: float,
             temp_C: float,
             alt_m: float,
             # NEW: actuator activity hints from your controller
             servo_active: bool,
             servo_effort: Optional[float] = None  # 0..1; scales “how hard” we’re moving
             ) -> Dict[str, float]:

        cfg = self.cfg

        # ---- GPS (unchanged) ----
        gps = self.gps.step(dt=dt, t=t, soc=soc, v_term=v_batt,
                            temp_C=temp_C, alt_m=alt_m, mission_phase=phase)
        P_gps = gps["P_W"]

        # ---- 3.3V rail loads (unchanged) ----
        shedding = (soc <= cfg.soc_shed) or (v_batt <= cfg.v_shed) or (temp_C <= cfg.temp_cold_C)
        mcu_on  = True
        imu_on  = True
        wind_on = (phase in ["float","descent","landing"]) and not shedding
        radio_tx = (phase in ["descent","landing"]) and not shedding

        I_mcu = cfg.I_mcu_active if mcu_on else cfg.I_mcu_idle
        I_imu = cfg.I_imu_active if imu_on else cfg.I_imu_idle
        I_wind = cfg.I_wind_active if wind_on else cfg.I_wind_idle
        I_radio = (cfg.I_radio_tx * cfg.radio_tx_duty) if radio_tx else cfg.I_radio_idle

        P_33 = (3.3*(I_mcu + I_imu + I_wind + I_radio))/max(0.90, 1e-3)

        # ---- Actuator duty based on command activity ----
        # base duty by phase
        #base_duty = cfg.duty_servo_landing if (phase == "landing") else cfg.duty_servo_nom
        if phase == "landing":
            base_duty = cfg.duty_servo_idle
        elif phase == "descent" or phase == "float":
            base_duty = cfg.duty_servo_landing
        else:
            base_duty = cfg.duty_servo_nom

        if servo_active:
            eff = 1.0 if (servo_effort is None) else float(max(0.0, min(1.0, servo_effort)))
            duty_servo = max(cfg.duty_servo_idle, base_duty * eff)

        else:
            duty_servo = cfg.duty_servo_idle

        # Optional: shed actuator workload further if power is tight
        if shedding:
            duty_servo *= 0.5  # or clamp to idle, depending on how aggressive you want

        I_act = cfg.I_act_stall * duty_servo
        P_act = v_batt * I_act
        P_total = P_gps + P_33 + P_act
        
        # save the history
        self.payload_history.time.append(t)
        self.payload_history.P_total.append(P_total)
        self.payload_history.P_gps.append(P_gps)
        self.payload_history.P_33.append(P_33)
        self.payload_history.P_act.append(P_act)
        self.payload_history.duty_servo.append(duty_servo)
        self.payload_history.gps_mode.append(gps["mode"])
        self.payload_history.servo_active.append(servo_active)
        self.payload_history.servo_effort.append(servo_effort)
        self.payload_history.shedding.append(shedding)
        
        return {
            "P_total_W": P_total,
            "P_gps_W": P_gps,
            "P_33_W": P_33,
            "P_act_W": P_act,
            "I_act_A": I_act,
            "duty_servo": duty_servo,
            "modes": {
                "gps_mode": self.gps.s.mode,
                "servo_active": servo_active,
                "servo_effort": 1.0 if (servo_effort is None) else float(max(0.0, min(1.0, servo_effort))),
                "shedding": shedding,
            }
        }
    