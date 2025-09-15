from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict
SIGMA = 5.670374419e-8  # Stefan–Boltzmann, W/m^2/K^4
# -----------------------
# History container
# -----------------------
@dataclass
class BatteryHistory:
    """
    Time-series log of the battery state for plotting/analysis.
    """
    t: List[float] = field(default_factory=list)          # time (s)
    V: List[float] = field(default_factory=list)          # terminal voltage (V)
    I: List[float] = field(default_factory=list)          # discharge current (A)
    SoC: List[float] = field(default_factory=list)        # state of charge [0..1]
    T: List[float] = field(default_factory=list)          # cell temperature (°C)
    P: List[float] = field(default_factory=list)          # delivered power (W)
    power_limited: List[bool] = field(default_factory=list)  # demanded P > Pmax?
    brownout: List[bool] = field(default_factory=list)       # V <= V_cutoff or SoC<=0


@dataclass
class BatteryState:
    """
    Minimal, simulation-friendly Li-ion cell model using a Thevenin source
    (voltage source Open-Circuit Voltage (OCV) in series with internal resistance R).

    Terminal equations (under load):
        V_term = OCV(SoC) - I * R(T)
        P_load = V_term * I

    We solve the quadratic that comes from P_load = (OCV - I*R) * I
    to find the physically valid discharge current I each step, then
    update SoC by coulomb counting, with capacity derated by temperature.

    This strikes a good balance between realism and numerical robustness
    for very small cells (100–130 mAh) where temperature and internal
    resistance dominate behavior.
    """

    # ------------- Nameplate & configuration -------------
    Q_rated_Ah: float = 0.10
    """
    Rated capacity at 25 °C and gentle discharge (e.g., 0.2C).
    Typical: 0.10–0.13 Ah for your cells. We derate this with temperature.
    """

    V_cutoff: float = 3.30
    """
    Minimum safe/usable terminal voltage under load.
    Below this we declare brownout (MCU, GPS, servos may fail).
    """

    R0_25: float = 0.35
    """
    Internal resistance (ohms) at 25 °C (DC small-signal).
    Small 100–130 mAh Li-ion pouches often sit in 0.2–0.6 Ω new.
    """

    SoC: float = 1.0
    """
    Initial state of charge [0..1]. If you cold-soak before launch, you
    still start near 1.0 but available capacity is reduced by temperature.
    """
    # --------- Thermal parameters (NEW) ----------
    m_cell_kg: float = 0.0025       # ~2.5 g tiny pouch cell
    c_p_J_per_kgK: float = 900.0    # effective specific heat
    area_m2: float = 6.5e-4         # ~20x12x3 mm pouch => ~6.5 cm^2
    emissivity: float = 0.8         # laminated polymer/aluminum pouch
    # Convection model: h = h_still + k_forced*(rho/rho0)^0.5 * V_rel^0.8
    h_still_W_m2K: float = 5.0
    k_forced_W_m2K: float = 8.0
    # Optional conduction to structure/internal air (use 0 if unknown)
    G_cond_W_per_K: float = 0.0
    T_struct_C: float = 25.0

    # Entropic heating coefficient (approx dOCV/dT). Set 0.0 to disable.
    dOCVdT_V_per_K: float = -4e-4

    # Initial cell temperature
    T_cell_C: float = 25.0

    # ------------- Running state (auto-updated) -------------
    V_ocv: float = 3.85       # interpolated open-circuit voltage (V)
    V_term: float = 3.85      # terminal voltage after IR drop (V)
    Wh_drawn: float = 0.0     # energy delivered so far (Wh)
    I_peak: float = 0.0       # max discharge current observed (A)
    hist: BatteryHistory = field(default_factory=BatteryHistory)
    
    # ------------- Curves -------------
    ocv_soc_pts: tuple = field(default_factory=lambda: (
        # SoC breakpoints (rested cell; mild load)
        np.array([0.00, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 1.00]),
        # Typical 1S Li-ion OCV (V) at those SoC points
        np.array([3.00, 3.45, 3.60, 3.70, 3.80, 3.95, 4.02, 4.15]),
    ))
    """
    OCV vs SoC curve. This is the "fuel gauge" at rest: the plateau around
    ~3.7–3.95 V reflects the flat region of Li-ion chemistry. Under load,
    V_term will be lower due to I*R.
    """

    cap_temp_pts: tuple = field(default_factory=lambda: (
        # Temperature (°C)
        np.array([-30,  -20,  -10,   0,   25,  40], dtype=float),
        # Fraction of rated capacity available at each temperature
        np.array([0.30, 0.50, 0.70, 0.85, 1.00, 0.95], dtype=float),
    ))
    """
    Available capacity vs temperature. Captures the well-known loss of
    capacity in the cold due to slowed diffusion and kinetics. Tunable.
    """
    mass_g: float = 0.003 # Approximate mass of the cell in grams (for reference)

    # --------- Helpers ----------
    @property
    def C_th_J_per_K(self) -> float:
        return max(1e-6, self.m_cell_kg * self.c_p_J_per_kgK)
    
    # ------------- Temperature effects -------------
    def R_internal(self, T_C: float) -> float:
        """
        Internal resistance vs temperature.
        Model: R(T) = R0_25 * exp(k*(25 - T))
        with k ~ 0.02–0.04 /°C for very small cells.
        Args:
            T_C: Temperature in °C
        Returns:
            Internal resistance in ohms
        Equations:
            R(T) = R0_25 * exp(k*(25 - T))
            25 °C is the reference temperature.
        
        """
        k = 0.03
        return self.R0_25 * np.exp(k * (25.0 - T_C))

    def capacity_Ah_at_T(self, T_C: float) -> float:
        """
        Available capacity (Ah) at temperature T by interpolation of the
        capacity derating curve.
        Args:
            T_C: Temperature in °C
        Returns:
            Available capacity in Ah
        Equations:
            Q(T) = Q_rated * f(T)
        """
        T, f = self.cap_temp_pts
        return self.Q_rated_Ah * float(np.interp(T_C, T, f))

    def ocv_from_soc(self, soc: float) -> float:
        """
        Lookup OCV at the given state of charge. We keep OCV independent of
        temperature in this minimal model; thermal dependence is smaller than
        IR/available-capacity effects for our use.
        """
        S, V = self.ocv_soc_pts
        return float(np.interp(np.clip(soc, 0.0, 1.0), S, V))

    def _h_conv(self, V_rel: float, rho_air: float) -> float:
            return self.h_still_W_m2K + self.k_forced_W_m2K * (max(V_rel, 0.0)**0.8) * np.sqrt(max(rho_air,1e-9)/rho_air)


    # ------------- One integration step -------------
    # def step(self,
    #          P_load_W: float,
    #          dt_s: float,
    #          T_env_C: float,
    #          V_rel_ms: float = 0.0,
    #          rho_air: float = RHO0) -> Dict[str, float]:
    #     """
    #     Advance the battery state by dt_s seconds while attempting to deliver
    #     P_load_W watts into your electronics.

    #     **Model Basis**
    #     This step function implements a minimal first-order Thevenin model of a
    #     Li-ion cell: an open-circuit voltage (OCV) source in series with a 
    #     temperature-dependent internal resistance R(T). This is a widely used
    #     reduced-order model for embedded simulation of batteries
    #     (see Tremblay & Dessaint, 2009; NASA BPS Li-ion models).

    #     **Equations**

    #     1) Terminal voltage under load:
    #     V_term = V_OCV(SoC) – I * R(T)

    #     2) Power balance quadratic (from P = V_term * I):
    #     P = (V_OCV – I R) * I
    #     => R I² – V_OCV I + P = 0

    #     Solve quadratic for I. Take the smaller positive root to ensure a 
    #     physically stable discharge branch.

    #     3) Maximum power limit (when discriminant D < 0):
    #     D = V_OCV² – 4 R P
    #     If D < 0, the demanded load power exceeds what the cell can provide.
    #     From dP/dI = 0:
    #         I_maxP = V_OCV / (2R)
    #         V_maxP = V_OCV / 2
    #         P_max  = V_OCV² / (4R)

    #     4) Coulomb counting (SoC update):
    #     ΔAh = (I * Δt) / 3600
    #     SoC_next = SoC – ΔAh / Q(T)

    #     where Q(T) is available capacity interpolated vs temperature.

    #     5) Internal resistance vs temperature (empirical exponential fit):
    #     R(T) = R_25 * exp(k * (25 – T))
    #     with k ≈ 0.02–0.04 /°C for very small pouch cells.

    #     6) Brownout condition:
    #     brownout = (V_term <= V_cutoff) or (SoC <= 0)

    #     **References**
    #     - Tremblay, O., & Dessaint, L.-A. (2009). "Experimental validation of a 
    #     battery dynamic model for EV applications." 
    #     https://www.mdpi.com/2032-6653/3/2/289
    #     - Shepherd, C. M. (1965). "Design of Primary and Secondary Cells II. 
    #     An equation describing battery discharge."
    #     https://iopscience.iop.org/article/10.1149/1.2423659/pdf
    #     - NASA Battery Performance and Safety (BPS) models: 
    #     https://ntrs.nasa.gov/citations/20140010375
    #     - General Li-ion OCV/SoC curves: 
    #     https://batteryuniversity.com/article/bu-503a-what-happens-under-load

    #     Returns
    #     -------
    #     dict with keys:
    #     V : float
    #         Terminal voltage (V)
    #     I : float
    #         Discharge current (A)
    #     SoC : float
    #         State of charge [0..1]
    #     P : float
    #         Power actually delivered (W)
    #     brownout : bool
    #         True if V <= V_cutoff or SoC depleted
    #     power_limited : bool
    #         True if demanded P > P_max
    #     """

    #     cap_Ah = self.capacity_Ah_at_T(T_env_C)
    #     self.V_ocv = self.ocv_from_soc(self.SoC, T_env_C)
    #     R = self.R_internal(T_env_C)

    #     # Quadratic coefficients: R I^2 - OCV I + P = 0
    #     D = self.V_ocv**2 - 4.0 * R * max(P_load_W, 0.0)
    #     power_limited = D < 0.0

    #     if power_limited:
    #         # The requested power is too high; clamp to P_max
    #         I = self.V_ocv / (2.0 * R)
    #         Vt = self.V_ocv - I * R             # = OCV/2
    #         P_deliv = max(Vt * I, 0.0)          # = OCV^2 / (4R)
    #     else:
    #         # Physically valid (smaller) root of the quadratic
    #         I = (self.V_ocv - np.sqrt(D)) / (2.0 * R)
    #         Vt = self.V_ocv - I * R
    #         P_deliv = P_load_W

    #     # Energy and charge integration
    #     self.Wh_drawn += (P_deliv * dt_s) / 3600.0               # Wh
    #     dAh = (I * dt_s) / 3600.0                                # Ah
    #     if cap_Ah > 0:
    #         self.SoC = max(0.0, self.SoC - dAh / cap_Ah)         # normalize by available capacity

    #     self.V_term = Vt
    #     self.I_peak = max(self.I_peak, I)
    #     brownout = (self.V_term <= self.V_cutoff) or (self.SoC <= 0.0)

    #     # Log
    #     t_next = (self.hist.t[-1] + dt_s) if self.hist.t else 0.0
    #     self.hist.t.append(t_next)
    #     self.hist.V.append(self.V_term)
    #     self.hist.I.append(I)
    #     self.hist.SoC.append(self.SoC)
    #     self.hist.T.append(T_env_C)
    #     self.hist.P.append(P_deliv)
    #     self.hist.power_limited.append(power_limited)
    #     self.hist.brownout.append(brownout)

    #     return {"V": self.V_term, "I": I, "SoC": self.SoC,
    #             "P": P_deliv, "brownout": brownout,
    #             "power_limited": power_limited}

    def is_dead(self) -> bool:
        """
        Check if battery is dead (brownout).
        """
        # check brownout
        return (self.V_term <= self.V_cutoff) or (self.SoC <= 0.0)

    # --------- One integration step with thermal update ----------
    def step(self,
             P_load_W: float,
             dt_s: float,
             T_env_C: float,
             V_rel_ms: float,
             rho_air: float) -> Dict[str, float]:
        """
        Args:
            P_load_W: Demanded load power (W). Negative values treated as 0.
            dt_s: Time step (s)
            T_env_C: Environment temperature (°C)
            V_rel_ms: Relative air velocity for convection (m/s)
            rho_air: Air density (kg/m³), default is sea level (1.225 kg/m³)
        
        Returns:
            dict with keys:
            V : float
                Terminal voltage (V)
            I : float
                Discharge current (A)
            SoC : float
                State of charge [0..1]
            P : float
                Power actually delivered (W)
            brownout : bool
                True if V <= V_cutoff or SoC depleted
            power_limited : bool
                True if demanded P > P_max
            T_C : float
                Cell temperature (°C)
        
        Advance the battery by dt_s with a demanded load power P_load_W and an
        environment at T_env_C. Convection scales with V_rel_ms and rho_air.

        ELECTRICAL:
          V_term = OCV(SoC) - I R(T_cell)
          P_load = V_term * I   -> solve quadratic for I
          SoC_{k+1} = SoC_k - (I*dt)/Q(T_cell)

        THERMAL:
          Q_gen = I^2 R(T_cell) + I * T_K * dOCVdT   [W]
          h = h_still + k_forced*(rho/rho0)^0.5 * V_rel^0.8
          Q_conv = h A (Tcell - Tenv)
          Q_rad  = eps σ A (T_K^4 - Tenv_K^4)
          Q_cond = G_cond (Tcell - Tstruct)
          Tcell_{k+1} = Tcell_k + (Q_gen - Q_conv - Q_rad - Q_cond) * dt / C_th
        """
        # Use cell temperature for temperature-dependent effects
        cap_Ah = self.capacity_Ah_at_T(self.T_cell_C)
        R = self.R_internal(self.T_cell_C)
        self.V_ocv = self.ocv_from_soc(self.SoC)

        # Solve power quadratic
        # Equations are derived from P = V_term * I = (OCV - I*R) * I
        P_dem = max(0.0, P_load_W)
        D = self.V_ocv**2 - 4.0 * R * P_dem
        power_limited = D < 0.0
        if power_limited:
            I = self.V_ocv / (2.0 * R)
            Vt = self.V_ocv - I * R
            P_deliv = Vt * I
        else:
            I = (self.V_ocv - np.sqrt(D)) / (2.0 * R)
            Vt = self.V_ocv - I * R
            P_deliv = P_dem

        # Coulomb counting
        dAh = (I * dt_s) / 3600.0
        if cap_Ah > 0:
            self.SoC = max(0.0, self.SoC - dAh / cap_Ah)

        # ---------- Thermal update ----------
        Tcell_K = self.T_cell_C + 273.15
        Tenv_K  = T_env_C     + 273.15
        # Joule + entropic (can set dOCVdT=0 to disable)
        Q_gen = I*I*R + I * Tcell_K * self.dOCVdT_V_per_K
        # Losses
        h = self._h_conv(V_rel_ms, rho_air)
        Q_conv = h * self.area_m2 * (Tcell_K - Tenv_K)
        Q_rad  = self.emissivity * SIGMA * self.area_m2 * (Tcell_K**4 - Tenv_K**4)
        Q_cond = self.G_cond_W_per_K * (self.T_cell_C - self.T_struct_C)
        # Integrate temperature
        dT = (Q_gen - Q_conv - Q_rad - Q_cond) * dt_s / self.C_th_J_per_K
        self.T_cell_C = float(np.clip(self.T_cell_C + dT, -50.0, 85.0))  # keep sane

        # Finish electrical bookkeeping
        self.V_term = Vt
        self.I_peak = max(self.I_peak, I)
        self.Wh_drawn += (P_deliv * dt_s) / 3600.0
        brownout = (self.V_term <= self.V_cutoff) or (self.SoC <= 0.0)

        # Log
        t_next = (self.hist.t[-1] + dt_s) if self.hist.t else 0.0
        self.hist.t.append(t_next)
        self.hist.V.append(self.V_term)
        self.hist.I.append(I)
        self.hist.SoC.append(self.SoC)
        self.hist.T.append(self.T_cell_C)
        self.hist.P.append(P_deliv)
        self.hist.power_limited.append(power_limited)
        self.hist.brownout.append(brownout)

        return {"V": self.V_term, "I": I, "SoC": self.SoC,
                "P": P_deliv, "brownout": brownout,
                "power_limited": power_limited, "T_C": self.T_cell_C}



