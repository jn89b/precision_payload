from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np
import json, gzip
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
# ---------- Design & Config Snapshots ----------

@dataclass
class DesignParams:
    mass_kg: float = 0.05
    Cd: float = 0.4
    S_min: float = 0.001
    S_max: float = 0.002
    emissivity: float = 0.3
    specific_heat_J_per_kgK: float = 600.0
    thermal_conductivity_W_per_mK: float = 0.1
    # geometry (if relevant to post-analysis)
    total_surface_area_m2: float = 0.008
    diameter_m: float = 0.058
    length_m: float = 0.068

@dataclass
class ControlParams:
    glide_ratio_min: float = 0.1
    glide_ratio_max: float = 0.2
    s_control_min: float = 0.002
    s_control_max: float = 0.003
    u_psi_min_rad: float = -np.deg2rad(10)
    u_psi_max_rad: float =  np.deg2rad(10)
    tau_psi_s: float = 0.01
    tau_gr_s: float = 0.2
    tau_s_s: float = 0.4
    control_frequency_hz: float = 20.0  # 1/dt_cmd if different from sim dt

@dataclass
class NoiseFailureParams:
    # measurement noise (1-sigma)
    gps_xy_sigma_m: float = 2.0
    gps_z_sigma_m: float = 5.0
    heading_sigma_deg: float = 2.0
    # command/telemetry
    cmd_drop_p: float = 0.01
    burst_drop_p: float = 0.001
    burst_len_s_min: float = 0.2
    burst_len_s_max: float = 1.0
    # actuator faults
    stiction_p_per_s: float = 0.003
    stiction_hold_min_s: float = 0.1
    stiction_hold_max_s: float = 0.3
    hard_fail_p_per_mission: float = 0.002

@dataclass
class PowerParams:
    voltage_cutoff_V: float = 3.3

@dataclass
class Meta:
    trial_id: int = -1
    seed: int = 0
    dt_s: float = 0.25
    base_module: Optional[str] = None
    pred_wind_file: Optional[str] = None
    actual_wind_file: Optional[str] = None
    code_version: Optional[str] = None  # git SHA/tag if you have it
    run_dir: Optional[str] = None

# ---------- Time Series ----------

@dataclass
class TimeSeries:
    t_s: List[float] = field(default_factory=list)

    # Kinematics
    x_m: List[float] = field(default_factory=list)
    y_m: List[float] = field(default_factory=list)
    z_m: List[float] = field(default_factory=list)
    psi_rad: List[float] = field(default_factory=list)
    vx_mps: List[float] = field(default_factory=list)
    vy_mps: List[float] = field(default_factory=list)
    vz_mps: List[float] = field(default_factory=list)

    # Winds
    wind_x_mps: List[float] = field(default_factory=list)
    wind_y_mps: List[float] = field(default_factory=list)
    wind_z_mps: List[float] = field(default_factory=list)
    gust_x_mps: List[float] = field(default_factory=list)
    gust_y_mps: List[float] = field(default_factory=list)
    gust_z_mps: List[float] = field(default_factory=list)

    # Commands (actual after filter)
    psi_cmd_rad: List[float] = field(default_factory=list)
    gr_cmd: List[float] = field(default_factory=list)
    S_cmd_m2: List[float] = field(default_factory=list)

    # Actuator state / duty proxy
    servo_effort: List[float] = field(default_factory=list)  # 0..1
    servo_active: List[bool] = field(default_factory=list)

    # Battery / payload
    batt_soc: List[float] = field(default_factory=list)
    batt_V: List[float] = field(default_factory=list)
    payload_P_total_W: List[float] = field(default_factory=list)
    payload_P_gps_W: List[float] = field(default_factory=list)
    payload_P_servo_W: List[float] = field(default_factory=list)
    payload_P_radio_W: List[float] = field(default_factory=list)

    # Thermal / aero
    accel_mps2: List[float] = field(default_factory=list)
    drag_force_N: List[float] = field(default_factory=list)
    Cd: List[float] = field(default_factory=list)
    q_dot_W_per_m2: List[float] = field(default_factory=list)
    T_surface_K: List[float] = field(default_factory=list)
    T_stag_K: List[float] = field(default_factory=list)
    Mach: List[float] = field(default_factory=list)
    Re: List[float] = field(default_factory=list)
    Kn: List[float] = field(default_factory=list)
    rho_kgpm3: List[float] = field(default_factory=list)
    T_air_K: List[float] = field(default_factory=list)
    mfp_m: List[float] = field(default_factory=list)
    g_mps2: List[float] = field(default_factory=list)
    q_solar_Wpm2: List[float] = field(default_factory=list)
    q_earthir_Wpm2: List[float] = field(default_factory=list)
    q_conv_Wpm2: List[float] = field(default_factory=list)
    q_rad_cooling_Wpm2: List[float] = field(default_factory=list)
    q_shock_Wpm2: List[float] = field(default_factory=list)
    q_rad_space_Wpm2: List[float] = field(default_factory=list)
    q_rad_atm_cooling_Wpm2: List[float] = field(default_factory=list)
    q_rad_atm_heating_Wpm2: List[float] = field(default_factory=list)
    T_interior_K: List[float] = field(default_factory=list)

    # Guidance references (optional)
    carp_x_m: List[float] = field(default_factory=list)
    carp_y_m: List[float] = field(default_factory=list)
    carp_z_m: List[float] = field(default_factory=list)
    carp_heading_rad: List[float] = field(default_factory=list)

# ---------- Events / Tallies ----------

@dataclass
class EventTallies:
    cmd_drops: int = 0
    burst_drop_seconds: float = 0.0
    stiction_events: int = 0
    hard_fail: bool = False
    brownout: bool = False
    cutoff_triggered: bool = False

# ---------- Summary / Outcomes ----------

@dataclass
class Summary:
    distance_to_goal_m: Optional[float] = None
    landing_time_s: Optional[float] = None
    flight_time_s: Optional[float] = None
    path_length_2d_m: Optional[float] = None

    energy_Wh: Optional[float] = None
    energy_servo_Wh: Optional[float] = None
    energy_radio_Wh: Optional[float] = None
    energy_gps_Wh: Optional[float] = None

    final_soc: Optional[float] = None
    min_voltage_V: Optional[float] = None
    max_surface_temp_K: Optional[float] = None
    max_stagnation_temp_K: Optional[float] = None
    max_heating_Wpm2: Optional[float] = None

    mission_score: Optional[float] = None
    success: Optional[bool] = None
    failure_reason: str = ""

# ---------- Top-level container ----------

@dataclass
class MonteCarloInformation:
    meta: Meta = field(default_factory=Meta)
    design: DesignParams = field(default_factory=DesignParams)
    control: ControlParams = field(default_factory=ControlParams)
    noise_fail: NoiseFailureParams = field(default_factory=NoiseFailureParams)
    power: PowerParams = field(default_factory=PowerParams)

    ts: TimeSeries = field(default_factory=TimeSeries)
    events: EventTallies = field(default_factory=EventTallies)
    summary: Summary = field(default_factory=Summary)

    # ---- helpers ----
    def record_step(self, **kwargs):
        """
        Append a timestep worth of data.
        Usage: info.record_step(t_s=t, x_m=x, y_m=y, ... )
        Only keys that match TimeSeries fields will be appended.
        """
        for k, v in kwargs.items():
            if hasattr(self.ts, k):
                getattr(self.ts, k).append(v)

    def finalize(self, goal_xy=(0.0, 0.0)):
        """Compute summary scalars from the time series."""
        ts = self.ts
        if ts.x_m and ts.y_m:
            dx = ts.x_m[-1] - goal_xy[0]
            dy = ts.y_m[-1] - goal_xy[1]
            self.summary.distance_to_goal_m = float(np.hypot(dx, dy))
        if ts.t_s:
            self.summary.flight_time_s = float(ts.t_s[-1])
            self.summary.landing_time_s = float(ts.t_s[-1])
        # path length
        if len(ts.x_m) > 1:
            x = np.array(ts.x_m); y = np.array(ts.y_m)
            self.summary.path_length_2d_m = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        # energies (approximate trapezoid integral)
        if ts.payload_P_total_W and ts.t_s:
            t = np.array(ts.t_s)
            def trapz(p): 
                p = np.array(p, dtype=float)
                return float(np.trapz(p, t) / 3600.0)  # Wh
            self.summary.energy_Wh = trapz(ts.payload_P_total_W)
            if ts.payload_P_servo_W: self.summary.energy_servo_Wh = trapz(ts.payload_P_servo_W)
            if ts.payload_P_radio_W: self.summary.energy_radio_Wh = trapz(ts.payload_P_radio_W)
            if ts.payload_P_gps_W:   self.summary.energy_gps_Wh   = trapz(ts.payload_P_gps_W)
        # battery/thermal extrema
        if ts.batt_soc: self.summary.final_soc = float(ts.batt_soc[-1])
        if ts.batt_V:   self.summary.min_voltage_V = float(np.min(ts.batt_V))
        if ts.T_surface_K: self.summary.max_surface_temp_K = float(np.max(ts.T_surface_K))
        if ts.T_stag_K:    self.summary.max_stagnation_temp_K = float(np.max(ts.T_stag_K))
        if ts.q_dot_W_per_m2: self.summary.max_heating_Wpm2 = float(np.max(ts.q_dot_W_per_m2))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_info_skeleton(
    trial_id:int,
    seed:int,
    dt_s:float,
    pred_wind_file:str,
    actual_wind_file:str,
    run_dir:str,
    design_kwargs:dict,
    control_kwargs:dict,
    noise_kwargs:dict,
    power_kwargs:dict
):
    info = MonteCarloInformation()
    info.meta.trial_id      = trial_id
    info.meta.seed          = seed
    info.meta.dt_s          = dt_s
    info.meta.pred_wind_file   = pred_wind_file
    info.meta.actual_wind_file = actual_wind_file
    info.meta.run_dir       = run_dir

    # snapshot configs at start
    info.design = replace(info.design, **design_kwargs)
    info.control = replace(info.control, **control_kwargs)
    info.noise_fail = replace(info.noise_fail, **noise_kwargs)
    info.power = replace(info.power, **power_kwargs)
    return info


def save_info(info: "MonteCarloInformation", out_dir: Path, fname_stem: str):
    """
    Save: (1) compact JSON.gz for metadata + scalars, (2) NPZ for big time-series.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split ts vs non-ts for efficient I/O
    payload = asdict(info)
    ts = payload.pop("ts")          # dict of lists
    # 1) JSON.gz for the light part
    with gzip.open(out_dir / f"{fname_stem}.json.gz", "wt") as f:
        json.dump(payload, f, separators=(",", ":"))

    # 2) NPZ for the heavy time-series (keeps types compact)
    # Convert lists to numpy arrays
    ts_np = {k: np.asarray(v) for k, v in ts.items()}
    np.savez_compressed(out_dir / f"{fname_stem}_ts.npz", **ts_np)
    
def save_montecarlo_bundle(run_dir: Path, info) -> None:
    """
    Save a rich bundle per run:
      - summary.json (nested dataclass dump)
      - timeseries.csv (wide table)
      - timeseries.npz (compressed numpy)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(asdict(info), f, indent=2)

    # time series as CSV + NPZ
    ts = info.ts
    # build a dict of 1D arrays in lockstep with t_s
    columns: Dict[str, Any] = {k: v for k, v in asdict(ts).items()}
    try:
        import pandas as pd
        df = pd.DataFrame(columns)
        df.to_csv(run_dir / "timeseries.csv", index=False)
    except Exception:
        # CSV-less envs: write a minimal CSV manually
        import csv
        keys = list(columns.keys())
        rows = zip(*[columns[k] for k in keys])
        with open(run_dir / "timeseries.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys); w.writerows(rows)

    #np.savez_compressed(run_dir / "timeseries.npz", **{k: _np(v) for k, v in columns.items()})