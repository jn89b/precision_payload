# wind_pipeline.py
# Pipeline:
# 1) Load prior wind profile from CSV (alt[m], wN[m/s], wE[m/s])
# 2) Build loiter-based wind pins from (A) real logs or (B) simulated rings
# 3) Fuse prior + pins into updated profile (mean + std vs altitude)
# 4) Compute CARP and simulate parafoil descent using updated profile
# 5) Monte-Carlo with altitude-varying uncertainty -> dispersion ellipse + region box
#
# Plots use map-like coordinates: x = East [m], y = North [m]

import os, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, List, Dict

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    
# --- add near your helpers ---
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct, WhiteKernel
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


@dataclass
class PinsSmoothCfg:
    z_min: float
    z_max: float
    dz: float = 25.0
    process_sigma_mps: float = 0.4   # per altitude-bin step (smoothness)
    sigma_floor_mps: float = 0.6     # floor for measurement/forecast std

def _assign_to_bins(z_grid: np.ndarray, zi: float):
    """Return (idxs, weights) for linear split of a measurement at zi to nearest two bins."""
    if zi <= z_grid[0]:
        return [0], [1.0]
    if zi >= z_grid[-1]:
        return [len(z_grid)-1], [1.0]
    j = int((zi - z_grid[0]) // (z_grid[1] - z_grid[0]))
    z0, z1 = z_grid[j], z_grid[j+1]
    t = (zi - z0) / max(1e-9, (z1 - z0))
    return [j, j+1], [1.0 - t, t]

def _rw_kf_smoother(z_grid: np.ndarray, meas: List[Tuple[float,float,float]],  # (z, value, var)
                    q: float, p0: float = 1e4):
    """
    Random-walk 1D KF + RTS smoother over altitude grid.
    meas: list of (z, value, variance) for a single component.
    q: process variance per step (m/s)^2.
    """
    K = len(z_grid)
    m_fwd = np.zeros(K); P_fwd = np.full(K, p0)
    # Collect measurements per-bin (allow multiple)
    bin_meas = [[] for _ in range(K)]  # each: list of (H, y, R)
    for z, y, R in meas:
        idxs, wts = _assign_to_bins(z_grid, z)
        for idx, wt in zip(idxs, wts):
            if wt < 1e-8: continue
            H = wt
            bin_meas[idx].append((H, y, R))

    # Forward filter
    for k in range(K):
        # predict
        if k > 0:
            m_fwd[k] = m_fwd[k-1]
            P_fwd[k] = P_fwd[k-1] + q
        # correct with all measurements at this bin (sequentially)
        for (H, y, R) in bin_meas[k]:
            S = H*P_fwd[k]*H + R + 1e-12
            Kk = (P_fwd[k]*H) / S
            innov = y - H*m_fwd[k]
            m_fwd[k] += Kk * innov
            P_fwd[k] = (1.0 - Kk*H) * P_fwd[k]
        P_fwd[k] = max(P_fwd[k], 1e-8)

    # RTS smoother (backward)
    m_smooth = m_fwd.copy()
    P_smooth = P_fwd.copy()
    for k in range(K-2, -1, -1):
        P_pred = P_fwd[k] + q
        A = 1.0  # random-walk
        Ck = P_fwd[k] * A / max(1e-12, P_pred)  # smoother gain
        m_smooth[k] = m_fwd[k] + Ck*(m_smooth[k+1] - m_fwd[k])  # since x̂_pred = m_fwd[k]
        P_smooth[k] = P_fwd[k] + Ck*(P_smooth[k+1] - P_pred)*Ck
        P_smooth[k] = max(P_smooth[k], 1e-8)

    return m_smooth, P_smooth

def profile_from_pins_random_walk(pins: List[Dict], cfg: PinsSmoothCfg):
    """Pins-only wind profile via 1-D random-walk Kalman RTS smoother (N and E independently)."""
    z_grid = np.arange(cfg.z_min, cfg.z_max + 1e-9, cfg.dz)

    # Build measurement lists for N and E components
    measN, measE = [], []
    for p in pins:
        z = float(p["z"])
        # R is covariance of the ring mean; guard tiny values with a floor
        Rn = max(cfg.sigma_floor_mps**2, float(p["R"][0,0]))
        Re = max(cfg.sigma_floor_mps**2, float(p["R"][1,1]))
        measN.append((z, float(p["wN"]), Rn))
        measE.append((z, float(p["wE"]), Re))

    q = cfg.process_sigma_mps**2

    mN, PN = _rw_kf_smoother(z_grid, measN, q)
    mE, PE = _rw_kf_smoother(z_grid, measE, q)

    # Interpolators for arbitrary z (nearest/linear clamp)
    fN, _ = _make_interp(z_grid, mN)
    fE, _ = _make_interp(z_grid, mE)
    fSN, _ = _make_interp(z_grid, np.sqrt(np.maximum(PN, cfg.sigma_floor_mps**2)))
    fSE, _ = _make_interp(z_grid, np.sqrt(np.maximum(PE, cfg.sigma_floor_mps**2)))

    def wind(z: float): return float(fN(z)), float(fE(z))
    def std(z: float):  return float(fSN(z)), float(fSE(z))
    dbg = dict(z_grid=z_grid, mN=mN, mE=mE, PN=PN, PE=PE)
    return wind, std, dbg

@dataclass
class PinsGPCfg:
    """
    Tuning cheat-sheet

    length_scale_m: ↓ (200–300 m) → curve follows pins tightly; ↑ (600–1000 m) → smoother.
    nu: 1.5 rougher; 2.5 smoother.
    use_rq=True if you see both gradual shear + small wiggles.
    add_linear=True if there’s an obvious linear trend with small deviations.
    clamp_margin_m 50–150 m is reasonable for your 2–3 ring strategy near release.
    Keep sigma_floor_mps ~ 0.5–1.0 to avoid overconfidence.
    """
    length_scale_m: float = 400.0       # vertical correlation (~200–600 m typical)
    nu: float = 2.0                     # Matern smoothness: 1.5 or 2.5
    use_rq: bool = False                # add RationalQuadratic
    add_linear: bool = False            # add DotProduct (linear trend)
    sigma_floor_mps: float = 0.6        # floor on noise / posterior std
    clamp_margin_m: float = 50.0        # clamp outside [min,max]±margin

def profile_from_pins_gp(
    pins: List[Dict],
    z_min: float, z_max: float, dz: float,
    cfg: PinsGPCfg,
    prior = None,
    blend_width_m: float = 150.0,   # how quickly to taper to the prior outside the pins band
):
    # Gather data
    z  = np.array([p["z"]  for p in pins], float)
    yN = np.array([p["wN"] for p in pins], float)
    yE = np.array([p["wE"] for p in pins], float)
    sN = np.sqrt(np.maximum(cfg.sigma_floor_mps**2, np.array([p["R"][0,0] for p in pins], float)))
    sE = np.sqrt(np.maximum(cfg.sigma_floor_mps**2, np.array([p["R"][1,1] for p in pins], float)))

    # Normalize altitude (km) for nicer length-scales
    z0 = float(z.mean())
    Z  = ((z - z0) / 1000.0).reshape(-1,1)
    ls_km = cfg.length_scale_m / 1000.0

    # Fallback if sklearn not present or too few points
    if not HAS_SKLEARN or len(z) < 2:
        return profile_from_pins_random_walk(
            pins,
            PinsSmoothCfg(z_min=z_min, z_max=z_max, dz=dz,
                          process_sigma_mps=0.4, sigma_floor_mps=cfg.sigma_floor_mps)
        )

    # ------- Kernel (optionally add linear trend for better extrapolation) -------
    kern = Matern(length_scale=ls_km, nu=cfg.nu)
    if cfg.use_rq:
        kern = kern + RationalQuadratic(length_scale=ls_km, alpha=1.0)
    if cfg.add_linear:
        kern = kern + DotProduct(sigma_0=1.0)

    # ------- Mean function from PRIOR; fit GP on residuals -------
    if prior is not None:
        mN_tr = np.array([prior.wind(zz)[0] for zz in z])
        mE_tr = np.array([prior.wind(zz)[1] for zz in z])
    else:
        mN_tr = np.zeros_like(yN)
        mE_tr = np.zeros_like(yE)

    yN_res = yN - mN_tr
    yE_res = yE - mE_tr

    gpN = GaussianProcessRegressor(kernel=kern, alpha=(sN**2),
                                   normalize_y=True, n_restarts_optimizer=2, copy_X_train=True)
    gpE = GaussianProcessRegressor(kernel=kern, alpha=(sE**2),
                                   normalize_y=True, n_restarts_optimizer=2, copy_X_train=True)
    gpN.fit(Z, yN_res)
    gpE.fit(Z, yE_res)

    # Prediction grid
    z_grid = np.arange(z_min, z_max + 1e-9, dz)
    Zg = ((z_grid - z0)/1000.0).reshape(-1,1)

    rN, sN_post = gpN.predict(Zg, return_std=True)   # residuals
    rE, sE_post = gpE.predict(Zg, return_std=True)

    # Add back prior mean on the grid
    if prior is not None:
        mN_grid = np.array([prior.wind(zz)[0] for zz in z_grid])
        mE_grid = np.array([prior.wind(zz)[1] for zz in z_grid])
    else:
        mN_grid = 0.0
        mE_grid = 0.0

    muN = (mN_grid + rN).astype(float)
    muE = (mE_grid + rE).astype(float)

    # ------- Blend to the PRIOR outside the pins band (no nearest-pin clamping) -------
    lo = float(z.min()) - cfg.clamp_margin_m
    hi = float(z.max()) + cfg.clamp_margin_m
    if prior is not None:
        muN = _blend_to_prior(muN, np.asarray(mN_grid, float), z_grid, lo, hi, blend_width_m)
        muE = _blend_to_prior(muE, np.asarray(mE_grid, float), z_grid, lo, hi, blend_width_m)

    # Floor the posterior std so MC doesn’t get overconfident
    sN_post = np.maximum(cfg.sigma_floor_mps, sN_post.astype(float))
    sE_post = np.maximum(cfg.sigma_floor_mps, sE_post.astype(float))

    # Interpolators for arbitrary z
    fN, _  = _make_interp(z_grid, muN)
    fE, _  = _make_interp(z_grid, muE)
    fSN, _ = _make_interp(z_grid, sN_post)
    fSE, _ = _make_interp(z_grid, sE_post)

    def wind(zq: float): return float(fN(zq)), float(fE(zq))
    def std(zq: float):  return float(fSN(zq)), float(fSE(zq))
    dbg = dict(z_grid=z_grid, mN=muN, mE=muE, sN=sN_post, sE=sE_post,
               kernel_N=str(gpN.kernel_), kernel_E=str(gpE.kernel_),
               lo=lo, hi=hi, blend_width=blend_width_m)
    return wind, std, dbg


# CONFIG
# =========================
@dataclass
class Config:
    # --- Files ---
    prior_csv_path: str = "Balloon Wind Files/10_matlab_desc.csv"  # CSV: alt, wN, wE
    # If you have real loiter samples, point this to a CSV with columns:
    # time, vgN, vgE, Va, psi_rad, z, ring_id   (psi in radians; speeds m/s; altitude m AGL)
    loiter_samples_csv: str | None = None      # e.g., "/mnt/data/loiter_logs.csv"

    # --- Loiter simulation (used only if loiter_samples_csv is None) ---
    sim_loiter_alts_m: List[float] = None      # if None, auto around release altitude
    sim_loiter_radius_m: float = 250.0
    sim_loiter_Va_mps: float = 22.0
    sim_samples_per_circle: int = 240
    sim_meas_noise_mps: float = 0.4            # noise on vg components

    # --- Fusion grid ---
    z_min: float = 1500.0
    z_max: float = 6000.0
    dz: float = 20.0
    prior_sigma_floor_mps: float = 1.0         # conservative prior std per component
    process_sigma_mps: float = 0.2             # random-walk per bin (smoothness)

    # --- Parafoil / mission ---
    target_NE: Tuple[float,float] = (0.0, 0.0)
    z_release: float = 2000.0
    z_ground: float = 1500.0
    dt: float = 0.2

    # Air/parafoil model (physics)  # NEW
    # Glide-ratio model (constant GR)
    glide_ratio: float = 0.4          # level-flight GR = Va / Vsink  (e.g., 4:1)
    bank_sink_exponent: float = 1.0   # α in (1/cos(phi))^α; 0 = no extra sink in turns
    phi_max_deg: float = 35.0         # bank angle limit for turns
    Va_trim_mps: float = 9.0          # trimmed airspeed magnitude
    bank_kp: float = 2.0              # bank cmd [rad] per rad of heading error
    
    # Turn/controls  # NEW
    bank_kp: float = 2.0               # bank angle command [rad] per rad of heading error    # NEW
    phi_max_deg: float = 35.0          # bank angle hard limit                                # NEW

    # (DEPRECATE these two if you like; they’re unused in physics model)
    Va_forward: float = 9.0            # (legacy) parafoil fwd airspeed
    V_sink: float = 2.5                # (legacy) constant sink

    psi_dot_max: float = math.radians(15.0)
    L1_dist: float = 60.0
    loiter_ring_R_m: float = 150.0
    
    # --- Monte Carlo ---
    N_mc: int = 100
    turb_sigma_mps: float = 0.25               # per-step white noise gust during descent
    ellipse_nsig: float = 2.0                  # ~95% for Gaussian
    box_margin_m: float = 10.0
    start_position_noise_m: float = 40.0               # simulate GPS noise on release point
    position_noise_m: float = 5.0               # simulate GPS noise on landing point
    velocity_noise_mps: float = 0.4            # simulate GPS noise on landing point
    heading_noise_deg: float = 5.0           # simulate compass noise on landing point  

    # --- Output dir ---
    out_dir: str = "/mnt/data"

CFG = Config()

# =========================
# Utilities
# =========================
def _make_interp(z: np.ndarray, y: np.ndarray):
    """Return an interpolator f(z) with PCHIP if available, else linear clamp."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    if HAS_SCIPY and len(z) >= 2:
        return PchipInterpolator(z, y, extrapolate=True), "pchip"
    def f(x):
        x = np.atleast_1d(x)
        yy = np.interp(x, z, y, left=y[0], right=y[-1])
        return yy if yy.size > 1 else float(yy)
    return f, "linear-clamped"

def _blend_to_prior(mu_gp: np.ndarray, mu_prior: np.ndarray,
                    z_grid: np.ndarray, lo: float, hi: float, width: float):
    """
    Inside [lo,hi] => weight=1 (pure GP). Outside, fade to the prior over 'width' meters.
    """
    w_lo = np.clip((z_grid - (lo - width))/max(1e-6, width), 0.0, 1.0)
    w_hi = np.clip(((hi + width) - z_grid)/max(1e-6, width), 0.0, 1.0)
    w = np.minimum(w_lo, w_hi)   # 1 inside, smoothly → 0 outside
    return w*mu_gp + (1.0 - w)*mu_prior


# =========================
# 1) PRIOR from CSV
# =========================
@dataclass
class Prior:
    wind: Callable[[float], Tuple[float,float]]
    std: Callable[[float], Tuple[float,float]]
    z: np.ndarray
    wN: np.ndarray
    wE: np.ndarray
    sigN: np.ndarray
    sigE: np.ndarray
    kind: str

def build_prior_from_csv(csv_path: str, sigma_floor: float = 1.0) -> Prior:
    df = pd.read_csv(csv_path, header=None, names=["z","wN","wE"]).dropna().sort_values("z")
    z = df["z"].to_numpy(float)
    wN = df["wN"].to_numpy(float)
    wE = df["wE"].to_numpy(float)

    # Heuristic prior std ~ local variability, but never below floor
    def movstd(y, win=7):
        y = np.asarray(y)
        if len(y) < 3: return np.full_like(y, sigma_floor, dtype=float)
        k = max(3, min(win, len(y)))
        pad = k//2
        ypad = np.r_[np.full(pad, y[0]), y, np.full(pad, y[-1])]
        return np.array([np.std(ypad[i:i+k]) for i in range(len(y))])
    sigN = np.maximum(sigma_floor, movstd(wN))
    sigE = np.maximum(sigma_floor, movstd(wE))

    fN, kindN = _make_interp(z, wN)
    fE, kindE = _make_interp(z, wE)
    fSN, _ = _make_interp(z, sigN)
    fSE, _ = _make_interp(z, sigE)

    def prior_wind(zz: float): return float(fN(zz)), float(fE(zz))
    def prior_std(zz: float):  return float(fSN(zz)), float(fSE(zz))

    return Prior(prior_wind, prior_std, z, wN, wE, sigN, sigE, kindN if kindN==kindE else "mixed")

# =========================
# 2) Loiter → pins
# =========================
def estimate_wind_pin_from_ring(vgN: np.ndarray, vgE: np.ndarray, Va: np.ndarray, psi: np.ndarray,
                                z: np.ndarray) -> Dict:
    """Least-squares wind for one ring (heading-only model)."""
    bN = vgN - Va*np.cos(psi)
    bE = vgE - Va*np.sin(psi)
    wN_i = float(np.mean(bN)); wE_i = float(np.mean(bE))
    rN = bN - wN_i; rE = bE - wE_i
    # average reduces variance ~1/K
    R11 = float(np.var(rN, ddof=1))/max(1, len(bN))
    R22 = float(np.var(rE, ddof=1))/max(1, len(bE))
    R12 = float(np.cov(rN, rE, ddof=1)[0,1])/max(1, len(bN)) if len(bN) > 2 else 0.0
    R = np.array([[R11, R12],[R12, R22]])
    return dict(z=float(np.mean(z)), wN=wN_i, wE=wE_i, R=R)

def load_loiter_pins_from_csv(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    assert {"vgN","vgE","Va","psi_rad","z","ring_id"}.issubset(df.columns), \
        "loiter_samples_csv must have columns: vgN, vgE, Va, psi_rad, z, ring_id"
    pins = []
    for ring_id, grp in df.groupby("ring_id"):
        pins.append(estimate_wind_pin_from_ring(
            vgN=grp["vgN"].to_numpy(float),
            vgE=grp["vgE"].to_numpy(float),
            Va=grp["Va"].to_numpy(float),
            psi=grp["psi_rad"].to_numpy(float),
            z=grp["z"].to_numpy(float),
        ))
    return pins

def simulate_loiter_ring(alt_m: float, Va_mps: float, R_m: float, samples: int,
                         true_wind: Callable[[float],Tuple[float,float]], noise_mps: float=0.4) -> Dict:
    thetas = np.linspace(0, 2*np.pi, samples, endpoint=False)  # heading ~ track
    psi = thetas.copy()
    wN, wE = true_wind(alt_m)
    vgN_true = Va_mps*np.cos(psi) + wN
    vgE_true = Va_mps*np.sin(psi) + wE
    vgN = vgN_true + np.random.randn(samples)*noise_mps
    vgE = vgE_true + np.random.randn(samples)*noise_mps
    Va = np.full(samples, Va_mps)
    z  = np.full(samples, alt_m)
    return estimate_wind_pin_from_ring(vgN, vgE, Va, psi, z)

def simulate_loiter_pins(alts: List[float], Va: float, R: float, samples: int,
                         true_wind: Callable[[float],Tuple[float,float]], noise: float) -> List[Dict]:
    pins = []
    for z in alts:
        pins.append(simulate_loiter_ring(z, Va, R, samples, true_wind, noise))
    return pins

# =========================
# 3) Fuse prior + pins into updated profile
# =========================
@dataclass
class FuseCfg:
    z_min: float
    z_max: float
    dz: float = 25.0
    process_sigma: float = 0.2     # m/s per altitude-bin step

# =========================
# 4) Parafoil guidance + CARP (internals N,E; plots in E=x, N=y)
# =========================
def crab_heading_for_track(chi_d: float, wN: float, wE: float, Va: float):
    W_along = wN*np.cos(chi_d) + wE*np.sin(chi_d)
    W_cross = -wN*np.sin(chi_d) + wE*np.cos(chi_d)
    arg = np.clip(W_cross/max(Va,1e-6), -1.0, 1.0)
    delta = math.asin(arg)
    psi_cmd = chi_d + delta
    Vg = Va*math.cos(delta) + W_along
    return psi_cmd, Vg


def _sink_from_gr(Va: float, phi: float, GR: float, alpha: float) -> float:
    """
    Constant-GR sink model with optional bank penalty.
    Vsink(phi) = (Va / GR) * (1/cos(phi))**alpha
    """
    phi = float(np.clip(phi, -np.pi/2 + 1e-3, np.pi/2 - 1e-3))
    base = Va / max(1e-6, GR)
    bank_factor = (1.0 / math.cos(phi)) ** float(alpha)
    return base * bank_factor

def simulate_parafoil_drop(release_NE, wind_func, turb_sigma=0.0):
    """
    Parafoil with config-defined glide ratio:
    - Airspeed magnitude = CFG.Va_trim_mps
    - Sink rate Vsink = Va/GR, optionally increased in turns by (1/cos(phi))^alpha
    - Bank command from heading error (P-control), limited to phi_max_deg
    - Heading rate from coordinated-turn kinematics
    """
    pN, pE = float(release_NE[0]), float(release_NE[1])
    z = CFG.z_release
    psi = 0.0
    tgtN, tgtE = CFG.target_NE

    Va = CFG.Va_trim_mps
    phi_max = math.radians(CFG.phi_max_deg)
    x_start_noise = np.random.uniform(-CFG.start_position_noise_m, CFG.start_position_noise_m)
    y_start_noise = np.random.uniform(-CFG.start_position_noise_m, CFG.start_position_noise_m)
    pN += y_start_noise
    pE += x_start_noise
    
    Xs, Ys = [pE], [pN]  # x=E, y=N

    while z > CFG.z_ground:
        # L1 point
        vN, vE = (tgtN - pN), (tgtE - pE)
        dist = max(math.hypot(vN, vE), 1e-6)
        if dist > CFG.L1_dist:
            laN = pN + (CFG.L1_dist/dist) * vN
            laE = pE + (CFG.L1_dist/dist) * vE
        else:
            laN, laE = tgtN, tgtE
        chi_d = math.atan2(laE - pE, laN - pN)

        # Wind (with optional gust)
        wN, wE = wind_func(z)
        if turb_sigma > 0.0:
            wN += np.random.randn() * turb_sigma
            wE += np.random.randn() * turb_sigma

        # Desired air heading to track desired ground course
        psi_cmd, _ = crab_heading_for_track(chi_d, wN, wE, Va)

        # Bank command from heading error, saturate
        err = (psi_cmd - psi + np.pi) % (2*np.pi) - np.pi
        phi_cmd = np.clip(CFG.bank_kp * err, -phi_max, +phi_max)

        # Coordinated-turn heading dynamics
        psi_dot = (9.80665 * math.tan(phi_cmd)) / max(1e-3, Va)
        psi += psi_dot * CFG.dt

        # Air-relative velocity
        air_velocity_noise = CFG.velocity_noise_mps 
        vel_random_noise: float = np.random.uniform(-air_velocity_noise, air_velocity_noise)
        v_aN = Va * math.cos(psi)  + vel_random_noise
        v_aE = Va * math.sin(psi) + vel_random_noise

        # Ground velocity
        gN = v_aN + wN + np.random.uniform(-air_velocity_noise, air_velocity_noise)
        gE = v_aE + wE + np.random.uniform(-air_velocity_noise, air_velocity_noise)

        # Sink from config-defined GR (+ bank penalty)
        Vsink = _sink_from_gr(Va, phi_cmd, CFG.glide_ratio, CFG.bank_sink_exponent)

        # Integrate
        pN += gN * CFG.dt
        pE += gE * CFG.dt
        z  -= Vsink * CFG.dt

        Xs.append(pE); Ys.append(pN)

    miss = math.hypot(pN - tgtN, pE - tgtE)
    return np.array(Xs), np.array(Ys), miss


def carp_from_layer_wind(wN_layer, wE_layer, n_candidates=90):
    R = CFG.loiter_ring_R_m
    tgtN, tgtE = CFG.target_NE
    best_d, best_NE = 1e9, None
    for th in np.linspace(0, 2*np.pi, n_candidates, endpoint=False):
        relN = tgtN + R*np.cos(th)
        relE = tgtE + R*np.sin(th)
        wind_const = lambda z: (wN_layer, wE_layer)
        _, _, miss = simulate_parafoil_drop(np.array([relN, relE]), wind_const, turb_sigma=0.0)
        if miss < best_d:
            best_d, best_NE = miss, np.array([relN, relE])
    return best_NE, best_d


# =========================
# 5) RUN THE PIPELINE
# =========================
def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    # Prior
    prior = build_prior_from_csv(CFG.prior_csv_path, sigma_floor=CFG.prior_sigma_floor_mps)

    # Pins: load real logs or simulate
    if CFG.loiter_samples_csv:
        pins = load_loiter_pins_from_csv(CFG.loiter_samples_csv)
        pins_src = "real"
    else:
        # Sim rings: by default, around release altitude ±200 m
        if CFG.sim_loiter_alts_m is None:
            # CFG.sim_loiter_alts_m = [CFG.z_release - 200, CFG.z_release, CFG.z_release + 200]
            CFG.sim_loiter_alts_m = np.arange(CFG.z_release - 200, CFG.z_release + 401, 50).tolist()
        pins = simulate_loiter_pins(
            alts=CFG.sim_loiter_alts_m,
            Va=CFG.sim_loiter_Va_mps,
            R=CFG.sim_loiter_radius_m,
            samples=CFG.sim_samples_per_circle,
            true_wind=prior.wind,             # you could inject deviations here
            noise=CFG.sim_meas_noise_mps
        )
        pins_src = "simulated"

    # Fusion grid bounds if not given
    # Pins-only vertical model (no fusion with prior)
    zmin = CFG.z_min
    zmax = CFG.z_max
    updated_wind, updated_std, pdbg = profile_from_pins_gp(
        pins,
        z_min=CFG.z_min, z_max=CFG.z_max, dz=CFG.dz,
        cfg=PinsGPCfg(
            length_scale_m=400.0,   # 200–600 typical
            nu=1.5,                 # 1.5 rougher, 2.5 smoother
            use_rq=False,
            add_linear=True,        # << enables linear trend extrapolation
            sigma_floor_mps=0.6,
            clamp_margin_m=50.0
        ),
        prior=prior,                # << residual GP around your prior
        blend_width_m=150.0         # << smooth fade to the prior outside pins
    )
    # CARP using layer wind at release
    wN_layer, wE_layer = updated_wind(CFG.z_release)
    carp_NE, carp_err = carp_from_layer_wind(wN_layer, wE_layer)

    # Deterministic drop with FULL updated profile
    X_det, Y_det, miss_det = simulate_parafoil_drop(carp_NE, updated_wind, turb_sigma=0.0)

    # Monte-Carlo with altitude-dependent uncertainty
    impacts = []
    for _ in range(CFG.N_mc):
        def wind_mc(z):
            muN, muE = updated_wind(z)
            sN, sE = updated_std(z)
            return np.random.normal(muN, sN), np.random.normal(muE, sE)
        X, Y, _ = simulate_parafoil_drop(carp_NE, wind_mc, turb_sigma=CFG.turb_sigma_mps)
        impacts.append([X[-1], Y[-1]])  # (x=E, y=N)
    impacts = np.array(impacts)

    # Ellipse + region box (computed in x=E, y=N space)
    mean_xy = impacts.mean(axis=0)
    cov_xy = np.cov(impacts.T)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:,order]
    axes_lengths = 2*CFG.ellipse_nsig*np.sqrt(eigvals)
    angle_rad = math.atan2(eigvecs[1,0], eigvecs[0,0])

    theta = np.linspace(0, 2*np.pi, 240)
    ellipse_local = np.vstack([
        (axes_lengths[0]/2)*np.cos(theta),
        (axes_lengths[1]/2)*np.sin(theta)
    ])
    Rmat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                     [np.sin(angle_rad),  np.cos(angle_rad)]])
    ellipse_global = (Rmat @ ellipse_local) + mean_xy.reshape(2,1)
    xmin, ymin = ellipse_global[0].min(), ellipse_global[1].min()
    xmax, ymax = ellipse_global[0].max(), ellipse_global[1].max()
    xmin -= CFG.box_margin_m; ymin -= CFG.box_margin_m
    xmax += CFG.box_margin_m; ymax += CFG.box_margin_m
    box_w, box_h = xmax - xmin, ymax - ymin

    # ---------------- PLOTS ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.ravel()

    # (1) Prior + updated profile (wN)
    # z_plot = np.linspace(min(prior.z.min(), fdbg["z_grid"].min()),
    #                      max(prior.z.max(), fdbg["z_grid"].max()), 300)
    z_plot = np.linspace(zmin, zmax, 300)
    prior_wN = np.array([prior.wind(z)[0] for z in z_plot])
    post_wN  = np.array([updated_wind(z)[0] for z in z_plot])
    post_sN  = np.array([updated_std(z)[0]  for z in z_plot])
    axs[0].plot(prior_wN, z_plot, 'C0--', alpha=0.5, label='prior wN (ref)')
    axs[0].plot(post_wN,  z_plot, 'C0-',  label='pins-only wN')
    axs[0].fill_betweenx(z_plot, post_wN - 2*post_sN, post_wN + 2*post_sN, color='C0', alpha=0.12, label='±2σ')
    # axs[0].plot(post_wN, z_plot, 'C0-',  label='updated wN')
    #axs[0].fill_betweenx(z_plot, post_wN - 2*post_sN, post_wN + 2*post_sN, color='C0', alpha=0.15, label='±2σ')
    axs[0].invert_yaxis(); axs[0].grid(True)
    axs[0].set_xlabel("wN [m/s]"); axs[0].set_ylabel("Altitude [m]")
    axs[0].set_title(f"North wind profile — pins ({pins_src})")
    for p in pins:
        axs[0].plot([p["wN"]], [p["z"]], 'ko', ms=4)

    # (2) Prior + updated profile (wE)
    prior_wE = np.array([prior.wind(z)[1] for z in z_plot])
    post_wE  = np.array([updated_wind(z)[1] for z in z_plot])
    post_sE  = np.array([updated_std(z)[1]  for z in z_plot])
    axs[1].plot(prior_wE, z_plot, 'C1--', alpha=0.5, label='prior wE (ref)')
    axs[1].plot(post_wE,  z_plot, 'C1-',  label='pins-only wE')
    axs[1].fill_betweenx(z_plot, post_wE - 2*post_sE, post_wE + 2*post_sE, color='C1', alpha=0.12, label='±2σ')

    axs[1].plot(post_wE, z_plot, 'C1-',  label='updated wE')
    axs[1].invert_yaxis(); axs[1].grid(True)
    axs[1].set_xlabel("wE [m/s]"); axs[1].set_ylabel("Altitude [m]")
    axs[1].set_title("East wind profile")
    for p in pins:
        axs[1].plot([p["wE"]], [p["z"]], 'ko', ms=4)

    # (3) CARP + deterministic path (x=E, y=N)
    ring_t = np.linspace(0,2*np.pi,361)
    ring_x = CFG.loiter_ring_R_m*np.sin(ring_t)  # x=E
    ring_y = CFG.loiter_ring_R_m*np.cos(ring_t)  # y=N
    axs[2].plot(ring_x, ring_y, color='0.7', lw=1, label='loiter ring')
    tgtN, tgtE = CFG.target_NE
    axs[2].scatter([tgtE],[tgtN], marker='x', s=80, label="Target")
    axs[2].scatter([carp_NE[1]],[carp_NE[0]], s=40, label="CARP")
    axs[2].plot(X_det, Y_det, lw=1.5, label=f"Deterministic drop (miss={miss_det:.1f} m)")
    axs[2].axis('equal'); axs[2].grid(True)
    axs[2].set_xlabel("East [m]"); axs[2].set_ylabel("North [m]")
    axs[2].set_title("CARP with updated profile")
    axs[2].legend(loc="best")

    # (4) Dispersion + ellipse + region box (x=E, y=N)
    axs[3].scatter(impacts[:,0], impacts[:,1], s=6, alpha=0.4, label="Impacts")
    axs[3].scatter([tgtE],[tgtN], marker='x', s=80, label="Target")
    from matplotlib.patches import Ellipse, Rectangle
    axs[3].add_patch(Ellipse(xy=mean_xy, width=axes_lengths[0], height=axes_lengths[1],
                             angle=np.degrees(angle_rad), fill=False, lw=2, label="~95% ellipse"))
    axs[3].add_patch(Rectangle((xmin, ymin), box_w, box_h, fill=False, lw=2, linestyle="--", label="A* region box"))
    axs[3].axis('equal'); axs[3].grid(True)
    axs[3].set_xlabel("East [m]"); axs[3].set_ylabel("North [m]")
    axs[3].set_title(f"Impact dispersion (N={CFG.N_mc})")
    axs[3].legend(loc="best")

    for a in axs:
        a.grid(True); a.legend()
        
    #create a 3d plot
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    # ax.plot3D(X_det, Y_det, np.linspace(CFG.z_release, CFG.z_ground, len(X_det)), 'gray', label='Deterministic Path')
    # ax.scatter(impacts[:,0], impacts[:,1], np.full(len(impacts), CFG.z_ground), c='r', s=10, alpha=0.5, label='Impact Points')
    # ax.set_xlabel('East [m]')
    # ax.set_ylabel('North [m]')
    # ax.set_zlabel('Altitude [m]')
    # ax.set_title('3D Parafoil Drop Simulation')
    # ax.legend()

    plt.tight_layout()
    fig_path = os.path.join(CFG.out_dir, "wind_pipeline_results.png")
    plt.savefig(fig_path, dpi=150)

if __name__ == "__main__":
    plt.close('all')
    for i in range(10):
        # randomize seed for each run
        np.random.seed()
        main()
