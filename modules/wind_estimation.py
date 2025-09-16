# === PRIOR + LOITER FUSION FOR WIND PROFILE =============================== #
# Inputs:
#   - prior CSV (alt, wN, wE)
#   - loiter samples (arrays of vgN, vgE, Va, psi, z) or per-ring pins
# Output:
#   - updated_wind(z) -> (wN, wE)
#   - updated_std(z)  -> (sigmaN, sigmaE)   # useful for Monte-Carlo

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Callable, List, Dict
import math

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# --------------------
# Helpers: smooth interpolators
# --------------------
def _make_interp(z: np.ndarray, y: np.ndarray):
    z = np.asarray(z, float); y = np.asarray(y, float)
    if HAS_SCIPY and len(z) >= 2:
        return PchipInterpolator(z, y, extrapolate=True), "pchip"
    # fallback linear clamp
    def f(x):
        x = np.atleast_1d(x)
        yy = np.interp(x, z, y, left=y[0], right=y[-1])
        return yy if yy.size > 1 else float(yy)
    return f, "linear-clamped"

# --------------------
# 1) Load PRIOR profile from CSV  (alt, wN, wE)  -> prior_wind(z), prior_std(z)
# --------------------
@dataclass
class PriorConfig:
    csv_path: str
    prior_sigma_floor: float = 1.0   # m/s floor on prior std per component (avoid overconfidence)
    prior_sigma_scale: float = 1.0   # multiply residual-based std if you provide one

def build_prior_from_csv(cfg: PriorConfig):
    df = pd.read_csv(cfg.csv_path, header=None, names=["z","wN","wE"]).dropna().sort_values("z")
    z = df["z"].to_numpy(float)
    wN = df["wN"].to_numpy(float)
    wE = df["wE"].to_numpy(float)

    # (optional) estimate rough variability vs z as prior std; else use constant floor
    # Here we compute a tiny moving std as a heuristic; you can replace with known model errors.
    def _movstd(y, win=7):
        y = np.asarray(y)
        k = max(3, min(win, len(y)))
        pad = k//2
        ypad = np.r_[y[0]*np.ones(pad), y, y[-1]*np.ones(pad)]
        s = np.array([np.std(ypad[i:i+k]) for i in range(len(y))])
        return s
    sigN_raw = _movstd(wN)
    sigE_raw = _movstd(wE)
    sigN = np.maximum(cfg.prior_sigma_floor, cfg.prior_sigma_scale*sigN_raw)
    sigE = np.maximum(cfg.prior_sigma_floor, cfg.prior_sigma_scale*sigE_raw)

    fN, _ = _make_interp(z, wN)
    fE, _ = _make_interp(z, wE)
    fSN, _ = _make_interp(z, sigN)
    fSE, _ = _make_interp(z, sigE)

    def prior_wind(zz: float) -> Tuple[float,float]:
        return float(fN(zz)), float(fE(zz))

    def prior_std(zz: float) -> Tuple[float,float]:
        return float(fSN(zz)), float(fSE(zz))

    return prior_wind, prior_std, dict(z=z, wN=wN, wE=wE, sigN=sigN, sigE=sigE)

# --------------------
# 2) Convert loiter samples -> per-altitude wind "pins"
#    Each pin: (z_i, wN_i, wE_i, R_i)  where R_i is 2x2 meas covariance
# --------------------
def estimate_wind_pin_from_ring(vgN: np.ndarray, vgE: np.ndarray, Va: np.ndarray, psi: np.ndarray,
                                z: np.ndarray) -> Dict:
    """Heading-only LS; attitude-comp if you have R_b2n available."""
    # w = vg - Va*[cos psi, sin psi]
    bN = vgN - Va*np.cos(psi)
    bE = vgE - Va*np.sin(psi)
    wN_i = float(np.mean(bN))
    wE_i = float(np.mean(bE))
    # residual covariance -> measurement noise
    rN = bN - wN_i
    rE = bE - wE_i
    R11 = float(np.var(rN, ddof=1))
    R22 = float(np.var(rE, ddof=1))
    R12 = float(np.cov(rN, rE, ddof=1)[0,1]) if len(bN) > 2 else 0.0
    R = np.array([[R11, R12],[R12, R22]])/max(1, len(bN))  # average reduces noise
    return dict(z=float(np.mean(z)), wN=wN_i, wE=wE_i, R=R)

# If you fly multiple rings at different altitudes, just call the function for each and collect pins.
# pins = [estimate_wind_pin_from_ring(...), estimate_wind_pin_from_ring(...), ...]

# --------------------
# 3) Fuse prior + pins with an altitude-bin Kalman filter
# --------------------
@dataclass
class FuseConfig:
    z_min: float
    z_max: float
    dz: float = 25.0      # bin size
    process_sigma: float = 0.2  # m/s per step random walk in altitude (smoothness prior)

def fuse_prior_with_pins(prior_wind: Callable[[float],Tuple[float,float]],
                         prior_std: Callable[[float],Tuple[float,float]],
                         pins: List[Dict],
                         cfg: FuseConfig) -> Tuple[Callable[[float],Tuple[float,float]],
                                                  Callable[[float],Tuple[float,float]],
                                                  Dict]:
    # Grid
    z_grid = np.arange(cfg.z_min, cfg.z_max + 1e-6, cfg.dz)

    # State mean and covariance per bin, independent for N and E (2 scalar filters)
    mN = np.array([prior_wind(z)[0] for z in z_grid])
    mE = np.array([prior_wind(z)[1] for z in z_grid])
    sN = np.array([prior_std(z)[0] for z in z_grid])  # std
    sE = np.array([prior_std(z)[1] for z in z_grid])
    Pn = (sN**2).copy()
    Pe = (sE**2).copy()

    Q = (cfg.process_sigma**2)  # process variance per “step” (encourages smoothness)

    # Sort pins by altitude (not required, but nice)
    pins = sorted(pins, key=lambda d: d["z"])

    for pin in pins:
        zi = pin["z"]; wN_i = pin["wN"]; wE_i = pin["wE"]; R = pin["R"]
        # Interpolate pin onto nearest two bins with linear weights
        if zi <= z_grid[0]:
            idxs = [0]; wts = [1.0]
        elif zi >= z_grid[-1]:
            idxs = [len(z_grid)-1]; wts = [1.0]
        else:
            j = int((zi - z_grid[0])//cfg.dz)
            z0 = z_grid[j]; z1 = z_grid[j+1]
            t = (zi - z0)/(z1 - z0 + 1e-9)
            idxs = [j, j+1]; wts = [1-t, t]

        # Do two scalar KF updates for N and E using weighted split of the same pin
        # (measurement covariance gets divided by weight^2 to keep information consistent)
        for idx, wt in zip(idxs, wts):
            if wt < 1e-6: 
                continue
            # Predict: (altitude-order) add a small process noise to promote smoothing
            Pn[idx] += Q
            Pe[idx] += Q

            # Measurement models (scalar each); use pin’s component variance
            Rn = float(R[0,0]) / (wt**2 + 1e-12)
            Re = float(R[1,1]) / (wt**2 + 1e-12)

            # Kalman gains
            Kn = Pn[idx] / (Pn[idx] + Rn + 1e-12)
            Ke = Pe[idx] / (Pe[idx] + Re + 1e-12)

            # Update means
            mN[idx] = mN[idx] + Kn * (wN_i - mN[idx])
            mE[idx] = mE[idx] + Ke * (wE_i - mE[idx])

            # Update covariances
            Pn[idx] = (1 - Kn) * Pn[idx]
            Pe[idx] = (1 - Ke) * Pe[idx]

    # Smooth pass (optional): apply a mild diffusion to neighboring bins
    # to avoid jagged profiles with sparse pins
    for _ in range(2):
        mN = 0.25*np.roll(mN,1) + 0.5*mN + 0.25*np.roll(mN,-1)
        mE = 0.25*np.roll(mE,1) + 0.5*mE + 0.25*np.roll(mE,-1)

    # Build interpolators for the fused profile + std
    fN, _ = _make_interp(z_grid, mN)
    fE, _ = _make_interp(z_grid, mE)
    fSN, _ = _make_interp(z_grid, np.sqrt(Pn))
    fSE, _ = _make_interp(z_grid, np.sqrt(Pe))

    def updated_wind(z: float) -> Tuple[float,float]:
        return float(fN(z)), float(fE(z))

    def updated_std(z: float) -> Tuple[float,float]:
        return float(fSN(z)), float(fSE(z))

    debug = dict(z_grid=z_grid, mN=mN, mE=mE, Pn=Pn, Pe=Pe)
    return updated_wind, updated_std, debug
