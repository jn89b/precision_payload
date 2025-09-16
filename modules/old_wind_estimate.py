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

# =========================
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
    z_min: float = 0.0
    z_max: float = 1200.0
    dz: float = 25.0
    prior_sigma_floor_mps: float = 1.0         # conservative prior std per component
    process_sigma_mps: float = 0.2             # random-walk per bin (smoothness)

    # --- Parafoil / mission ---
    target_NE: Tuple[float,float] = (0.0, 0.0) # internal (N,E)
    z_release: float = 1000.0                  # m AGL
    z_ground: float = 0.0
    dt: float = 0.2
    Va_forward: float = 9.0                    # parafoil fwd airspeed
    V_sink: float = 2.5                        # parafoil sink rate
    psi_dot_max: float = math.radians(15.0)    # max heading rate
    L1_dist: float = 60.0
    loiter_ring_R_m: float = 250.0             # for CARP search

    # --- Monte Carlo ---
    N_mc: int = 100
    turb_sigma_mps: float = 0.25               # per-step white noise gust during descent
    ellipse_nsig: float = 2.0                  # ~95% for Gaussian
    box_margin_m: float = 10.0

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

def fuse_prior_with_pins(prior: Prior, pins: List[Dict], cfg: FuseCfg):
    z_grid = np.arange(cfg.z_min, cfg.z_max + 1e-6, cfg.dz)

    mN = np.array([prior.wind(z)[0] for z in z_grid])
    mE = np.array([prior.wind(z)[1] for z in z_grid])
    sN = np.array([prior.std(z)[0]  for z in z_grid])
    sE = np.array([prior.std(z)[1]  for z in z_grid])
    Pn = (sN**2).copy()
    Pe = (sE**2).copy()
    Q  = cfg.process_sigma**2

    pins = sorted(pins, key=lambda d: d["z"])

    for pin in pins:
        zi, wN_i, wE_i, R = pin["z"], pin["wN"], pin["wE"], pin["R"]
        # Interpolate pin to nearest two bins
        if zi <= z_grid[0]:
            idxs, wts = [0], [1.0]
        elif zi >= z_grid[-1]:
            idxs, wts = [len(z_grid)-1], [1.0]
        else:
            j = int((zi - z_grid[0])//cfg.dz)
            z0, z1 = z_grid[j], z_grid[j+1]
            t = (zi - z0)/(z1 - z0 + 1e-9)
            idxs, wts = [j, j+1], [1-t, t]

        for idx, wt in zip(idxs, wts):
            if wt < 1e-6: continue
            # predict
            Pn[idx] += Q; Pe[idx] += Q
            # measurement variance per component (split by weight)
            Rn = float(R[0,0]) / (wt**2 + 1e-12)
            Re = float(R[1,1]) / (wt**2 + 1e-12)
            # Kalman update (scalar per component)
            Kn = Pn[idx] / (Pn[idx] + Rn + 1e-12)
            Ke = Pe[idx] / (Pe[idx] + Re + 1e-12)
            mN[idx] = mN[idx] + Kn*(wN_i - mN[idx])
            mE[idx] = mE[idx] + Ke*(wE_i - mE[idx])
            Pn[idx] = (1 - Kn)*Pn[idx]
            Pe[idx] = (1 - Ke)*Pe[idx]

    # optional gentle smoothing across neighboring bins
    for _ in range(2):
        mN = 0.25*np.roll(mN,1) + 0.5*mN + 0.25*np.roll(mN,-1)
        mE = 0.25*np.roll(mE,1) + 0.5*mE + 0.25*np.roll(mE,-1)

    fN, kindN = _make_interp(z_grid, mN)
    fE, kindE = _make_interp(z_grid, mE)
    fSN, _ = _make_interp(z_grid, np.sqrt(Pn))
    fSE, _ = _make_interp(z_grid, np.sqrt(Pe))

    def updated_wind(z: float): return float(fN(z)), float(fE(z))
    def updated_std(z: float):  return float(fSN(z)), float(fSE(z))

    dbg = dict(z_grid=z_grid, mN=mN, mE=mE, Pn=Pn, Pe=Pe, kind=kindN if kindN==kindE else "mixed")
    return updated_wind, updated_std, dbg

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

def rate_limit(cur: float, des: float, max_rate: float, dt: float):
    err = (des - cur + np.pi)%(2*np.pi) - np.pi
    return cur + np.clip(err, -max_rate*dt, max_rate*dt)

def simulate_parafoil_drop(release_NE, wind_func, turb_sigma=0.0):
    pN, pE = float(release_NE[0]), float(release_NE[1])
    z = CFG.z_release; psi = 0.0
    tgtN, tgtE = CFG.target_NE
    Xs, Ys = [pE], [pN]  # x=E, y=N
    while z > CFG.z_ground:
        vN, vE = (tgtN - pN), (tgtE - pE)
        dist = max(math.hypot(vN, vE), 1e-6)
        if dist > CFG.L1_dist:
            laN = pN + (CFG.L1_dist/dist)*vN
            laE = pE + (CFG.L1_dist/dist)*vE
        else:
            laN, laE = tgtN, tgtE
        chi_d = math.atan2(laE - pE, laN - pN)
        wN, wE = wind_func(z)
        if turb_sigma > 0.0:
            wN += np.random.randn()*turb_sigma
            wE += np.random.randn()*turb_sigma
        psi_cmd, _ = crab_heading_for_track(chi_d, wN, wE, CFG.Va_forward)
        psi = rate_limit(psi, psi_cmd, CFG.psi_dot_max, CFG.dt)
        v_aN = CFG.Va_forward*math.cos(psi)
        v_aE = CFG.Va_forward*math.sin(psi)
        gN = v_aN + wN; gE = v_aE + wE
        pN += gN*CFG.dt; pE += gE*CFG.dt; z -= CFG.V_sink*CFG.dt
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
            CFG.sim_loiter_alts_m = [CFG.z_release - 200, CFG.z_release, CFG.z_release + 200]
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
    zmin = CFG.z_min if CFG.z_min < CFG.z_max else min(prior.z.min(), min(p["z"] for p in pins)) - 50
    zmax = CFG.z_max if CFG.z_max > CFG.z_min else max(prior.z.max(), max(p["z"] for p in pins)) + 50

    updated_wind, updated_std, fdbg = fuse_prior_with_pins(
        prior,
        pins,
        FuseCfg(z_min=zmin, z_max=zmax, dz=CFG.dz, process_sigma=CFG.process_sigma_mps)
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
    z_plot = np.linspace(min(prior.z.min(), fdbg["z_grid"].min()),
                         max(prior.z.max(), fdbg["z_grid"].max()), 300)
    prior_wN = np.array([prior.wind(z)[0] for z in z_plot])
    post_wN  = np.array([updated_wind(z)[0] for z in z_plot])
    post_sN  = np.array([updated_std(z)[0]  for z in z_plot])
    axs[0].plot(prior.wN, prior.z, 'C0--', alpha=0.7, label='prior wN')
    axs[0].plot(post_wN, z_plot, 'C0-',  label='updated wN')
    axs[0].fill_betweenx(z_plot, post_wN - 2*post_sN, post_wN + 2*post_sN, color='C0', alpha=0.15, label='±2σ')
    axs[0].invert_yaxis(); axs[0].grid(True)
    axs[0].set_xlabel("wN [m/s]"); axs[0].set_ylabel("Altitude [m]")
    axs[0].set_title(f"North wind profile — pins ({pins_src})")
    for p in pins:
        axs[0].plot([p["wN"]], [p["z"]], 'ko', ms=4)

    # (2) Prior + updated profile (wE)
    prior_wE = np.array([prior.wind(z)[1] for z in z_plot])
    post_wE  = np.array([updated_wind(z)[1] for z in z_plot])
    post_sE  = np.array([updated_std(z)[1]  for z in z_plot])
    axs[1].plot(prior.wE, prior.z, 'C1--', alpha=0.7, label='prior wE')
    axs[1].plot(post_wE, z_plot, 'C1-',  label='updated wE')
    axs[1].fill_betweenx(z_plot, post_wE - 2*post_sE, post_wE + 2*post_sE, color='C1', alpha=0.15, label='±2σ')
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

    plt.tight_layout()
    fig_path = os.path.join(CFG.out_dir, "wind_pipeline_results.png")
    plt.show()
    plt.savefig(fig_path, dpi=150)

    # Save updated profile on dz grid
    z_grid = fdbg["z_grid"]
    up_wN  = np.array([updated_wind(z)[0] for z in z_grid])
    up_wE  = np.array([updated_wind(z)[1] for z in z_grid])
    up_sN  = np.array([updated_std(z)[0]  for z in z_grid])
    up_sE  = np.array([updated_std(z)[1]  for z in z_grid])
    prof_csv = os.path.join(CFG.out_dir, "updated_wind_profile.csv")
    pd.DataFrame(dict(altitude_m=z_grid, wN_mps=up_wN, wE_mps=up_wE,
                      sigN_mps=up_sN, sigE_mps=up_sE)).to_csv(prof_csv, index=False)

    # Save region box
    summary_txt = f"""Updated layer at z_release={CFG.z_release:.1f} m: wN={wN_layer:.2f} m/s, wE={wE_layer:.2f} m/s
Deterministic miss with full updated profile: {miss_det:.2f} m
Region box (x=E, y=N): xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}
"""
    with open(os.path.join(CFG.out_dir, "pipeline_summary.txt"), "w") as f:
        f.write(summary_txt)

    print(f"Saved figure: {fig_path}")
    print(f"Saved updated profile: {prof_csv}")
    print(summary_txt)

if __name__ == "__main__":
    main()
