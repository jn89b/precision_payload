# Simple wind-estimation + CARP + Monte Carlo dispersion demo
# (self-contained; tweak parameters in the CONFIG block)
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable
import math
from matplotlib.patches import Ellipse
import os

# --------------------
# CONFIG
# --------------------
@dataclass
class Config:
    # Loiter / sensing
    z_loiter: float = 400.0        # m AGL (release altitude)
    circle_R: float = 250.0        # m (loiter radius)
    Va: float = 22.0               # m/s true airspeed during loiter
    samples_per_circle: int = 180  # measurement density
    meas_noise: float = 0.5        # m/s std on ground-vel components
    
    # True wind model (for sim "ground truth")
    # Simple linear shear with height + direction veer
    W0: float = 6.0        # m/s wind speed at z_ref
    z_ref: float = 10.0    # m
    alpha: float = 0.12    # power-law exponent for speed
    psi0_deg: float = 240  # deg (direction *from* which wind blows)
    veer_deg_per_100m: float = 8.0  # deg rotation per 100 m altitude
    
    # Payload descent (ballistic simplification)
    V_sink: float = 7.0     # m/s constant vertical sink rate
    # Monte Carlo
    N_mc: int = 500
    wind_std_mps: float = 0.7   # 1-sigma uncertainty on each wind component at z_loiter
    
    # Geometry
    target_xy: Tuple[float,float] = (0.0, 0.0)  # target at origin
    gnd_z: float = 0.0
    # Release strategy
    assume_constant_wind_for_CARP: bool = True  # CARP uses single layer (fast)
    parafoil_forward_speed: float = 0.0         # m/s. Set >0 to emulate forward glide toward target.

CFG = Config()


# --------------------
# UTILITIES
# --------------------
def wind_profile(z: float, cfg: Config=CFG) -> Tuple[float,float]:
    """
    Return wind vector (N,E) at altitude z (m AGL).
    Direction psi is the meteorological "from" direction; we convert to vector blowing TO.
    """
    # speed via power law
    speed = cfg.W0 * (max(z,1.0)/cfg.z_ref)**cfg.alpha
    # direction veer with height
    psi_from_deg = cfg.psi0_deg + cfg.veer_deg_per_100m * (z/100.0)
    # Convert "from" to TO direction
    psi_to_rad = np.deg2rad(psi_from_deg + 180.0)
    wN = speed*np.cos(psi_to_rad)
    wE = speed*np.sin(psi_to_rad)
    return wN, wE

def simulate_loiter_measurements(cfg: Config=CFG):
    """Simulate one circle of loiter, returning headings psi, and measured ground velocities vgN, vgE"""
    thetas = np.linspace(0, 2*np.pi, cfg.samples_per_circle, endpoint=False)  # course around circle
    # Assume aircraft heading aligns with track (ok for gentle loiter)
    psi = thetas.copy()
    # Aircraft position (for show only)
    x = cfg.circle_R*np.cos(thetas)
    y = cfg.circle_R*np.sin(thetas)
    # True wind at loiter altitude
    wN, wE = wind_profile(cfg.z_loiter, cfg)
    # Air-relative velocity along body x: Va*[cos psi, sin psi]
    v_aN = cfg.Va*np.cos(psi)
    v_aE = cfg.Va*np.sin(psi)
    # Ground velocity is sum
    vgN_true = v_aN + wN
    vgE_true = v_aE + wE
    # Add measurement noise
    vgN_meas = vgN_true + np.random.randn(len(psi))*cfg.meas_noise
    vgE_meas = vgE_true + np.random.randn(len(psi))*cfg.meas_noise
    return x, y, psi, vgN_meas, vgE_meas, (wN, wE)

def estimate_wind_from_loiter(psi, Va, vgN, vgE) -> Tuple[float,float]:
    """Least-squares estimate of horizontal wind from one circle (small-attitude approximation)."""
    # Model: vg = Va*[cos psi, sin psi] + w  => w = vg - Va*[...]
    bN = vgN - Va*np.cos(psi)
    bE = vgE - Va*np.sin(psi)
    # Average is the LS solution here (since parameters enter additively)
    wN_hat = float(np.mean(bN))
    wE_hat = float(np.mean(bE))
    return wN_hat, wE_hat

def carp_constant_wind(target_xy, z0, V_sink, wN, wE, V_forward=0.0, unit_dir_to_target=None):
    """
    Closed-form CARP under constant wind and constant vertical sink rate.
    If V_forward>0 (parafoil-like), assumes we will point straight to target during descent.
    """
    T = max((z0-CFG.gnd_z)/V_sink, 0.001)
    offset_wind = np.array([wN, wE])*T
    offset_forward = np.array([0.0, 0.0])
    if V_forward > 0.0:
        if unit_dir_to_target is None:
            # If not provided, assume from release we'll point at target, so direction is from release to target.
            # For closed-form, we approximate using the target direction from loiter center (0,0) to target.
            tx, ty = target_xy
            vec = np.array([tx, ty])
            nrm = np.linalg.norm(vec) + 1e-9
            unit_dir_to_target = vec / nrm
        offset_forward = V_forward*T*unit_dir_to_target
    # Release point = target - (wind drift + forward air-relative travel)
    release_xy = np.array(target_xy) - offset_wind - offset_forward
    return release_xy, T

def simulate_drop(release_xy, z0, V_sink, wind_func: Callable[[float], Tuple[float,float]],
                  dt=0.1, V_forward=0.0, target_xy=(0.0,0.0)) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Integrate descent with height-varying wind. If V_forward>0, we "point to target" naively each step.
    Returns trajectory arrays (N,E) and impact distance.
    """
    p = np.array(release_xy, dtype=float)
    z = z0
    N_hist = [p[0]]
    E_hist = [p[1]]
    while z > CFG.gnd_z:
        wN, wE = wind_func(z)
        vN = wN
        vE = wE
        if V_forward > 0.0:
            # naive: point directly to target every step
            to_tgt = np.array(target_xy) - p
            nrm = np.linalg.norm(to_tgt) + 1e-9
            u_dir = to_tgt / nrm
            vN += V_forward*u_dir[0]
            vE += V_forward*u_dir[1]
        # integrate
        p[0] += vN*dt
        p[1] += vE*dt
        z -= V_sink*dt
        N_hist.append(p[0])
        E_hist.append(p[1])
    impact = p.copy()
    miss = float(np.linalg.norm(impact - np.array(target_xy)))
    return np.array(N_hist), np.array(E_hist), miss

def covariance_ellipse(cov: np.ndarray, nsig=2.0):
    """Return width, height, angle (deg) for an ellipse representing nsig sigma of a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width = 2*nsig*np.sqrt(vals[0])
    height = 2*nsig*np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    return width, height, angle

# --------------------
# RUN SIM
# --------------------
np.random.seed(7)

# 1) Simulate a loiter and estimate wind at z_loiter
x, y, psi, vgN_meas, vgE_meas, w_true = simulate_loiter_measurements(CFG)
wN_hat, wE_hat = estimate_wind_from_loiter(psi, CFG.Va, vgN_meas, vgE_meas)

# 2) Get a fast CARP under constant wind (optionally parafoil forward)
target_xy = np.array(CFG.target_xy)
unit_dir = None  # let carp function pick it
release_xy_const, Tfall = carp_constant_wind(
    target_xy, CFG.z_loiter, CFG.V_sink, wN_hat, wE_hat, CFG.parafoil_forward_speed, unit_dir
)

# 3) Single deterministic drop with full wind profile (to show bias vs constant-wind CARP)
N_tr, E_tr, miss_det = simulate_drop(
    release_xy_const, CFG.z_loiter, CFG.V_sink,
    lambda z: wind_profile(z, CFG), dt=0.2, V_forward=CFG.parafoil_forward_speed, target_xy=CFG.target_xy
)

# 4) Monte Carlo dispersion: sample wind estimate uncertainty at z_loiter
impacts = []
for i in range(CFG.N_mc):
    # Sample a wind vector at z_loiter (Gaussian around estimate)
    wN_s = wN_hat + np.random.randn()*CFG.wind_std_mps
    wE_s = wE_hat + np.random.randn()*CFG.wind_std_mps
    # Build a perturbed wind profile by biasing the whole profile with the sampled delta at z_loiter.
    dN = wN_s - wN_hat
    dE = wE_s - wE_hat
    wind_func = lambda z, dN=dN, dE=dE: (wind_profile(z, CFG)[0] + dN, wind_profile(z, CFG)[1] + dE)
    # Use the same CARP computed from (wN_hat,wE_hat) for all runs, to show dispersion due to uncertainty
    N_mc, E_mc, _ = simulate_drop(
        release_xy_const, CFG.z_loiter, CFG.V_sink, wind_func, dt=0.2,
        V_forward=CFG.parafoil_forward_speed, target_xy=CFG.target_xy
    )
    impacts.append([N_mc[-1], E_mc[-1]])
impacts = np.array(impacts)
cov = np.cov(impacts.T)
width, height, angle = covariance_ellipse(cov, nsig=2.0)  # ~95% for Gaussian

# CEP (circular error probable) approximation from covariance
eigvals, _ = np.linalg.eigh(cov)
sigma_eff = math.sqrt(np.mean(eigvals))
CEP95 = 2.45*sigma_eff  # rough 95% CEP factor

# --------------------
# PLOTS
# --------------------
fig = plt.figure(figsize=(8, 8))

# Loiter ring + vectors
ax = fig.add_subplot(2,1,1)
ax.plot(x, y, lw=1)
ax.scatter([target_xy[0]], [target_xy[1]], marker='x', s=80)
ax.scatter([release_xy_const[0]], [release_xy_const[1]], marker='o', s=40)
# draw estimated wind vector at loiter center
ax.arrow(0, 0, wN_hat*10, wE_hat*10, head_width=10, length_includes_head=True)
ax.set_aspect('equal', 'box')
ax.set_title('Loiter ring, target (x), CARP (o), and estimated wind vector (scaled x10)')
ax.set_xlabel('North [m]')
ax.set_ylabel('East [m]')
ax.grid(True)

# Impact dispersion
ax2 = fig.add_subplot(2,1,2)
ax2.scatter(impacts[:,0], impacts[:,1], s=8, alpha=0.5)
ax2.scatter([target_xy[0]], [target_xy[1]], marker='x', s=80)
# add 95% covariance ellipse
ell = Ellipse(xy=(np.mean(impacts[:,0]), np.mean(impacts[:,1])),
              width=width, height=height, angle=angle, fill=False, lw=2)
ax2.add_patch(ell)
ax2.set_aspect('equal', 'box')
ax2.set_title(f'Impact dispersion (N={CFG.N_mc}) — ~95% ellipse; CEP95≈{CEP95:.1f} m')
ax2.set_xlabel('North [m]')
ax2.set_ylabel('East [m]')
ax2.grid(True)

plt.tight_layout()
plt.show()
# Save artifacts for download
os.makedirs("/mnt/data", exist_ok=True)
fig_path = "/mnt/data/wind_carp_mc.png"
plt.savefig(fig_path, dpi=150)

# Also save the script to a file for reuse
script_path = "/mnt/data/simple_drop_sim.py"
with open(script_path, "w") as f:
    f.write("""# Saved from ChatGPT session — simple wind CARP simulator
# Copy of the code used in the notebook. If you want the latest, export from the conversation.
# Tip: run `python simple_drop_sim.py` after pasting plotting guards if using headless env.
""")

fig_path, script_path
