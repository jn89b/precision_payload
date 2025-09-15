# Parafoil L1/crab simulator + CARP + dispersion -> A* region box
# Self-contained. Edit CONFIG and re-run.
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable
import math
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
import os

# --------------------
# CONFIG
# --------------------
@dataclass
class ConfigPF:
    # Geometry
    target_xy: Tuple[float,float] = (0.0, 0.0)  # target at origin
    z_release: float = 400.0       # m AGL release altitude
    z_ground: float = 0.0
    dt: float = 0.2                # s integrator step
    
    # Parafoil kinematics
    Va_forward: float = 9.0        # m/s air-relative forward speed
    V_sink: float = 2.5            # m/s vertical sink
    psi_dot_max: float = np.deg2rad(15.0)  # rad/s maximum heading rate (turn authority)
    
    # L1 / guidance
    L1_dist: float = 60.0          # m lookahead distance (softens heading jumps)
    
    # Loiter sensing (to estimate wind before release)
    Va_loiter: float = 22.0
    circle_R: float = 250.0
    samples_per_circle: int = 180
    meas_noise: float = 0.4  # m/s on ground-vel components
    
    # Wind profile "truth" for sim
    W0: float = 6.0              # m/s at z_ref
    z_ref: float = 10.0
    alpha: float = 0.11          # speed shear exponent
    psi0_deg: float = 240.0      # "from" direction at z_ref (met.)
    veer_deg_per_100m: float = 7.0
    
    # Uncertainty (for Monte Carlo)
    N_mc: int = 800
    wind_std_layer: float = 0.7  # m/s 1-sigma per component @ release layer
    turb_sigma: float = 0.3      # m/s std of white-noise gust per step (small)
    
    # Region box sizing for A*
    ellipse_nsig: float = 2.0    # ~95% for Gaussian
    box_margin_m: float = 10.0   # add a little margin around ellipse-aligned box

CFG = ConfigPF()

# --------------------
# WIND UTILITIES
# --------------------
def wind_profile_truth(z: float, cfg: ConfigPF=CFG) -> Tuple[float,float]:
    """Ground-truth wind vector (N,E) at altitude z (blowing TO)."""
    spd = cfg.W0 * (max(z,1.0)/cfg.z_ref)**cfg.alpha
    psi_from = cfg.psi0_deg + cfg.veer_deg_per_100m*(z/100.0)
    psi_to = np.deg2rad(psi_from + 180.0)
    return spd*np.cos(psi_to), spd*np.sin(psi_to)

def simulate_loiter_measurements(cfg: ConfigPF=CFG):
    """One circle; return headings psi, and measured ground velocities vgN, vgE."""
    thetas = np.linspace(0, 2*np.pi, cfg.samples_per_circle, endpoint=False)
    psi = thetas.copy()
    wN, wE = wind_profile_truth(cfg.z_release, cfg)
    vgN_true = cfg.Va_loiter*np.cos(psi) + wN
    vgE_true = cfg.Va_loiter*np.sin(psi) + wE
    vgN = vgN_true + np.random.randn(len(psi))*cfg.meas_noise
    vgE = vgE_true + np.random.randn(len(psi))*cfg.meas_noise
    return psi, vgN, vgE, (wN, wE)

def estimate_layer_wind(psi, Va, vgN, vgE):
    """LS estimate of horizontal wind from loiter (small-attitude approx)."""
    bN = vgN - Va*np.cos(psi)
    bE = vgE - Va*np.sin(psi)
    return float(np.mean(bN)), float(np.mean(bE))

# --------------------
# GUIDANCE / CRAB
# --------------------
def desired_track_angle(p: np.ndarray, tgt: np.ndarray) -> float:
    """Ground track angle (chi_d) from position p to target."""
    v = tgt - p
    return math.atan2(v[1], v[0])

def crab_heading_for_track(chi_d: float, wN: float, wE: float, Va: float) -> Tuple[float,float]:
    """
    Given desired ground-track angle chi_d, wind (wN,wE), and airspeed Va,
    compute heading psi_cmd that cancels crosswind (classic crab) and the resulting Vg along-track.
    """
    # Wind components along/cross desired track
    W_along = wN*np.cos(chi_d) + wE*np.sin(chi_d)
    W_cross = -wN*np.sin(chi_d) + wE*np.cos(chi_d)
    # Crab angle (left positive); clamp to feasible asin range
    arg = np.clip(W_cross / max(Va,1e-6), -1.0, 1.0)
    delta = math.asin(arg)
    psi_cmd = chi_d + delta
    # Groundspeed magnitude along-track
    Vg = Va*math.cos(delta) + W_along
    return psi_cmd, Vg

def rate_limit(current: float, desired: float, max_rate: float, dt: float) -> float:
    """Limit rate of change of an angle (unwrap to shortest path)."""
    # unwrap difference to [-pi,pi]
    err = (desired - current + np.pi)%(2*np.pi) - np.pi
    step = np.clip(err, -max_rate*dt, max_rate*dt)
    return current + step

# --------------------
# DROP SIMULATOR (parafoil L1/crab)
# --------------------
def simulate_parafoil_drop(release_xy, cfg: ConfigPF, wind_func: Callable[[float], Tuple[float,float]],
                           turb_sigma: float=0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    p = np.array(release_xy, dtype=float)  # (N,E)
    z = cfg.z_release
    psi = 0.0  # initial heading arbitrary; will settle via rate limit
    tgt = np.array(cfg.target_xy, dtype=float)
    
    Ns, Es = [p[0]], [p[1]]
    while z > cfg.z_ground:
        # Desired ground-track to target with lookahead (L1-ish)
        vec = tgt - p
        dist = max(np.linalg.norm(vec), 1e-6)
        # Lookahead point along the line to target
        lookahead = p + (cfg.L1_dist/dist) * vec if dist > cfg.L1_dist else tgt
        chi_d = math.atan2(lookahead[1] - p[1], lookahead[0] - p[0])
        
        # Current wind (plus small random gust if enabled)
        wN, wE = wind_func(z)
        if turb_sigma > 0.0:
            wN += np.random.randn()*turb_sigma
            wE += np.random.randn()*turb_sigma
        
        # Crab to align track with chi_d
        psi_cmd, Vg_along = crab_heading_for_track(chi_d, wN, wE, cfg.Va_forward)
        # Rate limit heading (turn authority)
        psi = rate_limit(psi, psi_cmd, cfg.psi_dot_max, cfg.dt)
        
        # Air-relative velocity
        v_aN = cfg.Va_forward*math.cos(psi)
        v_aE = cfg.Va_forward*math.sin(psi)
        # Ground velocity
        vN = v_aN + wN
        vE = v_aE + wE
        
        # Integrate
        p[0] += vN*cfg.dt
        p[1] += vE*cfg.dt
        z -= cfg.V_sink*cfg.dt
        
        Ns.append(p[0])
        Es.append(p[1])
    miss = float(np.linalg.norm(p - tgt))
    return np.array(Ns), np.array(Es), miss

# --------------------
# CARP from estimated layer wind (shooting via root find around loiter ring)
# --------------------
def carp_from_estimate(wN_hat, wE_hat, cfg: ConfigPF=CFG, n_candidates:int=60) -> Tuple[np.ndarray, float]:
    """
    Pick a release point on a circle around target (your loiter radius) so that, under constant layer wind,
    the parafoil flying with crab-to-target will reach the target. We do a coarse search.
    """
    R = cfg.circle_R
    tgt = np.array(cfg.target_xy, dtype=float)
    best_d = 1e9
    best_xy = None
    # candidate points evenly spaced around target
    angles = np.linspace(0, 2*np.pi, n_candidates, endpoint=False)
    for th in angles:
        rel = tgt + np.array([R*np.cos(th), R*np.sin(th)])
        # simulate with constant wind = (wN_hat, wE_hat)
        wind_const = lambda z: (wN_hat, wE_hat)
        N,E,miss = simulate_parafoil_drop(rel, cfg, wind_const, turb_sigma=0.0)
        if miss < best_d:
            best_d = miss
            best_xy = rel.copy()
    return best_xy, best_d

# --------------------
# RUN: estimate wind, CARP, dispersion, and region box
# --------------------
np.random.seed(1)

# 1) Estimate layer wind from one loiter ring
psi_meas, vgN, vgE, w_true_layer = simulate_loiter_measurements(CFG)
wN_hat, wE_hat = estimate_layer_wind(psi_meas, CFG.Va_loiter, vgN, vgE)

# 2) Compute a practical CARP around a circle near the target
carp_xy, carp_err = carp_from_estimate(wN_hat, wE_hat, CFG)

# 3) Deterministic check with full wind profile
N_det, E_det, miss_det = simulate_parafoil_drop(carp_xy, CFG, wind_profile_truth, turb_sigma=0.0)

# 4) Monte Carlo dispersion with wind uncertainty + small turbulence
impacts = []
misses = []
for i in range(CFG.N_mc):
    # sample layer wind error and bias the whole profile with that delta
    dN = np.random.randn()*CFG.wind_std_layer
    dE = np.random.randn()*CFG.wind_std_layer
    wind_func = lambda z, dN=dN, dE=dE: (wind_profile_truth(z)[0] + dN, wind_profile_truth(z)[1] + dE)
    N, E, miss = simulate_parafoil_drop(carp_xy, CFG, wind_func, turb_sigma=CFG.turb_sigma)
    impacts.append([N[-1], E[-1]]); misses.append(miss)
impacts = np.array(impacts)
misses = np.array(misses)

# 5) Fit dispersion ellipse and derive A* region box (axis-aligned box covering nsig ellipse)
mean_impact = impacts.mean(axis=0)
cov = np.cov(impacts.T)
# ellipse params
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]; eigvals = eigvals[order]; eigvecs = eigvecs[:,order]
nsig = CFG.ellipse_nsig
axes_lengths = 2*nsig*np.sqrt(eigvals)  # width (major), height (minor)
angle_rad = math.atan2(eigvecs[1,0], eigvecs[0,0])

# Build oriented ellipse points then compute axis-aligned bounding box
theta = np.linspace(0, 2*np.pi, 200)
ellipse = np.vstack([ (axes_lengths[0]/2)*np.cos(theta), (axes_lengths[1]/2)*np.sin(theta) ])  # in ellipse frame
R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
              [np.sin(angle_rad),  np.cos(angle_rad)]])
ellipse_global = (R @ ellipse) + mean_impact.reshape(2,1)
xmin, ymin = ellipse_global[0].min(), ellipse_global[1].min()
xmax, ymax = ellipse_global[0].max(), ellipse_global[1].max()

# add margin
xmin -= CFG.box_margin_m; ymin -= CFG.box_margin_m
xmax += CFG.box_margin_m; ymax += CFG.box_margin_m
box_width = xmax - xmin; box_height = ymax - ymin

# 6) Plot everything
fig, axs = plt.subplots(1, 2, figsize=(13,6))

# Left: loiter-estimate + CARP 
axs[0].set_title("Loiter-derived wind → CARP (parafoil with crab)")
axs[0].plot([CFG.circle_R*np.cos(t) for t in np.linspace(0,2*np.pi,361)],
            [CFG.circle_R*np.sin(t) for t in np.linspace(0,2*np.pi,361)], color='0.6', lw=1)
axs[0].scatter([CFG.target_xy[0]],[CFG.target_xy[1]], marker='x', s=80, label="Target")
axs[0].scatter([carp_xy[0]],[carp_xy[1]], s=40, label="CARP")
axs[0].plot(N_det, E_det, lw=1.5, label=f"Deterministic drop (miss={miss_det:.1f} m)")
# draw estimated wind vector at center (scaled)
axs[0].arrow(0, 0, wN_hat*12, wE_hat*12, head_width=10, length_includes_head=True, label="Estimated wind")
axs[0].axis('equal'); axs[0].grid(True)
axs[0].set_xlabel("North [m]"); axs[0].set_ylabel("East [m]")
axs[0].legend(loc="best")

# Right: dispersion + ellipse + region box
axs[1].set_title(f"Impact dispersion (N={CFG.N_mc}) + {int(nsig*100/2)}% ellipse & A* box")
axs[1].scatter(impacts[:,0], impacts[:,1], s=6, alpha=0.4, label="Impacts")
axs[1].scatter([CFG.target_xy[0]],[CFG.target_xy[1]], marker='x', s=80, label="Target")
# ellipse
from matplotlib.patches import Ellipse, Rectangle
axs[1].add_patch(Ellipse(xy=mean_impact, width=axes_lengths[0], height=axes_lengths[1],
                         angle=np.degrees(angle_rad), fill=False, lw=2, label=f"{int( (math.erf(nsig/np.sqrt(2)))**2 *100)}% ellipse"))
# box
axs[1].add_patch(Rectangle((xmin, ymin), box_width, box_height, fill=False, lw=2, linestyle="--", label="A* region box"))
axs[1].axis('equal'); axs[1].grid(True); axs[1].set_xlabel("North [m]"); axs[1].set_ylabel("East [m]")
axs[1].legend(loc="best")
plt.tight_layout()
plt.show()

# 7) Save artifacts (figure, CSVs, JSON-ish text summary)
os.makedirs("/mnt/data", exist_ok=True)
fig_path = "/mnt/data/parafoil_carp_dispersion.png"
plt.savefig(fig_path, dpi=150)

impacts_df = pd.DataFrame(impacts, columns=["N","E"])
impacts_csv = "/mnt/data/impact_points.csv"
impacts_df.to_csv(impacts_csv, index=False)

summary_txt = f"""
CARP (coarse search on loiter ring):
  release_xy = ({carp_xy[0]:.2f}, {carp_xy[1]:.2f}) m; deterministic miss with true profile = {miss_det:.2f} m

Wind estimate at release layer (from loiter):
  w_hat = ({wN_hat:.2f} N, {wE_hat:.2f} E) m/s
  true  = ({w_true_layer[0]:.2f}, {w_true_layer[1]:.2f}) m/s

Dispersion (N={CFG.N_mc}) summary:
  Mean impact  = ({mean_impact[0]:.2f}, {mean_impact[1]:.2f}) m
  Covariance   = [[{cov[0,0]:.2f}, {cov[0,1]:.2f}],
                  [{cov[1,0]:.2f}, {cov[1,1]:.2f}]]
  Ellipse nsig = {CFG.ellipse_nsig} (width={axes_lengths[0]:.1f} m, height={axes_lengths[1]:.1f} m, angle={np.degrees(angle_rad):.1f} deg)

A* region (axis-aligned box around ellipse + margin {CFG.box_margin_m} m):
  xmin={xmin:.1f}, ymin={ymin:.1f}, xmax={xmax:.1f}, ymax={ymax:.1f}
  width={box_width:.1f} m, height={box_height:.1f} m
"""
summary_path = "/mnt/data/parafoil_region_summary.txt"
with open(summary_path, "w") as f:
    f.write(summary_txt)

# Show the text summary in the notebook output for convenience
summary_txt, fig_path, impacts_csv, summary_path
