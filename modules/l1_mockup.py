\
import math
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

def _wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2*math.pi)
    return a - math.pi

@dataclass
class L1Params:
    L1_dist: float
    kr_yaw: float
    diff_max: float
    kI: float = 0.0
    Imax: float = 0.2
    use_crab_ff: bool = True
    Va_est: float = None
    vz_ref: float = -5.0
    sym_min: float = 0.05
    sym_max: float = 0.6
    sym_kp: float = 0.20

@dataclass
class L1State:
    I_bias: float = 0.0

def l1_parafoil_step(px, py, chi, Vg, vz_meas,
                     Ax, Ay, Bx, By,
                     Wu=0.0, Wv=0.0,
                     st=None, p=None, dt=0.05):
    assert p is not None
    if st is None: st = L1State()
    tx, ty = Bx - Ax, By - Ay
    seg_len = math.hypot(tx, ty) + 1e-9
    tx, ty = tx/seg_len, ty/seg_len
    nx, ny = -ty, tx
    rx, ry = px - Ax, py - Ay
    ey = rx*nx + ry*ny
    print("ey:", ey)
    chi_path = math.atan2(ty, tx)
    def _wrap_pi(a: float) -> float:
        a = (a + math.pi) % (2*math.pi)
        return a - math.pi
    eta = _wrap_pi(chi_path - chi) + math.atan2(ey, max(1e-3, p.L1_dist))
    if p.use_crab_ff:
        Va = p.Va_est if (p.Va_est is not None) else max(0.1, Vg)
        Wperp = -Wu*math.sin(chi_path) + Wv*math.cos(chi_path)
        psi_ff = math.atan2(Wperp, max(0.1, Va))
        eta = _wrap_pi(eta + psi_ff)
    a_cmd = (2.0 * Vg * Vg / max(0.5, p.L1_dist)) * math.sin(eta)
    r_cmd = a_cmd / max(0.5, Vg)
    if p.kI > 0.0:
        st.I_bias += ey * p.kI * dt
        st.I_bias = max(-p.Imax, min(p.Imax, st.I_bias))
    else:
        st.I_bias = 0.0
    diff_cmd = (r_cmd / max(1e-3, p.kr_yaw)) + st.I_bias
    diff_cmd = max(-p.diff_max, min(p.diff_max, diff_cmd))
    evz = p.vz_ref - vz_meas
    sym_cmd = p.sym_min + p.sym_kp * evz
    sym_cmd = max(p.sym_min, min(p.sym_max, sym_cmd))
    return diff_cmd, sym_cmd, st

GR_POINTS = [
    # (0.05, 1.9, 5.8),
    (0.30, 1.9, 4.5),
    # (0.50, 1.0, 3.5),
]

def gr_polar(sym: float):
    pts = sorted(GR_POINTS, key=lambda x: x[0])
    sym = max(pts[0][0], min(pts[-1][0], sym))
    for i in range(len(pts)-1):
        s0, GR0, Vz0 = pts[i]
        s1, GR1, Vz1 = pts[i+1]
        if s0 <= sym <= s1:
            u = (sym - s0) / (s1 - s0 + 1e-9)
            GR = GR0 + u*(GR1 - GR0)
            Vz = Vz0 + u*(Vz1 - Vz0)
            Va = GR * Vz
            return Va, -Vz
    GR, Vz = pts[-1][1], pts[-1][2]
    return GR*Vz, -Vz

Ax, Ay = 0.0, -150.0
Bx, By = 0.0,  0.0

def run_once(Wu=4.0, Wv=1.0, alt0=120.0, px0=150, py0=-120.0,
             params=None, dt=0.1, max_t=300.0, psi0_deg=180.0,
             record=True):
    if params is None:
        params = L1Params(L1_dist=60.0, kr_yaw=0.03, diff_max=0.6,
                          kI=1e-4, Imax=0.1, use_crab_ff=True, Va_est=None,
                          vz_ref=-5.0, sym_kp=0.20, sym_min=0.05, sym_max=0.5)
    st = L1State()
    psi = math.radians(psi0_deg)
    px, py, alt = px0, py0, alt0
    sym_cmd = 0.2
    Va, vz = gr_polar(sym_cmd)
    t = 0.0
    traj = [(px, py)] if record else None
    while t < max_t and alt > 0.0:
        vax = Va*math.cos(psi); vay = Va*math.sin(psi)
        vx = vax + Wu;           vy = vay + Wv
        Vg = math.hypot(vx, vy); chi = math.atan2(vy, vx)
        diff_cmd, sym_cmd, st = l1_parafoil_step(px, py, chi, Vg, vz,
                                                 Ax, Ay, Bx, By,
                                                 Wu, Wv, st, params, dt)
        Va, vz = gr_polar(sym_cmd)
        r_cmd = params.kr_yaw * diff_cmd
        psi = (psi + r_cmd*dt + math.pi)%(2*math.pi) - math.pi
        px += vx*dt; py += vy*dt; alt += vz*dt
        if record: traj.append((px, py))
        t += dt
    return (px, py), (np.array(traj) if record else None)

def simulate_MC(N=300, alt0=120.0, Wu_mean=4.0, Wv_mean=1.0, Wu_std=2.0, Wv_std=2.0, seed=7):
    rng = np.random.default_rng(seed)
    landings = []
    trajectories = []
    for i in range(N):
        Wu = rng.normal(Wu_mean, Wu_std)
        Wv = rng.normal(Wv_mean, Wv_std)
        lp, traj = run_once(Wu=Wu, Wv=Wv, alt0=alt0, record=True)
        landings.append(lp)
        trajectories.append(traj)
    landings = np.array(landings)
    d = np.hypot(landings[:,0]-Bx, landings[:,1]-By)
    CEP = float(np.median(d))
    R95 = float(np.quantile(d, 0.95))
    trajectories = np.array(trajectories)
    return landings, CEP, R95, trajectories

if __name__ == "__main__":
    lp, traj = run_once()
    print(f"Single-run landing at: x={lp[0]:.1f} m, y={lp[1]:.1f} m")
    land, CEP, R95, traj_results = simulate_MC(N=300, alt0=200.0, Wu_std=2.0, Wv_std=2.0)
    print(f"Monte Carlo (120 m AGL, wind σ=2 m/s): CEP={CEP:.1f} m, R95={R95:.1f} m")

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(traj[:,0], traj[:,1], label="Trajectory")
    # plt.scatter([Ax, Bx], [Ay, By], marker='x', label="A & B")
    # plt.xlabel("x [m]"); plt.ylabel("y [m]")
    # plt.title("Trajectory (GR-driven polar)")
    # plt.axis('equal'); plt.grid(True); plt.legend(); plt.tight_layout()
    # plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    for t in traj_results:
        ax.plot(t[:,0], t[:,1], color='C0', alpha=0.1)
    ax.scatter([Ax, Bx], [Ay, By], marker='x', color='C1', s=100, label="A & B")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectories (GR-driven polar)")
    ax.axis('equal'); ax.grid(True); ax.legend(); plt.tight_layout()\

    plt.figure()
    plt.scatter(land[:,0], land[:,1], s=10, alpha=0.7)
    plt.scatter([Bx],[By], marker='x')
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Landing dispersion (σ=2 m/s)")
    plt.axis('equal'); plt.grid(True); plt.tight_layout()
    plt.show()
