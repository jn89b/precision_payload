import math
from dataclasses import dataclass

"""
https://www.research-collection.ethz.ch/server/api/core/bitstreams/091d4502-8f44-4fdc-a761-c1086c8b1c8b/content
"""

def _wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2*math.pi)
    return a - math.pi

@dataclass
class L1Params:
    L1_dist: float               # lookahead distance [m], ~1–2 × R_min
    kr_yaw: float                # yaw-rate per differential-brake unit [rad/s per command-unit]
    diff_max: float              # max |diff| command (e.g., 1.0 = 100%)
    kI: float = 0.0              # integral gain on cross-track [command/(m·s)]
    Imax: float = 0.2            # integral clamp
    use_crab_ff: bool = True     # wind feed-forward
    Va_est: float = None         # air-relative forward speed [m/s]; if None, uses Vg
    # symmetric brake (vertical) schedule:
    vz_ref: float = -5.0         # desired sink rate [m/s] (negative down)
    sym_min: float = 0.0         # min symmetric command
    sym_max: float = 0.6         # max symmetric command
    sym_kp: float = 0.08         # proportional on (vz_ref - vz_meas)

@dataclass
class L1State:
    I_bias: float = 0.0          # integral on cross-track

def l1_parafoil_step(
    # navigation inputs (ENU or NED; consistent frame):
    px: float, py: float,        # current position
    chi: float,                  # current ground-track (course) [rad]
    Vg: float,                   # ground speed [m/s]
    vz_meas: float,              # measured vertical speed [m/s] (negative down)
    # line AB definition:
    Ax: float, Ay: float,
    Bx: float, By: float,
    # (optional) wind estimate at canopy:
    Wu: float = 0.0, Wv: float = 0.0,   # wind components in same frame [m/s]
    # controller state & params:
    st: L1State = None,
    p: L1Params = None,
    dt: float = 0.05
):
    """
    Diff command is the "yaw-rate" command, mapped to differential brake.
    Sym command is the symmetric brake to regulate sink rate -> how fast the parafoil descends.
    Args:
        px (float): Current x position.
        py (float): Current y position.
        chi (float): Current ground-track (course) in radians.
        Vg (float): Ground speed in m/s.
        vz_meas (float): Measured vertical speed in m/s (negative down).
        Ax (float): x coordinate of point A defining the line.
        Ay (float): y coordinate of point A defining the line.
        Bx (float): x coordinate of point B defining the line.
        By (float): y coordinate of point B defining the line.
        Wu (float, optional): Wind component in the x direction in m/s. Defaults to 0.0.
        Wv (float, optional): Wind component in the y direction in m/s. Defaults to 0.0.
        st (L1State, optional): Controller state. If None, a new state is created. Defaults to None.
        p (L1Params, optional): Controller parameters. Must be provided. Defaults to None
        dt (float, optional): Time step in seconds. Defaults to 0.05.
    Returns: diff_cmd ([-diff_max..+diff_max]), sym_cmd ([sym_min..sym_max]), updated_state
    
    # Diff command 
    """
    assert p is not None, "Provide L1Params"
    if st is None: st = L1State()

    # 1) Path geometry
    tx, ty = Bx - Ax, By - Ay
    seg_len = math.hypot(tx, ty) + 1e-9
    tx, ty = tx/seg_len, ty/seg_len
    nx, ny = -ty, tx  # left normal
    # vector from A to current position
    rx, ry = px - Ax, py - Ay
    # cross-track error (+ left of line)
    ey = rx*nx + ry*ny
    # desired path course
    chi_path = math.atan2(ty, tx)

    # 2) Core L1 angle
    eta = _wrap_pi(chi_path - chi) + math.atan2(ey, max(1e-3, p.L1_dist))

    # 3) Optional wind crab feed-forward (use airspeed estimate if available)
    if p.use_crab_ff:
        Va = p.Va_est if (p.Va_est is not None) else max(0.1, Vg)  # fall back to Vg
        # wind components along/perp the path
        Wpar  =  Wu*math.cos(chi_path) + Wv*math.sin(chi_path)
        Wperp = -Wu*math.sin(chi_path) + Wv*math.cos(chi_path)
        psi_ff = math.atan2(Wperp, max(0.1, Va))
        eta = _wrap_pi(eta + psi_ff)

    # 4) Lateral command: a_cmd = 2*Vg^2/L1 * sin(eta); convert to yaw-rate then to diff brake
    a_cmd = (2.0 * Vg * Vg / max(0.5, p.L1_dist)) * math.sin(eta)
    r_cmd = a_cmd / max(0.5, Vg)  # yaw-rate command [rad/s]

    # 5) Integral trim on cross-track (very small; helps residual bias)
    if p.kI > 0.0:
        st.I_bias += ey * p.kI * dt
        st.I_bias = max(-p.Imax, min(p.Imax, st.I_bias))
    else:
        st.I_bias = 0.0

    # Map to differential brake command
    diff_cmd = (r_cmd / max(1e-3, p.kr_yaw)) + st.I_bias
    diff_cmd = max(-p.diff_max, min(p.diff_max, diff_cmd))  # saturate

    # 6) Symmetric brake to regulate sink rate
    # Error is (desired - measured); vz is negative down
    evz = p.vz_ref - vz_meas
    sym_cmd = p.sym_min + p.sym_kp * evz
    sym_cmd = max(p.sym_min, min(p.sym_max, sym_cmd))

    return diff_cmd, sym_cmd, st
