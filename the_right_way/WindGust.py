import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
def _clip01(x): return max(0.0, min(1.0, x))

@dataclass
class WindGust:
    """
    https://www.weather.gov/mfl/beaufort
    
    Altitude-scaled Ornstein–Uhlenbeck gust model for u/v (and optional w).
    sigma grows as you descend; tau is shorter near ground, longer aloft.
    
    The higher we are the less of the gusts 
    But once we get within 1 to 2km of the ground, the gusts increase rapidly.
    Args:
        dt: float
            Time step in seconds.
        h_low: float
            Altitude (m) where sigma is max and tau is min.
        h_high: float
            Altitude (m) where sigma is min and tau is max.
        sigma_min: float
            Minimum horizontal gust stddev (m/s) at h_high.
        sigma_max: float
            Maximum horizontal gust stddev (m/s) at h_low.
        tau_low: float
            Minimum time constant (s) at h_low.
        tau_high: float
            Maximum time constant (s) at h_high.
        rho_xy: float
            Correlation between u and v gusts (0 to 0.9).
        vert_frac: float
            Vertical gust stddev as fraction of horizontal stddev.
    """
    dt: float
    h_low: float = 1500.0
    h_high: float = 30000.0
    sigma_min: float = 0.5     # m/s near ~30 km
    sigma_max: float = 4.0     # m/s near ~1.5 km
    tau_low:   float = 4.0     # s near ground
    tau_high:  float = 60.0    # s aloft
    rho_xy:    float = 0.4     # correlation between x/y gusts (0..0.9)
    vert_frac: float = 0.15    # vertical sigma as fraction of horizontal

    def __post_init__(self):
        # OU state
        self.wx = 0.0; self.wy = 0.0; self.wz = 0.0
        # 2D correlated draws via Cholesky on [[1,r],[r,1]]
        r = np.clip(self.rho_xy, -0.95, 0.95)
        self._L = np.array([[1.0, 0.0], [r, np.sqrt(1 - r**2)]], dtype=float)

    def _params_at(self, alt: float) -> Tuple[float,float]:
        alt = float(np.clip(alt, self.h_low, self.h_high))
        frac = (alt - self.h_low) / (self.h_high - self.h_low)  # 0 at low, 1 at high
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - frac)
        tau   = self.tau_low   + (self.tau_high  - self.tau_low)  * (frac)
        return sigma, tau

    def step(self, alt: float) -> Tuple[float,float,float]:
        sigma, tau = self._params_at(alt)
        a = np.exp(-self.dt / max(tau, 1e-6))           # exact discretization
        q = sigma * np.sqrt(1.0 - a*a)                  # process std per step

        # correlated horizontal innovations
        z = self._L @ np.random.randn(2)
        self.wx = a*self.wx + q*z[0]
        self.wy = a*self.wy + q*z[1]

        # vertical (reduced sigma, same tau)
        qz = self.vert_frac * q
        self.wz = a*self.wz + qz*np.random.randn()
        return self.wx, self.wy, self.wz
