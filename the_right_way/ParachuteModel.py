import numpy as np
import casadi as ca
from typing import List, Dict
from optitraj.models.casadi_model import CasadiModel
from matplotlib import pyplot as plt
from the_right_way.constants import (g, rho_0, R, 
    T_0, gamma, stefan_boltzmann, earth_radius, mu_earth,
    k_boltzmann, avogadro, molecular_mass_air, solar_flux, T_exosphere,
    specific_heat)

# --- Constants to match your plant ---
EARTH_RADIUS = 6.371e6       # m
MU_EARTH     = 3.986e14      # m^3/s^2
g0           = 9.80665       # (used only in ISA exponent)
R_air        = 287.05
T0           = 288.15
L            = 0.0065
p0           = 101325.0


def rho(alt) -> float:
    C1 = -3.9142e-14
    C2 = 3.6272e-9
    C3 = -1.1357e-4
    C4 = 1.2204
    rho_poly = C1*alt**3 + C2*alt**2 + C3*alt + C4
    return rho_poly

def gravity_variation(alt):
    earth_radius = 6.371e6  # Earth radius (m)
    mu_earth = 3.986e14  # Earth's gravitational parameter (m³/s²)
    """Calculates gravitational acceleration as function of altitude."""
    return mu_earth / (earth_radius + alt) ** 2

def shuttlecock_drag_coefficient(velocity:float, 
                                 altitude:float, 
                                 diameter:float,
                                 T_exosphere=1200) -> float:
    """
    Calculates velocity and altitude-dependent drag coefficient.
    Includes compressibility effects and rarefied gas effects.
    """
    
    #rho, T, p, a, mean_free_path = compute_atmospheric_properties(altitude, T_exosphere)
    atmospheric_properties: Dict[str, float] = compute_atmospheric_properties(altitude, T_exosphere)
    rho = atmospheric_properties['rho']
    T = atmospheric_properties['T']
    p = atmospheric_properties['p']
    a = atmospheric_properties['speed_of_sound']
    mean_free_path = atmospheric_properties['mean_free_path']
    
    mach_number = abs(velocity) / a if a > 0 else 0

    # Knudsen number
    knudsen_number = mean_free_path / diameter if diameter > 0 else float("inf")

    abs_vel = abs(velocity)
    ##%% Problem change 
    if abs_vel < 5:
        Cd_base = 1.2
    elif abs_vel < 15:
        Cd_base = 1.2 - 0.04 * (abs_vel - 5)
    elif abs_vel < 30:
        Cd_base = 0.8 - 0.01 * (abs_vel - 15)
    else:
        Cd_base = 0.65

    # Compressibility correction
    if mach_number < 0.3:
        compressibility_factor = 1.0
    elif mach_number < 0.8:
        compressibility_factor = 1.0 + 0.2 * (mach_number - 0.3) ** 2
    elif mach_number < 1.2:
        compressibility_factor = 1.1 + 0.8 * (mach_number - 0.8)
    else:
        compressibility_factor = 1.5 - 0.1 * np.exp(-(mach_number - 1.2) / 2.0)

    # Rarefied gas effects
    if knudsen_number < 0.01:  # Continuum flow
        rarefaction_factor = 1.0
    elif knudsen_number < 0.1:  # Slip flow
        rarefaction_factor = 1.0 + 0.1 * knudsen_number
    elif knudsen_number < 10:  # Transition flow
        rarefaction_factor = 1.0 + 0.3 * np.log10(knudsen_number + 1)
    else:  # Free molecular flow
        rarefaction_factor = 2.0

    return Cd_base * compressibility_factor * rarefaction_factor

def compute_atmospheric_properties(alt:float, T_exosphere=1200) -> Dict[str, float]:
    """
    Calculates atmospheric properties at given altitude using enhanced ISA model
    extended to thermosphere (up to 350km).
    Returns: density, temperature, pressure, speed_of_sound, mean_free_path
    Discrete segments where it occurs
    Args:
        alt (float): Altitude in meters.
        T_exosphere (float): Exosphere temperature for thermosphere model (K).
    Returns:
        Dict[str, float]: A dictionary containing density (kg/m^3), temperature (K),
                          pressure (Pa), speed_of_sound (m/s), mean_free_path (m).
    """
    if alt <= 11000:  # Troposphere
        T = T_0 - 0.0065 * alt
        p = 101325 * (T / T_0) ** 5.256
        rho = p / (R * T)
    elif alt <= 20000:  # Lower Stratosphere (isothermal)
        T = 216.65
        p = 22632 * np.exp(-0.0001577 * (alt - 11000))
        rho = p / (R * T)
    elif alt <= 32000:  # Upper Stratosphere
        T = 216.65 + 0.001 * (alt - 20000)
        p = 5474 * np.exp(-(alt - 20000) / (R * T / gravity_variation(alt)))
        rho = p / (R * T)
    elif alt <= 47000:  # Stratopause
        T = 228.65 + 0.0028 * (alt - 32000)
        p = 868 * np.exp(-(alt - 32000) / (R * T / gravity_variation(alt)))
        rho = p / (R * T)
    elif alt <= 86000:  # Mesosphere
        T = 270.65 - 0.0028 * (alt - 47000)
        p = 110 * np.exp(-(alt - 47000) / (R * T / gravity_variation(alt)))
        rho = p / (R * T)
    else:  # Thermosphere (alt > 86000m)
        T_86km = 270.65 - 0.0028 * (86000 - 47000)  # = 161.45 K
        T = T_86km + (T_exosphere - T_86km) * (1 - np.exp(-(alt - 86000) / 42000))
        T = min(T, T_exosphere)

        T_86km_calc = 270.65 - 0.0028 * (86000 - 47000)
        p_86km = 110 * np.exp(
            -(86000 - 47000) / (R * T_86km_calc / gravity_variation(86000))
        )
        H_scale = R * T / gravity_variation(alt)
        p = p_86km * np.exp(-(alt - 86000) / H_scale)
        rho = p / (R * T)
        rho = max(rho, 1e-20)

    # Speed of sound
    if T > 0:
        speed_of_sound = np.sqrt(gamma * R * T)
    else:
        speed_of_sound = 0

    # Mean free path
    if rho > 0:
        molecular_density = rho * avogadro / molecular_mass_air
        mean_free_path = 1 / (
            np.sqrt(2) * np.pi * (3.7e-10) ** 2 * molecular_density
        )
    else:
        mean_free_path = float("inf")
        
    information = {
        "rho": rho,
        "T": T,
        "p": p,
        "speed_of_sound": speed_of_sound,
        "mean_free_path": mean_free_path,
    }

    return information

def equations_of_motion(t:np.array, y:np.array, T_exosphere:float,
                        area:float, mass:float) -> List[float]:
    """
    SNC
    Defines the system of ODEs for the simulation.
    y[0] = altitude, y[1] = velocity
    Returns derivatives [d(altitude)/dt, d(velocity)/dt] ie [velocity, acceleration]
    Args:
        y (np.array): State vector [altitude (m), velocity (m/s)]
        T_exosphere (float): Exosphere temperature for thermosphere model (K).
        area (float): Cross-sectional area of the payload (m^2).
        mass (float): Mass of the payload system (kg).
        
    Returns:
        List[float]: Derivatives [d(altitude)/dt, d(velocity)/dt]
    """
    alt, vel = y

    # Gravity
    #g_current = gravity_variation(alt)
    g_current = 9.81  # m/s^2, constant gravity

    # Drag
    #rho, _, _, _, _ = compute_atmospheric_properties(alt, T_exosphere)
    atmospheric_properties:Dict[str, float] = compute_atmospheric_properties(alt, T_exosphere)
    rho = atmospheric_properties['rho']
    current_Cd = shuttlecock_drag_coefficient(abs(vel), alt, T_exosphere)
    if abs(vel) > 0:
        F_drag_magnitude = 0.5 * rho * area * current_Cd * abs(vel) ** 2
        F_drag = -F_drag_magnitude * np.sign(vel)
    else:
        F_drag = 0

    accel = -g_current + F_drag / mass
    #print("mass:", mass, "F_drag:", F_drag, "vel:", vel, "alt:", alt, "acc:", accel, "rho:", rho, "Cd:", current_Cd)
    # print("mass:", mass, "F_drag:", F_drag, "vel:", vel, "alt:", alt, "acc:", accel, "rho:", rho, "Cd:", current_Cd, "g:", g_current, "area:", area)

    return [vel, accel]


def rho_isa_troposphere(z):
    """
    ISA troposphere density (0–11 km), same form as in your plant.
    Uses CasADi ops if z is SX.
    """
    # clamp z to [0, 11000] to avoid invalid exponents
    zc = ca.fmin(ca.fmax(z, 0.0), 11000.0)
    T  = T0 - L*zc
    expo = g0/(R_air*L)
    p  = p0 * ca.power(T/T0, expo)
    rho = p/(R_air*T)
    return rho

def gravity_var(z):
    return MU_EARTH / ca.power(EARTH_RADIUS + z, 2)


class ParachuteModel(CasadiModel):
    """
    MPC internal model that matches the plant's ODE:
      states:  x, y, z (altitude up), psi, vz (vertical speed, <0 in descent)
      inputs:  gr_control (>=0), s_control (>0), u_psi (yaw rate)
      wind:    wx, wy (world), wz (ignored here to match plant which uses 0)
    """
    def __init__(self, mass_kg=0.5, Cd=0.7, include_time=False, dt_val=0.125):
        super().__init__()
        self.include_time = include_time
        self.dt_val = dt_val
        self.mass_kg = mass_kg
        self.Cd_const = Cd        # you can replace with a CasADi Cd(v,z) later
        self.define_states()
        self.define_controls()
        self.define_wind()
        self.define_state_space()

    def define_states(self):
        self.x_f   = ca.SX.sym('x_f')
        self.y_f   = ca.SX.sym('y_f')
        self.z_f   = ca.SX.sym('z_f')     # altitude (up positive)
        self.psi_f = ca.SX.sym('psi_f')   # heading (rad)
        self.vz_f  = ca.SX.sym('vz_f')    # vertical velocity (m/s), <0 when descending
        self.states = ca.vertcat(self.x_f, self.y_f, self.z_f, self.psi_f, self.vz_f)
        self.n_states = int(self.states.size()[0])

    def define_wind(self):
        self.wind_x = ca.SX.sym('wind_x')  # world wind x (m/s)
        self.wind_y = ca.SX.sym('wind_y')  # world wind y (m/s)
        self.wind_z = ca.SX.sym('wind_z')  # (kept for API; set 0 to match plant)
        self.wind   = ca.vertcat(self.wind_x, self.wind_y, self.wind_z)
        self.n_wind = int(self.wind.size()[0])

    def define_controls(self):
        self.gr_control = ca.SX.sym('gr_control')   # >= 0
        self.s_control  = ca.SX.sym('s_control')    # > 0 (projected area)
        self.u_psi      = ca.SX.sym('u_psi')        # yaw rate (rad/s)
        self.controls   = ca.vertcat(self.gr_control, self.s_control, self.u_psi)
        self.n_controls = int(self.controls.size()[0])

    def _rho(self, z):
        # Use ISA troposphere (matches your plant at these altitudes).
        # Swap to a CasADi port of compute_atmospheric_properties(z)['rho'] if you need the full profile.
        return rho_isa_troposphere(z)

    def _Cd(self, v_abs, z):
        # For perfect matching, port your shuttlecock_drag_coefficient(|v|, z, ...) to CasADi here.
        # As an interim, use a constant (your plant often sits ~0.9–1.2 in logs).
        return self.Cd_const

    def define_state_space(self):
        # Guards (identical on both sides)
        S_min   = 1e-4
        rho_min = 1e-5
        S = ca.fmax(self.s_control, S_min)

        # Atmosphere and gravity (match plant)
        rho_air = ca.fmax(self._rho(self.z_f), rho_min)
        g_cur   = gravity_var(self.z_f)

        # Drag coefficient (replace with your CasADi Cd model when ready)
        Cd = self._Cd(ca.fabs(self.vz_f), self.z_f)

        # Vertical ODE (matches plant exactly):
        z_dot  = self.vz_f
        # F_drag/m = -(0.5*rho*S*Cd/m) * vz*|vz|
        drag_acc = -(0.5 * rho_air * S * Cd / self.mass_kg) * self.vz_f * ca.fabs(self.vz_f)
        vz_dot   = -g_cur + drag_acc

        # Horizontal airspeed from GR and vertical drop magnitude
        v_h_air = self.gr_control * (-self.vz_f)         # >= 0 during descent
        x_dot   = v_h_air * ca.cos(self.psi_f) + self.wind_x
        y_dot   = v_h_air * ca.sin(self.psi_f) + self.wind_y
        psi_dot = self.u_psi

        self.z_dot = ca.vertcat(x_dot, y_dot, z_dot, psi_dot, vz_dot)
        self.function = ca.Function('f', [self.states, self.controls, self.wind], [self.z_dot])

    # RK4 stepper (for discretization inside MPC or simulation)
    def rk4(self, x, u, dt, wind, use_numeric=True):
        k1 = self.function(x, u, wind)
        k2 = self.function(x + 0.5*dt*k1, u, wind)
        k3 = self.function(x + 0.5*dt*k2, u, wind)
        k4 = self.function(x + dt*k3,     u, wind)
        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return np.array(x_next).flatten() if use_numeric else x_next