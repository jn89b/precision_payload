import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Any

from optitraj.models.casadi_model import CasadiModel
from optitraj.close_loop import CloseLoopSim

from the_right_way.ParachuteModel import ParachuteModel
from the_right_way.OptimalParachute import MPCParachute

from the_right_way.Filters import ThirdOrderButterworthFilter, FirstOrderAngleFilter
from the_right_way.constants import (g, rho_0, T_0, R, gamma, stefan_boltzmann,
                                    earth_radius, mu_earth,
                                    k_boltzmann, avogadro, molecular_mass_air, solar_flux, T_exosphere)
from the_right_way.Battery import BatteryHistory, BatteryState
from the_right_way.Hardware import (GPSConfig, GPSPowerManager,
                                    PayloadConfig, PayloadPowerController)
from the_right_way.WindGust import WindGust
from the_right_way.MonteCarlo import MonteCarloInformation
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import List, Optional

# set configurations for plottings
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# set tight layout for plots
rcParams['figure.constrained_layout.use'] = True
rcParams['figure.constrained_layout.w_pad'] = 0.1
rcParams['figure.constrained_layout.h_pad'] = 0.1
# large font sizes
rcParams['font.size'] = 18


@dataclass
class ParachuteHistory:
    time: List[float] = field(default_factory=list)
    altitude: List[float] = field(default_factory=list)
    x : List[float] = field(default_factory=list)
    y : List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    vertical_velocity: List[float] = field(default_factory=list)
    gr: List[float] = field(default_factory=list)
    psi: List[float] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    success: List[bool] = field(default_factory=list)
    wind_x: List[float] = field(default_factory=list)
    wind_y: List[float] = field(default_factory=list)
    wind_z: List[float] = field(default_factory=list)
    pred_wind_x: List[float] = field(default_factory=list)
    pred_wind_y: List[float] = field(default_factory=list)
    desired_psi: List[float] = field(default_factory=list)
    desired_gr: List[float] = field(default_factory=list)
    desired_s: List[float] = field(default_factory=list)
    x_traj: List[float] = field(default_factory=list)
    y_traj: List[float] = field(default_factory=list)
    z_traj: List[float] = field(default_factory=list)
    
@dataclass
class ThermalSimulationHistory:
    acceleration: List[float] = field(default_factory=list)
    drag_force: List[float] = field(default_factory=list)
    drag_coefficient: List[float] = field(default_factory=list)
    heating_rate: List[float] = field(default_factory=list)
    surface_temp: List[float] = field(default_factory=list)
    stagnation_temp: List[float] = field(default_factory=list)
    mach_number: List[float] = field(default_factory=list)
    reynolds_number: List[float] = field(default_factory=list)
    knudsen_number: List[float] = field(default_factory=list)
    air_density: List[float] = field(default_factory=list)
    air_temperature: List[float] = field(default_factory=list)
    mean_free_path: List[float] = field(default_factory=list)
    gravity: List[float] = field(default_factory=list)
    solar_heating: List[float] = field(default_factory=list)
    earth_ir: List[float] = field(default_factory=list)
    convective_heating: List[float] = field(default_factory=list)
    radiative_cooling: List[float] = field(default_factory=list)
    shock_heating: List[float] = field(default_factory=list)
    q_rad_space: List[float] = field(default_factory=list)
    q_rad_atm_cooling: List[float] = field(default_factory=list)
    q_rad_atm_heating: List[float] = field(default_factory=list)
    interior_temp: List[float] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    
    def record_step(
        self,
        accel, F_drag, current_Cd, q_net,
        temp_surface_current, T_stag, mach, reynolds,
        knudsen, rho, T_amb, mean_free_path,
        g_current, q_solar, q_earth_ir, q_conv,
        q_rad_total, q_shock, q_rad_space,
        q_rad_atm_cooling, q_rad_atm_heating,
        interior_temp
    ):
        self.acceleration.append(accel)
        self.drag_force.append(abs(F_drag))
        self.drag_coefficient.append(current_Cd)
        self.heating_rate.append(q_net)
        self.surface_temp.append(temp_surface_current)
        self.stagnation_temp.append(T_stag)
        self.mach_number.append(mach)
        self.reynolds_number.append(reynolds)
        self.knudsen_number.append(knudsen)
        self.air_density.append(rho)
        self.air_temperature.append(T_amb)
        self.mean_free_path.append(mean_free_path)
        self.gravity.append(g_current)
        self.solar_heating.append(q_solar)
        self.earth_ir.append(q_earth_ir)
        self.convective_heating.append(q_conv)
        self.radiative_cooling.append(q_rad_total)
        self.shock_heating.append(q_shock)
        self.q_rad_space.append(q_rad_space)
        self.q_rad_atm_cooling.append(q_rad_atm_cooling)
        self.q_rad_atm_heating.append(q_rad_atm_heating)
        self.interior_temp.append(interior_temp)
    
def plot_aerothermal_from_history(history: ThermalSimulationHistory,
                                  time_arr: np.ndarray,
                                  altitude_arr: np.ndarray,
                                  velocity_arr: np.ndarray) -> None:
    """
    Recreate the 'comprehensive aerothermal analysis' plots from the class-based run.
    The arrays must be length-aligned with history (same number of time steps).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert lists -> arrays
    h = history
    time = np.asarray(time_arr)
    altitude = np.asarray(altitude_arr)
    velocity = np.asarray(velocity_arr)

    # 4x3 grid
    fig2 = plt.figure(figsize=(20, 16))

    # 1) Altitude vs Time
    ax1 = fig2.add_subplot(4, 3, 1)
    ax1.plot(time, altitude / 1000, linewidth=2)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Altitude vs Time"); ax1.grid(True)

    # 2) Velocity vs Time
    ax2 = fig2.add_subplot(4, 3, 2)
    ax2.plot(time, velocity, linewidth=2)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity vs Time"); ax2.grid(True)

    # 3) Surface & Air Temperature vs Time
    ax3 = fig2.add_subplot(4, 3, 3)
    ax3.plot(time, h.surface_temp, linewidth=2)
    ax3.plot(time, h.air_temperature, linewidth=2, alpha=0.7)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Temperature (K)")
    ax3.set_title("Surface & Air Temperature")
    ax3.legend(["Surface Temp", "Air Temp"]); ax3.grid(True)

    # 4) Heating Rate vs Time
    ax4 = fig2.add_subplot(4, 3, 4)
    ax4.plot(time, h.heating_rate, linewidth=2)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Heating Rate (W/m²)")
    ax4.set_title("Aerodynamic Heating Rate"); ax4.grid(True)

    # 5) Mach Number vs Time
    ax5 = fig2.add_subplot(4, 3, 5)
    ax5.plot(time, h.mach_number, linewidth=2)
    ax5.axhline(y=1.0, linestyle="--", alpha=0.7, label="Mach 1")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Mach Number")
    ax5.set_title("Mach Number vs Time"); ax5.legend(); ax5.grid(True)

    # 6) Drag Force vs Velocity
    ax6 = fig2.add_subplot(4, 3, 6)
    ax6.plot(velocity, h.drag_force, linewidth=2)
    ax6.set_xlabel("Velocity (m/s)"); ax6.set_ylabel("Drag Force (N)")
    ax6.set_title("Drag Force vs Velocity"); ax6.grid(True)

    # 7) Drag Coefficient vs Mach (colored by log10(Kn))
    ax7 = fig2.add_subplot(4, 3, 7)
    kn = np.asarray(h.knudsen_number)
    sc = ax7.scatter(h.mach_number, h.drag_coefficient,
                     c=np.log10(np.clip(kn, 1e-30, None)), cmap="plasma", s=20)
    ax7.set_xlabel("Mach Number"); ax7.set_ylabel("Drag Coefficient (Cd)")
    ax7.set_title("Drag Coefficient vs Mach (colored by log10(Kn))")
    ax7.axvline(x=0.8, linestyle="--", alpha=0.7, label="Transonic onset")
    ax7.axvline(x=1.0, linestyle="--", alpha=0.7, label="Sonic")
    ax7.axvline(x=1.2, linestyle="--", alpha=0.7, label="Supersonic")
    cbar7 = plt.colorbar(sc, ax=ax7); cbar7.set_label("log10(Knudsen Number)")
    ax7.legend(); ax7.grid(True)

    # 8) Reynolds Number vs Altitude
    ax8 = fig2.add_subplot(4, 3, 8)
    ax8.plot(altitude / 1000, h.reynolds_number, linewidth=2)
    ax8.set_xlabel("Altitude (km)"); ax8.set_ylabel("Reynolds Number")
    ax8.set_title("Reynolds Number vs Altitude")
    ax8.set_yscale("log"); ax8.grid(True)

    # 9) Velocity vs Altitude
    ax9 = fig2.add_subplot(4, 3, 9)
    ax9.plot(velocity, altitude / 1000, linewidth=2)
    ax9.set_xlabel("Velocity (m/s)"); ax9.set_ylabel("Altitude (km)")
    ax9.set_title("Velocity vs Altitude"); ax9.grid(True)

    # 10) Temperature vs Altitude
    ax10 = fig2.add_subplot(4, 3, 10)
    ax10.plot(h.surface_temp, altitude / 1000, linewidth=2, label="Surface")
    ax10.plot(h.stagnation_temp, altitude / 1000, linewidth=2, label="Stagnation")
    ax10.set_xlabel("Temperature (K)"); ax10.set_ylabel("Altitude (km)")
    ax10.set_title("Temperature Profiles"); ax10.legend(); ax10.grid(True)

    # 11) Knudsen Number vs Altitude
    ax11 = fig2.add_subplot(4, 3, 11)
    ax11.plot(altitude / 1000, h.knudsen_number, linewidth=2)
    ax11.axhline(y=0.01, linestyle="--", alpha=0.7, label="Continuum limit")
    ax11.axhline(y=0.1, linestyle="--", alpha=0.7, label="Slip flow")
    ax11.axhline(y=10, linestyle="--", alpha=0.7, label="Free molecular")
    ax11.set_xlabel("Altitude (km)"); ax11.set_ylabel("Knudsen Number")
    ax11.set_title("Flow Regime vs Altitude")
    ax11.set_yscale("log"); ax11.legend(); ax11.grid(True)

    # 12) Heating Components vs Time
    ax12 = fig2.add_subplot(4, 3, 12)
    ax12.plot(time, h.heating_rate, label="Net Heating", linewidth=2)
    ax12.plot(time, h.convective_heating, label="Convective", linestyle="--")
    ax12.plot(time, h.shock_heating, label="Shock Rad", linestyle=":")
    ax12.plot(time, -np.asarray(h.radiative_cooling), label="Total Cooling (-)", linestyle="--")
    ax12.set_xlabel("Time (s)"); ax12.set_ylabel("Heating Rate (W/m²)")
    ax12.set_title("Heating Components vs Time")
    ax12.legend(); ax12.grid(True); ax12.set_yscale("symlog")

    plt.tight_layout()
    plt.savefig("aerothermal_analysis.png", dpi=300, bbox_inches="tight")

    # 2-panel Cd analysis
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    sc_left = ax_left.scatter(h.mach_number, h.drag_coefficient,
                              c=altitude / 1000, cmap="viridis", s=30, alpha=0.7)
    ax_left.set_xlabel("Mach Number"); ax_left.set_ylabel("Drag Coefficient (Cd)")
    ax_left.set_title("Drag Coefficient vs Mach Number (Colored by Altitude)")
    cbar_left = plt.colorbar(sc_left, ax=ax_left); cbar_left.set_label("Altitude (km)")
    ax_left.grid(True)

    sc_right = ax_right.scatter(h.knudsen_number, h.drag_coefficient,
                                c=h.mach_number, cmap="plasma", s=30, alpha=0.7)
    ax_right.set_xlabel("Knudsen Number"); ax_right.set_ylabel("Drag Coefficient (Cd)")
    ax_right.set_title("Drag Coefficient vs Knudsen Number (Rarefaction Effects)")
    ax_right.set_xscale("log")
    ax_right.axvline(x=0.01, linestyle="--", alpha=0.7, label="Continuum limit")
    ax_right.axvline(x=0.1, linestyle="--", alpha=0.7, label="Slip flow")
    ax_right.axvline(x=10, linestyle="--", alpha=0.7, label="Free molecular")
    cbar_right = plt.colorbar(sc_right, ax=ax_right); cbar_right.set_label("Mach Number")
    ax_right.legend(); ax_right.grid(True)
    plt.tight_layout()
    plt.savefig("drag_coefficient_analysis.png", dpi=300, bbox_inches="tight")

    # Heat flux components
    fig4, ax13 = plt.subplots(figsize=(12, 8))
    ax13.plot(time, h.heating_rate, label="Net Heating (Result)", linewidth=2.5)
    ax13.plot(time, h.convective_heating, label="Convective (q_conv)", linestyle="--")
    ax13.plot(time, h.solar_heating, label="Solar (q_solar)", linestyle="--")
    ax13.plot(time, h.earth_ir, label="Earth IR (q_earth_ir)", linestyle="--")
    ax13.plot(time, h.q_rad_atm_heating, label="Atmospheric Rad In (q_rad_atm_in)", linestyle="--")
    ax13.plot(time, -np.asarray(h.q_rad_space), label="Space Rad Out (-q_rad_out_space)", linestyle=":")
    ax13.plot(time, -np.asarray(h.q_rad_atm_cooling), label="Atmospheric Rad Out (-q_rad_out_atm)", linestyle=":")
    ax13.plot(time, h.shock_heating, label="Shock Radiation (q_rad_shock)", linestyle="--")
    ax13.set_xlabel("Time (s)"); ax13.set_ylabel("Heating Rate (W/m²)")
    ax13.set_title("Individual Heat Flux Components vs Time")
    ax13.legend(); ax13.grid(True); ax13.set_yscale("symlog")
    plt.tight_layout()
    plt.savefig("heat_flux_components.png", dpi=300, bbox_inches="tight")

    # Interior temperature
    fig5, ax14 = plt.subplots(figsize=(12, 6))
    ax14.plot(time, h.interior_temp, label="Interior Temperature", linewidth=2)
    ax14.set_xlabel("Time (s)"); ax14.set_ylabel("Temperature (K)")
    ax14.set_title("Interior Temperature vs Time")
    ax14.legend(); ax14.grid(True)

    print("\nPlots saved:")
    print("- aerothermal_analysis.png")
    print("- drag_coefficient_analysis.png")
    print("- heat_flux_components.png")


    
def gravity_variation(alt:float) -> float:
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
    
def aerodynamic_heating_rate(velocity:float, 
                             altitude:float, 
                             surface_temp:float, 
                             diameter:float,
                             total_surface_area:float,
                             specific_heat:float,
                             emissivity:float,
                             T_exosphere=1200) -> Dict[str, float]:
    """
    Calculates aerodynamic heating rate with enhanced models for extreme altitude.
    Includes solar heating, Earth's IR radiation, and plasma effects.
    
    Args:
        velocity (float): Velocity of the object (m/s).
        altitude (float): Altitude of the object (m).
        surface_temp (float): Surface temperature of the object (K).
        diameter (float): Diameter of the object (m).
        total_surface_area (float): Total surface area of the object (m^2).
        specific_heat (float): Specific heat capacity of the object's material (J/kg·K).
        emissivity (float): Emissivity of the object's surface.
        T_exosphere (float): Exosphere temperature for thermosphere model (K).
    Returns:
        Dict[str, float]: A dictionary containing various heating/cooling rates and stagnation temperature.
        
    """
    atmospheric_properties: Dict[str, float] = compute_atmospheric_properties(altitude, T_exosphere)
    rho = atmospheric_properties['rho']
    T_amb = atmospheric_properties['T']
    p = atmospheric_properties['p']
    a = atmospheric_properties['speed_of_sound']
    mean_free_path = atmospheric_properties['mean_free_path']
    
    if velocity < 1:
        return {
            "q_net": 0.0,
            "T_stagnation": T_amb,
            "q_conv": 0.0,
            "total_cooling_for_plot": 0.0,
            "solar_heating": 0.0,
            "earth_ir": 0.0,
            "q_rad_shock": 0.0,
            "q_rad_space": 0.0,
            "q_rad_atm_cooling": 0.0,
            "q_rad_atm_heating": 0.0,
        }

    # Knudsen number
    knudsen_number = mean_free_path / diameter if diameter > 0 else float("inf")

    # Stagnation temperature
    mach_number = velocity / a if a > 0 else 0
    if mach_number < 5:
        T_stagnation = T_amb * (1 + (gamma - 1) / 2 * mach_number**2)
    else:
        T_stagnation = T_amb * (0.2 * mach_number**2 + 0.5 * mach_number + 1)

    # Heat transfer regimes
    q_conv_continuum = 0
    q_conv_transition = 0
    q_conv_molecular = 0

    nose_radius = diameter / 2
    if rho > 0 and nose_radius > 0 and velocity > 1 and T_stagnation > 0:
        h_continuum = (
            (1.83e-4 * np.sqrt(rho / nose_radius) * velocity**3) / T_stagnation
        )
        q_conv_continuum = h_continuum * (T_stagnation - surface_temp)

    if T_amb > 0 and velocity > 1:
        h_conv_transition = (
            0.5 * rho * velocity * specific_heat * np.sqrt(T_stagnation / T_amb)
        )
        q_conv_transition = h_conv_transition * (T_stagnation - surface_temp) * (
            1 + knudsen_number
        )

    if T_amb > 0 and velocity > 1:
        molecular_velocity = np.sqrt(
            8 * k_boltzmann * T_amb / (np.pi * molecular_mass_air / avogadro)
        )
        h_conv_molecular = 0.5 * rho * molecular_velocity * specific_heat
        q_conv_molecular = h_conv_molecular * (T_stagnation - surface_temp)

    # Weighted regime blending
    if knudsen_number < 0.01:
        weight_continuum, weight_transition, weight_molecular = 1.0, 0.0, 0.0
    elif knudsen_number < 0.1:
        weight_continuum = (0.1 - knudsen_number) / (0.1 - 0.01)
        weight_transition = 1.0 - weight_continuum
        weight_molecular = 0.0
    elif knudsen_number < 10:
        weight_continuum = 0.0
        weight_transition = (10 - knudsen_number) / (10 - 0.1)
        weight_molecular = 1.0 - weight_transition
    else:
        weight_continuum, weight_transition, weight_molecular = 0.0, 0.0, 1.0

    q_conv = (
        weight_continuum * q_conv_continuum
        + weight_transition * q_conv_transition
        + weight_molecular * q_conv_molecular
    )

    # Solar heating (above 100 km)
    if altitude > 100000:
        altitude_factor = min(1.0, (altitude - 100000) / 50000)
        optical_depth = max(0, rho * altitude / 10000)
        solar_absorption = np.exp(-optical_depth)
        solar_heating = (
            solar_flux
            * 0.3
            * solar_absorption
            * altitude_factor
            * (total_surface_area / 4)
            / total_surface_area
        )
    else:
        solar_heating = 0

    # Earth IR
    if altitude > 50000:
        if altitude < 120000:
            atm_fraction = np.exp(-altitude / 30000)
            earth_ir_atm = 50 * atm_fraction
            earth_angle = np.arctan(earth_radius / (earth_radius + altitude))
            view_factor = (1 - np.cos(earth_angle)) / 2
            earth_ir_direct = 240 * view_factor * (
                earth_radius / (earth_radius + altitude)
            ) ** 2
            altitude_factor = (altitude - 50000) / 70000
            earth_ir = earth_ir_atm + earth_ir_direct * altitude_factor
        else:
            earth_angle = np.arctan(earth_radius / (earth_radius + altitude))
            view_factor = (1 - np.cos(earth_angle)) / 2
            earth_ir = 240 * view_factor * (
                earth_radius / (earth_radius + altitude)
            ) ** 2
    else:
        earth_ir = 0

    # Radiative cooling/exchange
    scale_height = 8000.0
    kappa = 1.0
    space_view_factor = np.exp(-kappa * rho * scale_height)
    atm_view_factor = 1.0 - space_view_factor

    space_temp = 2.7
    q_rad_space = (
        space_view_factor
        * emissivity
        * stefan_boltzmann
        * (surface_temp**4 - space_temp**4)
    )

    q_rad_atm_cooling, q_rad_atm_heating = 0, 0
    if surface_temp > T_amb:
        q_rad_atm_cooling = (
            atm_view_factor
            * emissivity
            * stefan_boltzmann
            * (surface_temp**4 - T_amb**4)
        )
    else:
        q_rad_atm_heating = (
            atm_view_factor
            * emissivity
            * stefan_boltzmann
            * (T_amb**4 - surface_temp**4)
        )

    total_heating = q_conv + solar_heating + earth_ir + q_rad_atm_heating
    total_cooling = q_rad_space + q_rad_atm_cooling
    q_net = total_heating - total_cooling

    if abs(q_net) > 1000000:
        q_net = np.sign(q_net) * 1000000

    q_rad_shock = 0
    if mach_number > 5 and knudsen_number < 10:
        nose_radius = diameter / 2
        q_rad_shock = (
            1.83e-4 * (nose_radius**0.5) * (rho**1.22) * (velocity**3.05)
        )

    q_net += q_rad_shock
    total_cooling_for_plot = q_rad_space + q_rad_atm_cooling
    # print("alt:", altitude, "vel:", velocity, "rho:", rho, "Cd:", shuttlecock_drag_coefficient(velocity, altitude, T_exosphere), "q_net:", q_net, "q_conv:", q_conv, "q_rad_shock:", q_rad_shock)
    # print("q_net:", q_net, "q_conv:", q_conv, "q_rad_shock:", q_rad_shock, "solar_heating:", solar_heating, "earth_ir:", earth_ir, "q_rad_space:", q_rad_space, "q_rad_atm_cooling:", 
    #       q_rad_atm_cooling, "q_rad_atm_heating:", q_rad_atm_heating)
    information = {
        "q_net": q_net,
        "T_stagnation": T_stagnation,
        "q_conv": q_conv,
        "total_cooling_for_plot": total_cooling_for_plot,
        "solar_heating": solar_heating,
        "earth_ir": earth_ir,
        "q_rad_shock": q_rad_shock,
        "q_rad_space": q_rad_space,
        "q_rad_atm_cooling": q_rad_atm_cooling,
        "q_rad_atm_heating": q_rad_atm_heating,
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
    g_current = gravity_variation(alt)

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
    # print("mass:", mass, "F_drag:", F_drag, "vel:", vel, "alt:", alt, "acc:", accel, "rho:", rho, "Cd:", current_Cd)

    return [vel, accel]


def rho(alt:float) -> float:
    """
    Good enough for lower altitudes
    """
    C1 = -3.9142e-14
    C2 = 3.6272e-9
    C3 = -1.1357e-4
    C4 = 1.2204
    rho_poly = C1*alt**3 + C2*alt**2 + C3*alt + C4
    return rho_poly
    


class UpdatedCloseLoopSim(CloseLoopSim):
    """
    We're going to have to update the run_single_step method to handle the new
    dynamics since P is now a vector that includes the initial conditions
    and the threat parameters.
    
    """
    def __init__(self, optimizer: MPCParachute,
                    x_init: np.ndarray,
                    x_final: np.ndarray,
                    u0: np.ndarray,
                    N: int = 100,
                    log_data: bool = True,
                    stop_criteria: callable = None,
                    print_every: int = 10) -> None:
        
        self.optimizer: MPCParachute = optimizer 
        super().__init__(optimizer=optimizer, x_init=x_init,
                         x_final=x_final, u0=u0, N=N,
                         log_data=log_data, stop_criteria=stop_criteria)

    def run_single_step(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray,
                        wind_data:np.ndarray) -> None:
        """
        Run a single step of the closed loop simulation.
        """
        self.current_step += 1

        if xF is not None:
            self.update_x_final(xF)

        if x0 is not None:
            self.update_x_init(x0)

        if u0 is not None:
            self.update_u0(u0)

        if self.dynamics_adapter is not None and self.update_controller:
            # print("Updating controller")
            sol = self.optimizer.solve_and_get_solution(
                self.x_init, self.x_final, self.u0, wind_data)
            self.dynamics_adapter.set_controls(sol['states'],
                                               sol['controls'],
                                               self.ctrl_idx,
                                               xF=self.x_final)
            self.update_controller = False
        else:
            sol = self.optimizer.solve_and_get_solution(
                self.x_init, self.x_final, self.u0, wind_data)

        if self.current_step % self.print_every == 0:
            print("step: ", self.x_init)

        self.update_time()

        # run the dynamics system if it is there
        if self.dynamics_adapter is not None:
            self.dynamics_adapter.run()
            self.x_init = self.dynamics_adapter.get_state_information()
            self.u0 = self.dynamics_adapter.get_control_information()
        else:
            self.shift_next(sol)

        if self.log_data:
            self.save_data(sol)

        # def check criteria
        if self.stop_criteria is not None and self.stop_criteria(
                self.x_init, self.x_final):
            print('Stopping criteria met')
            self.report.save_everything()
            self.done = True
            return sol
        
        return sol

class ParachuteSimulator():
    def __init__(self,
                pred_wind_data_path:str,
                actual_wind_data_path:str,
                predictive_control: MPCParachute,
                starting_conditions: Dict[str, float],
                parachute_config: Dict[str, float],
                psi_filter: FirstOrderAngleFilter,
                glide_ratio_filter: ThirdOrderButterworthFilter,
                s_filter: ThirdOrderButterworthFilter,
                battery_cfg: BatteryState,
                gps_cfg: GPSConfig,
                payload_cfg: PayloadConfig,
                wind_gust: WindGust,
                dt: float = 0.05,
                info: Optional["MonteCarloInformation"]=None):
        self.pred_wind_data_path:str = pred_wind_data_path
        self.actual_wind_data_path:str = actual_wind_data_path
        self.predictive_control: MPCParachute = predictive_control
        
        self.pred_wind_data: np.ndarray = self.load_wind_data(use_pred=True)
        self.actual_wind_data: np.ndarray = self.load_wind_data(use_pred=False)
        self.starting_conditions: Dict[str, float] = starting_conditions
        self.parachute_config: Dict[str, float] = parachute_config
        self.dt: float = dt
        self.info = info
        
        # these two parameters are used to control/regulate how often 
        # we forecast the wind data  
        self.control_wind_forecast: bool = False
        self.wind_forecast_update_m:int = 1000   
        # self.psi_filter = FirstOrderAngleFilter(dt=self.dt,
        #                                     tau=0.7,                 # time constant (s)
        #                                     y0=0.0,                  # initial heading (rad)
        #                                     max_rate_rad_s=np.deg2rad(90))  # optional
        self.psi_filter: FirstOrderAngleFilter = psi_filter

        self.glide_ratio_filter: ThirdOrderButterworthFilter = glide_ratio_filter
        self.s_filter: ThirdOrderButterworthFilter = s_filter
        
        
        self.area = parachute_config['S']
        self.thermal_sim_history: ThermalSimulationHistory = ThermalSimulationHistory()
        self.interior_temp = parachute_config['temp_surface_current']
        self.battery_cfg: BatteryState = battery_cfg
        self.gps_cfg: GPSConfig = gps_cfg
        self.payload_cfg: PayloadConfig = payload_cfg
        
        self.gps = GPSPowerManager(cfg=gps_cfg)
        self.payload = PayloadPowerController(cfg=payload_cfg,
                                              gps_mgr=self.gps)
        self.phase = 'descent' # can be 'descent' or 'landing'
        self.mass = self.compute_sum_mass()
        self.parachute_history: ParachuteHistory = ParachuteHistory()
        self.wind_gust:WindGust = wind_gust
        
        self.command_failure_rate:float = self.starting_conditions['command_failure_rate']
        self.control_frequency_hz:float = self.starting_conditions['command_frequency_hz']
        print("Control frequency (Hz):", self.control_frequency_hz)
        self.control_interval_steps:int = int(1.0 / (self.control_frequency_hz * self.dt))
        self.init_monte_carlo_info()
        if self.info is None:
            self.info = MonteCarloInformation()
        
    def init_monte_carlo_info(self) -> None:
        self._design_snapshot = dict(
            mass_kg=self.parachute_config["M"],
            Cd=self.parachute_config["Cd"],
            S_min=self.parachute_config["Smin"],
            S_max=self.parachute_config["Smax"],
            emissivity=self.parachute_config.get("emissivity", 0.3),
            specific_heat_J_per_kgK=self.parachute_config.get("specific_heat", 600.0),
            thermal_conductivity_W_per_mK=self.parachute_config.get("thermal_conductivity", 0.1),
            total_surface_area_m2=self.parachute_config.get("total_surface_area", 0.008),
            diameter_m=self.parachute_config.get("diameter", 0.058),
            length_m=self.parachute_config.get("length", 0.068),
        )

        # control snapshot (from your limits + filters)
        self._control_snapshot = dict(
            glide_ratio_min=self.starting_conditions.get("GRmin", 0.1),
            glide_ratio_max=self.starting_conditions.get("GRmax", 0.4),
            s_control_min=self.parachute_config["Smin"],
            s_control_max=self.parachute_config["Smax"],
            u_psi_min_rad=-np.deg2rad(60),  # from control limits in your driver
            u_psi_max_rad= np.deg2rad(60),
            tau_psi_s=getattr(self.psi_filter, "tau", 0.5),
            tau_gr_s =getattr(self.glide_ratio_filter, "tau", 1.0),
            tau_s_s  =getattr(self.s_filter, "tau", 5.0),
            control_frequency_hz=float(self.starting_conditions["command_frequency_hz"]),
        )

        self._noise_snapshot = dict(
            gps_xy_sigma_m=float(self.starting_conditions["position_sensor_noise_std"]),
            gps_z_sigma_m=float(self.starting_conditions["vertical_position_sensor_noise_std"]),
            heading_sigma_deg=float(self.starting_conditions["heading_sensor_noise_std"]),
            cmd_drop_p=float(self.starting_conditions.get("command_failure_rate", 0.02)),
        )

        self._power_snapshot = dict(
            voltage_cutoff_V=float(self.battery_cfg.V_cutoff)
        )

        # bookkeeping for events
        self._cmd_drops = 0
        self._burst_drop_seconds = 0.0
        self._stiction_events = 0
        self._hard_fail = False
        self._brownout = False
        self._cutoff_triggered = False
        
    def compute_sum_mass(self) -> float:
        """
        Compute the total mass of the system.
        """
        divide_to_kg = 1000
        total_mass = self.parachute_config['M'] + \
            self.battery_cfg.mass_g/divide_to_kg + \
            self.payload_cfg.mass_board_g/divide_to_kg 
            
        self.parachute_config['M'] = total_mass

        return total_mass
    
    def carp_estimator(self, pred_wind) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CARP estimator using fixed time steps dt instead of fixed altitude steps dz.
        Builds the reference trajectory from the release point (start_alt) down to target_z.
        """
        
        starting_conditions:Dict[str,float] = self.starting_conditions
        start_alt:float = starting_conditions['start_alt']
        target_x:float = starting_conditions['target_x']
        target_y:float = starting_conditions['target_y']
        target_z:float = starting_conditions['target_z']
        
        S = self.parachute_config['S'] 
        Cd:float = self.parachute_config['Cd']
        M:float = self.parachute_config['M']
        dt:float = self.dt
        
        pred_wind: np.ndarray = self.pred_wind_data
        
        g:float = 9.81
        # initialize lists
        x_carp:List[float] = [target_x]
        y_carp:List[float] = [target_y]
        z_carp:List[float] = [target_z]

        # Integrate forward in time until reaching start_alt
        while z_carp[-1] < start_alt:
            # estimate density at midpoint of the vertical slice
            # first, compute terminal velocity at current altitude
            rho_mid = rho(z_carp[-1])
            v_term = np.sqrt((2 * M * g) / (rho_mid * Cd * S))
            # vertical distance fallen in this dt
            dz = v_term * dt

            # next altitude
            z_next = z_carp[-1] + dz

            # interpolate wind at midpoint altitude
            mid_alt = 0.5 * (z_carp[-1] + z_next)
            wind_x = np.interp(mid_alt, pred_wind[:, 0], pred_wind[:, 1])
            wind_y = np.interp(mid_alt, pred_wind[:, 0], pred_wind[:, 2])

            # update horizontal positions (negative wind drift since building backward)
            x_next = x_carp[-1] - wind_x * dt
            y_next = y_carp[-1] - wind_y * dt

            # append new points
            x_carp.append(x_next)
            y_carp.append(y_next)
            z_carp.append(z_next)

        # convert to arrays and flip order (top-down)
        x_carp = np.array(x_carp)[::-1]
        y_carp = np.array(y_carp)[::-1]
        z_carp = np.array(z_carp)[::-1]
        return x_carp, y_carp, z_carp

    def load_wind_data(self, use_pred:bool) -> np.ndarray:
        """
        Load the predicted wind data from the specified path.
        """
        # Load the predicted wind data
        if use_pred:
            wind_data = np.loadtxt(self.pred_wind_data_path, delimiter=',')
        else:
            wind_data = np.loadtxt(self.actual_wind_data_path, delimiter=',')
        
        return wind_data
    
    def interpolate_wind(self, prev_alt:float, current_alt:float,
                        use_actual:bool) -> Tuple[float, float]:
        """
        Get the predicted wind speed and direction at the midpoint altitude.
        Args:
            prev_alt (float): The previous altitude.
            current_alt (float): The current altitude.
        Returns:
            Tuple[float, float]: The wind speed in x and y directions at the midpoint altitude
            
        """
        mid_alt = 0.5 * (prev_alt + current_alt)
        if use_actual:
            # Use actual wind data if specified
            wind_data = self.actual_wind_data
        else:
            # Use predicted wind data
            wind_data = self.pred_wind_data
        wind_x = np.interp(mid_alt, wind_data[:, 0], wind_data[:, 1])
        wind_y = np.interp(mid_alt, wind_data[:, 0], wind_data[:, 2])
        
        return wind_x, wind_y

    def forecast_with_wind(self, x_current:float, y_current:float, 
                        pred_wind_x:float, pred_wind_y:float) -> Tuple[float, float]:
        """
        Forecast the next position based on the predicted wind. 
        Args:
            x_current (float): Current x position.
            y_current (float): Current y position.
            pred_wind_x (float): Predicted wind speed in x direction.
            pred_wind_y (float): Predicted wind speed in y direction.
        
        Returns:
            Tuple[float, float]: The next x and y positions after applying the predicted wind.
            
        """
        
        x_forecast = x_current + self.dt * pred_wind_x
        y_forecast = y_current + self.dt * pred_wind_y
        
        return x_forecast, y_forecast

    def add_wind_noise(self, altitude: float) -> Tuple[float, float]:
        """
        Add realistic altitude-dependent noise to wind data.
        
        Args:
            altitude (float): Current altitude in meters.
        Returns:
            Tuple[float, float]: Noise to be added to (wind_x, wind_y) in m/s.
        """
        # Bounds
        h_high, h_low = 30000, 1500.0
        sigma_min, sigma_max = 0.5, 5.0

        # Clamp altitude
        altitude = np.clip(altitude, h_low, h_high)

        # Scale σ with altitude (linear interpolation)
        frac = (altitude - h_low) / (h_high - h_low)
        sigma = sigma_min + (sigma_max - sigma_min) * (1 - frac)

        # Gaussian noise
        noise_x = np.random.normal(0.0, sigma)
        noise_y = np.random.normal(0.0, sigma)
        # plus or minus
        noise_x_sign = np.random.choice([-1, 1])
        noise_y_sign = np.random.choice([-1, 1])
        noise_x *= noise_x_sign
        noise_y *= noise_y_sign

        return noise_x, noise_y

    def step(self, x_current:float, y_current:float) -> None:
        """
        Peform a single simulation step with Euler's method to 
        update the position based on the predicted wind
        """
    def sim_thermal(self, vel:float, 
                    alt:float, 
                    temp_surface_current:float,
                    T_exosphere:float,
                    area:float) -> None:
        """
        SNC
        Simulate the thermal environment of the system during descent.
        Args:
            vel (float): Current velocity (m/s).
            alt (float): Current altitude (m).
            temp_surface_current (float): Current surface temperature (K).
            T_exosphere (float): Exosphere temperature for thermosphere model (K).

        """
        atmospheric_properties:Dict[str, float] = compute_atmospheric_properties(alt, T_exosphere)
        
        rho = atmospheric_properties['rho']
        T_amb = atmospheric_properties['T']
        p = atmospheric_properties['p']
        a = atmospheric_properties['speed_of_sound']
        mean_free_path = atmospheric_properties['mean_free_path']
        
        # Gravity
        g_current = gravity_variation(alt)

        # Atmosphere + Cd
        current_Cd = shuttlecock_drag_coefficient(abs(vel), alt, 
                                                  self.parachute_config['diameter'],
                                                  T_exosphere)


        heating_rate_results:Dict[str, Any] = aerodynamic_heating_rate(velocity=abs(vel),
                                 altitude=alt,
                                 surface_temp=temp_surface_current,
                                 diameter=self.parachute_config['diameter'],
                                 total_surface_area=self.parachute_config['total_surface_area'],
                                 specific_heat=self.parachute_config['specific_heat'],
                                 emissivity=self.parachute_config['emissivity'],
                                 T_exosphere=T_exosphere)
        
        q_net = heating_rate_results['q_net']
        T_stag = heating_rate_results['T_stagnation']
        q_conv = heating_rate_results['q_conv']
        q_rad_total = heating_rate_results['total_cooling_for_plot']
        q_solar = heating_rate_results['solar_heating']
        q_earth_ir = heating_rate_results['earth_ir']
        q_shock = heating_rate_results['q_rad_shock']
        q_rad_space = heating_rate_results['q_rad_space']
        q_rad_atm_cooling = heating_rate_results['q_rad_atm_cooling']
        q_rad_atm_heating = heating_rate_results['q_rad_atm_heating']
        
        surface_thickness:float = self.parachute_config['surface_thickness']
        specific_heat:float = self.parachute_config['specific_heat']
        total_surface_area:float = self.parachute_config['total_surface_area']
        surface_density:float = self.parachute_config['surface_density']
        thermal_conductivity:float = self.parachute_config['thermal_conductivity']
        length:float = self.parachute_config['length']
        diameter:float = self.parachute_config['diameter']
        mass = self.mass
        dt = self.dt
        
        surface_volume = total_surface_area * surface_thickness
        surface_mass = surface_volume * surface_density
        surface_thermal_capacity = surface_mass * specific_heat
        interior_temp = self.interior_temp
        
        # Drag
        if abs(vel) > 0:
            F_drag_magnitude = 0.5 * rho * area * current_Cd * abs(vel) ** 2
            F_drag = -F_drag_magnitude * np.sign(vel)
        else:
            F_drag = 0.0

        # Acceleration
        accel = -g_current + F_drag / mass
    
        if surface_thermal_capacity > 0:
            dT_surface = (q_net * total_surface_area * dt) / surface_thermal_capacity
            # Clamp only heating bursts, allow natural cooling
            if dT_surface > 0:
                dT_surface = min(dT_surface, 50.0)
        else:
            dT_surface = 0.0

        # Conduction to interior
        h_conduction = thermal_conductivity / (0.5 * length)
        q_conduction = h_conduction * (temp_surface_current - interior_temp)

        interior_mass = max(mass - surface_mass, 1e-6)
        interior_thermal_capacity = interior_mass * specific_heat
        if interior_thermal_capacity > 0:
            dT_interior = (q_conduction * total_surface_area * dt) / interior_thermal_capacity
            interior_temp += dT_interior

        # Update surface temperature (convective/radiative already in q_net)
        temp_surface_current += dT_surface - (q_conduction * total_surface_area * dt) / max(surface_thermal_capacity, 1e-9)

        # Physical caps to avoid runaway numerics
        temp_surface_current = max(temp_surface_current, 150)    # ~cosmic floor
        temp_surface_current = min(temp_surface_current, 2500)   # plausible upper bound

        # Dimensionless groups
        mach = abs(vel) / a if a > 0 else 0.0
        mu = 1.458e-6 * T_amb ** 1.5 / (T_amb + 110.4) if T_amb > 0 else 0.0
        reynolds = rho * abs(vel) * diameter / mu if mu > 0 else 0.0
        knudsen = mean_free_path / diameter if diameter > 0 else float("inf")
        
        # we want update the interior temp for next iteration
        self.interior_temp = interior_temp
        
        # Log data for analysis
        self.thermal_sim_history.record_step(
            accel, F_drag, current_Cd, q_net,
            temp_surface_current, T_stag, mach, reynolds,
            knudsen, rho, T_amb, mean_free_path,
            g_current, q_solar, q_earth_ir, q_conv,
            q_rad_total, q_shock, q_rad_space,
            q_rad_atm_cooling, q_rad_atm_heating,
            interior_temp
        )

    def sim_power_consumption(self, 
             T_env_C: float,
             V_rel_ms: float,
             rho_air: float,
             alt_m:float,
             servo_active:bool,
             servo_effort:float 
             ) -> None:
        """
        Simulates the power consumption and battery state over a time step.
        """
        loads = self.payload.step(
            dt= self.dt,
            t=self.thermal_sim_history.time[-1] if self.thermal_sim_history.time else 0.0,
            phase=self.phase,
            soc = self.battery_cfg.SoC,
            v_batt=self.battery_cfg.V_term,
            temp_C = T_env_C,
            alt_m = alt_m,
            servo_active=servo_active,
            servo_effort= servo_effort
        )
        
        b = self.battery_cfg.step(P_load_W=loads["P_total_W"], 
                      dt_s=self.dt, T_env_C=T_env_C,
                      V_rel_ms=V_rel_ms, 
                      rho_air=rho_air)  # rough air density model

        
    def is_dead(self, vel:float) -> bool:
        """
        Check if battery is dead or if we burn up
        """
        if self.battery_cfg.is_dead():
            print("Battery is dead!")
            return True
        
        return False

    def compute_control_effort(self,
                            desired_psi:float,
                            desired_gr:float,
                            desired_s:float,
                            current_psi:float,
                            current_gr:float,
                            current_s:float) -> Dict[str, Any]:
        """
        Compute the control effort required to reach the desired state.
        Returns whether the servo is active and the effort levels for psi, gr, and s.
        """
        psi_error = desired_psi - current_psi
        gr_error = desired_gr - current_gr
        s_error = desired_s - current_s
    
        # Simple proportional control for demonstration
        Kp_psi = 0.5
        Kp_gr = 0.5
        Kp_s = 0.5
        
        psi_effort = np.clip(Kp_psi * psi_error, -1.0, 1.0)
        gr_effort = np.clip(Kp_gr * gr_error, -1.0, 1.0)
        s_effort = np.clip(Kp_s * s_error, -1.0, 1.0)
        
        servo_active: bool = True
        if (abs(psi_effort) < 0.01 and
            abs(gr_effort) < 0.01 and
            abs(s_effort) < 0.01):
            servo_active = False
            
        summation = abs(psi_effort) + abs(gr_effort) + abs(s_effort)
        # clip the summation
        summation = 1.0
        if not servo_active:
            print("Servo inactive")
            return {
                "servo_active": False,
                "effort": 0.0
            }
        else:
            return {
                "servo_active": True,
                "effort": summation
            }
         
    def get_random_start_coordinate(self, config:List[float]) -> float:
        """
        Get a random starting coordinate within the specified noise bounds.
        Args:
            config (List[float]): List containing [min, max] bounds for the coordinate.
        Returns:
            float: Random starting coordinate.
        """
        rand_coord = np.random.uniform(config[0], config[1])
        rand_coord_sign = np.random.choice([-1, 1])
        rand_coord *= rand_coord_sign
        
        return rand_coord 
        
    def get_position_noise(self, noise_value:float) -> float:
        """
        Get a random position noise within the specified bounds.
        Args:
            config (List[float]): List containing [min, max] bounds for the noise.
        Returns:
            float: Random position noise.
        """
        pos_noise = np.random.uniform(-noise_value, noise_value)
        
        return pos_noise

         
    def to_control(self, i:int) -> bool:
        """
        Based on the control frequency as well as the command failure rate,
        we check if we should send a command at this time step.
        """
        # map this as time instead
        
        if i % self.control_interval_steps == 0:
            if np.random.rand() > self.command_failure_rate:
                return True
            else:
                self._cmd_drops += 1
                print("Command failed to send!")
                return False
        print("Not time to send command")
        return False
         
    def simulate(self, n_steps:int = 3000,
                 v_term:float=-10) -> None:
        
        x_des, y_des, z_des = self.carp_estimator(self.pred_wind_data)

        psi_filter: ThirdOrderButterworthFilter = self.psi_filter
        glide_ratio_filter: ThirdOrderButterworthFilter = self.glide_ratio_filter
        s_filter: ThirdOrderButterworthFilter = self.s_filter
        
        x_control:List[float] = []
        y_control:List[float] = []
        z_control:List[float] = []
        psi_control:List[float] = []
        gr_control:List[float] = []
        s_control:List[float] = []
        descent_rate:List[float] = []
        
        rand_x_drop = self.get_random_start_coordinate(self.starting_conditions['x_start_noise'])
        rand_y_drop = self.get_random_start_coordinate(self.starting_conditions['y_start_noise'])
        rand_z_drop = self.get_random_start_coordinate(self.starting_conditions['z_start_noise'])
        # rand_x_drop = -1000.0
        # rand_y_drop = 500.0
        start_x:float = x_des[0] + rand_x_drop  # starting 200m west of the desired path
        start_y:float = y_des[0] + rand_y_drop  # starting 200m north of the desired path
        start_alt:float = z_des[0] + rand_z_drop
        
        x_control.append(start_x)
        y_control.append(start_y)
        z_control.append(start_alt)

        current_x = start_x 
        current_y = start_y
        current_alt = start_alt
        
        desired_alt:float = z_des[-1]
        M = self.parachute_config['M']
        Cd = self.parachute_config['Cd']
        g = 9.81  # gravitational acceleration
        S = self.parachute_config['S']
        
        psi_actual:float = 0.0
        GR_actual:float = 0.3
        S_actual:float = S
        
        closed_loop_sim: UpdatedCloseLoopSim = UpdatedCloseLoopSim(
            optimizer=self.predictive_control,
            x_init=np.array([start_x, start_y, start_alt, 0, v_term]),
            x_final=np.array([0, 0, desired_alt, 0, v_term]),
            u0=np.array([GR_actual, S_actual, psi_actual]),
            N = n_steps)
        

        current_time: float = 0.0
        success: bool = True
        # used to track the cumulative descent on whether we want to 
        # update the wind forecast
        cumlative_descent:float = 0.0
        for i in range(n_steps):
            
            ## doing this since we are starting at the top of the descent
            if current_alt < desired_alt:
                print("Reached the desired altitude, stopping simulation.")
                break

            if i == 0:
                current_alt = z_des[0]
                prev_alt:float = start_alt
                temp_surface_current:float = self.parachute_config['temp_surface_current']
            else:
                prev_alt:float = z_control[i-1] 
                temp_surface_current = self.thermal_sim_history.surface_temp[-1]

            if not self.control_wind_forecast:
                # always update the wind forecast
                pred_wind_x, pred_wind_y = self.interpolate_wind(
                    prev_alt=prev_alt, current_alt=current_alt,
                    use_actual=False)
            else:
                if i == 0 or cumlative_descent >= self.wind_forecast_update_m:                        
                    pred_wind_x, pred_wind_y = self.interpolate_wind(
                            prev_alt=prev_alt, current_alt=current_alt,
                            use_actual=False)
                    cumlative_descent = 0.0
                
            ## This would be the forward step of the simulation
            actual_wind_x, actual_wind_y = self.interpolate_wind(
                prev_alt=prev_alt, current_alt=current_alt,
                use_actual=True)

            wind_x_noise, wind_y_noise, wind_z_noise = self.wind_gust.step(
                alt=current_alt)  
            
            ## MPC will live here
            # we want to feed a noise estimate of the mpc 
            x_noise:float = self.get_position_noise(
                self.starting_conditions['position_sensor_noise_std'])
            y_noise:float = self.get_position_noise(
                self.starting_conditions['position_sensor_noise_std'])
            z_noise:float = self.get_position_noise(
                self.starting_conditions['vertical_position_sensor_noise_std'])
            heading_noise = self.get_position_noise(
                self.starting_conditions['heading_sensor_noise_std'])
            vel_noise = self.get_position_noise(
                self.starting_conditions['velocity_sensor_noise_std'])
            psi_w_noise = psi_actual + np.deg2rad(heading_noise)
            psi_w_noise = (psi_w_noise + np.pi) % (2 * np.pi) - np.pi
            vel_w_noise = v_term + vel_noise
            x0 = np.array([current_x + x_noise, 
                           current_y + y_noise, 
                           current_alt + z_noise, 
                           psi_w_noise , 
                           vel_w_noise])
            
            # we want to use the projected altitude to get an idea 
            # of the ideal next position of the carp estimator
            projected_alt = current_alt + (vel_w_noise * self.dt)
            z_des_idx = np.argmin(np.abs(z_des - projected_alt))
            xF = np.array([x_des[z_des_idx], y_des[z_des_idx], z_des[z_des_idx], 0, 0.0])
            
            #xF = np.array([x_target, y_target, z_target, 0, v_term])
            u0 = np.array([GR_actual, S_actual, psi_actual])
            
            wind_data = np.array([
                [pred_wind_x, pred_wind_y, 0.0]  # assuming no vertical wind
            ])
            wind_data = wind_data.flatten()  # flatten to 1D array if needed
            # need to get wind data to pass into single step optimizer
            solution:Dict[str, Any] = closed_loop_sim.run_single_step(x0=x0, 
                xF=xF, u0=u0,
                wind_data=wind_data)
            states:Dict[str,float] = solution['states']
            controls:Dict[str,float] = solution['controls']
            control_idx: int = 1
            
            assumed_x = states['x'][control_idx]
            assumed_y = states['y'][control_idx]
            assumed_z = states['z'][control_idx]
    
            des_psi = states['psi'][control_idx]
            des_GR = controls['gr_control'][control_idx]
            des_S = controls['s_control'][control_idx]
            
            # introduce control frequency
            # This is where we actually control the system
            # TBD based on plant
            if self.to_control(i):
                # Update commands with filtering
                psi_actual = self.psi_filter.step(des_psi)    
                GR_actual = glide_ratio_filter.filter(des_GR)
                S_actual = s_filter.filter(des_S)

            print(f"Step {i}: assumed_x={assumed_x}, assumed_y={assumed_y},assumed_z={assumed_z}, current_x={current_x}, current_y={current_y}, \
                    current_alt={current_alt}")
        
            print(f"Step {i}: x={current_x}, y={current_y}, z={current_alt}, "
                    f"psi={des_psi}, GR={des_GR}, S={des_S}")
        
            x_error:float = assumed_x - current_x
            y_error:float = assumed_y - current_y
            z_error:float = assumed_z - current_alt
            print("x_error: ", x_error, " y_error: ", y_error,
                    " z_error: ", z_error)
            print("\n")
            
            #TODO: REPLACE This with SNC 
            #current_alt = current_alt - (v_term * self.dt)
            init_conditions = [current_alt, v_term]
            t_span = [current_time, current_time + self.dt]
            
            altitude_derivates = solve_ivp(
                fun=equations_of_motion,
                t_span=t_span,
                y0=init_conditions,
                args=(T_exosphere,S_actual, self.mass,),
                method="RK45",
                t_eval=[current_time + self.dt],
                dense_output=True,
            )
            
            altitude_sim, vz_sim = altitude_derivates.y[:, -1]   # vz_sim < 0 when descending
            alt_rate = vz_sim                                     # vertical rate (m/s)

            # Commit vertical state
            current_alt = altitude_sim
            v_term  = vz_sim

            # Horizontal speed = GR * altitude loss rate  (loss rate is -alt_rate)
            horiz_speed = GR_actual * (-alt_rate)             # >= 0 during descent

            glide_x = horiz_speed * np.cos(psi_actual)
            glide_y = horiz_speed * np.sin(psi_actual)

            wind_x = actual_wind_x + wind_x_noise
            wind_y = actual_wind_y + wind_y_noise
            current_x = current_x + ((glide_x + wind_x) * self.dt)
            current_y = current_y + ((glide_y + wind_y) * self.dt)
            current_alt = altitude_sim + (wind_z_noise * self.dt)
            # Append the new state to the control lists
            self.sim_thermal(vel=v_term,
                             alt=current_alt,
                             temp_surface_current=temp_surface_current,
                             T_exosphere=T_exosphere,
                             area=S_actual)
            
            if self.to_control(i):
                servo_info:Dict[str, Any] = self.compute_control_effort(
                    desired_psi=des_psi,
                    desired_gr=des_GR,
                    desired_s=des_S,
                    current_psi=psi_actual,
                    current_gr=GR_actual,
                    current_s=S_actual
                )
                
            self.sim_power_consumption(
                T_env_C=self.interior_temp - 273.15,  # convert K to C
                V_rel_ms = v_term,
                rho_air = rho(current_alt),
                alt_m= current_alt,
                servo_active= servo_info['servo_active'],
                servo_effort= servo_info['effort']   
            )
            self.thermal_sim_history.time.append(current_time)
            x_control.append(current_x)
            y_control.append(current_y)
            z_control.append(current_alt)
            psi_control.append(psi_actual)
            gr_control.append(GR_actual)
            s_control.append(S_actual)
            descent_rate.append(alt_rate)
            current_time += self.dt
            cumlative_descent += -alt_rate * self.dt

            self.parachute_history.x.append(current_x)
            self.parachute_history.y.append(current_y)
            self.parachute_history.altitude.append(current_alt)
            self.parachute_history.psi.append(psi_actual)
            self.parachute_history.gr.append(GR_actual)
            self.parachute_history.s.append(S_actual)
            self.parachute_history.time.append(current_time)
            self.parachute_history.vertical_velocity.append(alt_rate)
            self.parachute_history.wind_x.append(wind_x)
            self.parachute_history.wind_y.append(wind_y)
            self.parachute_history.wind_z.append(wind_z_noise)
            self.parachute_history.pred_wind_x.append(pred_wind_x)
            self.parachute_history.pred_wind_y.append(pred_wind_y)

            self.parachute_history.desired_psi.append(des_psi)
            self.parachute_history.desired_gr.append(des_GR)
            self.parachute_history.desired_s.append(des_S)
            
            self.info.ts.vx_mps.append(glide_x + wind_x)
            self.info.ts.vy_mps.append(glide_y + wind_y)
            self.info.ts.vz_mps.append(alt_rate)
            self.info.ts.gust_x_mps.append(wind_x_noise)
            self.info.ts.gust_y_mps.append(wind_y_noise)
            self.info.ts.gust_z_mps.append(wind_z_noise)
            
            self.parachute_history.x_traj.append(states['x'])
            self.parachute_history.y_traj.append(states['y'])
            self.parachute_history.z_traj.append(states['z'])

            if self.is_dead(vel=v_term):
                print("Simulation ended due to battery depletion or burn up.")
                success = False
                break
        
        if not success:
            self.parachute_history.success.append(False)
        else:
            self.parachute_history.success.append(True)
            
        self.info.ts.t_s = self.parachute_history.time
        self.info.ts.x_m = self.parachute_history.x
        self.info.ts.y_m = self.parachute_history.y
        self.info.ts.z_m = self.parachute_history.altitude
        self.info.ts.psi_rad = self.parachute_history.psi       
        
        self.info.ts.wind_x_mps = self.parachute_history.wind_x
        self.info.ts.wind_y_mps = self.parachute_history.wind_y
        self.info.ts.wind_z_mps = self.parachute_history.wind_z
        
        self.info.ts.psi_cmd_rad = self.parachute_history.desired_psi
        self.info.ts.gr_cmd = self.parachute_history.desired_gr
        self.info.ts.S_cmd_m2 = self.parachute_history.desired_s
        
        # actuator stuff
        battery_history = self.battery_cfg.hist
        payload_history = self.payload.payload_history
        
        self.info.ts.servo_effort = payload_history.servo_effort
        self.info.ts.servo_active = payload_history.servo_active
        self.info.ts.batt_soc = battery_history.SoC
        self.info.ts.batt_V = battery_history.V
        self.info.ts.payload_P_total_W = payload_history.P_total
        self.info.ts.payload_P_servo_W = payload_history.P_act
        self.info.ts.payload_P_gps_W = payload_history.P_gps
        
        thermal_sim_history = self.thermal_sim_history
        self.info.ts.accel_mps2          = thermal_sim_history.acceleration
        self.info.ts.drag_force_N        = thermal_sim_history.drag_force
        self.info.ts.Cd                  = thermal_sim_history.drag_coefficient
        self.info.ts.q_dot_W_per_m2      = thermal_sim_history.heating_rate
        self.info.ts.T_surface_K         = thermal_sim_history.surface_temp
        self.info.ts.T_stag_K            = thermal_sim_history.stagnation_temp
        self.info.ts.Mach                = thermal_sim_history.mach_number
        self.info.ts.Re                  = thermal_sim_history.reynolds_number
        self.info.ts.Kn                  = thermal_sim_history.knudsen_number
        self.info.ts.rho_kgpm3           = thermal_sim_history.air_density
        self.info.ts.T_air_K             = thermal_sim_history.air_temperature
        self.info.ts.mfp_m               = thermal_sim_history.mean_free_path
        self.info.ts.g_mps2              = thermal_sim_history.gravity
        self.info.ts.q_solar_Wpm2        = thermal_sim_history.solar_heating
        self.info.ts.q_earthir_Wpm2      = thermal_sim_history.earth_ir
        self.info.ts.q_conv_Wpm2         = thermal_sim_history.convective_heating
        self.info.ts.q_rad_cooling_Wpm2  = thermal_sim_history.radiative_cooling
        self.info.ts.q_shock_Wpm2        = thermal_sim_history.shock_heating
        self.info.ts.q_rad_space_Wpm2    = thermal_sim_history.q_rad_space
        self.info.ts.q_rad_atm_cooling_Wpm2 = thermal_sim_history.q_rad_atm_cooling
        self.info.ts.q_rad_atm_heating_Wpm2 = thermal_sim_history.q_rad_atm_heating
        self.info.ts.T_interior_K        = thermal_sim_history.interior_temp
        
        self.info.ts.x_traj = self.parachute_history.x_traj
        self.info.ts.y_traj = self.parachute_history.y_traj
        self.info.ts.z_traj = self.parachute_history.z_traj    
        
        # save all this information as monte carlo
        # ---- finalize MonteCarloInformation ----
        if self.info is not None:
            # events
            self.info.events.cmd_drops = self._cmd_drops
            self.info.events.brownout = self._brownout
            self.info.events.cutoff_triggered = self._cutoff_triggered
            self.info.events.hard_fail = self._hard_fail
            # summary
            goal_xy = (self.starting_conditions["target_x"], self.starting_conditions["target_y"])
            self.info.finalize(goal_xy=goal_xy)
            # mission success (simple rule, tune as needed)
            self.info.summary.success = bool(self.parachute_history.success and self.parachute_history.success[-1])
            if not self.info.summary.success:
                self.info.summary.failure_reason = "battery_cutoff" if self._cutoff_triggered else "unknown"

        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x_control, y_control, z_control, label='Simulated Trajectory',
        #         color='blue', linewidth=2)
        # ax.plot(x_des, y_des, z_des, label='CARP Trajectory', color='red', linestyle='--', linewidth=2)
        # ax.legend()
        
        x_last = x_control[-1]
        y_last = y_control[-1]
        
        distance_to_goal = np.sqrt((x_last - x_des[-1])**2 + (y_last - y_des[-1])**2)
        print(f"Distance to goal: {distance_to_goal:.2f} meters")
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_control, y_control, z_control, label='Trajectory',
                color='blue', linewidth=2)
        for i in range(len(self.parachute_history.x_traj)):
            ax.scatter(self.parachute_history.x_traj[i][0],
                          self.parachute_history.y_traj[i][0],
                          self.parachute_history.z_traj[i][0],
                          color='green', s=20, label='MPC Start' if i == 0 else "")
            ax.plot(self.parachute_history.x_traj[i], 
                    self.parachute_history.y_traj[i], 
                    self.parachute_history.z_traj[i], 
                    color='gray', alpha=1.0)
        ax.plot(x_des[:n_steps], y_des[:n_steps], z_des[:n_steps], label='CARP Trajectory', color='red', linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        
        plt.figure(2)
        plt.subplot(311)
        z_control = z_control[1:]
        psi_control = np.array(psi_control)
        plt.plot(self.parachute_history.time, gr_control)
        plt.plot(self.parachute_history.time, self.parachute_history.desired_gr, linestyle='--', label='Desired GR')
        plt.ylabel('Glide Ratio')
        plt.subplot(312)
        plt.plot(self.parachute_history.time, s_control)
        plt.plot(self.parachute_history.time, self.parachute_history.desired_s, linestyle='--', label='Desired Area')
        plt.ylabel('Drag Area, m^2')
        plt.subplot(313)
        plt.plot(self.parachute_history.time, psi_control*180/3.14)
        plt.plot(self.parachute_history.time, np.array(self.parachute_history.desired_psi)*180/3.14, 
                 linestyle='--', label='Desired Heading')
        plt.xlabel('Altitude, m')
        plt.ylabel('Heading, deg')

        fig, ax = plt.subplots()
        # plot the descent rate 
        # delta_z = np.diff(z_control)
        # desired_delta_z = np.diff(z_des)
        ax.plot(z_control, descent_rate, label='Descent Rate')
        ax.set_xlabel('Altitude, m')
        ax.set_ylabel('Descent Rate, m/s')
        
        fig, ax = plt.subplots(3,1)
        ax[0].plot(self.parachute_history.time,
                   self.parachute_history.wind_x, label='Wind X')
        ax[0].plot(self.parachute_history.time,
                     self.parachute_history.pred_wind_x, label='Pred Wind X', linestyle='--')
        ax[1].plot(self.parachute_history.time, 
                     self.parachute_history.wind_y, label='Wind Y')
        ax[1].plot(self.parachute_history.time,
                    self.parachute_history.pred_wind_y, label='Pred Wind Y', linestyle='--')
        ax[2].plot(self.parachute_history.time,
                        self.parachute_history.wind_z, label='Wind Z')
        ax[0].set_ylabel('Wind X (m/s)')
        ax[1].set_ylabel('Wind Y (m/s)')
        ax[2].set_ylabel('Wind Z (m/s)')
        for a in ax:
            a.set_xlabel('Time (s)')
            a.legend()
            a.grid(True)
        
        # save all this information into a pickle file
        info_dict = {
            'x_control': x_control,
            'y_control': y_control,
            'z_control': z_control,
            'psi_control': psi_control,
            'gr_control': gr_control,
            's_control': s_control,
            'x_des': x_des,
            'y_des': y_des,
            'z_des': z_des,
            'pred_wind_data': self.pred_wind_data,
            'actual_wind_data': self.actual_wind_data,
            'distance_to_goal': distance_to_goal,
            'descent_rate': descent_rate,
            'x': self.parachute_history.x,
            'y': self.parachute_history.y,
            'altitude': self.parachute_history.altitude,
            'x_traj': self.parachute_history.x_traj,
            'y_traj': self.parachute_history.y_traj,
            'z_traj': self.parachute_history.z_traj,
        }
        
        # Save the info_dict to a file
        import pickle
        with open('simulation_results_no_carp_mpc_real_wind.pkl', 'wb') as f:
            pickle.dump(info_dict, f)
        
        # Align lengths: history has one entry per sim_thermal call.
        n = len(self.thermal_sim_history.heating_rate)

        # z_control includes the initial seed at index 0; thermal logs start after the first step.
        altitude_for_plots = np.array(z_control)
        velocity_for_plots = np.array(descent_rate[:n])
        time_for_plots     = np.array(self.thermal_sim_history.time)

        plot_aerothermal_from_history(
            history=self.thermal_sim_history,
            time_arr=time_for_plots,
            altitude_arr=altitude_for_plots,
            velocity_arr=velocity_for_plots,
        )

        self.plot_battery() 
        plt.show()

        return self.info

    def plot_battery(self) -> None:
        """
        """
        battery_history = self.battery_cfg.hist
        payload_history = self.payload.payload_history
        
        fig, axs = plt.subplots(4,1, figsize=(11,12), sharex=True)
        axs[0].plot(battery_history.t, battery_history.V, label="V_batt")
        axs[0].axhline(self.battery_cfg.V_cutoff, ls="--", c="r", label="V_cutoff")
        axs[0].set_ylabel("Voltage (V)"); axs[0].legend();
        axs[0].grid(True)
        
        axs[1].plot(battery_history.t, np.array(battery_history.SoC)*100);
        axs[1].set_ylabel("SoC (%)"); axs[1].grid(True)
        
        axs[2].plot(payload_history.time, payload_history.P_total, label="P_total")
        axs[2].plot(payload_history.time, payload_history.P_gps, label="P_GPS")
        axs[2].plot(payload_history.time, payload_history.P_act,  label="P_actuator")
        axs[2].set_ylabel("Power (W)"); axs[2].legend(); axs[2].grid(True)
        
        axs[3].plot(payload_history.time, payload_history.servo_effort, label="servo duty")
        axs[3].set_ylabel("Duty"); axs[3].set_xlabel("Time (s)")
        axs[3].grid(True)
        print("Final SoC: ", battery_history.SoC[-1]*100, "%")