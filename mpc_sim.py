import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from the_right_way.ParachuteModel import ParachuteModel
from the_right_way.OptimalParachute import MPCParachute
from the_right_way.Simulator import ParachuteSimulator, UpdatedCloseLoopSim
from the_right_way.Filters import ThirdOrderButterworthFilter
from the_right_way.Battery import BatteryHistory, BatteryState
from the_right_way.Hardware import (GPSConfig, GPSPowerManager,
                                    PayloadConfig, PayloadPowerController)
from the_right_way.WindGust import WindGust
from the_right_way.Filters import FirstOrderAngleFilter, FirstOrderFilter
from optitraj.utils.data_container import MPCParams
from numpy import genfromtxt
from typing import Dict, Any

pred_wind = genfromtxt('Balloon_Wind_Data_Files/8_matlab_asc.csv', delimiter=',')
real_wind = genfromtxt('Balloon_Wind_Data_Files/15_matlab_asc.csv', delimiter=',')


def build_model(control_limits_dict: Dict[str, Dict[str, float]],
             state_limits_dict: Dict[str, Dict[str, float]],
             dt_val:float) -> ParachuteModel:
    """
    Build the parachute model with the given control and state limits.
    """
    return ParachuteModel(
        mass_kg=0.5,
        Cd=0.7,
        # inertia=0.5,
        include_time=True,
        dt_val=dt_val
    )


"""
Monte Carlo Parameters for this study:

Geometric profiles of payload:
mass: min in grams -> 10 grams increments 2 up to 20
    [10, 12, 14, 16, 18, 20] grams
- assumption is the controller -> can control the frontal area (expand/retract) 
- Cd: 0.45 -> 0.7 

## Thermal Properties to vary:
- emissivity = [0.3, 0.6, 0.85]   # Surface emissivity for radiation heat transfer
- specific_heat = 600 -> 1400 with increments of 200  # Specific heat capacity (J/kg·K) - composite material
- thermal_conductivity = [0.1, 1.0] steps of 0.1   # Thermal conductivity (W/m·K)
- specific heat capacity 
- emissivity

## Payload parameters to test:
- GRmax = [0.2, 0.3, 0.4]
- GRmin = [0.1, 0.2, 0.3]

We need to run multiple combinations of these parameters to see how they affect the trajectory and thermal response.
- Smax = [0.002, 0.003, 0.004] # Vehicle reference area max
- Smin = [0.001, 0.002, 0.003]
- Cd = [0.45, 0.55, 0.65] # Vehicle drag coefficient
- Mass = [0.01 to 0.02] # Mass of vehicle, kg in increments of 0.002


"""

####################################
##### Parachute Configuration ######
####################################
Smax = 0.003 # Vehicle reference area max
Smin = 0.002 # Vehicle reference area min
#midpoint s
midpoint_s = (Smax + Smin) / 2.0
S = midpoint_s # Nominal (for CARP calculations)
Cd = 0.45 # Vehicle drag coefficient
I = 0.1 # Mass moment of inertia of vehicle, kg-m^2
g = 9.81 # m/s^2

# Shuttlecock inertial characteristics (standard values)rams
area = 0.00267  # Cross-sectional area (m^2)
total_surface_area = 0.008  # Total surface area including feathers (m^2)

Cd = 0.6  # Base drag coefficient
emissivity = 0.85  # Surface emissivity for radiation heat transfer
specific_heat = 1200  # Specific heat capacity (J/kg·K) - composite material
thermal_conductivity = 0.15  # Thermal conductivity (W/m·K)

# Shuttlecock specific properties
diameter = 0.058  # Shuttlecock diameter (m) []
length = 0.068  # Shuttlecock total length (m)
cork_mass = 0.003  # Cork/rubber base mass (kg)
feather_mass = 0.002  # Feather mass (kg)
#mass = cork_mass + feather_mass  # Total mass (kg)
mass = 0.05
M = mass # Mass of vehicle, kg
temp_surface_current = 293.15  # Initial surface temperature (K)


parachute_config: Dict[str, float] = {
    'S': midpoint_s,
    'Cd': Cd,
    'M': M,
    'I': I,
    'g': g,
    'Smin': Smin,
    'Smax': Smax,
    'specific_heat': specific_heat,
    'emissivity': emissivity,
    'thermal_conductivity': thermal_conductivity,
    'total_surface_area': total_surface_area,
    'diameter': diameter,
    'length': length,
    'cork_mass': cork_mass,
    'feather_mass': feather_mass,
    'surface_thickness': 0.0005, # meters
    'surface_density': 1500.0, # kg/m^3,
    'temp_surface_current': temp_surface_current
}

gps_config_dict:Dict[str,Any] = {
    # Electrical
    # https://www.digikey.com/en/products/detail/u-blox/MAX-M8Q-0/6150636
    "v_out": 3.3,               # GPS rail voltage (V)
    "eta": 0.90,                # regulator efficiency
    "I_acquire": 0.02,         # A, cold acquisition current
    "I_track": 0.0018 ,           # A, tracking current
    "I_idle":  15e-6,           # A, idle current
    "fix_rate_hz": 1.0,         # nominal fix rate

    # Acquisition timing
    "t_hot_s": 3.0,             # hot-start time (s)
    "t_cold_s": 25.0,           # cold-start time (s)
    "hot_window_s": 60.0,       # hot-start valid window (s)

    # Low-power thresholds
    "soc_low": 0.25,            # low battery SoC trigger
    "temp_cold_C": -10.0,       # low temperature trigger
    "v_brown_hint": 3.45,       # brownout caution voltage (V)

    # Duty cycling (nominal)
    "duty_period_s_nom": 10.0,  # cycle length (s)
    "duty_on_s_nom": 2.0,       # active portion (s)

    # Duty cycling (low SoC / cold)
    "duty_period_s_low": 20.0,
    "duty_on_s_low": 2.0,

    # Landing boost
    "landing_alt_m": 1500.0,    # altitude threshold (m)
    "duty_period_s_landing": 5.0,
    "duty_on_s_landing": 2.0
}

payload_config_dict:Dict[str, Any] = {
    # 3.3V rail regulator efficiency
    "eta_33": 0.90,

    # MCU @3.3V
    "I_mcu_active": 0.012,   # A
    "I_mcu_idle": 0.0006,    # A

    # IMU @3.3V
    "I_imu_active": 0.0025,  # A
    "I_imu_idle": 0.0001,    # A

    # Wind sensor @3.3V
    "I_wind_active": 0.004,  # A
    "I_wind_idle": 0.0001,   # A

    # Radio/telemetry @3.3V based on
    # https://www.cdebyte.com/products/E22-900T33S/1
    "I_radio_tx": 1.2,     # A when transmitting
    "I_radio_idle": 0.014,   # A when idle
    "radio_tx_duty": 0.02,   # fraction of time in TX
    "mass_radio_g": 5.0,     # g

    # Actuator/servo on battery rail
    # These servos are heavy https://protosupplies.com/product/servo-motor-micro-sg90
    "I_act_stall": 0.9,        # A peak
    "duty_servo_nom": 0.04,      # fraction active in flight
    "duty_servo_landing": 0.5,  # stronger control near ground
    "duty_servo_idle": 0.002,    # idle/background duty

    # Load-shedding thresholds
    "soc_shed": 0.18,        # low SoC threshold
    "v_shed": 3.0,          # low-voltage threshold
    "temp_cold_C": -10.0     # cold-temperature threshold
}

battery_config_dict:Dict[str, Any] = {
    "Q_rated_Ah": 0.13,   # 130 mAh
    "R0_25": 0.35,        # Ohms at 25
    "V_cutoff": 3.0,      # cutoff voltage
    "SoC": 0.6            # initial SoC
}

payload_cfg = PayloadConfig(**payload_config_dict)
gps_cfg = GPSConfig(**gps_config_dict)
batt_cfg = BatteryState(**battery_config_dict)

####################################
##### Simulation Parameters ###### 
####################################
start_alt = 20000.0
end_alt = 1500.0
dz = 10.0
carp_dz = 1 # only compute CARP traj. every DZ meters
dt: float = 0.05
target_x = 0.0
target_y = 0.0
target_z = 1500 #Wind file does not go all the way to 0.0
GRmax = 0.4 # Maximum Glide Ratio
GRmin = 0.2 # Minimum Glide Ratio
simulation_config: Dict[str, Any] = {
    'start_alt': start_alt,
    'end_alt': end_alt,
    'dz': dz,
    'carp_dz': carp_dz,
    'dt': dt,
    'target_x': target_x,
    'target_y': target_y,
    'target_z': target_z,
    'GRmax': GRmax,
    'GRmin': GRmin,
    'x_start_noise': [5,15],
    'y_start_noise': [5,15],
    'z_start_noise': [5,45],
    'position_sensor_noise_std': 3, # meters
    'vertical_position_sensor_noise_std': 10, # meters
    'heading_sensor_noise_std': 5, # degrees
    'velocity_sensor_noise_std': 0.2, # m/s
    'command_failure_rate': 0.02, # probability of a command failure at each
    'command_frequency_hz': 20, # how often we issue commands
}
wind_gust: WindGust = WindGust(dt=dt)

####################################
##### Model Predictive Control ######
#################################### 
control_limits_dict: Dict[str, Dict[str, float]] = {
    'gr_control': {'min': GRmin, 'max': GRmax},
    's_control': {'min': Smin, 'max': Smax},
    'u_psi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
}
state_limits_dict: Dict[str, Dict[str, float]] = {
    'x': {'min': -np.inf, 'max': np.inf},
    'y': {'min': -np.inf, 'max': np.inf},
    'z': {'min': -np.inf, 'max': np.inf},
    'psi': {'min': -np.pi, 'max': np.pi},
    'v_term': {'min': -np.inf, 'max': -5},
}

parachute_model: ParachuteModel = build_model(
    control_limits_dict, state_limits_dict, dt)
# now we will set the MPC weights for the plane
# 0 means we don't care about the specific state variable 1 means we care about it
Q: np.diag = np.diag([1.0, 1.0, 0.0, 0, 0.0])
R: np.diag = np.diag([0.5, 0.5, 0.5])

# we will now slot the MPC weights into the MPCParams class
mpc_params: MPCParams = MPCParams(Q=Q, R=R, N=15, dt=dt)
# formulate your optimal control problem
parachute_model.set_control_limits(control_limits_dict)
parachute_model.set_state_limits(state_limits_dict)

optimal_parachute:MPCParachute  = MPCParachute(
    mpc_params=mpc_params,
    casadi_model=parachute_model)

# filter for design considerations, what is an adequate response time


####################################
##### Filters for Models ######
####################################
a1p = 0.001468
a2p = 0.005017
a3p = 0.001066
b1p = -2.447
b2p = 1.9862
b3p = -0.5273
a = np.array([a1p, a2p, a3p])
b = np.array([b1p, b2p, b3p])
psi_filt_in = np.zeros(3)
psi_filt_out = np.zeros(4)

third_order_psi_filter = ThirdOrderButterworthFilter(
    a=a,
    b=b,
    x0=np.zeros(3),
    y0=np.zeros(4)
)

# Glide Ratio LPF 
a1g = 0.001468
a2g = 0.005017
a3g = 0.001066
b1g = -2.447
b2g = 1.9862
b3g = -0.5273
a = np.array([a1g, a2g, a3g])
b = np.array([b1g, b2g, b3g])
third_order_gr_filter = ThirdOrderButterworthFilter(
    a= a,
    b= b,
    x0=np.zeros(3),
    y0=np.zeros(4)
)
GR_filt_in = np.zeros(3)
GR_filt_out = np.zeros(4)

S_filt_in = np.zeros(3)
S_filt_out = np.zeros(4)

# Drag Area LPF
a1s = 0.001468
a2s = 0.005017
a3s = 0.001066
b1s = -2.447
b2s = 1.9862
b3s = -0.5273
a = np.array([a1s, a2s, a3s])
b = np.array([b1s, b2s, b3s])
third_order_s_filter = ThirdOrderButterworthFilter(
    a=a,
    b=b,
    x0=np.zeros(3),
    y0=np.zeros(4)
)

first_order_psi_filter = FirstOrderAngleFilter(
    dt=dt,tau=0.5,y0=0.0,                  
    max_rate_rad_s=np.deg2rad(90))
first_order_gr_filter = FirstOrderFilter(
    dt=dt,
    tau=1.0,
    x0=GRmin)
first_order_s_filter = FirstOrderFilter(
    dt=dt,
    tau=5.0,
    x0=Smin)

####################################
##### Main Simulation/Program ######
####################################    
# Get the desired path that will be steered towards
  
simulator: ParachuteSimulator = ParachuteSimulator(
    pred_wind_data_path='Balloon_Wind_Data_Files/8_matlab_desc.csv',
    actual_wind_data_path='Balloon_Wind_Data_Files/15_matlab_desc.csv',
    predictive_control = optimal_parachute,
    starting_conditions=simulation_config,
    parachute_config=parachute_config,
    psi_filter=first_order_psi_filter,
    glide_ratio_filter=first_order_gr_filter,
    s_filter=first_order_s_filter,
    dt = dt,
    battery_cfg=batt_cfg,
    gps_cfg=gps_cfg,
    payload_cfg=payload_cfg,
    wind_gust=wind_gust,
)

simulator.simulate(n_steps=10000, v_term=-10.0)
