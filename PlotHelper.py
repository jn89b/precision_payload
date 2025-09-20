import matplotlib.pyplot as plt
# set font size and type
import numpy as np
import seaborn as sns
# PREVENT circular import
from l1_astar_loiter_wind import Prior, Config, Truth
from typing import Tuple, Callable, List, Optional, Dict
plt.rcParams.update({'font.size': 16})
# light grid lines
plt.rcParams['grid.color'] = 'lightgray'
# tight layout
plt.rcParams['figure.autolayout'] = True
GROUND_TRUTH_COLOR: str = 'black'  # color for ground truth wind profile in plots
GROUND_TRUTH_ALPHA : float = 0.5  # alpha for ground truth wind profile in plots


def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

def animate_loiter_path() -> None:
    """
    Animation of mothership loitering path
    Animatte the wind profile
    """

def plot_wind_profile(prior:Prior,
                      config:Config,
                      pins:List[Dict],
                      pins_src:str,
                      updated_wind:callable,
                      updated_std:callable,
                      zmin:float, 
                      zmax:float,
                      spacing:int,
                      truth:Truth = None,
                      save:bool = False,
                      file_name:str = "") -> Tuple[plt.Figure, plt.Axes]:
    """
    x axis will be altitude 
    y axis will be the wind speed
    dashed line is the ground truth
    - solid line is the prior from the gp 
    - dashed line is the 
    """
    
    fig , axs = plt.subplots(1, 2, figsize=(10, 6))
    z_plot = np.linspace(zmin, zmax, 300)
    prior_wN = np.array([prior.wind(z)[0] for z in z_plot])
    if prior is not None:
        prior_wN = np.array([prior.wind(z)[0] for z in z_plot])
        axs[0].plot(prior_wN, z_plot, 'C0--', alpha=0.5, label='prior wN (ref)')
    post_wN  = np.array([updated_wind(z)[0] for z in z_plot])
    post_sN  = np.array([updated_std(z)[0]  for z in z_plot])
    
    # axs[0].plot(prior_wN, z_plot, 'C0--', alpha=0.5, label='prior wN (ref)')
    axs[0].plot(post_wN,  z_plot, 'C0-',  label='pins-only wN')
    axs[0].fill_betweenx(z_plot, post_wN - 2*post_sN, post_wN + 2*post_sN, 
                         color='C0', alpha=0.12, label='±2σ')
    # axs[0].plot(post_wN, z_plot, 'C0-',  label='updated wN')
    #axs[0].fill_betweenx(z_plot, post_wN - 2*post_sN, post_wN + 2*post_sN, color='C0', alpha=0.15, label='±2σ')
    axs[0].invert_yaxis(); axs[0].grid(True)
    axs[0].set_xlabel("wN [m/s]"); axs[0].set_ylabel("Altitude [m]")
    axs[0].set_title(f"North wind profile — pins ({pins_src})")
    for p in pins:
        axs[0].plot([p["wN"]], [p["z"]], 'ko', ms=4)

    # (2) Prior + updated profile (wE)
    prior_wE = np.array([prior.wind(z)[1] for z in z_plot])
    post_wE  = np.array([updated_wind(z)[1] for z in z_plot])
    post_sE  = np.array([updated_std(z)[1]  for z in z_plot])
    axs[1].plot(prior_wE, z_plot, 'C1--', alpha=0.5, label='prior wE (ref)')
    axs[1].plot(post_wE,  z_plot, 'C1-',  label='pins-only wE')
    axs[1].fill_betweenx(z_plot, post_wE - 2*post_sE, post_wE + 2*post_sE, 
                         color='C1', alpha=0.12, label='±2σ')

    axs[1].plot(post_wE, z_plot, 'C1-',  label='updated wE')
    axs[1].invert_yaxis(); axs[1].grid(True)
    axs[1].set_xlabel("wE [m/s]"); axs[1].set_ylabel("Altitude [m]")
    axs[1].set_title("East wind profile")
    for p in pins:
        axs[1].plot([p["wE"]], [p["z"]], 'ko', ms=4)
        
    axs[0].legend(loc='best'); axs[1].legend(loc='best')
    
    if truth is not None:
        truth_wN = np.array([truth.wind(z)[0] for z in z_plot])
        truth_wE = np.array([truth.wind(z)[1] for z in z_plot])

        rmseN = rmse(post_wN, truth_wN); rmseE = rmse(post_wE, truth_wE)
        covN = float(np.mean((truth_wN >= post_wN - 2*post_sN) & (truth_wN <= post_wN + 2*post_sN)))
        covE = float(np.mean((truth_wE >= post_wE - 2*post_sE) & (truth_wE <= post_wE + 2*post_sE)))

        axs[0].plot(truth_wN, z_plot, color=GROUND_TRUTH_COLOR, lw=2, alpha=GROUND_TRUTH_ALPHA, ls='--', label='truth wN')
        axs[1].plot(truth_wE, z_plot, color=GROUND_TRUTH_COLOR, lw=2, alpha=GROUND_TRUTH_ALPHA, ls='--', label='truth wE')

        axs[0].set_title(f"North wind — GP vs truth (RMSE={rmseN:.2f} m/s, cov≈{100*covN:.0f}%)")
        axs[1].set_title(f"East wind — GP vs truth  (RMSE={rmseE:.2f} m/s, cov≈{100*covE:.0f}%)")
    
    if save:
        plt.savefig(file_name, dpi=400)
        plt.savefig(file_name+".svg")
        

    return fig, axs

def plot_carp_paths(config:Config, history_x:List[np.ndarray], 
                    history_y:List[np.ndarray], history_z:List[np.ndarray],
                    scores:List[float], save:bool=False) -> None:
    """
    Plot the CARP paths
    We want to find the best release point for our payload using 
    our updated Posterior wind profile
    Choose the Carp that minimizes the distance to the target location
    """
    # make a 3d plot
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(history_x)):
    #     ax.plot(history_x[i], history_y[i], history_z[i], alpha=0.3, label=f'Candidate {i+1} (miss={scores[i]:.1f} m)')
    #     ax.scatter(history_x[i][-1], history_y[i][-1], history_z[i][-1], marker='x', s=100, color='red')
    # ax.set_xlabel('East [m]')
    # ax.set_ylabel('North [m]')
    # ax.set_zlabel('Altitude [m]')
    # ax.set_title(f'CARP candidates from {release_altitude} m altitude')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ring_t = np.linspace(0,2*np.pi,361)
    ring_x = config.loiter_ring_R_m*np.sin(ring_t)  # x=E
    ring_y = config.loiter_ring_R_m*np.cos(ring_t)  # y=N
    ax.plot(ring_x, ring_y, color='0.7', lw=1, label='loiter ring')
    tgtN, tgtE = config.target_NE
    ax.scatter([tgtE],[tgtN], marker='x', s=80, label="Target")
    # ax.scatter([carp_NE[1]],[carp_NE[0]], s=40, label="CARP")
    # ax.plot(X_det, Y_det, lw=1.5, label=f"Deterministic drop (miss={miss_det:.1f} m)")
    for i, (x, y, z) in enumerate(zip(history_x, history_y, history_z)):
        ax.plot(x, y, alpha=0.3, label=f'Candidate {i+1} (miss={scores[i]:.1f} m)')
        ax.scatter(x[-1], y[-1], marker='x', s=100, color='red')
    ax.axis('equal'); ax.grid(True)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("CARP with updated profile")
    ax.legend(loc="best")
    
    return fig, ax