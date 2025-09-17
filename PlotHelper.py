import matplotlib.pyplot as plt
# set font size and type
import numpy as np
import seaborn as sns
from l1_astar_loiter_wind import

plt.rcParams.update({'font.size': 16})
# light grid lines
plt.rcParams['grid.color'] = 'lightgray'


def animate_loiter_path() -> None:
    """
    Animation of mothership loitering path
    Animatte the wind profile
    """

def plot_wind_profile() -> None:
    """
    x axis will be altitude 
    y axis will be the wind speed
    dashed line is the ground truth
    - solid line is the prior from the gp 
    - dashed line is the 
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    