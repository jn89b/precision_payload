"""
Baseline PID controller

Note Include a setpoint filter 
Setpoint Filter - Using a First Order Lag filter
https://blog.opticontrols.com/archives/1319#:~:text=A%20first%2Dorder%20lag%20filter%20is%20a%20type,more%20to%20the%20output%20than%20older%20samples**
https://cookierobotics.com/084/
"""
from typing import Tuple, Dict
import numpy as np

def tau_range_from_dt(dt: float) -> Dict[str, Tuple[float,float]]:
    return {
        "fast":   (2*dt, 4*dt),
        "medium": (5*dt, 20*dt),
        "heavy":  (20*dt, 100*dt),
    }


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

class FirstOrderAngleFilter:
    """
    y[k+1] = y[k] + k * wrap_pi(u[k] - y[k]),   k = 1 - exp(-dt/tau)
    Optional rate limit to mimic actuator dynamics.
    Args:
        dt: float
            Time step in seconds.
        tau: float
            Time constant in seconds the higher the tau the smoother the output, \
                the lower the tau the more responsive
        y0: float
            Initial angle in radians.
        max_rate_rad_s: Optional[float]
            Optional maximum slew rate in radians/second.
    """
    def __init__(self, dt: float, tau: float, y0: float = 0.0, max_rate_rad_s: float | None = None):
        self.dt = float(dt)
        self.tau = max(float(tau), 1e-6)
        self.k = 1.0 - np.exp(-self.dt / self.tau)   # exact FOH discretization
        self.y = wrap_pi(y0)
        self.max_step = None if max_rate_rad_s is None else float(max_rate_rad_s) * self.dt

    def reset(self, y0: float):
        self.y = wrap_pi(y0)

    def step(self, u: float) -> float:
        e = wrap_pi(u - self.y)          # shortest angular error
        dy = self.k * e
        if self.max_step is not None:    # optional slew-rate limit
            dy = np.clip(dy, -self.max_step, self.max_step)
        self.y = wrap_pi(self.y + dy)
        return self.y


class FirstOrderFilter:
    """
    First-order filter class for smoothing a setpoint signal.
    https://en.wikipedia.org/wiki/Low-pass_filter
    Args:
        # the higher the tau the smoother the output, the lower the tau the more responsive
        tau (float): Time constant of the filter.
        dt (float): Time step for the filter.
        x0 (float): Initial value of the filter.
    Methods:
        filter(x: float) -> float:
            Applies the first-order filter to the input signal.
    """
    def __init__(self, tau:float, dt:float, 
                 x0:float) -> None:
        self.tau:float = tau
        self.dt:float = dt
        self.x0:float = x0
        self.alpha:float = dt / (tau + dt)
        
    def filter(self, x:float) -> float:
        """
        Applies the first-order filter to the input signal.
        Args:
            x (float): Input signal to be filtered.
        Returns:
            float: Filtered output signal.
        """
        self.x0 = (1 - self.alpha) * self.x0 + self.alpha * x
        return self.x0

class ThirdOrderButterworthFilter:
    """
    https://gist.github.com/moorepants/bfea1dc3d1d90bdad2b5623b4a9e9bee
    
    Third-order Butterworth filter class for smoothing a setpoint signal.
    Args:
        a: np.array size 3
        b: np.array size 3
        x0: np.array size 3 the unfiltered signal
        y0: np.array size 4 the filtered signal
    Methods:
        filter(x: np.array) -> float:
            Applies the third-order Butterworth filter to the input signal.
    """
    def __init__(self, 
                 a:np.array,
                 b:np.array,
                 x0:np.array,
                 y0:np.array) -> None:
        self.a:np.array = a
        self.b:np.array = b
        self.x0:np.array = x0
        self.y0:np.array = y0
        
    def update_x0(self, x0:np.array) -> None:
        """
        Updates the internal state of the filter with a new x0 value.
        Args:
            x0 (np.array): New initial state for the filter.
        """
        self.x0 = x0
        
    def update_y0(self, y0:np.array) -> None:
        """
        Updates the internal state of the filter with a new y0 value.
        Args:
            y0 (np.array): New filtered state for the filter.
        """
        self.y0 = y0
        
    def filter(self, desired_in:float) -> float:
        """
        Applies the third-order Butterworth filter to the input signal.
        Args:
            desired_in (float): Input signal to be filtered.
        Returns:
            float: Filtered output signal.
        """
        a = self.a
        b = self.b
        # self.update_unfiltered_state(desired_in)
        self.x0[2] = self.x0[1]
        self.x0[1] = self.x0[0]
        self.x0[0] = desired_in
        
        self.y0[3] = self.y0[2]
        self.y0[2] = self.y0[1]
        self.y0[1] = self.y0[0]
        
        a_val: float = a[0] * desired_in + a[1] * self.x0[0] + a[2] * self.x0[1]
        b_val: float = -b[0] * self.y0[1] - b[1] * self.y0[2] - b[2] * self.y0[3]
        self.y0[0] = a_val + b_val

        return a_val + b_val
    
    
class PID:
    """
    PID controller class for controlling a system with a setpoint and current value.
    Args:
        min_constraint (float): Minimum constraint for the output.
        max_constraint (float): Maximum constraint for the output.
        use_integral (bool): Flag to use integral term in PID control.
        use_derivative (bool): Flag to use derivative term in PID control.
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step for the controller.
    
    Methods:
        compute(setpoint: float, current_value: float, dt: float) -> float:
            Computes the PID control output based on the setpoint and current value.
            
    """
    def __init__(self,
        min_constraint:float,
        max_constraint:float,
        use_integral:bool = False,
        use_derivative:bool = False,
        kp:float=0.05,
        ki:float=0.0,
        kd:float=0.0,
        dt:float=0.05) -> None:

        self.min_constraint:float = min_constraint
        self.max_constraint:float = max_constraint
        self.dt:float = dt
                
        self.use_integral:bool = use_integral
        self.use_derivative:bool = use_derivative
        
        self.kp:float = kp
        self.ki:float = ki
        self.kd:float = kd
        self.prev_error: float = None
        self.integral: float = 0.0
        
    def compute(self,
        setpoint:float,
        current_value:float,
        dt:float) -> float:
        
        error:float = setpoint - current_value
        derivative:float = (error - self.prev_error) / dt
        self.integral += error * dt
        
        if self.use_integral and self.use_derivative:
            output = (self.kp * error) + \
                (self.ki * self.integral) + (self.kd * derivative)
        elif self.use_integral:
            output:float = (self.kp * error) + \
                (self.ki * self.integral)
        elif self.use_derivative:
            output:float = (self.kp * error) + (self.kd * derivative)
        else:
            output:float = (self.kp * error)
        
        self.prev_error = error
        
        return output
    
    