
import numpy as np
import casadi as ca

from typing import Tuple, List, Dict
from dataclasses import dataclass
from casadi.casadi import MX
from optitraj.models.casadi_model import CasadiModel
from optitraj.utils.data_container import MPCParams
from optitraj import OptimalControlProblem

class MPCParachute(OptimalControlProblem):
    """
    Used for dynamic avoidance with threats in the environment
    """
    def __init__(self, mpc_params:MPCParams,
                 casadi_model:CasadiModel,
                 wind: int = 1,
                 wind_dim: int = 3,
                 num_positions: int = 1) -> None:
        
        self.mpc_params = mpc_params
        self.casadi_model = casadi_model
        self.num_wind    = wind # number of threats in the environment
        self.wind_dim     = wind_dim # x,y 
        self.num_positions  = num_positions # number of positions for each threat
        super().__init__(mpc_params, casadi_model)
        self.set_dynamic_constraints()
        

    def _parameter_length(self) -> int:
        # base has initial + final = 2*nstates
        base_len = 2 * self.casadi_model.n_states
        extra   = self.num_wind * self.wind_dim * self.num_positions
        return base_len + extra

    def set_dynamic_constraints(self):
        """
        Define dynamic constraints for the system using a Runge-Kutta 4th-order (RK4) integration scheme.
        This ensures the system dynamics are respected.
        Have to override this method to include the wind as the parameter to the model
        """
        n_states:int = self.casadi_model.n_states
        D    = self.wind_dim              # = 2 (x,y)
        Tn   = self.num_wind             # = 2
        Tp   = self.num_positions           # e.g. 5
        N    = self.N                       # horizon length
        
        #extract the wind location
        flat = self.P[2*n_states : 2*n_states + Tn*Tp*D]
        wind_matrix = ca.reshape(flat, D, Tn*Tp)
        
        self.g = self.X[:, 0] - self.P[:self.casadi_model.n_states]
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            k1 = self.casadi_model.function(states, controls, wind_matrix)
            k2 = self.casadi_model.function(states + self.dt/2 * k1, controls, wind_matrix)
            k3 = self.casadi_model.function(states + self.dt/2 * k2, controls, wind_matrix)
            k4 = self.casadi_model.function(states + self.dt * k3, controls, wind_matrix)
            state_next_rk4 = states + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Add the dynamic constraint to the constraints list
            self.g = ca.vertcat(self.g, self.X[:, k+1] - state_next_rk4)

    def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray,
              wind_info: np.ndarray) -> np.ndarray:
        """
        Solve the optimal control problem for the given initial state and control
        This method has to be updated since p is now a vector that includes the initial conditions
        and the threat parameters.
        
        Args:
            x0 (np.ndarray): Initial state of the system.
            xF (np.ndarray): Final state of the system.
            u0 (np.ndarray): Initial control input.
            wind_info (np.ndarray): wind position, 
            where positions is a 1d array of shape (num_wind * num_positions * wind_dimensions,).
        
        Returns:
            np.ndarray: Solution to the optimal control problem.
        """
        state_init = ca.DM(x0)
        state_final = ca.DM(xF)

        X0 = ca.repmat(state_init, 1, self.N+1)
        U0 = ca.repmat(u0, 1, self.N)

        n_states = self.casadi_model.n_states
        n_controls = self.casadi_model.n_controls

        num_constraints = n_states*(self.N+1)
        lbg = ca.DM.zeros((num_constraints, 1))
        ubg = ca.DM.zeros((num_constraints, 1))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }
        
        # TODO: add threat check
        
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_final,   # target state
            wind_info.flatten()  # threat positions
        )

        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(self.N+1), 1),
            ca.reshape(U0, n_controls*self.N, 1)
        )
        # init_time = time.time()
        solution = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        return solution

    def solve_and_get_solution(self, x0: np.ndarray,
                               xF: np.ndarray,
                               u0: np.ndarray,
                               wind_data:np.ndarray) -> Dict:
        """
        Solve the optimization problem and return the state and control trajectories.

        Parameters:
            x0 (np.ndarray): Initial state.
            xF (np.ndarray): Final state.
            u0 (np.ndarray): Initial control input.
            wind_data (np.ndarray): Positions of threats in the environment.

        Returns:
            Dict: Solution containing states and controls.
        """
        solution = self.solve(x0, xF, u0, wind_data)
        return self.get_solution(solution)

    def compute_dynamics_cost(self) -> ca.MX:
        """
        Compute the dynamics cost for the optimal control problem

        (xQx + uRU)^T -> Non
        """
        # initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R

        x_final = self.P[self.casadi_model.n_states:2*self.casadi_model.n_states]

        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                + (states - x_final).T @ Q @ (states - x_final) \
                + controls.T @ R @ controls

        return cost    

    def compute_goal_cost(self) -> ca.MX:
        # unpack sizes
        n_s  = self.casadi_model.n_states   # total state dim (here 3)
        D    = 2                             # x,y only
        N    = self.N                       # horizon length

        # 1) extract the goal (x_goal, y_goal) from P; 
        #    P layout: [ x0; y0; psi0;  x_goal; y_goal; psi_goal;  threat… ]
        x_goal = self.P[n_s + 0]
        y_goal = self.P[n_s + 1]

        # 2) grab all planned (x,y) over steps 1..N  → shape (2, N)
        #    (we skip X[:,0] since that's the current state)
        pos = self.X[0:D, 1:N+1]            # X[0:2,1:N+1]

        # 3) build a 2×N matrix of the goal repeated
        goal_mat = ca.repmat(ca.vertcat(x_goal, y_goal), 1, N)

        # 4) squared distances per step (→ 1×N row-vector)
        d2 = ca.sumsqr(pos - goal_mat)

        # 5) sum over all steps to get a scalar, and weight it
        w_goal = 1E-3  # tune this to trade off goal vs avoidance
        cost_goal = w_goal * ca.sum2(d2)

        return cost_goal


    def compute_total_cost(self) -> ca.MX:
        cost = 0
        cost += self.compute_dynamics_cost()
        return cost
        


# class MPCParachute(OptimalControlProblem):
#     """
#     Example of a class that inherits from OptimalControlProblem
#     for the Plane model using Casadi, can be used for 
#     obstacle avoidance
#     """

#     def __init__(self,
#                  mpc_params: MPCParams,
#                  casadi_model: CasadiModel) -> None:
#         super().__init__(mpc_params,
#                          casadi_model)

#     def compute_dynamics_cost(self) -> MX:
#         """
#         Compute the dynamics cost for the optimal control problem
#         """
#         # initialize the cost
#         cost = 0.0
#         Q = self.mpc_params.Q
#         R = self.mpc_params.R

#         x_final = self.P[self.casadi_model.n_states:]

#         for k in range(self.N):
#             states = self.X[:, k]
#             controls = self.U[:, k]
#             cost += cost \
#                 + (states - x_final).T @ Q @ (states - x_final) \
#                 + controls.T @ R @ controls

#         return cost

#     def compute_total_cost(self) -> MX:
#         cost = self.compute_dynamics_cost()
#         # cost = cost + self.compute_obstacle_avoidance_cost()
#         return cost

#     def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> np.ndarray:
#         """
#         Solve the optimal control problem for the given initial state and control

#         """
#         state_init = ca.DM(x0)
#         state_final = ca.DM(xF)

#         X0 = ca.repmat(state_init, 1, self.N+1)
#         U0 = ca.repmat(u0, 1, self.N)

#         n_states = self.casadi_model.n_states
#         n_controls = self.casadi_model.n_controls
#         # self.compute_obstacle_avoidance_cost()
#         num_constraints = n_states*(self.N+1)
#         lbg = ca.DM.zeros((num_constraints, 1))
#         ubg = ca.DM.zeros((num_constraints, 1))

#         args = {
#             'lbg': lbg,
#             'ubg': ubg,
#             'lbx': self.pack_variables_fn(**self.lbx)['flat'],
#             'ubx': self.pack_variables_fn(**self.ubx)['flat'],
#         }
#         args['p'] = ca.vertcat(
#             state_init,    # current state
#             state_final   # target state
#         )

#         args['x0'] = ca.vertcat(
#             ca.reshape(X0, n_states*(self.N+1), 1),
#             ca.reshape(U0, n_controls*self.N, 1)
#         )
#         # init_time = time.time()
#         solution = self.solver(
#             x0=args['x0'],
#             lbx=args['lbx'],
#             ubx=args['ubx'],
#             lbg=args['lbg'],
#             ubg=args['ubg'],
#             p=args['p']
#         )

#         return solution
