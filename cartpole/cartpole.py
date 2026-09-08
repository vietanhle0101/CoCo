import os
import cvxpy as cp
import yaml
import pickle
import numpy as np
import sys
import pdb

# The original notebooks expected a ``CoCo`` environment variable to contain
# the repository root.  Derive that path from this module when it is not set so
# Cartpole can also be imported directly from a project-local virtualenv.
project_root = os.environ.get(
    'CoCo', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(1, project_root)

from core import Problem

class Cartpole(Problem):
    """Class to setup + solve cartpole problems."""

    def __init__(self, config=None, solver=cp.GUROBI, prob_params=None,
                 sampled_params=None, state_slack_weight=0.0,
                 state_slack_max=None):
        """Constructor for Cartpole class.

        Args:
            config: full path to config file. if None, load default config.
            solver: solver object to be used by cvxpy
            prob_params: optional in-memory problem parameters. When supplied,
                no configuration file is loaded.
            sampled_params: names of varying parameters when ``prob_params``
                is supplied; defaults to ``['x0', 'xg']``.
            state_slack_weight: if positive, the state box constraints
                (``x_min``/``x_max``) become soft: a per-stage nonnegative
                slack widens them symmetrically, penalized quadratically by
                this weight. 0 (default) keeps the original hard constraints,
                so existing callers are unaffected.
            state_slack_max: optional length-``n`` cap on the slack. Needed
                whenever the wall-contact big-M constants (``delta_min/max``,
                ``sc_min/max`` in ``prob_params``) were not sized to already
                cover the slack range -- otherwise the state box becomes soft
                while the (still hard) wall logic does not, and a large slack
                excursion can make the wall constraints themselves infeasible.
        """
        super().__init__()
        self.state_slack_weight = state_slack_weight
        self.state_slack_max = (
            None if state_slack_max is None else np.asarray(state_slack_max, dtype=float)
        )

        ## TODO(pculbertson): allow different sets of params to vary.
        if prob_params is None:
            if config is None: #use default config
                relative_path = os.path.dirname(os.path.abspath(__file__))
                config = relative_path + '/config/default.p'

            config_file = open(config,"rb")
            _, prob_params, self.sampled_params = pickle.load(config_file)
            config_file.close()
        else:
            self.sampled_params = sampled_params or ['x0', 'xg']
        self.init_problem(prob_params)

    def init_problem(self,prob_params):
        # setup problem params
        self.n = 4; self.m = 3 

        self.N, self.Ak, self.Bk, self.Q, self.R, self.x_min, self.x_max, \
            self.uc_min, self.uc_max, self.sc_min, self.sc_max, \
            self.delta_min, self.delta_max, self.ddelta_min, self.ddelta_max, \
            self.dh, self.g, self.l, self.mc, self.mp, self.kappa, \
            self.nu, self.dist = prob_params
        # A zero affine term reproduces the original LTI dynamics.  It can be
        # replaced with a stage-wise term for sequentially linearized MPC.
        self.ck = np.zeros((self.n, self.N - 1))
        # Preserve the original terminal Q penalty unless MPC explicitly
        # replaces it by a full terminal quadratic matrix.
        self.terminal_weight = 1.0
        self.terminal_cost = self.Q.copy()

        self.init_bin_problem()
        self.init_mlopt_problem()

    def _dynamics_at(self, step):
        """Return the affine dynamics matrices at one horizon stage.

        Default configurations store LTI matrices with shapes ``(n, n)`` and
        ``(n, m)``.  Sequentially linearized MPC may instead provide one
        matrix per stage, stacked along axis 2.
        """
        ak = self.Ak if self.Ak.ndim == 2 else self.Ak[:, :, step]
        bk = self.Bk if self.Bk.ndim == 2 else self.Bk[:, :, step]
        ck = self.ck[:, step]
        return ak, bk, ck

    def set_time_varying_dynamics(self, ak, bk, ck):
        """Replace the dynamics with a stage-wise affine model and rebuild.

        Args:
            ak: State-transition matrices with shape ``(n, n, N-1)``.
            bk: Input matrices with shape ``(n, m, N-1)``.
            ck: Affine offsets with shape ``(n, N-1)``.
        """
        expected_stages = self.N - 1
        if ak.shape != (self.n, self.n, expected_stages):
            raise ValueError("ak must have shape (n, n, N-1)")
        if bk.shape != (self.n, self.m, expected_stages):
            raise ValueError("bk must have shape (n, m, N-1)")
        if ck.shape != (self.n, expected_stages):
            raise ValueError("ck must have shape (n, N-1)")
        self.Ak, self.Bk, self.ck = ak, bk, ck
        self.init_bin_problem()
        self.init_mlopt_problem()

    def set_terminal_weight(self, weight):
        """Set a positive scalar multiplier on the original terminal Q cost.

        This compatibility helper intentionally resets any custom terminal
        matrix installed with :meth:`set_terminal_cost`.
        """
        if weight <= 0:
            raise ValueError("terminal weight must be positive")
        self.terminal_weight = float(weight)
        self.terminal_cost = self.terminal_weight * self.Q
        self.init_bin_problem()
        self.init_mlopt_problem()

    def set_terminal_cost(self, terminal_cost):
        """Set the final-state quadratic penalty matrix.

        ``terminal_cost`` must be an ``n``-by-``n`` symmetric positive
        semidefinite matrix, e.g. the stabilizing solution of a discrete
        algebraic Riccati equation.  Stage costs remain unchanged.
        """
        terminal_cost = np.asarray(terminal_cost, dtype=float)
        if terminal_cost.shape != (self.n, self.n):
            raise ValueError("terminal_cost must have shape (n, n)")
        terminal_cost = 0.5 * (terminal_cost + terminal_cost.T)
        if np.min(np.linalg.eigvalsh(terminal_cost)) < -1e-10:
            raise ValueError("terminal_cost must be positive semidefinite")
        self.terminal_weight = None
        self.terminal_cost = terminal_cost
        self.init_bin_problem()
        self.init_mlopt_problem()

    def init_bin_problem(self):
        cons = []

        x = cp.Variable((self.n,self.N))
        u = cp.Variable((self.m, self.N-1))
        sc = u[1:,:]
        y = cp.Variable((4, self.N-1), boolean=True)
        self.bin_prob_variables = {'x': x, 'u' : u, 'y' : y}

        x0 = cp.Parameter(self.n)
        xg = cp.Parameter(self.n)
        self.bin_prob_parameters = {'x0': x0, 'xg': xg}

        # Initial condition
        cons += [x[:,0] == x0]

        # Dynamics constraints
        for kk in range(self.N-1):
            ak, bk, ck = self._dynamics_at(kk)
            cons += [x[:,kk+1] == ak @ x[:,kk] + bk @ u[:,kk] + ck]

        # State and control constraints
        state_slack = None
        if self.state_slack_weight > 0:
            state_slack = cp.Variable((self.n, self.N), nonneg=True)
            self.bin_prob_variables['state_slack'] = state_slack
        for kk in range(self.N):
            if state_slack is None:
                cons += [self.x_min - x[:,kk] <= np.zeros(self.n)]
                cons += [x[:,kk] - self.x_max <= np.zeros(self.n)]
            else:
                cons += [self.x_min - x[:,kk] <= state_slack[:,kk]]
                cons += [x[:,kk] - self.x_max <= state_slack[:,kk]]
                if self.state_slack_max is not None:
                    cons += [state_slack[:,kk] <= self.state_slack_max]

        for kk in range(self.N-1):
            cons += [self.uc_min - u[0,kk] <= 0.]
            cons += [u[0,kk] - self.uc_max <= 0.]

        # Binary variable constraints
        for kk in range(self.N-1):
            for jj in range(2):
                if jj == 0:
                    d_k    = -x[0,kk] + self.l*x[1,kk] - self.dist
                    dd_k   = -x[2,kk] + self.l*x[3,kk]
                else:
                    d_k    =  x[0,kk] - self.l*x[1,kk] - self.dist
                    dd_k   =  x[2,kk] - self.l*x[3,kk]

                y_l, y_r = y[2*jj:2*jj+2,kk]
                d_min, d_max = self.delta_min[jj], self.delta_max[jj]
                dd_min, dd_max = self.ddelta_min[jj], self.ddelta_max[jj]
                f_min, f_max = self.sc_min[jj], self.sc_max[jj]

                # Eq. (26a)
                cons += [d_min*(1-y_l) <= d_k]
                cons += [d_k <= d_max*y_l]

                # Eq. (26b)
                cons += [f_min*(1-y_r) <= self.kappa*d_k + self.nu*dd_k]
                cons += [self.kappa*d_k + self.nu*dd_k <= f_max*y_r]

                # Eq. (27)
                cons += [self.nu*dd_max*(y_l-1) <=
                         sc[jj,kk] - self.kappa*d_k - self.nu*dd_k]
                cons += [sc[jj,kk] - self.kappa*d_k - self.nu*dd_k <=
                         f_min*(y_r-1)]

                cons += [-sc[jj,kk] <= 0]
                cons += [sc[jj,kk] <= f_max*y_l]
                cons += [sc[jj,kk] <= f_max*y_r]

        # LQR objective
        lqr_cost = 0.
        for kk in range(self.N):
            cost_matrix = self.terminal_cost if kk == self.N - 1 else self.Q
            lqr_cost += cp.quad_form(x[:,kk]-xg, cost_matrix)
        for kk in range(self.N-1):
            lqr_cost += cp.quad_form(u[:,kk],self.R)
        if state_slack is not None:
            lqr_cost += self.state_slack_weight * cp.sum_squares(state_slack)

        self.bin_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def init_mlopt_problem(self):
        cons = []

        x = cp.Variable((self.n,self.N))
        u = cp.Variable((self.m, self.N-1))
        sc = u[1:,:]
        self.mlopt_prob_variables = {'x':x, 'u':u}

        x0 = cp.Parameter(self.n)
        xg = cp.Parameter(self.n)
        y = cp.Parameter((4, self.N-1))
        self.mlopt_prob_parameters = {'x0': x0, 'xg': xg, 'y': y}

        # Initial condition
        cons += [x[:,0] == x0]

        # Dynamics constraints
        for kk in range(self.N-1):
            ak, bk, ck = self._dynamics_at(kk)
            cons += [x[:,kk+1] == ak @ x[:,kk] + bk @ u[:,kk] + ck]

        # State and control constraints
        state_slack = None
        if self.state_slack_weight > 0:
            state_slack = cp.Variable((self.n, self.N), nonneg=True)
            self.mlopt_prob_variables['state_slack'] = state_slack
        for kk in range(self.N):
            if state_slack is None:
                cons += [self.x_min - x[:,kk] <= np.zeros(self.n)]
                cons += [x[:,kk] - self.x_max <= np.zeros(self.n)]
            else:
                cons += [self.x_min - x[:,kk] <= state_slack[:,kk]]
                cons += [x[:,kk] - self.x_max <= state_slack[:,kk]]
                if self.state_slack_max is not None:
                    cons += [state_slack[:,kk] <= self.state_slack_max]

        for kk in range(self.N-1):
            cons += [self.uc_min - u[0,kk] <= 0.]
            cons += [u[0,kk] - self.uc_max <= 0.]

        # Binary variable constraints
        for kk in range(self.N-1):
            for jj in range(2):
                if jj == 0:
                    d_k    = -x[0,kk] + self.l*x[1,kk] - self.dist
                    dd_k   = -x[2,kk] + self.l*x[3,kk]
                else:
                    d_k    =  x[0,kk] - self.l*x[1,kk] - self.dist
                    dd_k   =  x[2,kk] - self.l*x[3,kk]

                y_l, y_r = y[2*jj:2*jj+2,kk]
                d_min, d_max = self.delta_min[jj], self.delta_max[jj]
                dd_min, dd_max = self.ddelta_min[jj], self.ddelta_max[jj]
                f_min, f_max = self.sc_min[jj], self.sc_max[jj]

                # Eq. (26a)
                cons += [d_min*(1-y_l) <= d_k]
                cons += [d_k <= d_max*y_l]

                # Eq. (26b)
                cons += [f_min*(1-y_r) <= self.kappa*d_k + self.nu*dd_k]
                cons += [self.kappa*d_k + self.nu*dd_k <= f_max*y_r]

                # Eq. (27)
                cons += [self.nu*dd_max*(y_l-1) <=
                         sc[jj,kk] - self.kappa*d_k - self.nu*dd_k]
                cons += [sc[jj,kk] - self.kappa*d_k - self.nu*dd_k <=
                         f_min*(y_r-1)]

                cons += [-sc[jj,kk] <= 0]
                cons += [sc[jj,kk] <= f_max*y_l]
                cons += [sc[jj,kk] <= f_max*y_r]

        # LQR objective
        lqr_cost = 0.
        for kk in range(self.N):
            cost_matrix = self.terminal_cost if kk == self.N - 1 else self.Q
            lqr_cost += cp.quad_form(x[:,kk]-xg, cost_matrix)
        for kk in range(self.N-1):
            lqr_cost += cp.quad_form(u[:,kk],self.R)
        if state_slack is not None:
            lqr_cost += self.state_slack_weight * cp.sum_squares(state_slack)

        self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def solve_micp(self, params, solver=cp.MOSEK):
        """High-level method to solve parameterized MICP.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = params[p]
        
        ## TODO(pculbertson): allow different sets of params to vary.
        
        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.inf, np.inf
        if solver == cp.MOSEK:
            msk_param_dict = {}
            with open(os.path.join(project_root, 'config/mosek.yaml')) as file:
                msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, mosek_params=msk_param_dict)
        elif solver == cp.GUROBI:
            grb_param_dict = {}
            with open(os.path.join(project_root, 'config/gurobi.yaml')) as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, **grb_param_dict)
        solve_time = self.bin_prob.solver_stats.solve_time

        x_star, u_star, y_star = None, None, None
        if self.bin_prob.status in ['optimal', 'optimal_inaccurate'] and self.bin_prob.status not in ['infeasible', 'unbounded']:
            prob_success = True
            cost = self.bin_prob.value
            x_star = self.bin_prob_variables['x'].value
            u_star = self.bin_prob_variables['u'].value
            y_star = self.bin_prob_variables['y'].value.astype(int)

        # Clear any saved params
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)
        
    def solve_pinned(self, params, strat, solver=cp.GUROBI):
        """High-level method to solve MICP with pinned params & integer values.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            strat: numpy integer array, corresponding to integer values for the
                desired strategy.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy params to their values
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = params[p]

        self.mlopt_prob_parameters['y'].value = strat

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.inf, np.inf
        self.mlopt_prob.solve(solver=solver)

        solve_time = self.mlopt_prob.solver_stats.solve_time
        x_star, u_star, y_star = None, None, strat
        if self.mlopt_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value
            x_star = self.mlopt_prob_variables['x'].value
            u_star = self.mlopt_prob_variables['u'].value

        # Clear any saved params
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = None
        self.mlopt_prob_parameters['y'].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)

    def which_M(self, x, u, eq_tol=1e-5, ineq_tol=1e-5):
        """Method to check which big-M constraints are active.
        
        Args:
            x: numpy array of size [self.n, self.N], state trajectory.
            u: numpy array of size [self.m, self.N], input trajectory.
            eq_tol: tolerance for equality constraints, default of 1e-5.
            ineq_tol : tolerance for ineq. constraints, default of 1e-5.
            
        Returns:
            violations: list of which logical constraints are violated.
        """

        violations = []
        sc = u[1:,:]

        for kk in range(self.N-1):
            for jj in range(2):
                # Check for when Eq. (27) is strict equality
                if jj == 0:
                    d_k = -x[0,kk] + self.l*x[1,kk] - self.dist
                    dd_k = -x[2,kk] + self.l*x[3,kk]
                else:
                    d_k = x[0,kk] - self.l*x[1,kk] - self.dist
                    dd_k = x[2,kk] - self.l*x[3,kk]
                if abs(sc[jj,kk]-self.kappa*d_k-self.nu*dd_k) <= eq_tol:
                    violations.append(4*kk + 2*jj)
                    violations.append(4*kk + 2*jj + 1)

        return violations

    def construct_features(self, params, prob_features):
        """Helper function to construct feature vector from parameter vector.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
        """
        feature_vec = np.array([])
        x0, xg = params['x0'], params['xg'] 
        ## TODO(pculbertson): make this not hardcoded

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg))
            elif feature == "delta2_0":
                d_0 = -x0[0] + self.l*x0[1] - self.dist
                feature_vec = np.hstack((feature_vec, d_0))
            elif feature == "delta3_0":
                d_0 = x0[0] - self.l*x0[1] - self.dist
                feature_vec = np.hstack((feature_vec, d_0))
            elif feature == "delta2_g":
                d_g = -xg[0] + self.l*xg[1] - self.dist
                feature_vec = np.hstack((feature_vec, d_g))
            elif feature == "delta3_g":
                d_g = xg[0] - self.l*xg[1] - self.dist
                feature_vec = np.hstack((feature_vec, d_g))
            elif feature == "dist_to_goal":
                feature_vec = np.hstack((feature_vec, np.linalg.norm(x0-xg)))
            else:
                print('Feature {} is unknown'.format(feature))
        return feature_vec
