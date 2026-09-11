#!/usr/bin/env python3
"""Run closed-loop MINLP MPC for the CartPole soft-wall model.

This is the nonlinear counterpart to closed_loop_mpc.py: instead of freezing
one linearization per replan, it embeds the exact nonlinear dynamics (sin
theta, no approximation) as constraints at every horizon stage, solved
directly with Gurobi's native nonlinear support (gurobipy, not cvxpy -- cvxpy
requires DCP-convex atoms and cannot express sin(theta)). Everything else
(wall-contact big-M logic, cost structure, terminal LQR cost) mirrors
closed_loop_mpc.py so the two are a fair, apples-to-apples comparison.
"""

import argparse
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
from gurobipy import GRB

try:
    from .closed_loop_mpc import (
        CART_FORCE_LIMIT, THETA_LIMIT, WALL_DISTANCE, ROOT,
        nonlinear_step, terminal_lqr_cost, plot_trajectory, save_rollout,
    )
except ImportError:  # Supports direct execution: python cartpole/closed_loop_minlp.py
    from closed_loop_mpc import (
        CART_FORCE_LIMIT, THETA_LIMIT, WALL_DISTANCE, ROOT,
        nonlinear_step, terminal_lqr_cost, plot_trajectory, save_rollout,
    )

with (Path(__file__).with_name("config") / "default.p").open("rb") as _f:
    _, _PARAMS, _ = pickle.load(_f)
Q = _PARAMS[3]
GRAVITY, LENGTH = _PARAMS[16], _PARAMS[17]
CART_MASS, POLE_MASS = _PARAMS[18], _PARAMS[19]
KAPPA, NU = _PARAMS[20], _PARAMS[21]


def _bigm(theta_limit):
    """Same big-M sizing as closed_loop_mpc.make_mpc_problem's hard-bound case."""
    x_max = np.array([WALL_DISTANCE, theta_limit, 2.0, 1.0])
    x_min = -x_max
    delta_min = np.array([
        -x_max[0] + LENGTH * x_min[1] - WALL_DISTANCE,
        x_min[0] - LENGTH * x_max[1] - WALL_DISTANCE,
    ])
    delta_max = np.array([
        -x_min[0] + LENGTH * x_max[1] - WALL_DISTANCE,
        x_max[0] - LENGTH * x_min[1] - WALL_DISTANCE,
    ])
    ddelta_min = np.array([-x_max[2] + LENGTH * x_min[3], x_min[2] - LENGTH * x_max[3]])
    ddelta_max = np.array([-x_min[2] + LENGTH * x_max[3], x_max[2] - LENGTH * x_min[3]])
    sc_min = KAPPA * delta_min + NU * ddelta_min
    sc_max = KAPPA * delta_max + NU * ddelta_max
    return x_min, x_max, delta_min, delta_max, ddelta_min, ddelta_max, sc_min, sc_max


def _terminal_cost_matrix(dt):
    """Discrete LQR terminal cost about the upright/no-contact equilibrium,
    matching closed_loop_mpc.make_mpc_problem so the two controllers are
    comparable rather than one being handicapped by a missing terminal term.
    """
    continuous_a = np.zeros((4, 4))
    continuous_a[:2, 2:] = np.eye(2)
    continuous_a[2, 1] = GRAVITY * POLE_MASS / CART_MASS
    continuous_a[3, 1] = GRAVITY * (CART_MASS + POLE_MASS) / (LENGTH * CART_MASS)
    a_nominal = np.eye(4) + dt * continuous_a
    b_cart = np.zeros(4)
    b_cart[2] = dt / CART_MASS
    b_cart[3] = dt / (LENGTH * CART_MASS)
    return terminal_lqr_cost(a_nominal, b_cart, Q, 1.0)


def solve_minlp_step(x0, goal, horizon, dt, theta_limit, wall_force_weight,
                      terminal_cost, time_limit=30):
    """Solve one MINLP horizon from the given measured state."""
    x_min, x_max, delta_min, delta_max, _, ddelta_max, sc_min, sc_max = _bigm(theta_limit)
    n = horizon

    m = gp.Model("cartpole_minlp")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.NonConvex = 2

    x = m.addVars(n, 4, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    u = m.addVars(n - 1, 3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
    y = m.addVars(n - 1, 4, vtype=GRB.BINARY, name="y")
    sin_th = m.addVars(n - 1, lb=-1, ub=1, name="sin_theta")

    for i in range(4):
        m.addConstr(x[0, i] == x0[i])
    for k in range(n):
        for i in range(4):
            x[k, i].LB, x[k, i].UB = x_min[i], x_max[i]
    for k in range(n - 1):
        u[k, 0].LB, u[k, 0].UB = -CART_FORCE_LIMIT, CART_FORCE_LIMIT

    for k in range(n - 1):
        p, th, pd, thd = x[k, 0], x[k, 1], x[k, 2], x[k, 3]
        uc, sl, sr = u[k, 0], u[k, 1], u[k, 2]
        m.addConstr(sin_th[k] == gp.nlfunc.sin(th))
        accel = GRAVITY * POLE_MASS / CART_MASS * sin_th[k] + uc / CART_MASS
        ang_accel = (
            GRAVITY * (CART_MASS + POLE_MASS) / (LENGTH * CART_MASS) * sin_th[k]
            + uc / (LENGTH * CART_MASS) - sl / (LENGTH * POLE_MASS) + sr / (LENGTH * POLE_MASS)
        )
        m.addConstr(x[k + 1, 0] == p + dt * pd)
        m.addConstr(x[k + 1, 1] == th + dt * thd)
        m.addConstr(x[k + 1, 2] == pd + dt * accel)
        m.addConstr(x[k + 1, 3] == thd + dt * ang_accel)

    for k in range(n - 1):
        for jj in range(2):
            if jj == 0:
                d_k = -x[k, 0] + LENGTH * x[k, 1] - WALL_DISTANCE
                dd_k = -x[k, 2] + LENGTH * x[k, 3]
            else:
                d_k = x[k, 0] - LENGTH * x[k, 1] - WALL_DISTANCE
                dd_k = x[k, 2] - LENGTH * x[k, 3]
            y_l, y_r = y[k, 2 * jj], y[k, 2 * jj + 1]
            d_min, d_max = delta_min[jj], delta_max[jj]
            f_min, f_max = sc_min[jj], sc_max[jj]
            sc = u[k, 1 + jj]

            m.addConstr(d_min * (1 - y_l) <= d_k)
            m.addConstr(d_k <= d_max * y_l)
            m.addConstr(f_min * (1 - y_r) <= KAPPA * d_k + NU * dd_k)
            m.addConstr(KAPPA * d_k + NU * dd_k <= f_max * y_r)
            m.addConstr(NU * ddelta_max[jj] * (y_l - 1) <= sc - KAPPA * d_k - NU * dd_k)
            m.addConstr(sc - KAPPA * d_k - NU * dd_k <= f_min * (y_r - 1))
            m.addConstr(sc >= 0)
            m.addConstr(sc <= f_max * y_l)
            m.addConstr(sc <= f_max * y_r)

    cost = gp.QuadExpr()
    for k in range(n - 1):
        for i in range(4):
            cost += Q[i, i] * (x[k, i] - goal[i]) * (x[k, i] - goal[i])
        cost += 1.0 * u[k, 0] * u[k, 0]
        cost += wall_force_weight * u[k, 1] * u[k, 1]
        cost += wall_force_weight * u[k, 2] * u[k, 2]
    for i in range(4):
        for j in range(4):
            if terminal_cost[i, j] != 0:
                cost += terminal_cost[i, j] * (x[n - 1, i] - goal[i]) * (x[n - 1, j] - goal[j])
    m.setObjective(cost, GRB.MINIMIZE)

    start = time.time()
    m.optimize()
    solve_time = time.time() - start
    status_map = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
                  GRB.INF_OR_UNBD: "infeasible_or_unbounded", GRB.TIME_LIMIT: "time_limit"}
    status = status_map.get(m.Status, f"status_{m.Status}")
    if m.SolCount == 0:
        return status, solve_time, None, None
    u0 = np.array([u[0, i].X for i in range(3)])
    contact = [bool(round(y[0, 0].X)) and bool(round(y[0, 1].X)),
               bool(round(y[0, 2].X)) and bool(round(y[0, 3].X))]
    return status, solve_time, u0, contact


def run_mpc(initial_state, goal_state, steps, horizon, timestep, theta_limit,
            wall_force_weight=5.0, time_limit=30):
    """Closed-loop MINLP MPC, propagated through the same exact nonlinear
    dynamics the MINLP itself plans with (so plant == internal model here,
    unlike closed_loop_mpc.py's toy/mujoco plant options)."""
    terminal_cost = _terminal_cost_matrix(timestep)
    plant_params = SimpleNamespace(
        g=GRAVITY, l=LENGTH, mc=CART_MASS, mp=POLE_MASS, dh=timestep,
        N=horizon, x_min=_bigm(theta_limit)[0], x_max=_bigm(theta_limit)[1],
    )

    states = np.empty((steps + 1, 4))
    controls = np.empty((steps, 3))
    costs = np.empty(steps)
    solve_times = np.empty(steps)
    states[0] = initial_state

    for k in range(steps):
        status, solve_time, u0, contact = solve_minlp_step(
            states[k], goal_state, horizon, timestep, theta_limit,
            wall_force_weight, terminal_cost, time_limit,
        )
        if u0 is None:
            raise RuntimeError(f"MINLP status {status!r} at MPC step {k} from state {states[k]}.")

        controls[k] = u0
        states[k + 1] = nonlinear_step(plant_params, states[k], u0)
        costs[k] = np.nan  # per-step objective value not tracked (nonconvex, changes with reference)
        solve_times[k] = solve_time
        print(
            f"step {k + 1:02d}/{steps}: state={states[k]}, solve={solve_time:.4f}s, "
            f"u={controls[k]}, contact=[left={contact[0]}, right={contact[1]}]"
        )

    return plant_params, states, controls, costs, solve_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of MPC updates.")
    parser.add_argument("--horizon", type=int, default=20, help="Number of predicted states.")
    parser.add_argument("--dt", type=float, default=0.02, help="MPC and simulation timestep [s].")
    parser.add_argument(
        "--wall-force-weight", type=float, default=5.0,
        help="Quadratic cost weight on s_L/s_R (default: 5.0; use 0 to disable).",
    )
    parser.add_argument(
        "--time-limit", type=float, default=30.0,
        help="Per-step Gurobi time limit in seconds (default: 30).",
    )
    parser.add_argument(
        "--x0", type=float, nargs=4, default=[0.1, 0.0, 0.0, 0.0],
        metavar=("P", "THETA", "P_DOT", "THETA_DOT"), help="Initial state.",
    )
    parser.add_argument(
        "--goal", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0],
        metavar=("P", "THETA", "P_DOT", "THETA_DOT"), help="Tracking goal.",
    )
    parser.add_argument(
        "--plot", type=Path, default=ROOT / "cartpole" / "outputs" / "closed_loop_minlp.png",
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--data", type=Path, default=ROOT / "cartpole" / "data" / "closed_loop_minlp.npz",
        help="Destination compressed rollout archive for later replay.",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.horizon < 2 or args.dt <= 0:
        parser.error("--steps and --dt must be positive; --horizon must be at least 2")

    problem, states, controls, costs, solve_times = run_mpc(
        np.asarray(args.x0), np.asarray(args.goal), args.steps, args.horizon, args.dt,
        THETA_LIMIT, wall_force_weight=args.wall_force_weight, time_limit=args.time_limit,
    )
    plot_trajectory(problem, states, controls, np.asarray(args.goal), args.plot)
    save_rollout(
        args.data, problem, states, controls, costs, solve_times,
        np.asarray(args.goal), "minlp",
    )
    print(f"Saved plot to {args.plot}")
    print(f"Saved rollout data to {args.data}")
    print(f"Mean MIQP solve time: {np.mean(solve_times):.4f}s")
    print(f"Final state: {states[-1]}")


if __name__ == "__main__":
    main()
