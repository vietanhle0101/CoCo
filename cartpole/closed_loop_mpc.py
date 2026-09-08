#!/usr/bin/env python3
"""Run current-state-linearized MIQP MPC for the CartPole soft-wall model.

At every simulation step, the nonlinear model is linearized once at the
current measured state. That local affine model is used for the MIQP horizon,
Gurobi solves the MIQP, and only its first input is applied before repeating.
"""

import argparse
import os
import pickle
from pathlib import Path

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The default Matplotlib config directory may be read-only in a virtualenv or
# remote workspace.  Configure it before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt

from cartpole import Cartpole


def make_mpc_problem(horizon, timestep, theta_limit):
    """Create an MPC-only model override without changing saved datasets."""
    config_path = Path(__file__).with_name("config") / "default.p"
    with config_path.open("rb") as config_file:
        _, parameters, sampled_params = pickle.load(config_file)
    parameters = list(parameters)
    x_min, x_max = parameters[5], parameters[6]
    ddelta_min, ddelta_max = parameters[13], parameters[14]
    gravity, length = parameters[16], parameters[17]
    cart_mass, pole_mass = parameters[18], parameters[19]
    kappa, nu, distance = parameters[20], parameters[21], parameters[22]
    continuous_a = np.zeros((4, 4))
    continuous_a[:2, 2:] = np.eye(2)
    continuous_a[2, 1] = gravity * pole_mass / cart_mass
    continuous_a[3, 1] = gravity * (cart_mass + pole_mass) / (length * cart_mass)
    continuous_b = np.zeros((4, 3))
    continuous_b[2, 0] = 1 / cart_mass
    continuous_b[3, :] = np.array([
        1 / (length * cart_mass), -1 / (length * pole_mass),
        1 / (length * pole_mass),
    ])
    parameters[0] = horizon
    parameters[1] = np.eye(4) + timestep * continuous_a
    parameters[2] = timestep * continuous_b
    parameters[15] = timestep
    x_max = np.asarray(x_max, dtype=float).copy()
    x_max[1] = theta_limit
    x_min = -x_max
    delta_min = np.array([
        -x_max[0] + length * x_min[1] - distance,
        x_min[0] - length * x_max[1] - distance,
    ])
    delta_max = np.array([
        -x_min[0] + length * x_max[1] - distance,
        x_max[0] - length * x_min[1] - distance,
    ])
    parameters[5], parameters[6] = x_min, x_max
    parameters[9] = kappa * delta_min + nu * ddelta_min
    parameters[10] = kappa * delta_max + nu * ddelta_max
    parameters[11], parameters[12] = delta_min, delta_max
    return Cartpole(prob_params=parameters, sampled_params=sampled_params)


def nonlinear_step(problem, state, control):
    """Sine-based nonlinear extension whose Jacobian matches the LTI model."""
    _, angle, velocity, angular_rate = state
    cart_force, left_force, right_force = control
    acceleration = (
        problem.g * problem.mp / problem.mc * np.sin(angle)
        + cart_force / problem.mc
    )
    angular_acceleration = (
        problem.g * (problem.mc + problem.mp) / (problem.l * problem.mc) * np.sin(angle)
        + cart_force / (problem.l * problem.mc)
        - left_force / (problem.l * problem.mp)
        + right_force / (problem.l * problem.mp)
    )
    return state + problem.dh * np.array(
        [velocity, angular_rate, acceleration, angular_acceleration]
    )


def linearize_at_current_state(problem, state):
    """Return x+ = A x + B u + c linearized at the current state.

    The affine term c is necessary because A and B describe perturbations
    about the current state, whereas the MIQP optimizes absolute states.
    """
    angle = state[1]
    dt = problem.dh
    a = np.eye(problem.n)
    a[0, 2] = dt
    a[1, 3] = dt
    a[2, 1] = dt * problem.g * problem.mp / problem.mc * np.cos(angle)
    a[3, 1] = dt * problem.g * (problem.mc + problem.mp) / (problem.l * problem.mc) * np.cos(angle)
    b = np.zeros((problem.n, problem.m))
    b[2, 0] = dt / problem.mc
    b[3, :] = dt * np.array([
        1 / (problem.l * problem.mc),
        -1 / (problem.l * problem.mp),
        1 / (problem.l * problem.mp),
    ])
    c = nonlinear_step(problem, state, np.zeros(problem.m)) - a @ state
    return a, b, c


def run_mpc(initial_state, goal_state, steps, horizon, timestep, theta_limit):
    """Run MPC with one system linearization at each current state."""
    problem = make_mpc_problem(horizon, timestep, theta_limit)
    states = np.empty((steps + 1, problem.n))
    controls = np.empty((steps, problem.m))
    costs = np.empty(steps)
    solve_times = np.empty(steps)
    states[0] = initial_state

    for k in range(steps):
        # Linearize only at the measured x_k and keep this model fixed across
        # the prediction horizon, as in standard current-state LTV MPC.
        a, b, c = linearize_at_current_state(problem, states[k])
        stages = problem.N - 1
        problem.set_time_varying_dynamics(
            np.repeat(a[:, :, None], stages, axis=2),
            np.repeat(b[:, :, None], stages, axis=2),
            np.repeat(c[:, None], stages, axis=1),
        )
        parameters = {"x0": states[k], "xg": goal_state}
        success, cost, solve_time, (_, optimal_inputs, _) = problem.solve_micp(
            parameters, solver=cp.GUROBI
        )
        if not success:
            raise RuntimeError(
                f"MIQP status {problem.bin_prob.status!r} at MPC step {k} "
                f"from state {states[k]}."
            )

        # Receding-horizon control: apply the first input only, then replan.
        controls[k] = optimal_inputs[:, 0]
        states[k + 1] = nonlinear_step(problem, states[k], controls[k])
        costs[k] = cost
        solve_times[k] = solve_time
        print(
            f"step {k + 1:02d}/{steps}: cost={cost:.3f}, "
            f"solve={solve_time:.4f}s, u={controls[k]}"
        )

    return problem, states, controls, costs, solve_times


def plot_trajectory(problem, states, controls, goal_state, output_path):
    """Plot states, cart/contact inputs, and cart position against its bounds."""
    time_state = np.arange(len(states)) * problem.dh
    time_input = np.arange(len(controls)) * problem.dh
    labels = [r"cart position $p$", r"pole angle $\theta$", r"cart velocity $\dot p$", r"pole rate $\dot\theta$"]

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for index, label in enumerate(labels):
        axes[0].plot(time_state, states[:, index], label=label)
        axes[0].axhline(goal_state[index], color=f"C{index}", linestyle=":", alpha=0.7)
    axes[0].set(title="Closed-loop state trajectory", xlabel="time [s]", ylabel="state")
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(time_input, controls[:, 0], where="post", label=r"cart force $u^c$")
    axes[1].step(time_input, controls[:, 1], where="post", label=r"left-wall force $s^L$")
    axes[1].step(time_input, controls[:, 2], where="post", label=r"right-wall force $s^R$")
    axes[1].set(title="Applied MPC inputs", xlabel="time [s]", ylabel="input")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_state, states[:, 0], label="cart position")
    axes[2].axhline(problem.x_min[0], color="black", linestyle="--", label="cart bounds")
    axes[2].axhline(problem.x_max[0], color="black", linestyle="--")
    axes[2].set(title="Cart motion within track bounds", xlabel="time [s]", ylabel=r"$p$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40, help="Number of MPC updates.")
    parser.add_argument("--horizon", type=int, default=50, help="Number of predicted states.")
    parser.add_argument("--dt", type=float, default=0.02, help="MPC and simulation timestep [s].")
    parser.add_argument(
        "--theta-limit", type=float, default=np.pi / 2,
        help="Symmetric pole-angle state bound in radians (default: pi/2).",
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
        "--plot", type=Path, default=ROOT / "cartpole" / "outputs" / "closed_loop_mpc.png",
        help="Destination PNG path.",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.horizon < 2 or args.dt <= 0 or args.theta_limit <= 0:
        parser.error("--steps, --theta-limit, and --dt must be positive; --horizon must be at least 2")

    problem, states, controls, costs, solve_times = run_mpc(
        np.asarray(args.x0), np.asarray(args.goal), args.steps, args.horizon, args.dt,
        args.theta_limit,
    )
    plot_trajectory(problem, states, controls, np.asarray(args.goal), args.plot)
    print(f"Saved plot to {args.plot}")
    print(f"Mean MIQP solve time: {np.mean(solve_times):.4f}s")
    print(f"Final state: {states[-1]}")


if __name__ == "__main__":
    main()
