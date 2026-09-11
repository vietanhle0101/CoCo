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
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parents[1]
THETA_LIMIT = np.pi / 8
WALL_DISTANCE = 0.5
# The saved dataset config's +/-2N limit leaves almost no actuator margin for
# a 1kg cart + 1kg pole against gravity -- empirically, recovery from even a
# ~0.15 rad disturbance was already on the edge of infeasible. 10N (about
# half the combined weight force, mg~19.6N) gives real recovery headroom, more
# in line with what a small cart-pole rig's motor would actually provide.
CART_FORCE_LIMIT = 10.0
# The default Matplotlib config directory may be read-only in a virtualenv or
# remote workspace.  Configure it before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt

from cartpole import Cartpole
try:
    from .mujoco_soft_walls import SoftWallCartpole
except ImportError:  # Supports direct execution: python cartpole/closed_loop_mpc.py
    from mujoco_soft_walls import SoftWallCartpole


def terminal_lqr_cost(a, cart_input, q, cart_input_cost):
    """Return the stabilizing discrete-time LQR terminal-cost matrix.

    The terminal region is the upright, no-contact region.  Consequently the
    feedback law uses only the cart-force column of the input matrix: the
    wall reactions are identically zero there and are not free actuators.
    """
    b = np.asarray(cart_input, dtype=float).reshape(-1, 1)
    r = np.array([[float(cart_input_cost)]])
    return solve_discrete_are(a, b, q, r)


def make_mpc_problem(horizon, timestep, theta_limit, state_slack_weight=0.0,
                      wall_force_weight=5.0):
    """Create an MPC-only model override without changing saved datasets."""
    config_path = Path(__file__).with_name("config") / "default.p"
    with config_path.open("rb") as config_file:
        _, parameters, sampled_params = pickle.load(config_file)
    parameters = list(parameters)
    # The saved config gives s_L/s_R zero cost (R = diag([1, 0, 0])), so wall
    # force is "free" to the optimizer -- it will happily lean on the wall to
    # catch a falling pole even when that burns most of the track, then have
    # no incentive to spend real (R-priced) cart force clawing the cart back
    # afterward. Pricing wall force discourages using it more than needed.
    r = np.array(parameters[4], dtype=float).copy()
    r[1, 1] = wall_force_weight
    r[2, 2] = wall_force_weight
    parameters[4] = r
    x_min, x_max = parameters[5], parameters[6]
    ddelta_min, ddelta_max = parameters[13], parameters[14]
    gravity, length = parameters[16], parameters[17]
    cart_mass, pole_mass = parameters[18], parameters[19]
    kappa, nu = parameters[20], parameters[21]
    # Keep the MPC contact surfaces aligned with the MuJoCo wall locations.
    # The cart's own travel limits remain those stored in x_min/x_max.
    distance = WALL_DISTANCE
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
    parameters[7], parameters[8] = -CART_FORCE_LIMIT, CART_FORCE_LIMIT
    parameters[15] = timestep
    x_max = np.asarray(x_max, dtype=float).copy()
    x_max[1] = theta_limit
    x_min = -x_max

    # The wall-contact big-M constants below must stay valid over whatever
    # range the (nonnegative, capped) state slack can reach -- otherwise the
    # state box becomes soft while the wall logic built on top of it stays
    # hard, and a large slack excursion makes the wall constraints
    # themselves infeasible. Size the slack cap and the big-M's off the same
    # margin so that can't happen.
    state_slack_max = None
    bigm_min, bigm_max = x_min, x_max
    if state_slack_weight > 0:
        state_slack_max = x_max - x_min
        bigm_min = x_min - state_slack_max
        bigm_max = x_max + state_slack_max

    delta_min = np.array([
        -bigm_max[0] + length * bigm_min[1] - distance,
        bigm_min[0] - length * bigm_max[1] - distance,
    ])
    delta_max = np.array([
        -bigm_min[0] + length * bigm_max[1] - distance,
        bigm_max[0] - length * bigm_min[1] - distance,
    ])
    if state_slack_weight > 0:
        ddelta_min = np.array([
            -bigm_max[2] + length * bigm_min[3],
            bigm_min[2] - length * bigm_max[3],
        ])
        ddelta_max = np.array([
            -bigm_min[2] + length * bigm_max[3],
            bigm_max[2] - length * bigm_min[3],
        ])
    parameters[5], parameters[6] = x_min, x_max
    parameters[9] = kappa * delta_min + nu * ddelta_min
    parameters[10] = kappa * delta_max + nu * ddelta_max
    parameters[11], parameters[12] = delta_min, delta_max
    parameters[13], parameters[14] = ddelta_min, ddelta_max
    parameters[22] = distance
    problem = Cartpole(
        prob_params=parameters, sampled_params=sampled_params,
        state_slack_weight=state_slack_weight, state_slack_max=state_slack_max,
    )
    # Use the discrete LQR value function around the upright/no-contact
    # equilibrium as the terminal penalty.  This replaces the old Q-only
    # terminal stage cost and makes the finite-horizon controller much less
    # myopic without pretending that wall forces are terminal actuators.
    terminal_p = terminal_lqr_cost(
        parameters[1], parameters[2][:, 0], parameters[3], parameters[4][0, 0]
    )
    problem.set_terminal_cost(terminal_p)
    return problem


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


def make_mujoco_plant(timestep):
    """Create the physical MuJoCo plant and the MPC-dt substep count.

    ``contact_mode="explicit"`` disables physical wall collision and accepts
    the MIQP's explicit [s_L, s_R] inputs as pole-hinge torque. The MPC
    control period must land on a MuJoCo step boundary.
    """
    simulator = SoftWallCartpole(contact_mode="explicit")
    substeps = timestep / simulator.dt
    if abs(substeps - round(substeps)) > 1e-6:
        raise ValueError(
            f"--dt={timestep} must be a whole multiple of the MuJoCo model "
            f"timestep ({simulator.dt}s) for the --plant mujoco backend."
        )
    return simulator, round(substeps)


def run_mpc(initial_state, goal_state, steps, horizon, timestep, theta_limit,
            plant="toy", state_slack_weight=0.0, wall_force_weight=5.0):
    """Run MPC with one system linearization at each current state.

    Args:
        plant: "toy" rolls out the sine-based ``nonlinear_step`` model used to
            derive the linearization. "mujoco" executes the full first MIQP
            input [u_c, s_L, s_R] on the MuJoCo point-mass pole model. Its
            visual walls have no physical contact in this explicit-force mode.
        state_slack_weight: if positive, state bounds become soft (see
            ``Cartpole``) so a temporarily-unrecoverable state under this
            horizon/actuator limit no longer raises ``RuntimeError`` -- it is
            instead penalized and, if actually exceeded, means the plan
            could not keep the state in bounds even at that penalty.
        wall_force_weight: quadratic cost weight on s_L/s_R (0 in the saved
            dataset config). Without this, wall force is free to the
            optimizer, so it will happily burn the whole track leaning on
            the wall to catch a falling pole and never pays to return.
    """
    if plant not in ("toy", "mujoco"):
        raise ValueError("plant must be 'toy' or 'mujoco'")

    problem = make_mpc_problem(
        horizon, timestep, theta_limit, state_slack_weight, wall_force_weight,
    )
    states = np.empty((steps + 1, problem.n))
    controls = np.empty((steps, problem.m))
    costs = np.empty(steps)
    solve_times = np.empty(steps)
    states[0] = initial_state

    simulator = substeps = None
    if plant == "mujoco":
        simulator, substeps = make_mujoco_plant(timestep)
        simulator.reset(initial_state)

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
        success, cost, solve_time, (_, optimal_inputs, y_star) = problem.solve_micp(
            parameters, solver=cp.GUROBI
        )
        if not success:
            raise RuntimeError(
                f"MIQP status {problem.bin_prob.status!r} at MPC step {k} "
                f"from state {states[k]}."
            )

        # Receding-horizon control: apply the first input only, then replan.
        controls[k] = optimal_inputs[:, 0]
        if plant == "toy":
            states[k + 1] = nonlinear_step(problem, states[k], controls[k])
        else:
            # Apply all three values selected by the MIQP. The explicit-mode
            # simulator maps s_L/s_R to l * (s_R - s_L) hinge torque.
            next_state, _ = simulator.step(
                controls[k, 0], soft_wall_forces=controls[k, 1:], substeps=substeps
            )
            states[k + 1] = next_state
        costs[k] = cost
        solve_times[k] = solve_time
        # y_star[:,0] is this step's binary contact indicators [y_l, y_r] per
        # wall (jj=0 left, jj=1 right); s_L/s_R are only nonzero when both are
        # active, so "contact" here means the wall's y pair is fully engaged.
        left_contact = bool(y_star[0, 0]) and bool(y_star[1, 0])
        right_contact = bool(y_star[2, 0]) and bool(y_star[3, 0])
        print(
            f"step {k + 1:02d}/{steps}: state={states[k]}, cost={cost:.3f}, "
            f"solve={solve_time:.4f}s, u={controls[k]}, "
            f"contact=[left={left_contact}, right={right_contact}]"
        )

    return problem, states, controls, costs, solve_times


def plot_trajectory(problem, states, controls, goal_state, output_path):
    """Plot states, cart/contact inputs, and cart position against its bounds.
    """
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


def save_rollout(output_path, problem, states, controls, costs, solve_times,
                 goal_state, plant):
    """Save a self-describing MPC rollout for later replay or visualization.

    The compressed NumPy archive intentionally stores raw state samples rather
    than simulator-specific objects, so it can be replayed by the MuJoCo
    visualizer regardless of whether this rollout used the toy or MuJoCo
    plant.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time=np.arange(states.shape[0]) * problem.dh,
        input_time=np.arange(controls.shape[0]) * problem.dh,
        states=states,
        controls=controls,
        costs=costs,
        solve_times=solve_times,
        initial_state=states[0],
        goal_state=goal_state,
        timestep=np.array(problem.dh),
        horizon=np.array(problem.N),
        theta_limit=np.array(THETA_LIMIT),
        plant=np.array(plant),
        state_labels=np.array(["p", "theta", "p_dot", "theta_dot"]),
        control_labels=np.array(["cart_force", "left_wall_force", "right_wall_force"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of MPC updates.")
    parser.add_argument("--horizon", type=int, default=20, help="Number of predicted states.")
    parser.add_argument("--dt", type=float, default=0.02, help="MPC and simulation timestep [s].")
    parser.add_argument(
        "--plant", choices=("toy", "mujoco"), default="toy",
        help="Rollout the sine-based toy model (default) or a physically "
             "simulated MuJoCo cart-pole with explicit MIQP soft-wall forces.",
    )
    parser.add_argument(
        "--state-slack-weight", type=float, default=0.0,
        help="If positive, soften all state bounds with a quadratically "
             "penalized slack instead of a hard constraint, so the MIQP "
             "stays feasible when the horizon/actuator limits can't "
             "otherwise keep the plan in bounds (default: 0, hard bounds).",
    )
    parser.add_argument(
        "--wall-force-weight", type=float, default=5.0,
        help="Quadratic cost weight on s_L/s_R (0 in the saved dataset "
             "config, where wall force is a free actuator). Discourages "
             "burning the whole track leaning on the wall to catch the pole "
             "and never paying to return (default: 5.0; use 0 to disable).",
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
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "cartpole" / "data" / "closed_loop_mpc.npz",
        help="Destination compressed rollout archive for later replay.",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.horizon < 2 or args.dt <= 0:
        parser.error("--steps and --dt must be positive; --horizon must be at least 2")

    problem, states, controls, costs, solve_times = run_mpc(
        np.asarray(args.x0), np.asarray(args.goal), args.steps, args.horizon, args.dt,
        THETA_LIMIT, plant=args.plant, state_slack_weight=args.state_slack_weight,
        wall_force_weight=args.wall_force_weight,
    )
    plot_trajectory(problem, states, controls, np.asarray(args.goal), args.plot)
    save_rollout(
        args.data, problem, states, controls, costs, solve_times,
        np.asarray(args.goal), args.plant,
    )
    print(f"Saved plot to {args.plot}")
    print(f"Saved rollout data to {args.data}")
    print(f"Mean MIQP solve time: {np.mean(solve_times):.4f}s")
    print(f"Final state: {states[-1]}")


if __name__ == "__main__":
    main()
