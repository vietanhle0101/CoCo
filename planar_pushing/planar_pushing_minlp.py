#!/usr/bin/env python3
"""Run receding-horizon control for a small planar box-pushing contact MINLP.

The state is ``[x, y, yaw, vx, vy, yaw_rate]``. A circular pusher can be free
or contact either the left or right face of the rotating block. The integer
contact-mode sequence is enumerated;
each fixed sequence is solved as a nonlinear program with SciPy SLSQP.

At every MPC step, the binary contact schedule is enumerated and the best
fixed-schedule nonlinear program is solved with SciPy SLSQP.  Only the first
force and pusher motion are applied to the nonlinear rigid-body model, then
the MINLP is solved again from that new state.  This is intentionally small
(the default has 27 schedules per MPC step), not a globally certified
large-scale solver.
"""

import argparse
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))


@dataclass(frozen=True)
class PushingParameters:
    # Three stages are enough to demonstrate left -> free -> right switching.
    # Enumerating all modes is exponential (3**horizon), so retain a short
    # horizon for this educational implementation.
    horizon: int = 3
    dt: float = 0.10
    mass: float = 1.0
    inertia: float = 0.05
    half_width: float = 0.15
    pusher_radius: float = 0.03
    friction: float = 0.5
    force_max: float = 8.0
    pusher_speed_max: float = 3.0


FREE, LEFT, RIGHT = 0, 1, 2
MODE_NAMES = {FREE: "free", LEFT: "left", RIGHT: "right"}


def rotation(yaw):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]])


def contact_geometry(block_state, pusher_position, params, side=LEFT):
    """Return signed gap, contact arm, outward normal, and tangent.

    ``side`` is ``LEFT`` or ``RIGHT``. Positive gap means separation; zero
    gap means the pusher touches the selected face.
    """
    position, yaw = block_state[:2], block_state[2]
    rot = rotation(yaw)
    if side not in (LEFT, RIGHT):
        raise ValueError("contact geometry requires LEFT or RIGHT side")
    sign = -1.0 if side == LEFT else 1.0
    arm = rot @ np.array([sign * params.half_width, 0.0])
    normal = rot @ np.array([sign, 0.0])  # outward face normal
    tangent = rot @ np.array([0.0, 1.0])
    gap = normal @ (pusher_position - (position + arm)) - params.pusher_radius
    return gap, arm, normal, tangent


def box_clearance(block_state, pusher_position, params):
    """Signed circular-pusher clearance from the oriented box boundary."""
    position, yaw = block_state[:2], block_state[2]
    local = rotation(yaw).T @ (pusher_position - position)
    distance_from_box = np.abs(local) - params.half_width
    outside_distance = np.linalg.norm(np.maximum(distance_from_box, 0.0))
    inside_distance = min(max(distance_from_box[0], distance_from_box[1]), 0.0)
    return outside_distance + inside_distance - params.pusher_radius


def dynamics_step(state, force, mode, params):
    """One semi-implicit nonlinear rigid-body step under a contact force."""
    if mode == FREE:
        world_force, torque = np.zeros(2), 0.0
    else:
        _, arm, normal, tangent = contact_geometry(state, np.zeros(2), params, mode)
        normal_force, tangential_force = force
        # A positive normal force pushes inward, from the selected outside
        # face toward the block center.
        world_force = -normal_force * normal + tangential_force * tangent
        torque = arm[0] * world_force[1] - arm[1] * world_force[0]
    velocity = state[3:5] + params.dt * world_force / params.mass
    yaw_rate = state[5] + params.dt * torque / params.inertia
    return np.hstack((
        state[:2] + params.dt * velocity,
        state[2] + params.dt * yaw_rate,
        velocity,
        yaw_rate,
    ))


def rollout(initial_state, forces, modes, params):
    """Roll out any-length force sequence with ``dynamics_step``."""
    states = np.empty((len(forces) + 1, 6))
    states[0] = initial_state
    for step, force in enumerate(forces):
        states[step + 1] = dynamics_step(states[step], force, modes[step], params)
    return states


def unpack(decision, params):
    """Decode pusher positions q[1:N] and contact forces f[0:N-1]."""
    n = params.horizon
    pusher_tail = decision[: 2 * n].reshape(n, 2)
    forces = decision[2 * n :].reshape(n, 2)
    return pusher_tail, forces


def solve_fixed_mode(initial_state, goal_state, initial_pusher, mode, params):
    """Solve the nonlinear program associated with one integer mode vector."""
    mode = np.asarray(mode, dtype=int)
    n = params.horizon

    # ``pusher_tail[k]`` is the position reached over control interval k.
    # Start each contact interval on the initial left-face trajectory;
    # separation intervals begin just outside it. This is a useful SLSQP
    # initialization.
    pusher_guess = np.empty((n, 2))
    state_guess = np.asarray(initial_state, dtype=float)
    for step in range(n):
        if mode[step] == FREE:
            # Reposition above the block when not in contact. This supplies a
            # collision-free warm start for moving between its two faces.
            rot = rotation(state_guess[2])
            pusher_guess[step] = state_guess[:2] + rot @ np.array(
                [0.0, params.half_width + params.pusher_radius + 0.04]
            )
        else:
            _, arm, normal, _ = contact_geometry(
                state_guess, np.zeros(2), params, mode[step]
            )
            pusher_guess[step] = state_guess[:2] + arm + params.pusher_radius * normal
    force_guess = np.zeros((n, 2))
    force_guess[mode != FREE, 0] = 2.0
    initial_guess = np.hstack((pusher_guess.ravel(), force_guess.ravel()))

    bounds = []
    bounds.extend([(-1.0, 1.0)] * (2 * n))
    for contact_mode in mode:
        active = contact_mode != FREE
        bounds.append((0.0, params.force_max) if active else (0.0, 0.0))
        bounds.append((-params.force_max, params.force_max) if active else (0.0, 0.0))

    def evaluate(decision):
        pusher_tail, forces = unpack(decision, params)
        pushers = np.vstack((initial_pusher, pusher_tail))
        states = rollout(initial_state, forces, mode, params)
        return pushers, forces, states

    def objective(decision):
        pushers, forces, states = evaluate(decision)
        terminal_error = states[-1, :3] - goal_state[:3]
        velocity_error = states[-1, 3:] - goal_state[3:]
        pusher_velocity = np.diff(pushers, axis=0) / params.dt
        return (
            terminal_error @ np.diag([250.0, 250.0, 40.0]) @ terminal_error
            + velocity_error @ np.diag([8.0, 8.0, 2.0]) @ velocity_error
            + 0.02 * np.sum(forces**2)
            + 0.002 * np.sum(pusher_velocity**2)
        )

    constraints = []
    for step, contact_mode in enumerate(mode):
        def gap(decision, step=step):
            pushers, _, states = evaluate(decision)
            # The kinematic pusher moves during interval k.  Its end position
            # q[k+1] is therefore the position paired with f[k].  Using q[k]
            # here creates a one-step contact delay and makes an MPC controller
            # repeatedly choose an unforced "move into contact" first step.
            return contact_geometry(
                states[step], pushers[step + 1], params, mode[step]
            )[0]

        if contact_mode != FREE:
            constraints.append({"type": "eq", "fun": gap})
        else:
            def free_clearance(decision, step=step):
                pushers, _, states = evaluate(decision)
                return box_clearance(states[step], pushers[step + 1], params)

            constraints.append({"type": "ineq", "fun": free_clearance})

        def friction(decision, step=step):
            _, forces, _ = evaluate(decision)
            return params.friction * forces[step, 0] - abs(forces[step, 1])

        if contact_mode != FREE:
            constraints.append({"type": "ineq", "fun": friction})

    def pusher_speed(decision):
        pushers, _, _ = evaluate(decision)
        return params.pusher_speed_max - np.linalg.norm(
            np.diff(pushers, axis=0) / params.dt, axis=1
        )

    constraints.append({"type": "ineq", "fun": pusher_speed})
    result = minimize(
        objective, initial_guess, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": 600, "ftol": 1e-8, "disp": False},
    )
    if not result.success:
        return None
    pushers, forces, states = evaluate(result.x)
    # Reject solutions that only satisfy constraints through a loose SLSQP
    # tolerance, especially important for contact equalities.
    gaps = np.full(n, np.nan)
    for k, contact_mode in enumerate(mode):
        if contact_mode == FREE:
            gaps[k] = box_clearance(states[k], pushers[k + 1], params)
        else:
            gaps[k] = contact_geometry(
                states[k], pushers[k + 1], params, contact_mode
            )[0]
    if (np.any(gaps[mode == FREE] < -1e-5)
            or np.any(np.abs(gaps[mode != FREE]) > 1e-4)):
        return None
    return {
        "mode": mode,
        "cost": float(result.fun),
        "states": states,
        "pushers": pushers,
        "forces": forces,
        "gaps": gaps,
    }


def solve_minlp(initial_state, goal_state, initial_pusher, params):
    """Enumerate free/left/right contact modes and retain the best NLP."""
    best = None
    for mode in product((FREE, LEFT, RIGHT), repeat=params.horizon):
        candidate = solve_fixed_mode(
            initial_state, goal_state, initial_pusher, mode, params
        )
        if candidate is not None and (best is None or candidate["cost"] < best["cost"]):
            best = candidate
    if best is None:
        raise RuntimeError("no feasible contact mode found")
    return best


def run_receding_horizon(initial_state, goal_state, initial_pusher, params, steps):
    """Apply a contact-implicit MINLP in receding-horizon fashion.

    The optimizer predicts ``params.horizon`` steps each time.  The nonlinear
    plant receives only stage zero of that solution; therefore all saved
    forces, modes, and gaps describe the *applied* closed-loop trajectory.
    """
    states = np.empty((steps + 1, 6))
    pushers = np.empty((steps + 1, 2))
    forces = np.empty((steps, 2))
    modes = np.empty(steps, dtype=int)
    gaps = np.empty(steps)
    solve_costs = np.empty(steps)
    states[0] = initial_state
    pushers[0] = initial_pusher

    for step in range(steps):
        plan = solve_minlp(states[step], goal_state, pushers[step], params)
        forces[step] = plan["forces"][0]
        modes[step] = plan["mode"][0]
        gaps[step] = plan["gaps"][0]
        solve_costs[step] = plan["cost"]

        # The pusher is kinematic: q[1] is the position reached after the
        # first planned interval and is paired with the applied contact force.
        pushers[step + 1] = plan["pushers"][1]
        states[step + 1] = dynamics_step(
            states[step], forces[step], modes[step], params
        )
        print(
            f"MPC step {step + 1:02d}/{steps}: mode={modes[step]}, "
            f"force={forces[step]}, predicted cost={solve_costs[step]:.4f}"
        )

    return {
        "states": states,
        "pushers": pushers,
        "forces": forces,
        "mode": modes,
        "gaps": gaps,
        "solve_costs": solve_costs,
    }


def save_solution(path, solution, initial_state, goal_state, initial_pusher, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=np.arange(len(solution["states"])) * params.dt,
        states=solution["states"],
        pusher_positions=solution["pushers"],
        contact_forces=solution["forces"],
        contact_mode=solution["mode"],
        gaps=solution["gaps"],
        initial_state=initial_state,
        goal_state=goal_state,
        initial_pusher=initial_pusher,
        timestep=np.array(params.dt),
        block_half_width=np.array(params.half_width),
        pusher_radius=np.array(params.pusher_radius),
        mpc_horizon=np.array(params.horizon),
        solve_costs=solution.get("solve_costs", np.array([solution.get("cost", np.nan)])),
    )


def plot_solution(path, solution, goal_state, params):
    import matplotlib.pyplot as plt

    states, forces = solution["states"], solution["forces"]
    time = np.arange(len(states)) * params.dt
    input_time = time[:-1]
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(states[:, 0], states[:, 1], marker="o", label="block center")
    axes[0].scatter(goal_state[0], goal_state[1], marker="*", s=140, label="goal")
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Planar pushing path", aspect="equal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].step(input_time, forces[:, 0], where="post", label="normal force")
    axes[1].step(input_time, forces[:, 1], where="post", label="tangential force")
    axes[1].set(xlabel="time [s]", ylabel="force [N]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].step(input_time, solution["mode"], where="post", label="mode (0 free, 1 left, 2 right)")
    axes[2].plot(input_time, solution["gaps"], marker="o", label="gap")
    axes[2].set(xlabel="time [s]", title="Contact schedule and gap")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "planar_pushing" / "outputs" / "planar_pushing.npz")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "planar_pushing" / "outputs" / "planar_pushing.png")
    parser.add_argument("--steps", type=int, default=3,
                        help="number of closed-loop control steps (default: 3)")
    parser.add_argument("--open-loop", action="store_true",
                        help="solve once for comparison instead of running MPC")
    args = parser.parse_args()

    params = PushingParameters()
    initial_state = np.zeros(6)
    goal_state = np.array([0.18, 0.02, 0.12, 0.0, 0.0, 0.0])
    initial_pusher = np.array([-params.half_width - params.pusher_radius, 0.0])
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.open_loop:
        solution = solve_minlp(initial_state, goal_state, initial_pusher, params)
        print(f"Best open-loop contact schedule: {solution['mode'].tolist()}")
        print(f"Objective: {solution['cost']:.5f}")
    else:
        solution = run_receding_horizon(
            initial_state, goal_state, initial_pusher, params, args.steps
        )
    save_solution(args.output, solution, initial_state, goal_state, initial_pusher, params)
    plot_solution(args.plot, solution, goal_state, params)
    print(f"Final state: {solution['states'][-1]}")
    print(f"Saved solution to {args.output}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
