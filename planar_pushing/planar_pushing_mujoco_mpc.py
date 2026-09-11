#!/usr/bin/env python3
"""Run the contact-MINLP MPC against a physical MuJoCo planar-pushing plant.

The MINLP remains a deliberately compact prediction model.  MuJoCo is the
plant: the orange pusher is driven by force-limited position actuators and
contact forces are created by MuJoCo's collision solver, not injected into the
box state.  Consequently this script is also useful for observing model
mismatch and tuning the controller.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planar_pushing.planar_pushing_minlp import (
    FREE,
    LEFT,
    MODE_NAMES,
    PushingParameters,
    contact_geometry,
    plot_solution,
    solve_minlp,
)


XML_PATH = Path(__file__).with_name("assets") / "planar_pushing_mpc.xml"


def joint_value(model, data, name, velocity=False):
    """Read the scalar qpos or qvel associated with a named MuJoCo joint."""
    joint = model.joint(name)
    address = joint.dofadr[0] if velocity else joint.qposadr[0]
    return float(data.qvel[address] if velocity else data.qpos[address])


def measured_state(model, data):
    """Return the controller state [x, y, yaw, vx, vy, yaw_rate]."""
    names = ("block_x", "block_y", "block_yaw")
    return np.array(
        [*(joint_value(model, data, name) for name in names),
         *(joint_value(model, data, name, velocity=True) for name in names)]
    )


def pusher_position(model, data):
    return np.array([
        joint_value(model, data, "pusher_x"),
        joint_value(model, data, "pusher_y"),
    ])


def set_pusher_state(model, data, position):
    for name, value in zip(("pusher_x", "pusher_y"), position):
        data.qpos[model.joint(name).qposadr[0]] = value
    data.ctrl[:] = position


def normal_contact_force(model, data):
    """Sum normal forces over contacts involving the pusher and block."""
    import mujoco

    pusher_id = model.geom("pusher_geom").id
    block_id = model.geom("block_geom").id
    total = 0.0
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if {contact.geom1, contact.geom2} == {pusher_id, block_id}:
            mujoco.mj_contactForce(model, data, index, wrench)
            total += abs(wrench[0])
    return total


def run_mujoco_mpc(params, steps, penetration):
    """Solve MINLP MPC, execute each action on MuJoCo, and log measurements."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    initial_state = np.zeros(6)
    goal_state = np.array([0.18, 0.02, 0.12, 0.0, 0.0, 0.0])
    initial_pusher = np.array([-params.half_width - params.pusher_radius, 0.0])
    set_pusher_state(model, data, initial_pusher)
    mujoco.mj_forward(model, data)

    states = np.empty((steps + 1, 6))
    pushers = np.empty((steps + 1, 2))
    forces = np.empty((steps, 2))
    modes = np.empty(steps, dtype=int)
    gaps = np.empty(steps)
    solve_costs = np.empty(steps)
    measured_normal_forces = np.empty(steps)
    states[0], pushers[0] = measured_state(model, data), pusher_position(model, data)
    substeps = round(params.dt / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, params.dt):
        raise ValueError("MPC timestep must be an integer number of MuJoCo timesteps")

    for step in range(steps):
        state = measured_state(model, data)
        pusher = pusher_position(model, data)
        plan = solve_minlp(state, goal_state, pusher, params)
        mode, force = int(plan["mode"][0]), plan["forces"][0]
        target = plan["pushers"][1].copy()

        # The optimizer uses an ideal force-contact model.  On an active
        # contact, command a small inward displacement to let the physical
        # MuJoCo contact generate the corresponding reaction force.  When
        # inactive, retract slightly to avoid an unintended lingering force.
        contact_side = mode if mode != FREE else LEFT
        _, _, normal, _ = contact_geometry(state, target, params, contact_side)
        target += (-penetration if mode else penetration) * normal
        data.ctrl[:] = target
        contact_force_sum = 0.0
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            contact_force_sum += normal_contact_force(model, data)

        states[step + 1] = measured_state(model, data)
        pushers[step + 1] = pusher_position(model, data)
        forces[step], modes[step] = force, mode
        gaps[step] = contact_geometry(state, target, params)[0]
        solve_costs[step] = plan["cost"]
        # A contact force can be impulsive while the position servo catches
        # the block.  The interval mean is more informative than only reading
        # the final MuJoCo substep, which may already be at zero separation.
        measured_normal_forces[step] = contact_force_sum / substeps
        print(
            f"MPC step {step + 1:02d}/{steps}: mode={MODE_NAMES[mode]}, "
            f"MINLP force={force}, MuJoCo normal force={measured_normal_forces[step]:.3f}, "
            f"state={states[step + 1, :3]}"
        )

    return {
        "states": states,
        "pushers": pushers,
        "forces": forces,
        "mode": modes,
        "gaps": gaps,
        "solve_costs": solve_costs,
        "measured_normal_forces": measured_normal_forces,
        "initial_state": initial_state,
        "goal_state": goal_state,
        "initial_pusher": initial_pusher,
    }


def save_rollout(path, rollout, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=np.arange(len(rollout["states"])) * params.dt,
        states=rollout["states"],
        pusher_positions=rollout["pushers"],
        contact_forces=rollout["forces"],
        contact_mode=rollout["mode"],
        gaps=rollout["gaps"],
        minlp_solve_costs=rollout["solve_costs"],
        mujoco_normal_contact_forces=rollout["measured_normal_forces"],
        initial_state=rollout["initial_state"],
        goal_state=rollout["goal_state"],
        initial_pusher=rollout["initial_pusher"],
        timestep=np.array(params.dt),
        block_half_width=np.array(params.half_width),
        pusher_radius=np.array(params.pusher_radius),
        mpc_horizon=np.array(params.horizon),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--penetration", type=float, default=0.004,
                        help="inward pusher target offset during an active contact [m]")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "planar_pushing" / "outputs" / "planar_pushing_mujoco_mpc.npz")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "planar_pushing" / "outputs" / "planar_pushing_mujoco_mpc.png")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 < args.penetration <= 0.02:
        parser.error("--penetration must be in (0, 0.02]")

    params = PushingParameters()
    rollout = run_mujoco_mpc(params, args.steps, args.penetration)
    save_rollout(args.output, rollout, params)
    plot_solution(args.plot, rollout, rollout["goal_state"], params)
    print(f"Final MuJoCo state: {rollout['states'][-1]}")
    print(f"Saved rollout to {args.output}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
