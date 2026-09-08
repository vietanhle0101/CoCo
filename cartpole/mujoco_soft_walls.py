#!/usr/bin/env python3
"""MuJoCo cart-pole with either physical or MIQP-supplied wall forces.

This is a new physics simulator, not the linear plant used by the repository's
MIQP.  It has a 1 kg cart, 1 kg / 1 m pole, horizontal cart force limited to
[-2, 2] and track limits at +/-0.5 m. ``explicit`` mode applies MIQP force
inputs s_L,s_R as hinge torque l*(s_R-s_L). ``mujoco`` mode enables compliant
physical pole-wall contact and rejects explicit s_L,s_R inputs to avoid double
counting contact forces.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import mujoco


XML_PATH = Path(__file__).with_name("assets") / "soft_wall_cartpole.xml"


class SoftWallCartpole:
    """Small Python interface around the MuJoCo cart-pole model."""

    def __init__(self, xml_path=XML_PATH, contact_mode="explicit"):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        hinge_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole_hinge"
        )
        self._hinge_dof = self.model.jnt_dofadr[hinge_joint]
        self.pole_length = 1.0
        self._applied_soft_wall_forces = np.zeros(2)
        self._wall_ids = {
            "left": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_wall"),
            "right": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_wall"),
        }
        self._contact_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("pole_geom", "tip_mass", "left_wall", "right_wall")
        ]
        self.set_contact_mode(contact_mode)

    @property
    def dt(self):
        return self.model.opt.timestep

    @property
    def time(self):
        return self.data.time

    def reset(self, state=(0.0, 0.15, 0.0, 0.0)):
        """Set [cart position, pole angle, cart velocity, pole angular rate]."""
        state = np.asarray(state, dtype=float)
        if state.shape != (4,):
            raise ValueError("state must have four entries: p, theta, p_dot, theta_dot")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state[:2]
        self.data.qvel[:] = state[2:]
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_contact_mode(self, contact_mode):
        """Select explicit MIQP forces or MuJoCo's physical wall contacts."""
        if contact_mode not in {"explicit", "mujoco"}:
            raise ValueError("contact_mode must be 'explicit' or 'mujoco'")
        enabled = int(contact_mode == "mujoco")
        for geom_id in self._contact_geom_ids:
            self.model.geom_contype[geom_id] = enabled
            self.model.geom_conaffinity[geom_id] = enabled
        self.contact_mode = contact_mode

    def state(self):
        return np.hstack((self.data.qpos[:2], self.data.qvel[:2])).copy()

    def soft_wall_forces(self):
        """Return the most recently applied [s_L, s_R] MIQP inputs."""
        return self._applied_soft_wall_forces.copy()

    def physical_wall_forces(self):
        """Return summed [left, right] MuJoCo normal contact-force magnitudes."""
        forces = {"left": 0.0, "right": 0.0}
        contact_force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for wall, wall_id in self._wall_ids.items():
                if contact.geom1 == wall_id or contact.geom2 == wall_id:
                    mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                    forces[wall] += contact_force[0]
        return np.array([forces["left"], forces["right"]])

    def wall_forces(self):
        """Return forces supplied by the selected contact mode."""
        if self.contact_mode == "explicit":
            return self.soft_wall_forces()
        return self.physical_wall_forces()

    def step(self, cart_force, soft_wall_forces=(0.0, 0.0), substeps=1):
        """Advance with cart force and explicit nonnegative [s_L, s_R]."""
        soft_wall_forces = np.asarray(soft_wall_forces, dtype=float)
        if soft_wall_forces.shape != (2,):
            raise ValueError("soft_wall_forces must contain [s_L, s_R]")
        if self.contact_mode == "mujoco" and not np.allclose(soft_wall_forces, 0.0):
            raise ValueError("explicit soft-wall forces are invalid in 'mujoco' contact mode")
        self._applied_soft_wall_forces = np.maximum(soft_wall_forces, 0.0)
        self.data.ctrl[0] = np.clip(cart_force, -2.0, 2.0)
        for _ in range(substeps):
            self.data.qfrc_applied[:] = 0.0
            if self.contact_mode == "explicit":
                self.data.qfrc_applied[self._hinge_dof] = self.pole_length * (
                    self._applied_soft_wall_forces[1] - self._applied_soft_wall_forces[0]
                )
            mujoco.mj_step(self.model, self.data)
        return self.state(), self.wall_forces()


def simulate(simulator, duration, initial_state, cart_force, soft_wall_forces):
    simulator.reset(initial_state)
    steps = int(np.ceil(duration / simulator.dt))
    states = np.empty((steps + 1, 4))
    forces = np.empty((steps + 1, 2))
    times = np.empty(steps + 1)
    states[0], forces[0], times[0] = simulator.state(), simulator.wall_forces(), simulator.time
    for step in range(steps):
        states[step + 1], forces[step + 1] = simulator.step(cart_force, soft_wall_forces)
        times[step + 1] = simulator.time
    return times, states, forces


def plot(times, states, wall_forces, output_path):
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(times, states[:, 0], label="cart position")
    axes[0].axhline(-0.5, color="black", linestyle="--", label="track limits")
    axes[0].axhline(0.5, color="black", linestyle="--")
    axes[0].set(xlabel="time [s]", ylabel="position [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(times, states[:, 1], label="pole angle")
    axes[1].set(xlabel="time [s]", ylabel="angle [rad]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(times, wall_forces[:, 0], label="left wall")
    axes[2].plot(times, wall_forces[:, 1], label="right wall")
    axes[2].set(xlabel="time [s]", ylabel="wall force [N]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def launch_viewer(simulator, duration, initial_state, cart_force, soft_wall_forces):
    """Run the same simulation in MuJoCo's interactive viewer."""
    import mujoco.viewer

    simulator.reset(initial_state)
    with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        stop_time = simulator.time + duration
        while viewer.is_running() and simulator.time < stop_time:
            start = time.time()
            simulator.step(cart_force, soft_wall_forces)
            viewer.sync()
            time.sleep(max(0.0, simulator.dt - (time.time() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--force", type=float, default=0.0, help="Constant cart force in N.")
    parser.add_argument("--left-force", type=float, default=0.0, help="Constant MIQP soft-wall force s_L in N.")
    parser.add_argument("--right-force", type=float, default=0.0, help="Constant MIQP soft-wall force s_R in N.")
    parser.add_argument(
        "--contact-mode", choices=("explicit", "mujoco"), default="explicit",
        help="Use MIQP-supplied forces or MuJoCo-computed physical contacts.",
    )
    parser.add_argument("--x0", type=float, nargs=4, default=[0.0, 0.35, 0.0, 0.0])
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    parser.add_argument(
        "--plot", type=Path,
        default=ROOT / "cartpole" / "outputs" / "mujoco_soft_walls.png",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    simulator = SoftWallCartpole(contact_mode=args.contact_mode)
    soft_wall_forces = np.array([args.left_force, args.right_force])
    if args.viewer:
        launch_viewer(simulator, args.duration, args.x0, args.force, soft_wall_forces)
    else:
        times, states, forces = simulate(
            simulator, args.duration, args.x0, args.force, soft_wall_forces
        )
        plot(times, states, forces, args.plot)
        print(f"Saved plot to {args.plot}")
        print(f"Final state: {states[-1]}")
        if args.contact_mode == "explicit":
            print(f"Applied wall forces [left, right]: {soft_wall_forces}")
        else:
            print(f"Peak physical wall forces [left, right]: {np.max(forces, axis=0)}")


if __name__ == "__main__":
    main()
