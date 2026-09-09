#!/usr/bin/env python3
"""Replay a saved closed-loop Cartpole trajectory in the MuJoCo viewer.

The viewer is driven by the saved states, rather than re-integrating the
saved controls.  This makes playback faithful to the recorded MPC rollout
whether it was generated using the toy plant or the MuJoCo plant.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    from .mujoco_soft_walls import SoftWallCartpole
except ImportError:  # Supports direct execution from the repository root.
    from mujoco_soft_walls import SoftWallCartpole


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "cartpole" / "data" / "closed_loop_mpc.npz"


def load_rollout(data_path):
    """Load and validate the state trajectory required for MuJoCo playback."""
    with np.load(data_path, allow_pickle=False) as archive:
        required = {"states", "time", "timestep"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{data_path} is missing required fields: {sorted(missing)}")
        states = np.asarray(archive["states"], dtype=float)
        times = np.asarray(archive["time"], dtype=float)
        timestep = float(archive["timestep"])
        plant = str(archive["plant"]) if "plant" in archive.files else "unknown"
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("rollout states must have shape (num_samples, 4)")
    if times.shape != (states.shape[0],):
        raise ValueError("rollout time must have one sample per state")
    if timestep <= 0:
        raise ValueError("rollout timestep must be positive")
    return states, times, timestep, plant


def set_viewer_state(simulator, state):
    """Set MuJoCo's generalized coordinates to one saved Cartpole state."""
    simulator.data.qpos[:2] = state[:2]
    simulator.data.qvel[:2] = state[2:]
    # No integration occurs during playback. mj_forward updates the geometry
    # and camera-visible kinematics for this exact saved state.
    mujoco.mj_forward(simulator.model, simulator.data)


def replay(states, timestep, speed=1.0, loop=True):
    """Display states in real time, scaled by ``speed``."""
    simulator = SoftWallCartpole(contact_mode="explicit")
    simulator.reset(states[0])
    frame_duration = timestep / speed

    with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        frame = 0
        while viewer.is_running():
            frame_start = time.perf_counter()
            set_viewer_state(simulator, states[frame])
            viewer.sync()
            frame += 1
            if frame == len(states):
                if not loop:
                    break
                frame = 0
            time.sleep(max(0.0, frame_duration - (time.perf_counter() - frame_start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Rollout .npz file from closed_loop_mpc.py.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback-rate multiplier (default: real time).")
    parser.add_argument("--once", action="store_true",
                        help="Play once instead of looping until the viewer closes.")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if not args.data.is_file():
        parser.error(f"rollout archive not found: {args.data}")

    states, times, timestep, plant = load_rollout(args.data)
    print(
        f"Replaying {len(states)} states ({times[-1]:.3f}s) from {args.data} "
        f"[recorded plant: {plant}, speed: {args.speed:g}x]"
    )
    replay(states, timestep, speed=args.speed, loop=not args.once)


if __name__ == "__main__":
    main()
