#!/usr/bin/env python3
"""Replay a saved planar-pushing MINLP solution in MuJoCo."""

import argparse
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = Path(__file__).with_name("assets") / "planar_pushing.xml"
DEFAULT_DATA = ROOT / "planar_pushing" / "outputs" / "planar_pushing.npz"


def yaw_quaternion(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def set_frame(model, data, state, pusher, goal):
    """Update the same recorded geometry for windowed and offscreen replay."""
    import mujoco

    block_qpos = model.jnt_qposadr[model.joint('block_free').id]
    data.qpos[block_qpos:block_qpos + 7] = np.hstack(
        (state[:2], 0.15, yaw_quaternion(state[2]))
    )
    goal_qpos = model.jnt_qposadr[model.joint('goal_free').id]
    data.qpos[goal_qpos:goal_qpos + 7] = np.hstack(
        (goal[:2], 0.006, yaw_quaternion(goal[2]))
    )
    # New two-pusher archives store [left,right]; retain one-pusher replay.
    positions = np.asarray(pusher, dtype=float)
    if positions.shape == (2,):
        positions = np.vstack((positions, np.array([0.18, 0.0])))
    for name, position in zip(('left_pusher_free', 'right_pusher_free'), positions):
        qpos = model.jnt_qposadr[model.joint(name).id]
        data.qpos[qpos:qpos + 7] = np.hstack(
            (position, 0.03, np.array([1.0, 0.0, 0.0, 0.0]))
        )
    mujoco.mj_forward(model, data)


def save_gif(states, pushers, goal, timestep, speed, loop, output):
    """Render MuJoCo frames without creating a GLFW/X11 window."""
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for state, pusher in zip(states, pushers):
            set_frame(model, data, state, pusher, goal)
            renderer.update_scene(data, camera='overview')
            frames.append(Image.fromarray(renderer.render().copy()))
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {'loop': 0} if loop else {}
    frames[0].save(
        output, save_all=True, append_images=frames[1:],
        duration=max(10, round(1000 * timestep / speed)), **options,
    )
    print(f'Saved MuJoCo playback to {output}')


def replay(states, pushers, goal, timestep, speed, loop):
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            start = time.perf_counter()
            with viewer.lock():
                set_frame(model, data, states[frame], pushers[frame], goal)
            viewer.sync()
            frame += 1
            if frame == len(states):
                if not loop:
                    break
                frame = 0
            time.sleep(max(0.0, timestep / speed - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument('--software', action='store_true',
                        help='Use Mesa software rendering (Linux graphics-driver fallback).')
    parser.add_argument('--gif', type=Path,
                        help='Export a MuJoCo-rendered GIF through EGL without opening a window.')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive")
    if not args.data.is_file():
        parser.error(f"solution archive not found: {args.data}")

    with np.load(args.data, allow_pickle=False) as solution:
        states = np.asarray(solution["states"], dtype=float)
        pushers = np.asarray(solution["pusher_positions"], dtype=float)
        goal = np.asarray(solution["goal_state"], dtype=float)
        timestep = float(solution["timestep"])
    if (states.ndim != 2 or states.shape[1] != 6 or len(states) == 0
            or pushers.shape not in ((len(states), 2), (len(states), 2, 2))
            or goal.shape != (6,)):
        parser.error("archive has invalid state or pusher-position dimensions")
    if (not np.all(np.isfinite(states)) or not np.all(np.isfinite(pushers))
            or not np.isfinite(timestep) or timestep <= 0):
        parser.error('archive must contain finite samples and a positive timestep')
    if args.gif is not None and args.gif.suffix.lower() != '.gif':
        parser.error('--gif must name a .gif file')

    # These must be set BEFORE importing MuJoCo, GLFW, or PyOpenGL.
    if args.software:
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'mesa'
    if args.gif is not None:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        save_gif(states, pushers, goal, timestep, args.speed, not args.once, args.gif)
        return
    replay(states, pushers, goal, timestep, args.speed, loop=not args.once)


if __name__ == "__main__":
    main()
