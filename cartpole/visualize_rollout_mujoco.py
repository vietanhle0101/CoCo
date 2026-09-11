#!/usr/bin/env python3
"""Replay a saved closed-loop Cartpole trajectory in MuJoCo.

The viewer is driven by the saved states, rather than re-integrating the
saved controls.  This makes playback faithful to the recorded MPC rollout
whether it was generated using the toy plant or the MuJoCo plant.

Default output is a headless-rendered GIF (works with no display / no GPU,
via MuJoCo's EGL backend). Pass --viewer for the old interactive GLFW window,
which needs a real GLX-capable display and will fail in most remote/sandboxed
environments (this is why the interactive path used to be the only option).
"""

import argparse
import os
import pickle
import time
from pathlib import Path

# Must be set before mujoco is imported. Defaults to the headless EGL backend
# (works with no display/GPU passthrough); export MUJOCO_GL=glfw yourself
# before running if you want --viewer on a machine with a real display.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

try:
    from .mujoco_soft_walls import SoftWallCartpole
except ImportError:  # Supports direct execution from the repository root.
    from mujoco_soft_walls import SoftWallCartpole


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "cartpole" / "data" / "closed_loop_mpc.npz"

with (Path(__file__).with_name("config") / "default.p").open("rb") as _f:
    _, _CONFIG_PARAMS, _ = pickle.load(_f)
WALL_KAPPA = _CONFIG_PARAMS[20]  # soft-wall spring stiffness; see cartpole.py

# How far the wall visually compresses per Newton of recorded contact force
# (displacement = force / kappa, from the same spring law the MIQP uses),
# capped so a large force doesn't retreat the wall an implausible distance.
MAX_WALL_COMPRESSION = 0.05
# Force [N] at which the wall's contact-highlight color reaches full intensity.
COLOR_SATURATION_FORCE = 20.0
WALL_BASE_COLOR = np.array([0.7, 0.7, 0.7])
WALL_CONTACT_COLOR = np.array([0.95, 0.35, 0.1])
WALL_ALPHA = 0.55  # semi-transparent so the pole is never fully hidden

# Number of stacked height segments the wall panels are visually split into
# (see the XML's left/right_wall_segN geoms) so a contact event bulges the
# panel locally around the pole tip's height, rather than translating the
# whole rigid slab uniformly.
WALL_HEIGHT_SEGMENTS = 7
# Gaussian falloff (in meters) of the bulge away from the tip's height.
WALL_BULGE_FALLOFF = 0.06


def load_rollout(data_path):
    """Load and validate the state/control trajectory required for playback."""
    with np.load(data_path, allow_pickle=False) as archive:
        required = {"states", "controls", "time", "timestep"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{data_path} is missing required fields: {sorted(missing)}")
        states = np.asarray(archive["states"], dtype=float)
        controls = np.asarray(archive["controls"], dtype=float)
        times = np.asarray(archive["time"], dtype=float)
        timestep = float(archive["timestep"])
        plant = str(archive["plant"]) if "plant" in archive.files else "unknown"
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("rollout states must have shape (num_samples, 4)")
    if controls.ndim != 2 or controls.shape[1] != 3:
        raise ValueError("rollout controls must have shape (num_samples - 1, 3)")
    if times.shape != (states.shape[0],):
        raise ValueError("rollout time must have one sample per state")
    if timestep <= 0:
        raise ValueError("rollout timestep must be positive")
    return states, controls, times, timestep, plant


def _wall_reference_geom_ids(model):
    """The original single-slab wall geoms (still used for physics contact
    elsewhere, e.g. SoftWallCartpole's 'mujoco' contact mode) -- hidden
    during replay in favor of the segmented visual panels below."""
    return {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wall"),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wall"),
    }


def _wall_segment_geom_ids(model):
    """Visual-only stacked segments (left/right_wall_seg0..N-1 in the XML),
    ordered bottom-to-top, used to render a localized contact "bulge"."""
    return {
        side: [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_wall_seg{i}")
            for i in range(WALL_HEIGHT_SEGMENTS)
        ]
        for side in ("left", "right")
    }


def _wall_body_ids(model):
    return {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wall_frame"),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wall_frame"),
    }


def _pivot_height(model):
    """World height of the pole's hinge (cart body offset + pole body offset)."""
    cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    pole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
    return model.body_pos[cart_id][2] + model.body_pos[pole_id][2]


def _clamp_theta_to_walls(p, theta, pole_length, inner_face_left, inner_face_right):
    """Clamp theta so the rendered tip never visually crosses a wall face.

    Exact tip kinematics for this model: tip_x = p - pole_length*sin(theta)
    (verified against MuJoCo's own hinge rotation, not a small-angle
    approximation). The MIQP's own soft-contact model tolerates some virtual
    penetration by design (that's what makes it a *soft* wall) -- this clamp
    only affects the rendered angle, never the recorded state/data.
    """
    upper = (p - inner_face_left) / pole_length      # tip_x >= inner_face_left
    lower = (p - inner_face_right) / pole_length      # tip_x <= inner_face_right
    sin_theta = np.clip(np.sin(theta), max(-1.0, lower), min(1.0, upper))
    return float(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))


def set_viewer_state(simulator, state, control, wall_reference_ids, wall_segment_ids,
                      wall_body_ids, base_segment_pos, pivot_height):
    """Set MuJoCo's generalized coordinates to one saved Cartpole state, and
    animate the wall panels (localized bulge + color) from the recorded
    s_L/s_R contact force so soft contact is visible instead of a rigid,
    occluding wall. ``contact_mode="explicit"`` (used for playback) disables
    real collision, since the MIQP already supplies this force -- the wall
    must be animated manually to show it, rather than relying on physics.

    The wall panel is split into WALL_HEIGHT_SEGMENTS stacked geoms (see the
    XML); each segment's inward displacement and color are weighted by a
    Gaussian falloff in height from the pole tip's current position, so
    contact reads as a localized bulge around the contact point rather than
    the whole rigid slab translating uniformly.

    The soft-wall MIQP model also tolerates some virtual tip/wall
    penetration by design (that's the "soft" part), which otherwise renders
    as the pole swinging visibly through the panel. The rendered theta is
    separately clamped so the tip stops at the nearest segment's current
    (bulged) face instead -- display-only; the recorded state is untouched.
    """
    for geom_id in wall_reference_ids.values():
        simulator.model.geom_rgba[geom_id, 3] = 0.0  # hide the reference slab

    p, theta_recorded = state[0], state[1]
    tip_z = pivot_height + simulator.pole_length * np.cos(theta_recorded)

    s_left, s_right = max(0.0, control[1]), max(0.0, control[2])
    inner_face = {}
    for side, force in (("left", s_left), ("right", s_right)):
        sign = +1.0 if side == "left" else -1.0
        peak_compression = min(MAX_WALL_COMPRESSION, force / WALL_KAPPA)
        intensity = min(1.0, force / COLOR_SATURATION_FORCE)
        body_x = simulator.model.body_pos[wall_body_ids[side]][0]

        best_weight, best_face = -1.0, None
        for seg_id, seg_base_pos in zip(wall_segment_ids[side], base_segment_pos[side]):
            weight = np.exp(-0.5 * ((seg_base_pos[2] - tip_z) / WALL_BULGE_FALLOFF) ** 2)
            pos = seg_base_pos.copy()
            pos[0] += sign * peak_compression * weight
            simulator.model.geom_pos[seg_id] = pos

            color = WALL_BASE_COLOR + (weight * intensity) * (WALL_CONTACT_COLOR - WALL_BASE_COLOR)
            simulator.model.geom_rgba[seg_id, :3] = color
            simulator.model.geom_rgba[seg_id, 3] = WALL_ALPHA

            if weight > best_weight:
                half_thickness = simulator.model.geom_size[seg_id][0]
                best_weight = weight
                best_face = body_x + pos[0] + sign * half_thickness
        inner_face[side] = best_face

    theta = _clamp_theta_to_walls(
        p, theta_recorded, simulator.pole_length, inner_face["left"], inner_face["right"]
    )
    simulator.data.qpos[0] = p
    simulator.data.qpos[1] = theta
    simulator.data.qvel[:2] = state[2:]

    # No integration occurs during playback. mj_forward updates the geometry
    # and camera-visible kinematics for this exact saved state.
    mujoco.mj_forward(simulator.model, simulator.data)


def _control_at(controls, frame):
    """states has one more sample than controls (the final propagated state
    has no corresponding control); hold the last control for that frame."""
    return controls[min(frame, len(controls) - 1)]


def _setup_wall_visuals(model):
    """Gather the (mostly static, computed once) geom/body lookups and base
    segment positions that set_viewer_state needs every frame."""
    wall_reference_ids = _wall_reference_geom_ids(model)
    wall_segment_ids = _wall_segment_geom_ids(model)
    wall_body_ids = _wall_body_ids(model)
    base_segment_pos = {
        side: [model.geom_pos[gid].copy() for gid in ids]
        for side, ids in wall_segment_ids.items()
    }
    pivot_height = _pivot_height(model)
    return wall_reference_ids, wall_segment_ids, wall_body_ids, base_segment_pos, pivot_height


def replay_interactive(states, controls, timestep, speed=1.0, loop=True):
    """Display states live in the interactive GLFW viewer, scaled by ``speed``.

    Requires a real GLX-capable display; raises the same GLFWError this
    module's docstring warns about on most remote/sandboxed environments.
    """
    import mujoco.viewer

    simulator = SoftWallCartpole(contact_mode="explicit")
    simulator.reset(states[0])
    wall_visuals = _setup_wall_visuals(simulator.model)
    frame_duration = timestep / speed

    with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        frame = 0
        while viewer.is_running():
            frame_start = time.perf_counter()
            set_viewer_state(simulator, states[frame], _control_at(controls, frame), *wall_visuals)
            viewer.sync()
            frame += 1
            if frame == len(states):
                if not loop:
                    break
                frame = 0
            time.sleep(max(0.0, frame_duration - (time.perf_counter() - frame_start)))


def replay_headless(states, controls, timestep, output_path, speed=1.0, loop=True,
                     camera="overview", width=480, height=360, max_fps=15):
    """Render states offscreen (MuJoCo's EGL backend) and save an animated
    GIF. Works with no display and no GPU passthrough, unlike the interactive
    viewer. Frames are subsampled to ``max_fps`` so long rollouts stay a
    reasonably sized file rather than rendering every physics timestep.
    """
    from PIL import Image

    simulator = SoftWallCartpole(contact_mode="explicit")
    simulator.reset(states[0])
    wall_visuals = _setup_wall_visuals(simulator.model)
    renderer = mujoco.Renderer(simulator.model, height=height, width=width)

    playback_dt = timestep / speed
    stride = max(1, round(1.0 / (max_fps * playback_dt))) if playback_dt > 0 else 1
    frame_duration_ms = max(20, round(playback_dt * stride * 1000))

    frames = []
    for frame in range(0, len(states), stride):
        set_viewer_state(simulator, states[frame], _control_at(controls, frame), *wall_visuals)
        renderer.update_scene(simulator.data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))
    renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0 if loop else 1,
    )
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Rollout .npz file from closed_loop_mpc.py.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback-rate multiplier (default: real time).")
    parser.add_argument("--once", action="store_true",
                        help="Play once instead of looping (viewer: until closed; "
                             "headless: the saved GIF loops when viewed unless set).")
    parser.add_argument("--viewer", action="store_true",
                        help="Use the interactive GLFW window instead of headless "
                             "rendering. Requires a real display; will fail with a "
                             "GLFWError on most remote/sandboxed machines.")
    parser.add_argument("--video", type=Path,
                         default=ROOT / "cartpole" / "outputs" / "rollout.gif",
                         help="Headless mode only: output GIF path.")
    parser.add_argument("--camera", default="overview",
                         help="Headless mode only: camera name from the MJCF (default: overview).")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if not args.data.is_file():
        parser.error(f"rollout archive not found: {args.data}")

    states, controls, times, timestep, plant = load_rollout(args.data)
    print(
        f"Replaying {len(states)} states ({times[-1]:.3f}s) from {args.data} "
        f"[recorded plant: {plant}, speed: {args.speed:g}x]"
    )
    if args.viewer:
        replay_interactive(states, controls, timestep, speed=args.speed, loop=not args.once)
    else:
        num_frames = replay_headless(
            states, controls, timestep, args.video, speed=args.speed, loop=not args.once,
            camera=args.camera,
        )
        print(f"Saved {num_frames}-frame GIF to {args.video}")


if __name__ == "__main__":
    main()
