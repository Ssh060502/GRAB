"""Render a retargeted RB-Y1A+Sharpa ConTrack clip inside PyBullet -- same kinematic replay as
visualize_rby1a_sharpa.py (no physics: no gravity, no stepSimulation, joints/object are just
teleported to the precomputed qpos/pose each frame), but using PyBullet's own shaded renderer
and the real URDF visual meshes instead of a bare line-skeleton plot.

Still NOT a physics validation: "does the finger really touch the object" is not answered any
more authoritatively here than the numeric residual checks in retarget_rby1a_sharpa.py -- for
that, qpos would need to be fed into ConTrack's actual Isaac Sim environment.

Difference from visualize_pybullet.py (the xArm7+XHand version): RB-Y1A+Sharpa is ONE bimanual
URDF (both qpos[:,0] and qpos[:,1] drive the SAME robot body, left and right arm/hand joints,
not two separate robots), and the joint names follow sharpa_geometry.py's ARM_NAMES/HAND_NAMES
convention (right_*, mirrored as left_* for the other side) instead of xArm7/XHand's names.

Runs in the `retarget` env (or any env with pybullet/h5py/trimesh/imageio -- this script itself
doesn't need pinocchio/dex_retargeting, same as retarget_rby1a_sharpa.py).

Example (full clip -> mp4):
    python contrack/visualize_sharpa_pybullet.py \
        --h5 out/grab-s1_teapot_pour_1-rby1a_sharpa.h5 \
        --assets-dir ~/ConTrack/assets --out out/sharpa_teapot.mp4

Example (one frame, zoomed in on the object -> png):
    python contrack/visualize_sharpa_pybullet.py \
        --h5 out/grab-s1_teapot_pour_1-rby1a_sharpa.h5 \
        --assets-dir ~/ConTrack/assets --out out/sharpa_zoom.png --zoom-frame 400
"""

import argparse
import os
import tempfile

import h5py
import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation as R

from sharpa_geometry import ARM_NAMES, HAND_NAMES, FINGERS


def joint_name_to_index(body_id):
    """PyBullet joint index for every joint in the URDF, actuated or not, keyed by name."""
    out = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        out[info[1].decode()] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="render every Nth frame")
    ap.add_argument("--zoom-frame", type=int, default=None,
                    help="render a single frame as a PNG, camera close on the object, instead of a full video")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--azim", type=float, default=-45,
                    help="camera azimuth in degrees around the vertical axis.")
    ap.add_argument("--elev", type=float, default=23,
                    help="camera elevation in degrees above the horizontal plane (0=straight on, 90=top-down)")
    ap.add_argument("--dist-scale", type=float, default=1.0, help="multiply the auto camera distance (zoom)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        if f["qpos"].shape[1:] != (2, 29):
            raise SystemExit("Expected a Sharpa H5 (qpos shape (T,2,29)) -- wrong file for this script")
        qpos = f["qpos"][:]  # (T, 2, 29): index 0 = left side, 1 = right side
        obj_v = f["object_tracks/0/vertices"][:]
        obj_f = f["object_tracks/0/faces"][:]
        obj_t = f["object_tracks/0/translations"][:]
        obj_q = f["object_tracks/0/orientations_xyzw"][:]
        base = f["base_translation"][:]
        if not np.allclose(base[0], base[1]):
            raise SystemExit("Sharpa is one bimanual robot; both sides must share one base_translation")
        base = base[0].astype(np.float64)

    T = qpos.shape[0]
    print(f"robot base = {base.tolist()} (one bimanual body, shared by both sides)")

    client = p.connect(p.DIRECT)  # headless; works without a display on the server
    p.setGravity(0, 0, 0)  # kinematic replay only -- nothing should ever fall or collide-react

    urdf_path = os.path.join(args.assets_dir, "urdf", "rby1a_sharpa.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=base.tolist(), useFixedBase=True)
    name_to_idx = joint_name_to_index(robot_id)

    right_names = ARM_NAMES + HAND_NAMES
    left_names = [n.replace("right_", "left_") for n in right_names]
    missing = [n for n in right_names + left_names if n not in name_to_idx]
    if missing:
        raise SystemExit(f"URDF loaded by PyBullet is missing expected joints: {missing}")
    right_idx = [name_to_idx[n] for n in right_names]
    left_idx = [name_to_idx[n] for n in left_names]
    print(f"loaded {urdf_path}: {p.getNumJoints(robot_id)} total joints in URDF, "
          f"{len(right_idx)} actuated per side (x2 sides)")

    # color every link whose name mentions "hand" or a finger name bright red, so the hand is
    # visually unmistakable against the plain-gray arm links (both default to similar shades of
    # gray otherwise, making it hard to tell from a render alone what's touching what).
    hand_markers = ("hand",) + FINGERS
    n_colored = 0
    for i in range(p.getNumJoints(robot_id)):
        link_name = p.getJointInfo(robot_id, i)[12].decode().lower()
        if any(m in link_name for m in hand_markers):
            p.changeVisualShape(robot_id, i, rgbaColor=[0.9, 0.1, 0.1, 1])
            n_colored += 1
    print(f"colored {n_colored} hand/finger links red (arm links left at URDF default)")

    # the object: write its rest mesh to a temp .obj once, PyBullet needs a mesh FILE, not raw arrays
    with tempfile.TemporaryDirectory() as tmp:
        obj_path = os.path.join(tmp, "object.obj")
        trimesh.Trimesh(vertices=obj_v, faces=obj_f, process=False).export(obj_path)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, rgbaColor=[0.85, 0.45, 0.1, 1])
        object_id = p.createMultiBody(baseVisualShapeIndex=vis_shape, baseCollisionShapeIndex=-1)

        frame_ids = [args.zoom_frame] if args.zoom_frame is not None else list(range(0, T, args.stride))

        # camera target/distance: zoomed on the object alone, or a wider view covering object+base
        if args.zoom_frame is not None:
            center = obj_t[args.zoom_frame]
            dist = 0.25
        else:
            all_pos = np.concatenate([obj_t, base[None]])
            center = all_pos.mean(0)
            dist = float(np.max(np.linalg.norm(all_pos - center, axis=-1))) + 0.6
        # same 1.5362 correction as visualize_pybullet.py: a unit azim/elev direction vector needs
        # this factor to reproduce the equivalent of the old un-normalized [1,-1,0.6]-style offset.
        dist *= 1.5362 * args.dist_scale

        azim, elev = np.radians(args.azim), np.radians(args.elev)
        direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        view = p.computeViewMatrix(cameraEyePosition=(center + dist * direction).tolist(),
                                   cameraTargetPosition=center.tolist(), cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=args.width / args.height, nearVal=0.01, farVal=10.0)

        frames = []
        for t in frame_ids:
            for i, idx in enumerate(right_idx):
                p.resetJointState(robot_id, idx, float(qpos[t, 1, i]))
            for i, idx in enumerate(left_idx):
                p.resetJointState(robot_id, idx, float(qpos[t, 0, i]))
            p.resetBasePositionAndOrientation(object_id, obj_t[t].tolist(), obj_q[t].tolist())  # pybullet quats are xyzw too

            _, _, rgb, _, _ = p.getCameraImage(args.width, args.height, viewMatrix=view, projectionMatrix=proj,
                                               renderer=p.ER_TINY_RENDERER)
            img = np.reshape(rgb, (args.height, args.width, 4))[..., :3].astype(np.uint8)
            frames.append(img)
            if args.zoom_frame is not None or t % (args.stride * 20) == 0:
                print(f"rendered {t}/{T}")

    p.disconnect(client)

    import imageio.v2 as imageio
    if args.zoom_frame is not None:
        imageio.imwrite(args.out, frames[0])
        print("saved", args.out)
        return
    fps = 120.0 / args.stride
    try:
        imageio.mimsave(args.out, frames, fps=fps)
        print("saved", args.out)
    except Exception as exc:
        print(f"[warn] mp4 encode failed ({exc}); saving PNG frames instead")
        frame_dir = args.out + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        for i, img in enumerate(frames):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:05d}.png"), img)
        print("saved", frame_dir)


if __name__ == "__main__":
    main()
