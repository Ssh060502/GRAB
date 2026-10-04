"""Render a retargeted ConTrack clip inside PyBullet instead of matplotlib -- same kinematic
replay as visualize_retarget.py (no physics is actually run: no gravity, no stepSimulation,
joints/object are just teleported to our precomputed qpos/pose each frame), but using a real
physics-engine's own renderer (lighting, shading, materials) instead of a bare 3D scatter plot.

Still NOT a physics validation: "does the finger really touch the object" is not answered any
more authoritatively here than in visualize_retarget.py or the numeric residual checks -- for
that, qpos would need to be fed into ConTrack's actual Isaac Sim environment and let its own
collision engine judge contact, which this script does not attempt.

Advantage over visualize_retarget.py: PyBullet loads the URDF directly (p.loadURDF), so there is
no need for the hand-written <visual> mesh parser -- PyBullet resolves the full kinematic tree,
materials, and mesh files on its own.

Runs in the `retarget` env, needs one more package:
    pip install pybullet

Headless by default (p.DIRECT, software or EGL rendering, no display needed on the server).

Example (full clip -> mp4):
    python contrack/visualize_pybullet.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --out out/teapot_pybullet.mp4

Example (one frame, zoomed in on the object -> png):
    python contrack/visualize_pybullet.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --out out/teapot_pybullet_zoom.png --zoom-frame 400
"""

import argparse
import os
import tempfile

import h5py
import numpy as np
import pybullet as p
import trimesh
from scipy.spatial.transform import Rotation as R

XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]


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
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        qpos = f["qpos"][:, 1, :]  # right hand only, matches visualize_retarget.py's scope
        obj_v = f["object_tracks/0/vertices"][:]
        obj_f = f["object_tracks/0/faces"][:]
        obj_t = f["object_tracks/0/translations"][:]
        obj_q = f["object_tracks/0/orientations_xyzw"][:]
        right_base = f["base_translation"][1].astype(np.float64)

    T = qpos.shape[0]
    print(f"robot base = {right_base.tolist()}  (same base_translation fact as visualize_retarget.py)")

    client = p.connect(p.DIRECT)  # headless; works without a display on the server
    p.setGravity(0, 0, 0)  # kinematic replay only -- nothing should ever fall or collide-react

    urdf_path = os.path.join(args.assets_dir, "urdf", "xarm_xhand_right.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=right_base.tolist(), useFixedBase=True)
    name_to_idx = joint_name_to_index(robot_id)
    joint_names = XARM_JOINT_NAMES + HAND_JOINT_NAMES
    missing = [n for n in joint_names if n not in name_to_idx]
    if missing:
        raise SystemExit(f"URDF loaded by PyBullet is missing expected joints: {missing}")
    joint_idx = [name_to_idx[n] for n in joint_names]
    print(f"loaded {urdf_path}: {p.getNumJoints(robot_id)} total joints in URDF, "
          f"{len(joint_idx)} of them are our 19 actuated joints")

    # the object: write its rest mesh to a temp .obj once, PyBullet needs a mesh FILE, not raw arrays
    with tempfile.TemporaryDirectory() as tmp:
        obj_path = os.path.join(tmp, "object.obj")
        trimesh.Trimesh(vertices=obj_v, faces=obj_f, process=False).export(obj_path)
        vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, rgbaColor=[0.85, 0.45, 0.1, 1])
        object_id = p.createMultiBody(baseVisualShapeIndex=vis_shape, baseCollisionShapeIndex=-1)

        obj_R = R.from_quat(obj_q)
        frame_ids = [args.zoom_frame] if args.zoom_frame is not None else list(range(0, T, args.stride))

        # camera target/distance: zoomed on the object alone, or a wider view covering object+base
        if args.zoom_frame is not None:
            center = obj_t[args.zoom_frame]
            dist = 0.25
        else:
            all_pos = np.concatenate([obj_t, right_base[None]])
            center = all_pos.mean(0)
            dist = float(np.max(np.linalg.norm(all_pos - center, axis=-1))) + 0.6

        view = p.computeViewMatrix(cameraEyePosition=(center + np.array([dist, -dist, dist * 0.6])).tolist(),
                                   cameraTargetPosition=center.tolist(), cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=args.width / args.height, nearVal=0.01, farVal=10.0)

        frames = []
        for t in frame_ids:
            for i, idx in enumerate(joint_idx):
                p.resetJointState(robot_id, idx, float(qpos[t, i]))
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
