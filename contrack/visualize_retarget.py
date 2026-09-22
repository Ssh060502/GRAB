"""Render an offline MP4 of a retargeted ConTrack clip: object + right xArm7+XHand.

No Isaac Sim, no SAPIEN, no trained policy needed -- just forward kinematics (pinocchio, via
the same RobotWrapper used by retarget_xarm_xhand.py) plus matplotlib, following the exact same
rendering approach ConTrack's own scripts/tools/inspect_dataset.py --video uses for the object
alone (ax.plot_trisurf per mesh per frame, encoded with imageio, falling back to a PNG sequence
if no mp4 codec is available). This just adds the robot's own visual meshes to that picture.

dex_retargeting's own vendored `yourdfpy` module has no forward-kinematics API (checked: no
update_cfg/link_fk in this trimmed copy, unlike the standalone yourdfpy package) -- so mesh
placement uses a small local URDF <visual> parser (stdlib xml.etree only) combined with
RobotWrapper.get_link_pose() for each link's pose.

Runs in the `retarget` env (needs pinocchio, i.e. dex_retargeting) plus trimesh/matplotlib/imageio:
    pip install trimesh matplotlib imageio

Example:
    python contrack/visualize_retarget.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --out out/teapot_retarget.mp4
"""

import argparse
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
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


def rpy_to_matrix(rpy):
    return R.from_euler("xyz", rpy).as_matrix()


def parse_visual_meshes(urdf_path):
    """link name -> list of (trimesh.Trimesh, local_origin 4x4) for every <visual><mesh> in the URDF."""
    base_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    out = {}
    for link in tree.findall("link"):
        entries = []
        for visual in link.findall("visual"):
            mesh_el = visual.find("geometry/mesh")
            if mesh_el is None:
                continue
            fn = os.path.normpath(os.path.join(base_dir, mesh_el.get("filename")))
            if not os.path.isfile(fn):
                print(f"  [warn] mesh not found, skipping: {fn}")
                continue
            origin_el = visual.find("origin")
            xyz = np.array([float(v) for v in origin_el.get("xyz", "0 0 0").split()]) if origin_el is not None else np.zeros(3)
            rpy = np.array([float(v) for v in origin_el.get("rpy", "0 0 0").split()]) if origin_el is not None else np.zeros(3)
            origin = np.eye(4)
            origin[:3, :3] = rpy_to_matrix(rpy)
            origin[:3, 3] = xyz
            mesh = trimesh.load(fn, process=False, force="mesh")
            entries.append((mesh, origin))
        if entries:
            out[link.get("name")] = entries
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="render every Nth frame")
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    urdf_path = os.path.join(args.assets_dir, "urdf", "xarm_xhand_right.urdf")
    print("parsing visual meshes from", urdf_path)
    visuals = parse_visual_meshes(urdf_path)
    print(f"  {sum(len(v) for v in visuals.values())} meshes on {len(visuals)} links")

    robot = RobotWrapper(urdf_path)
    joint_idx = np.array([robot.get_joint_index(n) for n in XARM_JOINT_NAMES + HAND_JOINT_NAMES])
    link_ids = {name: robot.get_link_index(name) for name in visuals}

    with h5py.File(args.h5, "r") as f:
        qpos = f["qpos"][:, 1, :]  # right hand only
        obj_v = f["object_tracks/0/vertices"][:]
        obj_f = f["object_tracks/0/faces"][:]
        obj_t = f["object_tracks/0/translations"][:]
        obj_q = f["object_tracks/0/orientations_xyzw"][:]

    obj_R = R.from_quat(obj_q).as_matrix()
    frame_ids = list(range(0, qpos.shape[0], args.stride))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_pos = obj_t
    center = all_pos.mean(0)
    radius = float(np.max(np.linalg.norm(all_pos - center, axis=-1))) + 0.4

    full = np.zeros(robot.dof)
    frames = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    for t in frame_ids:
        ax.cla()
        obj_w = obj_v @ obj_R[t].T + obj_t[t]
        ax.plot_trisurf(obj_w[:, 0], obj_w[:, 1], obj_w[:, 2], triangles=obj_f, color="tab:orange", alpha=0.9, linewidth=0)

        full[joint_idx] = qpos[t]
        robot.compute_forward_kinematics(full)
        for link_name, entries in visuals.items():
            pose = robot.get_link_pose(link_ids[link_name])
            for mesh, origin in entries:
                M = pose @ origin
                v = mesh.vertices @ M[:3, :3].T + M[:3, 3]
                ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=mesh.faces, color="tab:blue", alpha=0.6, linewidth=0)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(f"frame {t}/{qpos.shape[0]}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if t % (args.stride * 20) == 0:
            print(f"rendered {t}/{qpos.shape[0]}")
    plt.close(fig)

    fps = 120.0 / args.stride
    try:
        import imageio.v2 as imageio
        imageio.mimsave(args.out, frames, fps=fps)
        print("saved", args.out)
    except Exception as exc:
        print(f"[warn] mp4 encode failed ({exc}); saving PNG frames instead")
        frame_dir = args.out + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        import imageio.v2 as imageio
        for i, img in enumerate(frames):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:05d}.png"), img)
        print("saved", frame_dir)


if __name__ == "__main__":
    main()
