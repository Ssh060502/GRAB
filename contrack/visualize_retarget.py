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
    ap.add_argument("--zoom-frame", type=int, default=None,
                    help="render a single frame as a PNG (ignores --stride/--out's video encoding), zoomed tightly "
                         "on the object so any hand-object gap is actually visible, instead of the whole arm+room "
                         "view where a few cm gap is hard to see. --out is still used as the file path.")
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
        right_base = f["base_translation"][1]  # is_rhand=[0,1] -> index 1 is the right hand

    # ConTrack spawns the robot's own root at base_translation in world space (same fact used to
    # fix retarget_xarm_xhand.py's Step 1); get_link_pose() below reports poses relative to the
    # URDF's own root at (0,0,0), so every rendered link position needs +right_base to land in the
    # same world frame the object (obj_t, already an absolute ConTrack-world position) is drawn in.
    # Omitting this drew the robot roughly `right_base` (0.4 m) away from where the object actually is.
    print(f"robot base (base_translation, right hand) = {right_base.tolist()}  <- ground truth, not a screenshot guess")

    obj_R = R.from_quat(obj_q).as_matrix()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zoom = args.zoom_frame is not None
    if zoom:
        # tight close-up on the object at one specific frame, so a few-cm hand-object gap is
        # actually visible -- the whole-arm+room view averages/zooms out too much to see it clearly
        frame_ids = [args.zoom_frame]
        center = obj_t[args.zoom_frame]
        radius = 0.12
    else:
        frame_ids = list(range(0, qpos.shape[0], args.stride))
        all_pos = np.concatenate([obj_t, right_base[None]])  # include the arm's base so it stays in frame
        center = all_pos.mean(0)
        radius = float(np.max(np.linalg.norm(all_pos - center, axis=-1))) + 0.3

    # explicit ground plane at z=0 (skipped in zoom mode -- irrelevant at this scale, and the base
    # itself is likely well outside a 12 cm close-up), so "is the base actually at ground level" is
    # a visible fact in the wide view, not something to eyeball off two differently-zoomed screenshots
    if not zoom:
        gx, gy = np.meshgrid(np.linspace(center[0] - radius, center[0] + radius, 2),
                              np.linspace(center[1] - radius, center[1] + radius, 2))
        gz = np.zeros_like(gx)

    full = np.zeros(robot.dof)
    frames = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    for t in frame_ids:
        ax.cla()
        if not zoom:
            ax.plot_surface(gx, gy, gz, color="gray", alpha=0.15, linewidth=0)
            ax.scatter(*right_base, color="black", s=40, marker="^", label="robot base (z=0)")
        obj_w = obj_v @ obj_R[t].T + obj_t[t]
        ax.plot_trisurf(obj_w[:, 0], obj_w[:, 1], obj_w[:, 2], triangles=obj_f, color="tab:orange", alpha=0.9, linewidth=0)

        full[joint_idx] = qpos[t]
        robot.compute_forward_kinematics(full)
        for link_name, entries in visuals.items():
            pose = robot.get_link_pose(link_ids[link_name])
            for mesh, origin in entries:
                M = pose @ origin
                v = mesh.vertices @ M[:3, :3].T + M[:3, 3] + right_base
                ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=mesh.faces, color="tab:blue", alpha=0.6, linewidth=0)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))  # matplotlib 3D doesn't enforce equal x/y/z scale by default,
        # which was making the (real, meter-scale) robot mesh look artificially compressed/undersized
        ax.set_title(f"frame {t}/{qpos.shape[0]}" + ("  (zoom)" if zoom else ""))
        if not zoom:
            ax.legend(loc="upper left", fontsize=7)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if zoom or t % (args.stride * 20) == 0:
            print(f"rendered {t}/{qpos.shape[0]}")
    plt.close(fig)

    if zoom:
        import imageio.v2 as imageio
        imageio.imwrite(args.out, frames[0])
        print("saved", args.out)
        return

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
