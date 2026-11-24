"""Headless CRX-5iA / Sharpa skeleton + object replay; no physics or robot meshes needed.

Reads a file written by retarget_crx_sharpa.py. Draws each arm's URDF skeleton at
its own base pose, the object mesh, the MANO fingertip targets and the table plane.
This is a kinematic geometry check, not a hand-surface/contact validation.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from crx_sharpa_geometry import CrxSharpaKinematics, FINGERS, SIDE_DOF


def load(path):
    import h5py
    with h5py.File(path, "r") as f:
        q = f["qpos"][:]
        if q.shape[1:] != (2, SIDE_DOF) or not np.isfinite(q).all() or not np.array_equal(f["is_rhand"][:], [0, 1]):
            raise ValueError("Expected finite left/right CRX+Sharpa qpos (T,2,28); run retarget_crx_sharpa.py first")
        bases = np.tile(np.eye(4), (2, 1, 1))
        bases[:, :3, 3] = f["base_translation"][:]
        bases[:, :3, :3] = Rotation.from_quat(f["base_quat_wxyz"][:][:, [1, 2, 3, 0]]).as_matrix()
        objects = [(g["vertices"][:], g["faces"][:], g["translations"][:],
                    Rotation.from_quat(g["orientations_xyzw"][:]).as_matrix())
                   for g in f["object_tracks"].values()]
        tips = None
        if "grab_source/keypoints/tips" in f:
            tips = np.stack([f["grab_source/keypoints/tips"][n][:] for n in FINGERS], axis=1)
        scene = f.attrs.get("scene_transform_xyz_yaw_deg")
        table_z = None if scene is None else float(scene[2])
        return q, bases, objects, tips, float(np.max(f["fps"][:])), table_z


def render(models, q, bases, objects, tips, fps, table_z, out, frame=None, stride=4, elev=25., azim=-45.):
    """models: {side_index: CrxSharpaKinematics}; sides without a model are not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    def world_poses(side, t):
        model = models[side]
        local = model.fk(dict(zip(model.arm_names + model.hand_names, q[t, side])))
        return {name: bases[side] @ pose for name, pose in local.items()}
    # Fixed equal-aspect camera bounds covering both robots and the object trajectory.
    samples = np.unique(np.linspace(0, len(q)-1, min(10, len(q))).astype(int))
    points = [np.stack([p[:3, 3] for p in world_poses(s, t).values()]) for s in models for t in samples]
    points += [obj[2] for obj in objects]
    points = np.concatenate(points)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * .55, .2)
    ids = [frame] if frame is not None else range(0, len(q), stride)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = None if frame is not None else imageio.get_writer(out, fps=fps / stride)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    try:
        for t in ids:
            ax.clear()
            for side, model in models.items():
                poses = world_poses(side, t)
                color = "tab:blue" if side == 1 else "gray"
                for joint in model.joints:
                    p = np.stack([poses[n][:3, 3] for n in (joint["parent"], joint["child"])])
                    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=1.5)
                ax.scatter(*bases[side][:3, 3], color="black", s=30, marker="s")
                if side == 1:
                    robot_tips = np.stack([poses[n][:3, 3] for n in model.tip_links])
                    ax.scatter(*robot_tips.T, color="tab:blue", s=15, label="robot tips")
                    wrist = poses[model.palm_link]
                    for i, axis_color in enumerate(("red", "green", "blue")):
                        p = np.stack([wrist[:3, 3], wrist[:3, 3] + .06 * wrist[:3, i]])
                        ax.plot(*p.T, color=axis_color, linewidth=2)
            if tips is not None:
                ax.scatter(*tips[t].T, color="red", s=12, label="MANO targets")
            for vertices, faces, translation, rotation in objects:
                world = vertices @ rotation[t].T + translation[t]
                ax.add_collection3d(Poly3DCollection(world[faces], facecolor="orange", alpha=.4, linewidth=0))
            if table_z is not None:
                corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * radius + center[:2]
                ax.add_collection3d(Poly3DCollection([np.c_[corners, np.full(4, table_z)]],
                                                     facecolor="lightgray", alpha=.25, linewidth=0))
            ax.set_xlim(center[0]-radius, center[0]+radius)
            ax.set_ylim(center[1]-radius, center[1]+radius)
            ax.set_zlim(center[2]-radius, center[2]+radius)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"CRX-5iA / Sharpa frame {t}/{len(q)}, t = {t/fps:.2f} s (kinematic skeleton)")
            ax.legend(loc="upper right")
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            if writer is None:
                imageio.imwrite(out, image)
            else:
                writer.append_data(image)
    finally:
        if writer is not None:
            writer.close()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--right-urdf", required=True)
    ap.add_argument("--left-urdf", help="Optional; without it only the right arm is drawn")
    ap.add_argument("--out", required=True, help="PNG with --frame, otherwise MP4")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--elev", type=float, default=25.)
    ap.add_argument("--azim", type=float, default=-45.)
    args = ap.parse_args()
    q, bases, objects, tips, fps, table_z = load(args.h5)
    if args.stride < 1 or (args.frame is not None and not 0 <= args.frame < len(q)):
        ap.error("Invalid frame/stride")
    models = {1: CrxSharpaKinematics(args.right_urdf, "right")}
    if args.left_urdf:
        models[0] = CrxSharpaKinematics(args.left_urdf, "left")
    render(models, q, bases, objects, tips, fps, table_z, args.out, args.frame, args.stride, args.elev, args.azim)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
