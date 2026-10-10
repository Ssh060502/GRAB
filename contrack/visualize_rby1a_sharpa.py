"""Headless RB-Y1A/Sharpa skeleton + object replay; no physics or mesh assets needed.

Uses all joint/link origins from the full URDF and the reference object's mesh.
This is a kinematic geometry check, not a shaded hand-surface/contact validation.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sharpa_geometry import SharpaKinematics, ARM_NAMES, HAND_NAMES, PALM_LINK, TIP_LINKS


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5",required=True)
    ap.add_argument("--assets-dir",required=True)
    ap.add_argument("--out",required=True,help="PNG with --frame, otherwise MP4")
    ap.add_argument("--frame",type=int)
    ap.add_argument("--stride",type=int,default=4)
    ap.add_argument("--elev",type=float,default=25.)
    ap.add_argument("--azim",type=float,default=-45.)
    args=ap.parse_args()
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio
    model=SharpaKinematics(Path(args.assets_dir)/"urdf"/"rby1a_sharpa.urdf")
    with h5py.File(args.h5,"r") as f:
        q=f["qpos"][:]
        if q.shape[1:] != (2,29) or not np.isfinite(q).all() or not np.array_equal(f["is_rhand"][:],[0,1]):
            raise ValueError("Expected finite left/right Sharpa qpos; run retargeting first")
        base=f["base_translation"][:]
        if not np.allclose(base[0],base[1]):
            raise ValueError("Whole robot must share one base")
        base=base[0]
        fps=float(np.max(f["fps"][:]))
        objects=[]
        for group in f["object_tracks"].values():
            objects.append((group["vertices"][:],group["faces"][:],group["translations"][:],
                            Rotation.from_quat(group["orientations_xyzw"][:]).as_matrix()))
        source=None
        if "grab_source/keypoints/tips" in f:
            source=np.stack([f["grab_source/keypoints/tips"][n][:] for n in ("thumb","index","middle","ring","pinky")],axis=1)
    if args.stride < 1 or (args.frame is not None and not 0 <= args.frame < len(q)):
        ap.error("Invalid frame/stride")
    names=ARM_NAMES+HAND_NAMES
    def poses_at(t):
        angles=dict(zip(names,q[t,1]))
        angles.update(zip([n.replace("right_","left_") for n in names],q[t,0]))
        return model.fk(angles)
    # Fixed equal-aspect camera bounds covering root, robot and object trajectory.
    all_points=[np.stack([p[:3,3]+base for p in poses_at(t).values()])
                for t in np.unique(np.linspace(0,len(q)-1,min(10,len(q))).astype(int))]
    all_points += [obj[2] for obj in objects]
    points=np.concatenate(all_points)
    center=(points.min(axis=0)+points.max(axis=0))/2
    radius=max(float(np.ptp(points,axis=0).max())*.55,.2)
    ids=[args.frame] if args.frame is not None else range(0,len(q),args.stride)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    writer=None if args.frame is not None else imageio.get_writer(args.out,fps=fps/args.stride)
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection="3d")
    try:
        for t in ids:
            ax.clear()
            poses=poses_at(t)
            for joint in model.joints:
                p=np.stack([poses[n][:3,3]+base for n in (joint["parent"],joint["child"])])
                color="tab:blue" if joint["name"].startswith("right_") else "gray"
                ax.plot(p[:,0],p[:,1],p[:,2],color=color,linewidth=1.5)
            tips=np.stack([poses[n][:3,3]+base for n in TIP_LINKS])
            ax.scatter(*tips.T,color="tab:blue",s=15,label="robot tips")
            if source is not None:
                ax.scatter(*source[t].T,color="red",s=12,label="MANO targets")
            wrist=poses[PALM_LINK]
            for i,color in enumerate(("red","green","blue")):
                p=np.stack([wrist[:3,3]+base,wrist[:3,3]+base+.06*wrist[:3,i]])
                ax.plot(*p.T,color=color,linewidth=2)
            for vertices,faces,translation,rotation in objects:
                world=vertices@rotation[t].T+translation[t]
                ax.add_collection3d(Poly3DCollection(world[faces],facecolor="orange",alpha=.4,linewidth=0))
            ax.set_xlim(center[0]-radius,center[0]+radius)
            ax.set_ylim(center[1]-radius,center[1]+radius)
            ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=args.elev,azim=args.azim)
            ax.set_title(f"RB-Y1A / Sharpa frame {t}/{len(q)} (kinematic skeleton)")
            ax.legend(loc="upper right")
            fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            if writer is None:
                imageio.imwrite(args.out,frame)
            else:
                writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()
        plt.close(fig)
    print("Saved",args.out)


if __name__=="__main__":
    main()
