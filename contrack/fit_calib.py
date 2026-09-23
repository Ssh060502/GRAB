"""Find --calib-rpy by directly minimizing Step 2's own vector-fit residual, instead of the
coarse zero-pose finger-splay heuristic in estimate_calib.py.

Why: estimate_calib.py fits one rigid rotation to make the robot's zero-pose finger directions
resemble GRAB's wrist-local fingertip directions -- a reasonable starting guess, but the 5
fingers disagreed by 5.6 to 36.3 degrees on what that rotation should be, and plugging the
resulting compromise into retarget_xarm_xhand.py still left Step 2 with 100-290mm residuals
(bigger than some fingers' entire physical length) and every frame pinned at a joint limit. That
points at direction, not scale (the scale check there shows most fingers' reach brackets overlap
GRAB's numbers) -- so the rotation itself is still off, and a proxy heuristic isn't precise enough.

This script instead treats calib as the unknown of an outer optimization whose objective IS
Step 2's real residual: for a handful of sampled frames, solve the 12 finger joints (reusing
retarget_xarm_xhand.solve_fingers verbatim) and sum up the leftover fit error; scipy.optimize
.minimize (Nelder-Mead, gradient-free since the inner IK isn't differentiable end-to-end here)
searches over the 3 Euler angles to minimize that sum. Slower than estimate_calib.py (each outer
step re-solves several 12-DOF IK problems) but optimizes the thing we actually care about,
instead of a stand-in for it. Seed it with estimate_calib.py's answer since it's a reasonable
starting point even if not precise enough on its own.

v2: sample frames from the CONTACT-PRIORITY frames specifically (where GRAB records a real
finger-object touch), and fit against the contact-point target (same swap solve_fingers' own
contact_targets does), not the plain wrist-to-fingertip vector. Reason: fitting v1 against the
plain vector gave a calib that, once plugged into retarget_xarm_xhand.py's contact-priority mode,
still left a stubborn ~25-30mm residual with 0% of contact frames under 10mm and 100% of frames
pinned at a joint limit -- a sign the earlier calib was optimized for the wrong target (a coarse
proxy vector) rather than the thing that actually matters for grasping (the literal recorded
contact point). Re-fitting directly against contact points, for contact frames only, gives calib
its best shot at finding an overall hand orientation that leaves the fingers enough room to reach
those specific points, instead of averaging error over hundreds of irrelevant non-contact frames.

Runs in the `retarget` env.

Example:
    python contrack/fit_calib.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --init-rpy -163.40 -28.35 -145.48
"""

import argparse
import os

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from retarget_xarm_xhand import HAND_JOINT_NAMES, ORIGIN_LINK, TIP_LINKS, solve_fingers

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--init-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--num-frames", type=int, default=20,
                    help="contact-priority frames sampled evenly for the fit (across all fingers combined)")
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xhand_right.urdf"))
    finger_idx = np.array([robot.get_joint_index(n) for n in HAND_JOINT_NAMES])
    origin_id = robot.get_link_index(ORIGIN_LINK)
    tip_ids = [robot.get_link_index(TIP_LINKS[f]) for f in FINGERS]
    finger_limits = robot.joint_limits[finger_idx]

    with h5py.File(args.h5, "r") as f:
        k = f["grab_source/keypoints"]
        wrist_pos, wrist_rotmat = k["wrist_pos"][:], k["wrist_rotmat"][:]
        tips = {name: k["tips"][name][:] for name in FINGERS}
        seg_names = [s.decode() for s in f["contacts/segment_names"][:]]
        is_contact = f["contacts/0/is_contact"][:]
        contact_pts = f["contacts/0/points"][:]
        obj_t = f["object_tracks/0/translations"][:]
        obj_R = R.from_quat(f["object_tracks/0/orientations_xyzw"][:]).as_matrix()

    T = wrist_pos.shape[0]

    def contact_targets_for(calib):
        """(T,5,3) array, NaN where that finger has no distal contact this frame -- same rule as
        retarget_xarm_xhand.py's contact-priority block, recomputed here since calib is the
        unknown being searched over."""
        out = np.full((T, 5, 3), np.nan)
        for i, name in enumerate(FINGERS):
            seg = seg_names.index(f"{name}_distal")
            touching = is_contact[:, 1, seg].astype(bool)
            if not touching.any():
                continue
            world_pt = np.einsum("tij,tj->ti", obj_R[touching], contact_pts[touching, 1, seg]) + obj_t[touching]
            R_hand_world_t = np.einsum("tij,jk->tik", wrist_rotmat[touching], calib)
            out[touching, i] = np.einsum("tji,tj->ti", R_hand_world_t, world_pt - wrist_pos[touching])
        return out

    # sample frames from wherever contact-priority actually applies (any finger), not evenly across
    # the whole clip -- non-contact frames don't matter for what we're optimizing here
    any_contact = np.isfinite(contact_targets_for(np.eye(3))).all(axis=2).any(axis=1)
    contact_frames = np.where(any_contact)[0]
    if len(contact_frames) == 0:
        raise SystemExit("no contact-priority frames at all -- nothing to fit against")
    sample = contact_frames[np.linspace(0, len(contact_frames) - 1, min(args.num_frames, len(contact_frames))).astype(int)]
    print(f"fitting against {len(sample)} contact-priority frames: {sample.tolist()}")

    def total_residual(calib_rpy):
        calib = R.from_euler("xyz", calib_rpy, degrees=True).as_matrix()
        ct_all = contact_targets_for(calib)
        total = 0.0
        x0 = np.zeros(12)
        for t in sample:
            tips_t = {f: tips[f][t] for f in FINGERS}
            ct_t = ct_all[t]
            x = solve_fingers(robot, wrist_pos[t], wrist_rotmat[t], calib, tips_t,
                               finger_idx, origin_id, tip_ids, finger_limits, x0, contact_targets=ct_t)
            R_hand_world = wrist_rotmat[t] @ calib
            target_vecs = np.stack([R_hand_world.T @ (tips_t[f] - wrist_pos[t]) for f in FINGERS])
            use = np.isfinite(ct_t).all(axis=1)
            target_vecs[use] = ct_t[use]
            full = np.zeros(robot.dof)
            full[finger_idx] = x
            robot.compute_forward_kinematics(full)
            origin_pos = robot.get_link_pose(origin_id)[:3, 3]
            vecs = np.stack([robot.get_link_pose(i)[:3, 3] - origin_pos for i in tip_ids])
            total += float(np.sum((vecs - target_vecs) ** 2))
        return total

    print("initial total squared residual:", total_residual(args.init_rpy))
    res = minimize(total_residual, np.array(args.init_rpy), method="Nelder-Mead",
                    options={"xatol": 0.1, "fatol": 1e-8, "maxiter": 300, "disp": True})
    print("\nfinal total squared residual:", res.fun)
    print(f"--calib-rpy {res.x[0]:.2f} {res.x[1]:.2f} {res.x[2]:.2f}")
    print("\nplug this into retarget_xarm_xhand.py --calib-rpy and check the Step 2 diagnostics again.")


if __name__ == "__main__":
    main()
