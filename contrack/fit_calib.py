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

Runs in the `retarget` env.

Example:
    python contrack/fit_calib.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --init-rpy -96.21 40.95 -61.16
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
    ap.add_argument("--num-frames", type=int, default=12, help="frames sampled evenly across the clip for the fit")
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

    T = wrist_pos.shape[0]
    sample = np.linspace(0, T - 1, args.num_frames).astype(int)
    print(f"fitting against {len(sample)} sampled frames: {sample.tolist()}")

    def total_residual(calib_rpy):
        calib = R.from_euler("xyz", calib_rpy, degrees=True).as_matrix()
        total = 0.0
        x0 = np.zeros(12)
        for t in sample:
            tips_t = {f: tips[f][t] for f in FINGERS}
            x = solve_fingers(robot, wrist_pos[t], wrist_rotmat[t], calib, tips_t,
                               finger_idx, origin_id, tip_ids, finger_limits, x0)
            R_hand_world = wrist_rotmat[t] @ calib
            target_vecs = np.stack([R_hand_world.T @ (tips_t[f] - wrist_pos[t]) for f in FINGERS])
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
