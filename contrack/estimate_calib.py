"""Estimate --calib-rpy for retarget_xarm_xhand.py from geometry instead of guessing.

Why this is needed: retarget_xarm_xhand.py's Step 1 diagnostic ("orientation error") only
checks whether the arm reached the target orientation IT WAS GIVEN -- it says nothing about
whether that target (GRAB wrist rotation x --calib-rpy) is anatomically correct in the first
place. A 0.00 deg number there is compatible with --calib-rpy being wrong. The real symptom of
a wrong --calib-rpy is downstream, in Step 2: if the whole hand is planted facing the wrong way,
the fingers are asked to reach world-frame vectors that are physically behind/sideways of where
their joints can bend to, showing up as a large vector-fit residual AND every frame pinned at a
joint limit (exactly what a first run produced: 30-120mm residual, 100% of frames saturated,
versus a hand roughly 15-20cm across -- residuals that size mean the fingers are pointing in
roughly the wrong general direction, not just slightly mis-curled).

Method: build a "which way do the fingers splay from the wrist/palm" direction for 1 thumb + 4
fingers on both sides, then find the single rotation that best rotates the human pattern onto the
robot pattern (Kabsch fit on direction vectors, no translation -- same technique already used to
find Rm for the object in grab_to_contrack.py, just without the centering step since these are
directions from a shared origin, not point clouds).

  Robot side: fingertip position relative to the wrist link (right_hand_link), computed by
  forward kinematics with all finger joints at 0 (fingers straight) -- a fixed, once-only fact
  about the URDF, not run per frame.

  Human side: GRAB fingertip position relative to the wrist, rotated into the wrist's own local
  frame (removing the wrist's own rotation, i.e. "which way does each finger point relative to
  the back of the hand"), averaged over every frame of the clip (curling changes finger length
  but not much its general splay direction, so the frame average is a reasonable summary).

This gives a good STARTING GUESS, not a guaranteed-exact answer -- it is only as good as
"human finger splay direction roughly resembles robot finger splay direction at zero pose",
which is a coarse anatomical assumption, not something verified here. The per-finger fit angles
this script prints tell you how consistent that assumption is across fingers: if they scatter a
lot (e.g. one finger fits within 10 deg and another is 60 deg off), don't trust the single
average rotation -- something else is likely wrong (e.g. a finger-identity mixup) and the
calib-rpy is not going to fix it by itself.

Runs in the `retarget` env (needs pinocchio for the robot-side FK).

Example:
    python contrack/estimate_calib.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets
"""

import argparse
import os

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

ORIGIN_LINK = "right_hand_link"
TIP_LINKS = {
    "thumb": "right_hand_thumb_rota_tip", "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip", "ring": "right_hand_ring_tip", "pinky": "right_hand_pinky_tip",
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def kabsch_rotation_only(P, Q):
    """Rotation R minimising sum ||R @ P_i - Q_i||^2 for direction vectors sharing an origin (no translation)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    # ---- robot side: fingertip directions at qpos=0 (fingers straight) ----
    robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xhand_right.urdf"))
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    origin_pos = robot.get_link_pose(robot.get_link_index(ORIGIN_LINK))[:3, 3]
    robot_dirs = np.stack([
        robot.get_link_pose(robot.get_link_index(TIP_LINKS[f]))[:3, 3] - origin_pos for f in FINGERS
    ])
    robot_dirs /= np.linalg.norm(robot_dirs, axis=1, keepdims=True)

    # ---- human side: wrist-local fingertip directions, averaged over the whole clip ----
    with h5py.File(args.h5, "r") as f:
        k = f["grab_source/keypoints"]
        wrist_pos, wrist_rotmat = k["wrist_pos"][:], k["wrist_rotmat"][:]
        tips = {name: k["tips"][name][:] for name in FINGERS}

    human_dirs = []
    for f in FINGERS:
        local = np.einsum("tji,tj->ti", wrist_rotmat, tips[f] - wrist_pos)  # wrist_rotmat^T @ vec, per frame
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        human_dirs.append(local.mean(0))
    human_dirs = np.stack(human_dirs)
    human_dirs /= np.linalg.norm(human_dirs, axis=1, keepdims=True)

    # ---- fit ----
    calib = kabsch_rotation_only(human_dirs, robot_dirs)
    fitted = human_dirs @ calib.T
    angle_err = np.degrees(np.arccos(np.clip((fitted * robot_dirs).sum(1), -1, 1)))

    print("per-finger fit quality (should mostly be well under 30 deg if the coarse assumption holds):")
    for f, e in zip(FINGERS, angle_err):
        print(f"  {f:7s} {e:6.1f} deg")
    rpy = R.from_matrix(calib).as_euler("xyz", degrees=True)
    print(f"\n--calib-rpy {rpy[0]:.2f} {rpy[1]:.2f} {rpy[2]:.2f}")
    print("\nrerun retarget_xarm_xhand.py with the line above and check whether Step 2's")
    print("residual and joint-limit-saturation number drop a lot.")


if __name__ == "__main__":
    main()
