"""Cross-check PyBullet's own forward kinematics against the TARGET pose retarget_xarm_xhand.py
solved for, to rule in/out a visualization-layer bug as the cause of "wrist looks wrong" --
separate from "is the retargeting math wrong".

Why this is needed: everything verified so far (position error, orientation error, the direction/
roll checks) was computed using pinocchio (via dex_retargeting's RobotWrapper), the SAME library
used to solve the IK in the first place -- so those checks can only catch "pinocchio disagrees
with itself", not "pinocchio and PyBullet parse/interpret the same URDF + qpos differently".
PyBullet is a completely separate library loading the same URDF independently; if it has even a
subtly different convention for some joint's zero-reference or sign, the exact same qpos numbers
could produce a visibly different pose in PyBullet's render than what pinocchio (and therefore
every diagnostic so far) reports -- which would explain "the numbers say it's fine but the video
looks wrong" without any bug in retarget_xarm_xhand.py itself.

Method: load the robot in PyBullet (same as visualize_pybullet.py), set a frame's qpos, and ask
PyBullet directly (p.getLinkState) where IT thinks right_hand_link ended up -- then compare
against the known target (GRAB's recorded wrist position/orientation, the same target
retarget_xarm_xhand.py's Step 1 was solving for). If PyBullet's own answer disagrees with the
target by anywhere near the same margin pinocchio reports as ~0, that is a visualization-layer
bug, not a retargeting bug.

Runs in the `retarget` env (needs pybullet, same as visualize_pybullet.py).

Example:
    python contrack/check_pybullet_fk.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir ~/ConTrack/assets --calib-rpy 94.59 -55.28 -58.61
"""

import argparse
import os

import h5py
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--calib-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="must match whatever --calib-rpy was used for the qpos currently saved in --h5")
    ap.add_argument("--num-frames", type=int, default=6)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        qpos = f["qpos"][:, 1, :]  # right hand
        k = f["grab_source/keypoints"]
        wrist_pos, wrist_rotmat = k["wrist_pos"][:], k["wrist_rotmat"][:]
        right_base = f["base_translation"][1].astype(np.float64)

    T = qpos.shape[0]
    calib = R.from_euler("xyz", args.calib_rpy, degrees=True).as_matrix()

    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    urdf_path = os.path.join(args.assets_dir, "urdf", "xarm_xhand_right.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=right_base.tolist(), useFixedBase=True)

    # map joint NAME -> pybullet joint index (for setting qpos), and LINK name -> index (to find
    # right_hand_link specifically, which is a child link, not necessarily named like a joint)
    name_to_joint_idx, name_to_link_idx = {}, {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name_to_joint_idx[info[1].decode()] = i
        name_to_link_idx[info[12].decode()] = i

    joint_names = XARM_JOINT_NAMES + HAND_JOINT_NAMES
    joint_idx = [name_to_joint_idx[n] for n in joint_names]
    wrist_link_idx = name_to_link_idx["right_hand_link"]
    print(f"right_hand_link is pybullet link index {wrist_link_idx}")

    sample = np.linspace(0, T - 1, args.num_frames).astype(int)
    print(f"\n{'frame':>6} {'pos err (mm)':>13} {'orient err (deg)':>17}")
    for t in sample:
        for i, idx in enumerate(joint_idx):
            p.resetJointState(robot_id, idx, float(qpos[t, i]))
        link_state = p.getLinkState(robot_id, wrist_link_idx, computeForwardKinematics=True)
        pb_pos = np.array(link_state[4])          # linkWorldPosition (urdf frame, not COM frame)
        pb_quat_xyzw = np.array(link_state[5])     # linkWorldOrientation
        pb_rot = R.from_quat(pb_quat_xyzw).as_matrix()

        target_pos = wrist_pos[t]  # already in ConTrack world frame, robot base included via basePosition
        target_rot = wrist_rotmat[t] @ calib

        pos_err = np.linalg.norm(pb_pos - target_pos) * 1000
        orient_err = R.from_matrix(pb_rot.T @ target_rot).magnitude() * 180 / np.pi
        print(f"{t:6d} {pos_err:13.2f} {orient_err:17.2f}")

    print("\nif these errors are anywhere near 0 (matching what retarget_xarm_xhand.py's own")
    print("pinocchio-based diagnostics reported), PyBullet agrees with pinocchio -- the")
    print("visualization is trustworthy and the problem is elsewhere (finger shape, etc).")
    print("If these errors are LARGE (notably bigger than pinocchio's ~0), PyBullet is")
    print("interpreting the URDF/qpos differently from pinocchio -- a visualization-layer bug.")


if __name__ == "__main__":
    main()
