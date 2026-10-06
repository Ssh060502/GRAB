"""Take an existing --calib-rpy and add a 180-degree ROLL on top of it -- rotating specifically
around the robot's own "reach direction" axis (the same axis check_wrist_direction.py's roll check
measures against, using the thumb), so this changes ONLY roll and leaves the already-validated
"does the hand reach toward the object" direction completely untouched. Use this to directly test
a labmate's hypothesis ("looks flipped 180 degrees along the wrist direction") instead of guessing.

Math: compose the existing calib with an extra 180-degree rotation about robot_forward_local (a
FIXED vector in XHand's own local frame, the mean reach direction of index/middle/ring/pinky at
the robot's zero pose) -- calib_new = calib @ R(robot_forward_local, 180deg). Because a 180-degree
rotation about an axis leaves that same axis unchanged (R @ axis == axis), applying this new calib
produces EXACTLY the same check_wrist_direction.py human_dot/robot_dot numbers as before (forward
direction untouched) while flipping roll by 180 degrees -- an isolated, clean test of roll alone.

Runs in the `retarget` env (needs pinocchio for the robot-side FK, same as estimate_calib.py).

Example:
    python contrack/add_roll_180.py --assets-dir ~/ConTrack/assets --calib-rpy 94.59 -55.28 -58.61
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

ORIGIN_LINK = "right_hand_link"
TIP_LINKS = {
    "index": "right_hand_index_rota_tip", "middle": "right_hand_mid_tip",
    "ring": "right_hand_ring_tip", "pinky": "right_hand_pinky_tip",
}
FIT_FINGERS = ("index", "middle", "ring", "pinky")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--calib-rpy", type=float, nargs=3, required=True,
                    help="the --calib-rpy currently in use, to add 180 degrees of roll on top of")
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xhand_right.urdf"))
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    origin_pos = robot.get_link_pose(robot.get_link_index(ORIGIN_LINK))[:3, 3]
    robot_dirs = np.stack([
        robot.get_link_pose(robot.get_link_index(TIP_LINKS[f]))[:3, 3] - origin_pos for f in FIT_FINGERS
    ])
    robot_dirs /= np.linalg.norm(robot_dirs, axis=1, keepdims=True)
    robot_forward_local = robot_dirs.mean(0)
    robot_forward_local /= np.linalg.norm(robot_forward_local)
    print(f"robot_forward_local (XHand-local, the roll axis) = {robot_forward_local.round(4).tolist()}")

    calib = R.from_euler("xyz", args.calib_rpy, degrees=True).as_matrix()
    roll180 = R.from_rotvec(np.pi * robot_forward_local).as_matrix()  # 180deg about this fixed axis
    calib_new = calib @ roll180

    # sanity check: this must leave robot_forward_local itself unchanged (pure roll, nothing else)
    unchanged = np.allclose(roll180 @ robot_forward_local, robot_forward_local, atol=1e-6)
    print(f"sanity check -- roll180 leaves the reach axis fixed: {unchanged} (must be True)")

    rpy_new = R.from_matrix(calib_new).as_euler("xyz", degrees=True)
    print(f"\noriginal --calib-rpy {args.calib_rpy[0]:.2f} {args.calib_rpy[1]:.2f} {args.calib_rpy[2]:.2f}")
    print(f"+180deg roll ->   --calib-rpy {rpy_new[0]:.2f} {rpy_new[1]:.2f} {rpy_new[2]:.2f}")
    print("\nrerun retarget_xarm_xhand.py and check_wrist_direction.py with this new value.")
    print("Expected if this is a clean roll-only change: human_dot/robot_dot in check_wrist_direction.py")
    print("should be UNCHANGED from before; the roll-check number (using the thumb) should flip by")
    print("~180 degrees from whatever it was before.")


if __name__ == "__main__":
    main()
