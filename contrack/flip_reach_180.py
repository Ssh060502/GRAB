"""Take an existing --calib-rpy and flip the "reach direction" axis by 180 degrees -- the SAME
axis check_wrist_direction.py's human_dot/robot_dot checks (the one that came back consistently
matching, not flipped). This is the complementary test to add_roll_180.py: that one changed roll
only and left reach direction untouched (confirmed visually: wrist direction didn't change);
this one does the opposite -- flips reach direction, tested directly instead of trusting the
earlier numeric check blindly.

Math: a 180-degree rotation about an axis PERPENDICULAR to robot_forward_local reverses
robot_forward_local (points the opposite way) while a 180-degree rotation ABOUT
robot_forward_local itself (add_roll_180.py) leaves it unchanged -- these are complementary tests
of the two different things that could be "backwards". The perpendicular axis used here is picked
via Gram-Schmidt from a fixed reference vector -- any perpendicular axis works to flip reach
direction (that part is unambiguous and is what check_wrist_direction.py will confirm), though the
resulting roll around the new, flipped direction depends on which perpendicular axis was chosen,
so don't read too much into the exact roll check number after this -- the thing to check is
whether human_dot/robot_dot actually flip sign, and then look at the render.

Runs in the `retarget` env (needs pinocchio for the robot-side FK, same as estimate_calib.py).

Example:
    python contrack/flip_reach_180.py --assets-dir ~/ConTrack/assets --calib-rpy 94.59 -55.28 -58.61
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
                    help="the --calib-rpy currently in use, to flip the reach direction of")
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
    print(f"robot_forward_local (the axis being flipped) = {robot_forward_local.round(4).tolist()}")

    # Gram-Schmidt: any fixed reference not parallel to robot_forward_local, projected perpendicular
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ robot_forward_local) > 0.9:  # too close to parallel, pick a different reference
        ref = np.array([0.0, 1.0, 0.0])
    perp = ref - (ref @ robot_forward_local) * robot_forward_local
    perp /= np.linalg.norm(perp)

    calib = R.from_euler("xyz", args.calib_rpy, degrees=True).as_matrix()
    flip180 = R.from_rotvec(np.pi * perp).as_matrix()  # 180deg about perp -> reverses robot_forward_local
    calib_new = calib @ flip180

    flipped = flip180 @ robot_forward_local
    reversed_ok = np.allclose(flipped, -robot_forward_local, atol=1e-6)
    print(f"sanity check -- this reverses the reach axis: {reversed_ok} (must be True)")

    rpy_new = R.from_matrix(calib_new).as_euler("xyz", degrees=True)
    print(f"\noriginal --calib-rpy {args.calib_rpy[0]:.2f} {args.calib_rpy[1]:.2f} {args.calib_rpy[2]:.2f}")
    print(f"reach flipped ->  --calib-rpy {rpy_new[0]:.2f} {rpy_new[1]:.2f} {rpy_new[2]:.2f}")
    print("\nrerun retarget_xarm_xhand.py and check_wrist_direction.py with this new value.")
    print("Expected: human_dot should stay the same as before (it doesn't depend on calib at all),")
    print("but robot_dot should now flip sign (go NEGATIVE where it used to be positive) -- if it")
    print("doesn't flip sign, something is wrong with this script, not with the hypothesis.")


if __name__ == "__main__":
    main()
