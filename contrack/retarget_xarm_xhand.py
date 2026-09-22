"""Fill in the qpos of a converted ConTrack clip (xArm7+XHand) via two-step retargeting.

*** Runs in its OWN conda env, separate from `grab` *** -- pinocchio (pip package `pin`) has no
wheel for Python 3.9 past version 2.6.18, and dex_retargeting wants numpy>=2.0, which conflicts
with smplx's MANO loader needing chumpy needing numpy<1.24. So keypoint extraction is a separate
earlier step (compute_hand_keypoints.py, run in the existing `grab` env) that writes wrist/
fingertip positions into the h5's grab_source/keypoints group; this script only reads that group
back -- it never imports smplx/torch/chumpy, so it's free to live in a fresh Python>=3.10 env:
    conda create -n retarget python=3.11 -y && conda activate retarget
    pip install dex_retargeting pin scipy h5py

Fills the right arm+hand (the hand that actually touches the object in our clips so far);
the left arm+hand is held at ConTrack's own resting pose (source/ConTrack/ConTrack/robots/
xarm_xhand_left.py: DEFAULT_XARM_QPOS, fingers at 0) -- per the earlier decision that an idle,
non-contacting hand doesn't need to be retargeted for a first pass.

Two independent least-squares fits per frame, both implemented directly with pinocchio
(via dex_retargeting's small RobotWrapper, used only for URDF loading + forward kinematics --
none of dex_retargeting's own nlopt-based Optimizer classes are reused, see "why" below) and
scipy.optimize.least_squares (numerical Jacobian, bounded by the URDF's joint limits, each
frame warm-started from the previous one's solution for a smooth trajectory):

  Step 1 (arm, "position"-type): places the wrist. The 7 arm joints of xarm_xhand_right.urdf
  are solved so that 3 points rigidly attached to the wrist match 3 target points: the GRAB
  wrist position, plus 2 synthetic points offset by 5 cm along the wrist's local x/y axes
  (constructed from GRAB's own wrist rotation, global_orient). Position alone leaves the arm's
  orientation about the wrist unconstrained (redundant DOF) -- the 2 extra points pin it down.
  (The URDF does have 2 extra fixed links right at the wrist, right_hand_ee_link and
  right_hand_back_link, but both sit at the exact same offset (0,0,-0.065) from right_hand_link,
  i.e. they coincide and add no orientation information -- confirmed by dumping the URDF's joint
  origins -- hence the synthetic points instead.)

  Step 2 (fingers, "vector"-type): the 12 finger joints of xhand_right.urdf (hand only, no arm)
  are solved so that the 5 vectors from the hand's own origin link (right_hand_link) to its 5
  fingertip links match the 5 corresponding vectors from the GRAB wrist to the GRAB fingertips.
  This never looks at absolute position, so it is completely decoupled from Step 1 -- exactly the
  "wrist is fixed, only the finger shape is fit" behaviour confirmed by reading
  dex_retargeting.optimizer.VectorOptimizer.

Why not call dex_retargeting.optimizer.PositionOptimizer/VectorOptimizer directly: their
PositionOptimizer only matches a link's own origin, not "a link's origin plus a local offset"
(needed for the synthetic orientation points), and hand-deriving an analytic Jacobian for that
case is easy to get subtly wrong. scipy.optimize.least_squares' numerical Jacobian sidesteps
that at the cost of a bit more compute -- fine at 7-12 unknowns per frame.

*** ONE UNVERIFIED ASSUMPTION (flagged, not hidden) ***
The mapping between GRAB/MANO's wrist local axes and the robot end-effector's own local axes
is NOT known to be identity. --calib-rpy lets you apply a fixed correction rotation to the two
synthetic orientation points; the default (0,0,0) assumes the two frames already line up. This
can only really be checked by looking at the retargeted motion (Isaac Sim / SAPIEN / the object
video from inspect_dataset.py) -- if the hand's palm faces the wrong way relative to the object,
adjust --calib-rpy and rerun Step 1 (Step 2 does not need to change).

Joint/link names and offsets below are read directly from ConTrack's assets/urdf/*.urdf files
(not guessed): xarm_xhand_right.urdf, xhand_right.urdf.

Example (after running compute_hand_keypoints.py in the `grab` env):
    python contrack/retarget_xarm_xhand.py \
        --h5 out/grab-s1_teapot_pour_1.h5 --assets-dir /path/to/ConTrack/assets
"""

import argparse
import os

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# xarm_xhand_left.py / xarm_xhand_right.py (ConTrack robots module)
XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]
DEFAULT_XARM_QPOS = [0.0, 0.0, 0.0, 0.22, 3.14, 1.33, 0.0]  # resting pose for the idle left arm

ORIGIN_LINK = "right_hand_link"
TIP_LINKS = {  # xhand_right.urdf fingertip links; NB "mid" not "middle" in the URDF
    "thumb": "right_hand_thumb_rota_tip", "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip", "ring": "right_hand_ring_tip", "pinky": "right_hand_pinky_tip",
}
WRIST_OFFSET_M = 0.05  # size of the 2 synthetic orientation-pinning points, in meters


def _qpos_indices(robot, joint_names):
    return np.array([robot.get_joint_index(n) for n in joint_names])


def solve_arm(robot, wrist_pos, wrist_rotmat, calib, arm_idx, hand_idx, link_id, joint_limits, x0):
    """One frame of Step 1. Returns the 7 arm joint values.

    axis0/axis1 must be FIXED reference directions, not derived from calib: calib belongs only on
    the target side (the desired world orientation is wrist_rotmat @ calib). Using calib's own
    columns as axis0/axis1 was a bug -- since the same axis then appears on both the "robot" and
    "target" side of the residual, it cancels out algebraically and the fit always converges to
    rot = wrist_rotmat regardless of calib, silently ignoring --calib-rpy entirely.
    """
    axis0 = np.array([1.0, 0.0, 0.0])
    axis1 = np.array([0.0, 1.0, 0.0])
    target_rot = wrist_rotmat @ calib
    full = np.zeros(robot.dof)
    full[hand_idx] = 0.0

    def residual(x):
        full[arm_idx] = x
        robot.compute_forward_kinematics(full)
        pose = robot.get_link_pose(link_id)
        pos, rot = pose[:3, 3], pose[:3, :3]
        p1 = pos + WRIST_OFFSET_M * (rot @ axis0)
        p2 = pos + WRIST_OFFSET_M * (rot @ axis1)
        target1 = wrist_pos + WRIST_OFFSET_M * (target_rot @ axis0)
        target2 = wrist_pos + WRIST_OFFSET_M * (target_rot @ axis1)
        return np.concatenate([pos - wrist_pos, p1 - target1, p2 - target2])

    res = least_squares(residual, x0, bounds=joint_limits.T, method="trf", xtol=1e-10, ftol=1e-10)
    return res.x


def solve_fingers(robot, wrist_pos, wrist_rotmat, calib, tips, finger_idx, origin_id, tip_ids, joint_limits, x0):
    """One frame of Step 2. Returns the 12 finger joint values.

    The isolated hand URDF sits at the world origin with identity orientation, so the vector
    it computes (tip - origin) via forward kinematics is expressed in the hand's OWN local frame.
    GRAB's wrist-to-fingertip vector is in world frame, so it must be rotated back by the inverse
    of the same rotation Step 1 used to place the hand (wrist_rotmat @ calib) before the two are
    comparable -- omitting this was a bug: Step 2's residual used to be identical no matter what
    --calib-rpy was, because calib never reached this function at all.
    """
    R_hand_world = wrist_rotmat @ calib
    target_vecs = np.stack([R_hand_world.T @ (tips[f] - wrist_pos) for f in ("thumb", "index", "middle", "ring", "pinky")])
    full = np.zeros(robot.dof)

    def residual(x):
        full[finger_idx] = x
        robot.compute_forward_kinematics(full)
        origin_pos = robot.get_link_pose(origin_id)[:3, 3]
        vecs = np.stack([robot.get_link_pose(i)[:3, 3] - origin_pos for i in tip_ids])
        return (vecs - target_vecs).ravel()

    res = least_squares(residual, x0, bounds=joint_limits.T, method="trf", xtol=1e-10, ftol=1e-10)
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True,
                    help="the file produced by grab_to_contrack.py + compute_hand_keypoints.py; qpos is filled in place")
    ap.add_argument("--assets-dir", required=True, help="ConTrack's assets/ folder (urdf/xarm_xhand_right.urdf, urdf/xhand_right.urdf)")
    ap.add_argument("--calib-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="correction rotation (deg) applied to the wrist's local axes before matching the arm; see module docstring")
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    with h5py.File(args.h5, "r+") as f:
        g = f["grab_source"]
        if "keypoints" not in g:
            raise SystemExit("no grab_source/keypoints group -- run compute_hand_keypoints.py first (in the `grab` env)")
        kp_grp = g["keypoints"]
        kp = {"wrist_pos": kp_grp["wrist_pos"][:], "wrist_rotmat": kp_grp["wrist_rotmat"][:],
              "tips": {name: kp_grp["tips"][name][:] for name in kp_grp["tips"]}}
        T = kp["wrist_pos"].shape[0]
        calib = R.from_euler("xyz", args.calib_rpy, degrees=True).as_matrix()

        arm_robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xarm_xhand_right.urdf"))
        arm_idx = _qpos_indices(arm_robot, XARM_JOINT_NAMES)
        hand_idx_in_arm_robot = _qpos_indices(arm_robot, HAND_JOINT_NAMES)
        wrist_link_id = arm_robot.get_link_index(ORIGIN_LINK)
        arm_limits = arm_robot.joint_limits[arm_idx]

        hand_robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xhand_right.urdf"))
        finger_idx = _qpos_indices(hand_robot, HAND_JOINT_NAMES)
        origin_id = hand_robot.get_link_index(ORIGIN_LINK)
        tip_ids = [hand_robot.get_link_index(TIP_LINKS[name]) for name in ("thumb", "index", "middle", "ring", "pinky")]
        finger_limits = hand_robot.joint_limits[finger_idx]

        arm_qpos = np.zeros((T, 7))
        finger_qpos = np.zeros((T, 12))
        x_arm = np.clip(np.zeros(7), arm_limits[:, 0], arm_limits[:, 1])
        x_fin = np.clip(np.zeros(12), finger_limits[:, 0], finger_limits[:, 1])
        for t in range(T):
            x_arm = solve_arm(arm_robot, kp["wrist_pos"][t], kp["wrist_rotmat"][t], calib,
                               arm_idx, hand_idx_in_arm_robot, wrist_link_id, arm_limits, x_arm)
            arm_qpos[t] = x_arm
            tips_t = {k: v[t] for k, v in kp["tips"].items()}
            x_fin = solve_fingers(hand_robot, kp["wrist_pos"][t], kp["wrist_rotmat"][t], calib,
                                   tips_t, finger_idx, origin_id, tip_ids, finger_limits, x_fin)
            finger_qpos[t] = x_fin
            if t % 100 == 0:
                print(f"frame {t}/{T}")

        qpos = np.zeros((T, 2, 19), np.float32)
        qpos[:, 0, :7] = DEFAULT_XARM_QPOS
        qpos[:, 1, :7] = arm_qpos
        qpos[:, 1, 7:] = finger_qpos

        del f["qpos"]
        f.create_dataset("qpos", data=qpos)
        f.attrs["qpos_is_placeholder"] = 0
        f.attrs["retarget_calib_rpy_deg"] = args.calib_rpy

        # ---- diagnostics over ALL frames (no Isaac Sim needed) ------------------------------
        pos_err = np.zeros(T)
        rot_err_deg = np.zeros(T)
        finger_err = np.zeros((T, 5))  # per-finger vector-matching residual, meters
        full_arm = np.zeros(arm_robot.dof)
        full_hand = np.zeros(hand_robot.dof)
        finger_names = ("thumb", "index", "middle", "ring", "pinky")
        for t in range(T):
            full_arm[arm_idx] = arm_qpos[t]
            arm_robot.compute_forward_kinematics(full_arm)
            pose = arm_robot.get_link_pose(wrist_link_id)
            pos_err[t] = np.linalg.norm(pose[:3, 3] - kp["wrist_pos"][t])
            target_rot = kp["wrist_rotmat"][t] @ calib
            rot_err_deg[t] = R.from_matrix(pose[:3, :3].T @ target_rot).magnitude() * 180 / np.pi

            full_hand[finger_idx] = finger_qpos[t]
            hand_robot.compute_forward_kinematics(full_hand)
            origin_pos = hand_robot.get_link_pose(origin_id)[:3, 3]
            R_hand_world = kp["wrist_rotmat"][t] @ calib
            for i, name in enumerate(finger_names):
                tip_pos = hand_robot.get_link_pose(tip_ids[i])[:3, 3]
                target_vec = R_hand_world.T @ (kp["tips"][name][t] - kp["wrist_pos"][t])
                finger_err[t, i] = np.linalg.norm((tip_pos - origin_pos) - target_vec)

        eps = 1e-3
        arm_sat = np.mean(np.any((arm_qpos - arm_limits[:, 0] < eps) | (arm_limits[:, 1] - arm_qpos < eps), axis=1))
        fin_sat = np.mean(np.any((finger_qpos - finger_limits[:, 0] < eps) | (finger_limits[:, 1] - finger_qpos < eps), axis=1))

        print("\n=== Step 1 (arm, position): wrist match over all frames ===")
        print(f"  position error   mean {pos_err.mean()*1000:.2f} mm   max {pos_err.max()*1000:.2f} mm")
        print(f"  orientation error mean {rot_err_deg.mean():.2f} deg  max {rot_err_deg.max():.2f} deg  (large mean here usually means --calib-rpy is wrong)")
        print(f"  frames with an arm joint at its limit: {arm_sat*100:.1f}%")
        print("=== Step 2 (fingers, vector): per-finger fit residual (mm) ===")
        for i, name in enumerate(finger_names):
            print(f"  {name:7s} mean {finger_err[:, i].mean()*1000:6.2f}  max {finger_err[:, i].max()*1000:6.2f}")
        print(f"  frames with a finger joint at its limit: {fin_sat*100:.1f}%")
    print("\nsaved qpos into", args.h5)


if __name__ == "__main__":
    main()
