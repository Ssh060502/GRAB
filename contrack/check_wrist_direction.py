"""Directly test whether --calib-rpy points the robot's hand TOWARD or AWAY from the object,
instead of inferring it from a rendered video. Answers the specific question raised after
estimate_calib.py's thumb-excluded fit: did excluding the thumb (the one finger that breaks the
near-planar symmetry of index/middle/ring/pinky) let the fit converge to a 180-degrees-flipped
rotation that still looks like a good fit on paper (small per-finger angle errors) but actually
points the palm the wrong way?

Method: for a handful of frames, compute two dot products against the same reference direction
(wrist -> object center, normalized) and compare them side by side:
  1. human_dot:  GRAB's own MANO hand's "forward" direction (mean reach direction of
     index/middle/ring/pinky from the wrist, computed and kept in WORLD frame throughout -- no
     calib involved, and no local-frame detour). This MUST be meaningfully positive: it's a plain
     physical fact that a hand actually grasping an object reaches toward it. Trusted baseline.
  2. robot_dot:  the SAME local "forward" direction (this time the robot's own, from
     estimate_calib.py's robot_dirs, averaged over the same 4 fingers), rotated into world by
     wrist_rotmat @ calib -- i.e. exactly the orientation retarget_xarm_xhand.py's Step 1 asks
     the arm to place the wrist at.

If robot_dot tracks human_dot (both positive, comparable magnitude) at frames where the hand is
near the object, calib is pointing the right way. If robot_dot is negative while human_dot is
positive, the fitted rotation is flipped (even though estimate_calib.py's own per-finger angle
errors looked small) -- rotate calib by ~180 degrees about the axis this script prints and retry.

Runs in the `retarget` env (needs pinocchio for the robot-side FK, same as estimate_calib.py).

Example:
    python contrack/check_wrist_direction.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir ~/ConTrack/assets --calib-rpy 94.59 -55.28 -58.61
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
FIT_FINGERS = ("index", "middle", "ring", "pinky")  # same subset estimate_calib.py fits on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--calib-rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--num-frames", type=int, default=8, help="frames sampled evenly across the whole clip")
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
    robot_thumb_local = robot.get_link_pose(robot.get_link_index(TIP_LINKS["thumb"]))[:3, 3] - origin_pos
    robot_thumb_local /= np.linalg.norm(robot_thumb_local)

    with h5py.File(args.h5, "r") as f:
        k = f["grab_source/keypoints"]
        wrist_pos, wrist_rotmat = k["wrist_pos"][:], k["wrist_rotmat"][:]
        tips = {name: k["tips"][name][:] for name in FIT_FINGERS}
        thumb_tip = k["tips"]["thumb"][:]
        obj_t = f["object_tracks/0/translations"][:]  # ConTrack world frame, same as wrist_pos -- no
        # base_translation needed here, we're only comparing two world-frame points/directions,
        # never feeding anything into the arm's own base-relative FK like retarget_xarm_xhand.py does.

    T = wrist_pos.shape[0]
    calib = R.from_euler("xyz", args.calib_rpy, degrees=True).as_matrix()

    # human_forward_world: MANO hand's own "reach direction", kept in WORLD frame throughout (no
    # local-frame detour) -- directly comparable to `ref` (also world frame) below. An earlier
    # version of this script rotated the fingertip vectors into the wrist's local frame first
    # (wrist_rotmat.T @ ...) and then dotted that LOCAL-frame vector against `ref`, a WORLD-frame
    # vector -- comparing vectors expressed in two different frames, which is not meaningful (this
    # is exactly the local-vs-world mixing mistake flagged earlier in this conversation, just
    # reintroduced in a new script). Fixed by never leaving world frame for this comparison.
    human_dirs = []
    for name in FIT_FINGERS:
        v = tips[name] - wrist_pos
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        human_dirs.append(v)
    human_forward_world = np.mean(human_dirs, axis=0)  # (T,3), world frame, per-frame (finger curl moves this a bit)
    human_forward_world /= np.linalg.norm(human_forward_world, axis=1, keepdims=True)
    human_thumb_world = thumb_tip - wrist_pos
    human_thumb_world /= np.linalg.norm(human_thumb_world, axis=1, keepdims=True)

    def signed_angle_around_axis(a, b, axis):
        """Signed angle (deg) from a to b, measured in the plane perpendicular to axis -- i.e. the
        ROLL between a and b, with any component of their disagreement ALONG axis projected out
        first. Positive/negative sign follows the right-hand rule around axis."""
        a = a - (a @ axis) * axis
        a /= np.linalg.norm(a)
        b = b - (b @ axis) * axis
        b /= np.linalg.norm(b)
        s = np.cross(a, b) @ axis
        c = a @ b
        return np.degrees(np.arctan2(s, c))

    sample = np.linspace(0, T - 1, args.num_frames).astype(int)
    print(f"{'frame':>6} {'dist(mm)':>9} {'human_dot':>10} {'robot_dot':>10}  agree?")
    for t in sample:
        ref = obj_t[t] - wrist_pos[t]
        dist = np.linalg.norm(ref)
        ref /= dist
        human_dot = float(human_forward_world[t] @ ref)
        target_rot = wrist_rotmat[t] @ calib
        robot_world_forward = target_rot @ robot_forward_local
        robot_dot = float(robot_world_forward @ ref)
        agree = "OK" if (human_dot > 0) == (robot_dot > 0) else "*** FLIPPED ***"
        print(f"{t:6d} {dist*1000:9.1f} {human_dot:10.3f} {robot_dot:10.3f}  {agree}")

    print("\nhuman_dot should be clearly positive whenever the hand is actually near/reaching the")
    print("object (small dist). If robot_dot is negative there while human_dot is positive, the")
    print("fitted hand orientation is pointing away from the object -- i.e. flipped relative to")
    print("the correct one, consistent with 'the hand looks exactly backwards' in the video.")

    # ---- roll check: the ONE rotational DOF the dot-product check above never tested -----------
    # human_dot/robot_dot above only validate ONE axis of the rotation (how well the "reach
    # direction" lines up). A full 3D rotation has 2 more DOF beyond that: rotating the whole hand
    # around that same reach-direction axis (roll -- "palm up" vs "palm down" vs anything in
    # between) changes nothing about human_dot/robot_dot but completely changes which way
    # structures rigidly attached near the wrist (e.g. the fixed ee_link/back_link connector
    # visible in the PyBullet render) end up pointing. The thumb is the only finger NOT in
    # estimate_calib.py's fit (excluded for being anatomically different, see that script), and
    # it is also the only one of the 5 that sits off the other 4 fingers' common plane -- which is
    # exactly what makes it useful here: project both the human and robot thumb directions onto
    # the plane perpendicular to the (already-validated) reach axis, and compare the angle between
    # them WITHIN that plane. That isolates roll specifically, instead of mixing it back together
    # with "thumb's CMC joint just bends differently" the way the raw 43.8 deg angle error from
    # estimate_calib.py's fit-quality printout does.
    print("\n=== roll check (uses thumb, excluded from the fit, as an off-plane reference) ===")
    print(f"{'frame':>6} {'roll error (deg)':>18}")
    rolls = []
    for t in sample:
        target_rot = wrist_rotmat[t] @ calib
        robot_world_thumb = target_rot @ robot_thumb_local
        roll = signed_angle_around_axis(human_thumb_world[t], robot_world_thumb, human_forward_world[t])
        rolls.append(roll)
        print(f"{t:6d} {roll:18.1f}")
    print(f"\nmean roll error: {np.mean(rolls):.1f} deg (std {np.std(rolls):.1f} deg across sampled frames)")
    print("if this is consistently large (and consistent in SIGN across frames, not scattered),")
    print("that's a real roll offset in --calib-rpy -- rotate calib by roughly this many degrees")
    print("around the reach-direction axis and recheck. If it scatters with no consistent sign,")
    print("it's more likely just the thumb's own CMC joint behaving differently frame to frame,")
    print("not a fixed calib error.")


if __name__ == "__main__":
    main()
