"""Test whether MANO's wrist joint and the URDF's right_hand_link correspond to the SAME
anatomical reference point, or whether there's a fixed translation offset between them that no
amount of --calib-rpy (a pure rotation) could ever fix.

Why this matters: the scale check in retarget_xarm_xhand.py showed thumb/index/middle's human
wrist-to-fingertip distances comfortably inside the robot's own physical reach bracket, but
ring/pinky's human distances falling AT OR BELOW the robot's own minimum reach (i.e. even fully
curled, the robot's ring/pinky can't get as close to right_hand_link as GRAB's data is asking
for). A fixed offset between the two models' "wrist origin" affects SHORT fingers proportionally
far more than long ones (the same 1-2cm offset is a small fraction of a 12cm thumb reach, but a
huge fraction of a 6cm pinky reach) -- which is exactly the pattern observed. A pure proportional
size difference between the two hands would instead affect all 5 fingers roughly equally in
relative terms, which is NOT what was observed. This script checks the offset hypothesis directly
instead of continuing to guess from that indirect pattern.

Method: classic Procrustes/Kabsch registration WITH translation (and optionally scale), fit
between two REAL point sets (not normalized direction vectors, actual mm positions) --
  Robot side: the 4 fingertip positions (index/middle/ring/pinky) relative to right_hand_link,
  at the robot's own zero pose (fingers straight).
  Human side: the SAME 4 fingertips' positions relative to the MANO wrist, averaged over the
  pre-contact frames (same frames estimate_calib.py uses), expressed in the wrist's own local
  frame (so it's directly comparable to the robot side, which is also in its own local frame).
Both point sets describe "where do open-hand fingertips sit relative to the wrist", in their own
hand's local coordinate convention -- if the two models' wrist origins coincide, the translation
term `t` should come out near zero; if it's a real, sizeable, consistent offset, that is the
answer, independent of (and not fixable by) --calib-rpy.

Runs in the `retarget` env (needs pinocchio for the robot-side FK, same as estimate_calib.py).

Example:
    python contrack/check_wrist_offset.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir ~/ConTrack/assets
"""

import argparse
import os

import h5py
import numpy as np

ORIGIN_LINK = "right_hand_link"
TIP_LINKS = {
    "thumb": "right_hand_thumb_rota_tip", "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip", "ring": "right_hand_ring_tip", "pinky": "right_hand_pinky_tip",
}
FIT_FINGERS = ("index", "middle", "ring", "pinky")  # same subset estimate_calib.py's rotation fit uses
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def procrustes_with_translation_and_scale(P, Q):
    """Find scale s, rotation R, translation t minimising sum ||s*R@P_i + t - Q_i||^2.
    Standard Umeyama algorithm (Kabsch + centering + optional scale)."""
    p_mean, q_mean = P.mean(0), Q.mean(0)
    Pc, Qc = P - p_mean, Q - q_mean
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    var_p = (Pc ** 2).sum() / len(P)
    s = (S * np.array([1, 1, d])).sum() / (var_p * len(P))
    t = q_mean - s * R @ p_mean
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    # ---- robot side: REAL (not normalized) fingertip positions relative to right_hand_link ----
    robot = RobotWrapper(os.path.join(args.assets_dir, "urdf", "xhand_right.urdf"))
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    origin_pos = robot.get_link_pose(robot.get_link_index(ORIGIN_LINK))[:3, 3]
    robot_pts_all = {f: robot.get_link_pose(robot.get_link_index(TIP_LINKS[f]))[:3, 3] - origin_pos for f in FINGERS}
    robot_pts = np.stack([robot_pts_all[f] for f in FIT_FINGERS])  # meters, XHand-local frame

    # ---- human side: REAL fingertip positions relative to wrist, pre-contact frames, MANO-local ----
    with h5py.File(args.h5, "r") as f:
        k = f["grab_source/keypoints"]
        wrist_pos, wrist_rotmat = k["wrist_pos"][:], k["wrist_rotmat"][:]
        tips = {name: k["tips"][name][:] for name in FINGERS}
        T = wrist_pos.shape[0]
        is_contact = f["contacts/0/is_contact"][:]
        contact_frames = np.flatnonzero(is_contact[:, 1, :].any(axis=1))

    end = int(contact_frames[0]) if len(contact_frames) and contact_frames[0] >= 5 else T
    print(f"using frames [0:{end}] (before first recorded contact) out of {T} total for the human-side fit")

    human_pts_all = {}
    for name in FINGERS:
        local = np.einsum("tji,tj->ti", wrist_rotmat[:end], tips[name][:end] - wrist_pos[:end])
        human_pts_all[name] = local.mean(0)  # meters, MANO-local frame
    human_pts = np.stack([human_pts_all[f] for f in FIT_FINGERS])

    print("\nraw per-finger distance from wrist/origin (mm), robot zero-pose vs human (pre-contact avg):")
    for f in FINGERS:
        rd = np.linalg.norm(robot_pts_all[f]) * 1000
        hd = np.linalg.norm(human_pts_all[f]) * 1000
        print(f"  {f:7s}  robot {rd:6.1f}   human {hd:6.1f}   ratio human/robot {hd/rd:.2f}")

    # ---- fit WITHOUT translation (rotation + scale only, i.e. assume origins coincide) ----
    # NOT the same as procrustes_with_translation_and_scale with t discarded -- that function
    # centers both point sets first (which implicitly assumes/absorbs a translation), so skipping
    # its `t` output would silently still use a translated fit. Re-derive from the UNCENTERED
    # points instead, which is the correct way to force "no translation allowed" into the fit.
    H = robot_pts.T @ human_pts
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_only = Vt.T @ np.diag([1, 1, d]) @ U.T
    var_p = (robot_pts ** 2).sum() / len(robot_pts)
    s_only = (S * np.array([1, 1, d])).sum() / (var_p * len(robot_pts))
    res_rot_scale_only = np.sqrt(np.mean(np.sum((s_only * (robot_pts @ R_only.T) - human_pts) ** 2, axis=1))) * 1000

    # ---- full fit: rotation + translation + scale ----
    s, R, t = procrustes_with_translation_and_scale(robot_pts, human_pts)
    res_full = np.sqrt(np.mean(np.sum((s * (robot_pts @ R.T) + t - human_pts) ** 2, axis=1))) * 1000

    print(f"\n=== fit WITHOUT translation (rotation + scale only, origin assumed to coincide) ===")
    print(f"  scale factor: {s_only:.3f}")
    print(f"  RMS residual: {res_rot_scale_only:.1f} mm")

    print(f"\n=== fit WITH translation (rotation + scale + offset) ===")
    print(f"  scale factor: {s:.3f}")
    print(f"  translation t (robot-local frame, mm): {(t*1000).round(1).tolist()}")
    print(f"  |t| = {np.linalg.norm(t)*1000:.1f} mm")
    print(f"  RMS residual: {res_full:.1f} mm")

    print(f"\nresidual dropped from {res_rot_scale_only:.1f}mm to {res_full:.1f}mm by adding a translation term.")
    if np.linalg.norm(t) * 1000 > 10 and res_full < res_rot_scale_only * 0.5:
        print("-> |t| is sizeable (>1cm) AND adding it substantially improved the fit: this looks like a")
        print("   real reference-point offset between MANO's wrist and right_hand_link, not just noise.")
    else:
        print("-> translation isn't doing much work here -- the reference points likely already coincide")
        print("   reasonably well; the ring/pinky shortfall is probably a genuine size/proportion")
        print("   difference between the two hands, not a mismatched origin.")


if __name__ == "__main__":
    main()
