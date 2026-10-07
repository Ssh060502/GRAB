"""Retarget right XArm7/XHand: stable palm pose -> arm IK -> fixed-wrist fingers.

Palm geometry requires freshly extracted MCP/palm keypoints. Robot palm axes
come from fixed MCP pivots in the URDF. Explicit offset is in human palm axes (m).
Finger targets are transformed using the actual arm FK wrist, and diagnostics
measure world-space fingertip error against the targets used for optimization.
Assumes the robot base has identity rotation in ConTrack world.
"""

import argparse
import os

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from palm_geometry import robot_palm_frame, wrist_targets

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
WRIST_OFFSET_M = 0.05  # orientation residual scale, meters per radian


def _qpos_indices(robot, joint_names):
    return np.array([robot.get_joint_index(n) for n in joint_names])


def solve_arm(robot, wrist_pos, target_rot, arm_idx, hand_idx, link_id, joint_limits, x0):
    """Match the supplied wrist position and orientation in the robot base frame."""
    full = np.zeros(robot.dof)
    full[hand_idx] = 0.0

    def residual(x):
        full[arm_idx] = x
        robot.compute_forward_kinematics(full)
        pose = robot.get_link_pose(link_id)
        pos, rot = pose[:3, 3], pose[:3, :3]
        rotation_error = R.from_matrix(target_rot.T @ rot).as_rotvec()
        return np.concatenate([pos - wrist_pos, WRIST_OFFSET_M * rotation_error])

    res = least_squares(residual, x0, bounds=joint_limits.T, method="trf", xtol=1e-10, ftol=1e-10)
    return res.x


def solve_fingers(robot, wrist_pos, wrist_rotmat, tips, finger_idx, origin_id, tip_ids, joint_limits, x0,
                   contact_targets=None):
    """Fit world targets relative to the actual fixed wrist pose.

    contact_targets, when present, are world positions, NaN for no override.
    wrist_rotmat is the actual wrist orientation from arm FK.
    """
    R_hand_world = wrist_rotmat
    target_points = np.stack([tips[f] for f in ("thumb", "index", "middle", "ring", "pinky")])
    target_vecs = (target_points - wrist_pos) @ R_hand_world
    if contact_targets is not None:
        use = np.isfinite(contact_targets).all(axis=1)
        target_vecs[use] = (contact_targets[use] - wrist_pos) @ R_hand_world
    full = np.zeros(robot.dof)

    def residual(x):
        full[finger_idx] = x
        robot.compute_forward_kinematics(full)
        origin = robot.get_link_pose(origin_id)
        vecs = np.stack([origin[:3, :3].T @ (robot.get_link_pose(i)[:3, 3] - origin[:3, 3]) for i in tip_ids])
        return (vecs - target_vecs).ravel()

    res = least_squares(residual, x0, bounds=joint_limits.T, method="trf", xtol=1e-10, ftol=1e-10)
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True,
                    help="the file produced by grab_to_contrack.py + mano_keypoints.py; qpos is filled in place")
    ap.add_argument("--assets-dir", required=True, help="ConTrack's assets/ folder (urdf/xarm_xhand_right.urdf, urdf/xhand_right.urdf)")
    ap.add_argument("--contact-priority", action="store_true",
                    help="on contact frames, use GRAB distal contact world positions as fingertip targets")
    ap.add_argument("--wrist-offset-palm", type=float, nargs=3, default=(0., 0., 0.),
                    help="robot origin minus MANO wrist in stable palm axes, meters")
    args = ap.parse_args()
    from dex_retargeting.robot_wrapper import RobotWrapper

    with h5py.File(args.h5, "r+") as f:
        g = f["grab_source"]
        if "keypoints" not in g:
            raise SystemExit("no grab_source/keypoints group -- run mano_keypoints.py first (in the `grab` env)")
        kp_grp = g["keypoints"]
        kp = {"wrist_pos": kp_grp["wrist_pos"][:],
              "tips": {name: kp_grp["tips"][name][:] for name in kp_grp["tips"]}}
        if not kp_grp.attrs.get("is_rhand", 1):
            raise SystemExit("Right-arm retargeting requires right-hand keypoints")
        T = kp["wrist_pos"].shape[0]
        if "palm_rotmat" not in kp_grp:
            raise SystemExit("Missing palm geometry: rerun mano_keypoints.py")
        hand_urdf = os.path.join(args.assets_dir, "urdf", "xhand_right.urdf")
        basis, _ = robot_palm_frame(RobotWrapper(hand_urdf), hand_urdf)
        wrist_target_pos, wrist_target_rot = wrist_targets(
            kp["wrist_pos"], kp_grp["palm_rotmat"][:], basis, args.wrist_offset_palm)
        actual_pos = np.zeros((T, 3))
        actual_rot = np.zeros((T, 3, 3))

        # ConTrack spawns xarm_xhand_right's own root (link_base) at base_translation in the world
        # (confirmed by reading xarm_xhand_env_cfg.py's init_state=...pos=base_translation), but
        # RobotWrapper/pinocchio's forward kinematics reports link poses relative to the URDF's own
        # root sitting at (0,0,0) -- i.e. relative to the robot's OWN base, not ConTrack world. Step
        # 1 must therefore solve in that base-relative frame: subtract base_translation from the
        # wrist target before matching it against arm FK output. Step 2 does not need this: its
        # vectors (tip - wrist) are unaffected by adding/subtracting the same constant to both ends.
        right_base = f["base_translation"][1]  # is_rhand = [0, 1] -> index 1 is the right hand
        wrist_pos_arm_frame = wrist_target_pos - right_base

        # ---- contact-priority targets for Step 2 (see solve_fingers' contact_targets doc) --------
        FINGERS = ("thumb", "index", "middle", "ring", "pinky")
        if args.contact_priority:
            seg_names = [s.decode() for s in f["contacts/segment_names"][:]]
            is_contact = f["contacts/0/is_contact"][:]      # (T, 2, 15)
            contact_pts = f["contacts/0/points"][:]          # (T, 2, 15, 3), object-local frame
            obj_t = f["object_tracks/0/translations"][:]
            obj_R = R.from_quat(f["object_tracks/0/orientations_xyzw"][:]).as_matrix()

            contact_targets_all = np.full((T, 5, 3), np.nan)
            for i, name in enumerate(FINGERS):
                seg = seg_names.index(f"{name}_distal")  # the segment closest to what our tip_ids target
                touching = is_contact[:, 1, seg].astype(bool)  # right hand
                if not touching.any():
                    continue
                world_pt = np.einsum("tij,tj->ti", obj_R[touching], contact_pts[touching, 1, seg]) + obj_t[touching]
                contact_targets_all[touching, i] = world_pt
            n_used = int(np.isfinite(contact_targets_all).all(axis=2).sum())
            print(f"contact-priority: {n_used} (frame, finger) pairs will target GRAB's own contact point "
                  f"instead of the plain fingertip vector")
        else:
            contact_targets_all = None
            print("contact-priority off (default): every finger uses the plain MANO wrist-to-fingertip vector")

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
            x_arm = solve_arm(arm_robot, wrist_pos_arm_frame[t], wrist_target_rot[t],
                               arm_idx, hand_idx_in_arm_robot, wrist_link_id, arm_limits, x_arm)
            arm_qpos[t] = x_arm
            arm_full = np.zeros(arm_robot.dof)
            arm_full[arm_idx] = x_arm
            arm_robot.compute_forward_kinematics(arm_full)
            achieved = arm_robot.get_link_pose(wrist_link_id)
            actual_pos[t] = achieved[:3, 3] + right_base
            actual_rot[t] = achieved[:3, :3]
            tips_t = {k: v[t] for k, v in kp["tips"].items()}
            ct_t = contact_targets_all[t] if contact_targets_all is not None else None
            x_fin = solve_fingers(hand_robot, actual_pos[t], actual_rot[t],
                                   tips_t, finger_idx, origin_id, tip_ids, finger_limits, x_fin,
                                   contact_targets=ct_t)
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
        # Remove obsolete metadata when rewriting an older output file.
        for obsolete in ("retarget_calib_rpy_deg", "retarget_wrist_mode"):
            if obsolete in f.attrs:
                del f.attrs[obsolete]
        f.attrs["retarget_method"] = "wrist_mcp_palm"
        f.attrs["retarget_wrist_offset_palm_m"] = args.wrist_offset_palm
        if "grab_source/wrist_targets" in f:
            del f["grab_source/wrist_targets"]
        target_group = f.create_group("grab_source/wrist_targets")
        target_group.attrs["method"] = "wrist_mcp_palm"
        target_group.attrs["wrist_offset_palm_m"] = args.wrist_offset_palm
        target_group.create_dataset("robot_palm_basis", data=basis)
        for name, arr in (("positions", wrist_target_pos), ("rotations", wrist_target_rot),
                          ("actual_positions", actual_pos), ("actual_rotations", actual_rot)):
            target_group.create_dataset(name, data=arr)

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
            pos_err[t] = np.linalg.norm(pose[:3, 3] - wrist_pos_arm_frame[t])
            target_rot = wrist_target_rot[t]
            rot_err_deg[t] = R.from_matrix(pose[:3, :3].T @ target_rot).magnitude() * 180 / np.pi

            full_hand[finger_idx] = finger_qpos[t]
            hand_robot.compute_forward_kinematics(full_hand)
            origin = hand_robot.get_link_pose(origin_id)
            for i, name in enumerate(finger_names):
                tip_pos = hand_robot.get_link_pose(tip_ids[i])[:3, 3]
                tip_local = origin[:3, :3].T @ (tip_pos - origin[:3, 3])
                tip_world = actual_pos[t] + actual_rot[t] @ tip_local
                target = kp["tips"][name][t]
                if contact_targets_all is not None and np.isfinite(contact_targets_all[t, i]).all():
                    target = contact_targets_all[t, i]
                finger_err[t, i] = np.linalg.norm(tip_world - target)

        eps = 1e-3
        arm_sat = np.mean(np.any((arm_qpos - arm_limits[:, 0] < eps) | (arm_limits[:, 1] - arm_qpos < eps), axis=1))
        fin_sat = np.mean(np.any((finger_qpos - finger_limits[:, 0] < eps) | (finger_limits[:, 1] - finger_qpos < eps), axis=1))

        # ---- sampled reach diagnostic (not a rigorous workspace bound) ----
        # For each finger, bracket the robot's own achievable tip-to-origin distance by evaluating
        # that finger's 2 joints at {lower limit, 0, upper limit} (other joints held at 0), and
        # compare against the human vector length actually being asked for (mean/min/max over the
        # whole clip). This sample does not prove feasibility or infeasibility.
        print("\n=== scale check: human wrist-to-fingertip distance vs robot's sampled reach (mm) ===")
        import itertools
        finger_joints = {
            "thumb": ("right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2"),
            "index": ("right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2"),
            "middle": ("right_hand_mid_joint1", "right_hand_mid_joint2"),
            "ring": ("right_hand_ring_joint1", "right_hand_ring_joint2"),
            "pinky": ("right_hand_pinky_joint1", "right_hand_pinky_joint2"),
        }
        full_probe = np.zeros(hand_robot.dof)
        for i, name in enumerate(finger_names):
            human_len = np.linalg.norm(kp["tips"][name] - kp["wrist_pos"], axis=1) * 1000
            js = [hand_robot.get_joint_index(n) for n in finger_joints[name]]
            bounds = [hand_robot.joint_limits[j] for j in js]  # every joint of THIS finger, none left fixed at 0
            reach = []
            for combo in itertools.product(*[(lo, 0.0, hi) for lo, hi in bounds]):
                full_probe[:] = 0.0
                for j, v in zip(js, combo):
                    full_probe[j] = v
                hand_robot.compute_forward_kinematics(full_probe)
                reach.append(np.linalg.norm(hand_robot.get_link_pose(tip_ids[i])[:3, 3] - hand_robot.get_link_pose(origin_id)[:3, 3]))
            reach = np.array(reach) * 1000
            print(f"  {name:7s} human [{human_len.min():6.1f}, {human_len.mean():6.1f}, {human_len.max():6.1f}]"
                  f"   robot reach bracket [{reach.min():6.1f}, {reach.max():6.1f}]")

        print("\n=== Step 1 (arm, position): wrist match over all frames ===")
        print(f"  position error   mean {pos_err.mean()*1000:.2f} mm   max {pos_err.max()*1000:.2f} mm")
        print(f"  orientation error mean {rot_err_deg.mean():.2f} deg  max {rot_err_deg.max():.2f} deg  (IK tracking error; does not validate anatomical calibration)")
        print(f"  frames with an arm joint at its limit: {arm_sat*100:.1f}%")
        print("=== Step 2 (fingers, vector): per-finger WORLD position residual (mm), ALL frames (mixes plain-vector and contact-point targets) ===")
        for i, name in enumerate(finger_names):
            print(f"  {name:7s} mean {finger_err[:, i].mean()*1000:6.2f}  max {finger_err[:, i].max()*1000:6.2f}")
        print(f"  frames with a finger joint at its limit: {fin_sat*100:.1f}%")

        if contact_targets_all is not None:
            print("=== Step 2: residual on ONLY the contact-priority frames (mm) -- the number that actually matters for grasping ===")
            for i, name in enumerate(finger_names):
                mask = np.isfinite(contact_targets_all[:, i]).all(axis=1)
                if not mask.any():
                    print(f"  {name:7s} (no contact-priority frames for this finger)")
                    continue
                e = finger_err[mask, i] * 1000
                print(f"  {name:7s} n={int(mask.sum()):4d}  mean {e.mean():6.2f}  max {e.max():6.2f}  frac<10mm {(e<10).mean()*100:5.1f}%")
    print("\nsaved qpos into", args.h5)


if __name__ == "__main__":
    main()
