"""Right FANUC CRX-5iA + Sharpa: palm wrist IK, then 22 finger joints at the actual wrist.

Same method as retarget_rby1a_sharpa.py (palm-basis wrist target, least-squares arm
IK, tip + PIP/IP + DIP finger fit). What differs is the robot:

  * input : a Sharpa H5 that already has grab_source/keypoints
            (grab_to_contrack_sharpa.py, then sharpa_keypoints.py). Its qpos is ignored.
  * output: a NEW H5 (--out); the input is never modified.
  * the whole scene (object, keypoints) is first moved by one rigid scene transform
    into the CRX world, whose origin is the LEFT arm base.
  * each arm has its own base pose (xyz + roll/pitch/yaw); the right base is not identity.
  * qpos is (T, 2, 28): J1..J6 then 22 Sharpa joints, left/right order.
  * --time-scale slows the clip by dividing the stored fps. Human motion at
    original speed can exceed the CRX joint velocity limits.

Only the right hand is retargeted, as in the RB-Y1A route. The left arm holds a
rest pose with open fingers. No collision avoidance or physics validation.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from crx_sharpa_geometry import (CrxSharpaKinematics, FINGERS, ARM_DOF, HAND_DOF, SIDE_DOF,
                                 URDF_NAMES, REST_ARM_DEG, pose_matrix, scene_matrix)

# Scene placement used for the reviewed hammer replay: object near (0.65, -0.40),
# operation surface 0.175 m below the left base.
DEFAULT_SCENE = (0.15, -0.40, -0.175, 0.0)
DEFAULT_LEFT_BASE = (0., 0., 0., 0., 0., 0.)
# Current dual_crx_control xacro default (urdf/dual_crx.urdf.xacro on GitHub). A tape
# measurement of the base spacing gave about 1.0 m, consistent with this value and
# not with the older 0.9116 m still loaded on the lab PC as of its Oct 9 checkout.
# It is a calibration result, not re-verified here: pass --right-base if it changes.
DEFAULT_RIGHT_BASE = (0.03340744, -0.99546257, 0.01093060, 0.01335405, -0.02065696, -1.54875984)
PREVIOUS_RIGHT_BASE = (0.03058936, -0.91156264, 0.00873476, 0.00994493, -0.01029918, -1.54694055)
# First-frame / recovery seeds in ROS/URDF degrees. The lab rest pose comes first
# so the chosen IK branch matches how the real arm is normally configured.
ARM_SEEDS_DEG = (REST_ARM_DEG["right"], (-20., -60., 215., -70., -95., -30.),
                 (0., 30., -30., 0., 60., 0.), (-90., 30., -30., 0., 60., 0.),
                 (90., 30., -30., 0., 60., 0.), (180., 30., -30., 0., 60., 0.))
GOOD_POSITION_M, GOOD_ROTATION_RAD = 1e-3, np.deg2rad(.5)


def arm_errors(model, q, target_pos, target_rot):
    pose = model.palm_pose(q)
    return (np.linalg.norm(pose[:3, 3] - target_pos),
            Rotation.from_matrix(target_rot.T @ pose[:3, :3]).magnitude())


def solve_arm(model, target_pos, target_rot, seeds, orientation_scale=.05):
    """Targets are in the arm base frame. Seeds are tried in order.

    The first seed is the continuity seed (previous frame). Later seeds are only
    tried while tracking is poor; the first one that tracks well wins, otherwise
    the lowest residual. A 6-axis arm has no elbow redundancy to trade off.
    """
    bounds = model.limits(model.arm_names)
    def residual(q):
        pose = model.palm_pose(q)
        return np.r_[pose[:3, 3] - target_pos, orientation_scale *
                     Rotation.from_matrix(target_rot.T @ pose[:3, :3]).as_rotvec()]
    best = None
    for seed in seeds:
        start = np.clip(seed, bounds[:, 0] + 1e-6, bounds[:, 1] - 1e-6)
        result = least_squares(residual, start, bounds=bounds.T,
                               max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
        position, rotation = arm_errors(model, result.x, target_pos, target_rot)
        if position < GOOD_POSITION_M and rotation < GOOD_ROTATION_RAD:
            return result
    return best


def solve_fingers(model, targets, previous, intermediate_weight=.35, regularization=.001):
    """Targets are palm-local positions; regularization has units m/rad."""
    bounds = model.limits(model.hand_names)
    def residual(q):
        tip, pip, dip = model.hand_points(q)
        return np.r_[(tip-targets["tips"]).ravel(),
                     intermediate_weight*(pip-targets["pip"]).ravel(),
                     intermediate_weight*(dip-targets["dip"]).ravel(),
                     regularization*(q-previous)]
    # Exact zero starts put many flexion joints on their lower bound.
    initial = np.clip(previous, bounds[:, 0]+.01, bounds[:, 1]-.01)
    result = least_squares(residual, initial, bounds=bounds.T,
                           max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if np.max(np.linalg.norm(result.fun[:15].reshape(5, 3), axis=1)) > .005:
        candidate = least_squares(residual, bounds.mean(axis=1), bounds=bounds.T,
                                  max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if np.linalg.norm(candidate.fun) < np.linalg.norm(result.fun):
            result = candidate
    return result


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def retarget_right(model, arrays, right_base, wrist_offset=(0., 0., 0.), orientation_scale=.05,
                   intermediate_weight=.35, regularization=.001, progress=None):
    """arrays: wrist (T,3), palm (T,3,3), tips (T,5,3), pip (T,5,3), dip (T,4,3) in the CRX world.

    Returns right-side qpos (T,28) plus world-frame targets, achieved poses and errors.
    """
    basis, _ = model.palm_basis()
    target_pos = arrays["wrist"] + np.einsum("tij,j->ti", arrays["palm"], np.asarray(wrist_offset, float))
    target_rot = arrays["palm"] @ basis.T
    base_inv = np.linalg.inv(right_base)
    count = len(target_pos)
    qpos = np.zeros((count, SIDE_DOF))
    actual_pos, actual_rot = np.zeros((count, 3)), np.zeros((count, 3, 3))
    pos_err, rot_err, finger_err = np.zeros(count), np.zeros(count), np.zeros((count, 5))
    successes = np.zeros((count, 2), dtype=np.uint8)
    seeds = [np.deg2rad(s) for s in ARM_SEEDS_DEG]
    arm_q, hand_q = None, np.zeros(HAND_DOF)
    for t in range(count):
        local_pos = base_inv[:3, :3] @ target_pos[t] + base_inv[:3, 3]
        local_rot = base_inv[:3, :3] @ target_rot[t]
        arm = solve_arm(model, local_pos, local_rot, seeds if arm_q is None else [arm_q] + seeds,
                        orientation_scale)
        arm_q = arm.x
        achieved = right_base @ model.palm_pose(arm_q)
        actual_pos[t], actual_rot[t] = achieved[:3, 3], achieved[:3, :3]
        local = {name: (arrays[name][t]-actual_pos[t]) @ actual_rot[t] for name in ("tips", "pip", "dip")}
        fingers = solve_fingers(model, local, hand_q, intermediate_weight, regularization)
        hand_q = fingers.x
        if not np.isfinite(np.r_[arm_q, hand_q]).all():
            raise RuntimeError(f"Nonfinite optimization result at frame {t}")
        qpos[t] = np.r_[arm_q, hand_q]
        tip_world = actual_pos[t] + model.hand_points(hand_q)[0] @ actual_rot[t].T
        finger_err[t] = np.linalg.norm(tip_world-arrays["tips"][t], axis=1)
        pos_err[t] = np.linalg.norm(actual_pos[t]-target_pos[t])
        rot_err[t] = Rotation.from_matrix(target_rot[t].T @ actual_rot[t]).magnitude()*180/np.pi
        successes[t] = [arm.success, fingers.success]
        if progress is not None and t % 50 == 0:
            progress(f"frame {t}/{count}: wrist {pos_err[t]*1000:.2f} mm / {rot_err[t]:.2f} deg")
    return dict(qpos=qpos, target_pos=target_pos, target_rot=target_rot, actual_pos=actual_pos,
                actual_rot=actual_rot, pos_err=pos_err, rot_err=rot_err, finger_err=finger_err,
                successes=successes, basis=basis)


def read_keypoints(f):
    if not np.array_equal(f["is_rhand"][:], [0, 1]):
        raise ValueError("Expected a two-side (left, right) Sharpa H5")
    if "grab_source/keypoints" not in f:
        raise ValueError("Missing grab_source/keypoints: run sharpa_keypoints.py first")
    k = f["grab_source/keypoints"]
    if not k.attrs.get("is_rhand", False):
        raise ValueError("Right-hand keypoints with explicit is_rhand are required")
    count = f["qpos"].shape[0]
    if count == 0:
        raise ValueError("Empty sequence")
    arrays = {"wrist": k["wrist_pos"][:].astype(float), "palm": k["palm_rotmat"][:].astype(float)}
    for group, fingers in (("tips", FINGERS), ("pip", FINGERS), ("dip", FINGERS[1:])):
        if group not in k:
            raise ValueError("Missing intermediate joints: run sharpa_keypoints.py first")
        arrays[group] = np.stack([k[group][n][:] for n in fingers], axis=1).astype(float)
    for name, array in arrays.items():
        shape = (count, 3, 3) if name == "palm" else ((count, 3) if name == "wrist" else
                (count, 4 if name == "dip" else 5, 3))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} targets: expected {shape}")
    if not np.allclose(arrays["palm"].transpose(0, 2, 1) @ arrays["palm"], np.eye(3), atol=1e-4) or not np.allclose(
            np.linalg.det(arrays["palm"]), 1., atol=1e-4):
        raise ValueError("Invalid palm rotations")
    return arrays


def move_scene(f, scene):
    """Rewrite every world-frame quantity of an opened (r+) copy into the CRX world.

    Contact points are object-local vertex means and are left untouched.
    """
    rotation = scene[:3, :3]
    for group in f["object_tracks"].values():
        translations = transform_points(scene, group["translations"][:].astype(float))
        quat = (Rotation.from_matrix(rotation) * Rotation.from_quat(group["orientations_xyzw"][:])).as_quat()
        flip = np.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1., 1.)
        quat = quat * np.concatenate([[1.], np.cumprod(flip)])[:, None]
        group["translations"][:] = translations.astype(np.float32)
        group["orientations_xyzw"][:] = quat.astype(np.float32)
    source = f["grab_source"]
    rm, tm = source["Rm"][:].astype(float), source["tm"][:].astype(float)
    source["Rm"][:] = rotation @ rm
    source["tm"][:] = rotation @ tm + scene[:3, 3]
    k = source["keypoints"]
    k["wrist_pos"][:] = transform_points(scene, k["wrist_pos"][:].astype(float))
    for name in ("wrist_rotmat", "palm_rotmat"):
        if name in k:
            k[name][:] = rotation @ k[name][:].astype(float)
    for name in ("mcp", "tips", "pip", "dip"):
        if name in k:
            for dataset in k[name].values():
                dataset[:] = transform_points(scene, dataset[:].astype(float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True, help="Sharpa H5 with grab_source/keypoints; read only")
    ap.add_argument("--out", required=True, help="New CRX+Sharpa H5")
    ap.add_argument("--right-urdf", required=True,
                    help="Single-arm right model; for training use ConTrack assets/urdf/crx_sharpa_right.urdf")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--scene-transform", type=float, nargs=4, default=DEFAULT_SCENE,
                    metavar=("X", "Y", "Z", "YAW_DEG"), help="Source world -> CRX world (left base origin)")
    ap.add_argument("--left-base", type=float, nargs=6, default=DEFAULT_LEFT_BASE,
                    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    ap.add_argument("--right-base", type=float, nargs=6, default=DEFAULT_RIGHT_BASE,
                    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    help="Meters/radians. Default is the current dual_crx_control calibration")
    ap.add_argument("--left-rest-deg", type=float, nargs=6, default=REST_ARM_DEG["left"],
                    help="Left arm J1..J6 in ROS/URDF degrees; left fingers stay at zero")
    ap.add_argument("--time-scale", type=float, default=1.,
                    help="Duration multiplier >= 1. Frames are unchanged; the stored fps is divided")
    ap.add_argument("--wrist-offset-palm", type=float, nargs=3, default=(0., 0., 0.))
    ap.add_argument("--orientation-scale", type=float, default=.05, help="arm orientation weight, m/rad")
    ap.add_argument("--intermediate-weight", type=float, default=.35)
    ap.add_argument("--regularization", type=float, default=.001, help="previous finger pose weight, m/rad")
    args = ap.parse_args()
    numbers = np.r_[args.scene_transform, args.left_base, args.right_base, args.left_rest_deg,
                    args.wrist_offset_palm, args.orientation_scale, args.intermediate_weight, args.regularization]
    if not np.isfinite(args.time_scale) or args.time_scale < 1.:
        ap.error("--time-scale must be finite and >= 1")
    if (not np.isfinite(numbers).all() or args.orientation_scale <= 0
            or args.intermediate_weight < 0 or args.regularization < 0):
        ap.error("All values must be finite; weights nonnegative, orientation scale positive")
    source, out = Path(args.h5).resolve(), Path(args.out).resolve()
    if source == out:
        ap.error("--out must differ from --h5; the input is never modified")
    if out.exists() and not args.overwrite:
        ap.error(f"{out} exists; pass --overwrite or choose a new name")
    import h5py
    model = CrxSharpaKinematics(args.right_urdf, "right")
    scene = scene_matrix(*args.scene_transform[:3], np.deg2rad(args.scene_transform[3]))
    bases = np.stack([pose_matrix(args.left_base), pose_matrix(args.right_base)])
    left_rest = np.deg2rad(args.left_rest_deg)
    arm_bounds = model.limits(model.arm_names)  # both CRX-5iA arms share joint limits
    if np.any(left_rest < arm_bounds[:, 0]) or np.any(left_rest > arm_bounds[:, 1]):
        ap.error("--left-rest-deg is outside the CRX joint limits")
    with h5py.File(source, "r") as f:
        arrays = read_keypoints(f)
        source_fps = f["fps"][:].astype(float)
        if not np.isfinite(source_fps).all() or np.any(source_fps <= 0):
            raise ValueError("Invalid source fps")
        if "contacts/0/is_contact" in f:
            left_frames = int(f["contacts/0/is_contact"][:, 0].any(axis=1).sum())
            if left_frames:
                print(f"Note: GRAB has left-hand contact in {left_frames} frames; this right-only solver keeps the left arm at rest.")
    arrays = {name: (scene[:3, :3] @ value if name == "palm" else transform_points(scene, value))
              for name, value in arrays.items()}
    result = retarget_right(model, arrays, bases[1], args.wrist_offset_palm, args.orientation_scale,
                            args.intermediate_weight, args.regularization, progress=print)
    count = len(result["qpos"])
    qpos = np.zeros((count, 2, SIDE_DOF), dtype=np.float32)
    qpos[:, 0, :ARM_DOF] = left_rest
    qpos[:, 1] = result["qpos"]
    names = model.arm_names + model.hand_names
    bounds = model.limits(names)
    lower = qpos[:, 1]-bounds[:, 0] < 1e-3
    upper = bounds[:, 1]-qpos[:, 1] < 1e-3
    arm_delta = np.abs(np.diff(result["qpos"][:, :ARM_DOF], axis=0)) if count > 1 else np.zeros((1, ARM_DOF))
    arm_step = arm_delta.max(axis=1)
    fps = source_fps / args.time_scale
    velocity_limit = model.velocity_limits(model.arm_names)
    arm_speed = arm_delta * float(fps.max())
    # Smallest total slowdown, relative to the source clip, that keeps every arm joint in limits.
    needed_scale = float((arm_delta * float(source_fps.max()) / velocity_limit).max())

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    try:
        with h5py.File(out, "r+") as f:
            move_scene(f, scene)
            for name in ("qpos", "urdf_name", "base_translation", "base_rpy", "base_quat_wxyz"):
                if name in f:
                    del f[name]
            f.create_dataset("qpos", data=qpos)
            f["fps"][:] = fps.astype(f["fps"].dtype)
            f.create_dataset("urdf_name", data=np.array([s.encode() for s in URDF_NAMES], dtype="S40"))
            f.create_dataset("base_translation", data=bases[:, :3, 3].astype(np.float32))
            f.create_dataset("base_rpy", data=np.array([args.left_base[3:], args.right_base[3:]], np.float32))
            f.create_dataset("base_quat_wxyz", data=Rotation.from_matrix(
                bases[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]].astype(np.float32))
            f.attrs["qpos_is_placeholder"] = 0
            f.attrs["robot"] = "crx5ia_sharpa_wave"
            f.attrs["qpos_layout"] = "per side: J1..J6 then 22 Sharpa joints (thumb,index,middle,ring,pinky)"
            f.attrs["retarget_method"] = "crx_sharpa_right_palm_intermediate"
            f.attrs["scene_transform_xyz_yaw_deg"] = np.asarray(args.scene_transform, float)
            f.attrs["right_base_verified"] = 0
            # LF-normalized, matching ConTrack's asset record: ties this file to one arm model.
            f.attrs["right_urdf_sha256"] = hashlib.sha256(
                Path(args.right_urdf).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
            f.attrs["right_urdf_name"] = Path(args.right_urdf).name
            f.attrs["time_scale"] = args.time_scale
            f.attrs["source_fps"] = source_fps
            f.attrs["retarget_wrist_offset_palm_m"] = args.wrist_offset_palm
            f.attrs["retarget_intermediate_weight"] = args.intermediate_weight
            f.attrs["retarget_regularization_m_per_rad"] = args.regularization
            f.attrs["retarget_orientation_scale_m_per_rad"] = args.orientation_scale
            for path in ("grab_source/wrist_targets", "grab_source/sharpa_diagnostics"):
                if path in f:
                    del f[path]
            g = f.create_group("grab_source/wrist_targets")
            for name, data in (("positions", result["target_pos"]), ("rotations", result["target_rot"]),
                               ("actual_positions", result["actual_pos"]),
                               ("actual_rotations", result["actual_rot"]),
                               ("robot_palm_basis", result["basis"])):
                g.create_dataset(name, data=data)
            d = f.create_group("grab_source/sharpa_diagnostics")
            for name, data in (("wrist_position_error_m", result["pos_err"]),
                               ("wrist_rotation_error_deg", result["rot_err"]),
                               ("tip_error_m", result["finger_err"]), ("lower_limit", lower),
                               ("upper_limit", upper), ("optimizer_success", result["successes"]),
                               ("arm_step_max_rad", arm_step)):
                d.create_dataset(name, data=data)
            d.create_dataset("joint_names", data=np.array(names, dtype="S40"))
            d.create_dataset("finger_names", data=np.array(FINGERS, dtype="S10"))
    except Exception:
        out.unlink(missing_ok=True)  # never leave a half-converted file that looks valid
        raise
    pos_err, rot_err, finger_err = result["pos_err"], result["rot_err"], result["finger_err"]
    print("Wrist position mean/max (mm):", pos_err.mean()*1000, pos_err.max()*1000)
    print("Wrist orientation mean/max (deg):", rot_err.mean(), rot_err.max())
    for i, name in enumerate(FINGERS):
        print(f"{name}: tip mean {finger_err[:, i].mean()*1000:.4f}, max {finger_err[:, i].max()*1000:.4f} mm")
    print("Per-joint lower/upper saturation (%):")
    for i, name in enumerate(names):
        print(f"{name}: {lower[:, i].mean()*100:.1f} / {upper[:, i].mean()*100:.1f}")
    print("Optimizer success counts (arm/fingers):", result["successes"].sum(axis=0), "/", count)
    print(f"Largest arm joint change between frames: {arm_step.max():.4f} rad at frame {int(arm_step.argmax())+1}")
    print(f"Time scale {args.time_scale:g}: stored fps {fps.max():g}, duration {(count-1)/fps.max():.2f} s")
    print("Arm joint speed max (rad/s):", np.round(arm_speed.max(axis=0), 2),
          "limits:", np.round(velocity_limit, 2))
    over = (arm_speed > velocity_limit).any(axis=1)
    print(f"Frames over an arm velocity limit: {int(over.sum())} of {len(over)}; "
          f"smallest --time-scale that removes them: {needed_scale:.2f}")
    if over.any():
        print("WARNING: the arm cannot follow this clip at the stored speed. "
              "Rerun with a larger --time-scale.")
    if pos_err.max() > .01 or rot_err.max() > 5.:
        print("WARNING: wrist target not reached in some frames (unreachable or at a joint limit). "
              "Adjust --scene-transform or the base pose before training on this file.")
    if arm_step.max() > .3:
        print("WARNING: the arm jumps between IK branches. Inspect before use; a 6-axis arm cannot "
              "blend between branches without leaving the wrist target.")
    print("Saved", out)


if __name__ == "__main__":
    main()
