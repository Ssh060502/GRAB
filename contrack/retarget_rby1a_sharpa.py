"""Right RB-Y1A+Sharpa: palm wrist IK, then 22 finger joints at actual wrist.

Uses full assembly URDF, scipy least_squares and ordered ConTrack joint names.
Left side stays at repository default. No contact substitution or XHand imports.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sharpa_geometry import (SharpaKinematics, ARM_NAMES, HAND_NAMES, DEFAULT_ARM,
                             PALM_LINK, FINGERS, palm_basis, wrist_targets, REFERENCE_COMMIT)


def solve_arm(model, target_pos, target_rot, previous, orientation_scale=.05):
    bounds = model.limits(ARM_NAMES)
    def residual(q):
        pose = model.fk(dict(zip(ARM_NAMES, q)))[PALM_LINK]
        return np.r_[pose[:3, 3] - target_pos, orientation_scale *
                     Rotation.from_matrix(target_rot.T @ pose[:3, :3]).as_rotvec()]
    result = least_squares(residual, np.clip(previous, *bounds.T), bounds=bounds.T,
                           max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    # Retry poor tracking with distinct feasible arm configurations; choose by same objective.
    if np.linalg.norm(result.fun[:3]) > .01 or np.linalg.norm(result.fun[3:]) > orientation_scale*.15:
        for seed in (DEFAULT_ARM, np.array([-.6, -.8, .5, -1.8, .2, .6, 0.])):
            candidate = least_squares(residual, np.clip(seed, *bounds.T), bounds=bounds.T,
                                      max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if np.linalg.norm(candidate.fun) < np.linalg.norm(result.fun):
                result = candidate
    return result


def solve_fingers(model, targets, previous, intermediate_weight=.35, regularization=.001):
    """Targets are hand-local positions; regularization has units m/rad.

    Intermediate joints constrain curl distribution; a small previous-pose penalty
    discourages drift in the remaining redundancy. Tips keep weight 1.
    """
    bounds = model.limits(HAND_NAMES)
    def residual(q):
        tip, pip, dip = model.hand_points(q)
        return np.r_[(tip-targets["tips"]).ravel(),
                     intermediate_weight*(pip-targets["pip"]).ravel(),
                     intermediate_weight*(dip-targets["dip"]).ravel(),
                     regularization*(q-previous)]
    # Exact zero starts put many flexion joints at their lower bound. A small
    # interior start avoids premature termination with an almost-zero trust radius.
    initial = np.clip(previous, bounds[:,0]+.01, bounds[:,1]-.01)
    result = least_squares(residual, initial, bounds=bounds.T,
                           max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if np.max(np.linalg.norm(result.fun[:15].reshape(5,3),axis=1)) > .005:
        candidate = least_squares(residual, bounds.mean(axis=1), bounds=bounds.T,
                                  max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if np.linalg.norm(candidate.fun) < np.linalg.norm(result.fun):
            result = candidate
    return result


def validate_input(f):
    sides = f["is_rhand"][:]
    if not np.array_equal(sides, [0, 1]) or f["qpos"].shape[1:] != (2, 29):
        raise ValueError("Expected Sharpa H5 with left/right qpos (T,2,29)")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["urdf_name"][:]]
    if names != ["rby1a_sharpa.urdf"]*2:
        raise ValueError("This file is not an RB-Y1A+Sharpa reference")
    base = f["base_translation"][:]
    if base.shape != (2, 3) or not np.isfinite(base).all() or not np.allclose(base[0], base[1]):
        raise ValueError("Both sides must share one finite whole-robot base translation")
    k = f["grab_source/keypoints"]
    if not k.attrs.get("is_rhand", False):
        raise ValueError("Right-hand keypoints with explicit is_rhand are required")
    count = f["qpos"].shape[0]
    if count == 0:
        raise ValueError("Empty sequence")
    arrays = {"wrist": k["wrist_pos"][:], "palm": k["palm_rotmat"][:]}
    for group, fingers in (("tips", FINGERS), ("pip", FINGERS), ("dip", FINGERS[1:])):
        if group not in k:
            raise ValueError("Missing intermediate joints: run sharpa_keypoints.py first")
        arrays[group] = np.stack([k[group][n][:] for n in fingers], axis=1)
    for name, array in arrays.items():
        shape = (count, 3, 3) if name == "palm" else ((count, 3) if name == "wrist" else
                (count, 4 if name == "dip" else 5, 3))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} targets: expected {shape}")
    if not np.allclose(arrays["palm"].transpose(0,2,1) @ arrays["palm"], np.eye(3), atol=1e-5) or not np.allclose(
            np.linalg.det(arrays["palm"]), 1., atol=1e-5):
        raise ValueError("Invalid palm rotations")
    return arrays, base[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True, help="Sharpa file, updated in place after solving all frames")
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--wrist-offset-palm", type=float, nargs=3, default=(0.,0.,0.))
    ap.add_argument("--orientation-scale", type=float, default=.05, help="arm orientation weight, m/rad")
    ap.add_argument("--intermediate-weight", type=float, default=.35)
    ap.add_argument("--regularization", type=float, default=.001, help="previous finger pose weight, m/rad")
    args = ap.parse_args()
    if (not np.isfinite(args.wrist_offset_palm).all() or not np.isfinite(
        [args.orientation_scale,args.intermediate_weight,args.regularization]).all() or
        args.orientation_scale <= 0 or args.intermediate_weight < 0 or args.regularization < 0):
        ap.error("Weights must be finite/nonnegative, orientation scale positive; offset finite")
    import h5py
    model = SharpaKinematics(Path(args.assets_dir)/"urdf"/"rby1a_sharpa.urdf")
    basis, anchors = palm_basis(model)
    print("Sharpa zero-reference MCP anchors (palm local, m):", anchors)
    with h5py.File(args.h5, "r") as f:
        arrays, base = validate_input(f)
        raw_qpos_shape = f["qpos"].shape
        if "contacts/0/is_contact" in f:
            left_frames = int(f["contacts/0/is_contact"][:,0].any(axis=1).sum())
            if left_frames:
                print(f"Note: GRAB has left-hand contact in {left_frames} frames; this right-only solver keeps the left side at rest.")
    target_pos, target_rot = wrist_targets(arrays["wrist"], arrays["palm"], basis, args.wrist_offset_palm)
    count = len(target_pos)
    qpos = np.zeros(raw_qpos_shape, dtype=np.float32)
    qpos[:,0,:7] = DEFAULT_ARM
    actual_pos, actual_rot = np.zeros((count,3)), np.zeros((count,3,3))
    pos_err, rot_err, finger_err = np.zeros(count), np.zeros(count), np.zeros((count,5))
    successes = np.zeros((count,2), dtype=np.uint8)
    arm_q, hand_q = DEFAULT_ARM.copy(), np.zeros(22)
    for t in range(count):
        arm = solve_arm(model, target_pos[t]-base, target_rot[t], arm_q, args.orientation_scale)
        arm_q = arm.x
        achieved = model.fk(dict(zip(ARM_NAMES,arm_q)))[PALM_LINK]
        actual_pos[t], actual_rot[t] = achieved[:3,3]+base, achieved[:3,:3]
        local = {name: (arrays[name][t]-actual_pos[t]) @ actual_rot[t] for name in ("tips","pip","dip")}
        fingers = solve_fingers(model, local, hand_q, args.intermediate_weight, args.regularization)
        hand_q = fingers.x
        if not np.isfinite(np.r_[arm_q,hand_q]).all():
            raise RuntimeError(f"Nonfinite optimization result at frame {t}; input H5 not updated")
        qpos[t,1] = np.r_[arm_q,hand_q]
        tip_world = actual_pos[t] + model.hand_points(hand_q)[0] @ actual_rot[t].T
        finger_err[t] = np.linalg.norm(tip_world-arrays["tips"][t], axis=1)
        pos_err[t] = np.linalg.norm(actual_pos[t]-target_pos[t])
        rot_err[t] = Rotation.from_matrix(target_rot[t].T @ actual_rot[t]).magnitude()*180/np.pi
        successes[t] = [arm.success,fingers.success]
        if t%50 == 0:
            print(f"frame {t}/{count}: wrist {pos_err[t]*1000:.2f} mm / {rot_err[t]:.2f} deg")
    bounds = model.limits(ARM_NAMES+HAND_NAMES)
    lower = qpos[:,1]-bounds[:,0] < 1e-3
    upper = bounds[:,1]-qpos[:,1] < 1e-3
    with h5py.File(args.h5,"r+") as f:
        f["qpos"][:] = qpos
        f.attrs["qpos_is_placeholder"] = 0
        f.attrs["retarget_method"] = "rby1a_sharpa_right_palm_intermediate"
        f.attrs["retarget_wrist_offset_palm_m"] = args.wrist_offset_palm
        f.attrs["retarget_intermediate_weight"] = args.intermediate_weight
        f.attrs["retarget_regularization_m_per_rad"] = args.regularization
        f.attrs["retarget_orientation_scale_m_per_rad"] = args.orientation_scale
        f.attrs["retarget_reference_commit"] = REFERENCE_COMMIT
        for path in ("grab_source/wrist_targets", "grab_source/sharpa_diagnostics"):
            if path in f:
                del f[path]
        g=f.create_group("grab_source/wrist_targets")
        for name, data in (("positions",target_pos),("rotations",target_rot),
                           ("actual_positions",actual_pos),("actual_rotations",actual_rot),("robot_palm_basis",basis)):
            g.create_dataset(name,data=data)
        d=f.create_group("grab_source/sharpa_diagnostics")
        for name,data in (("wrist_position_error_m",pos_err),("wrist_rotation_error_deg",rot_err),
                          ("tip_error_m",finger_err),("lower_limit",lower),("upper_limit",upper),
                          ("optimizer_success",successes)):
            d.create_dataset(name,data=data)
        d.create_dataset("joint_names",data=np.array(ARM_NAMES+HAND_NAMES,dtype="S40"))
        d.create_dataset("finger_names",data=np.array(FINGERS,dtype="S10"))
    print("Wrist position mean/max (mm):",pos_err.mean()*1000,pos_err.max()*1000)
    print("Wrist orientation mean/max (deg):",rot_err.mean(),rot_err.max())
    for i,name in enumerate(FINGERS):
        print(f"{name}: tip mean {finger_err[:,i].mean()*1000:.4f}, max {finger_err[:,i].max()*1000:.4f} mm")
    print("Per-joint lower/upper saturation (%):")
    for i,name in enumerate(ARM_NAMES+HAND_NAMES):
        print(f"{name}: {lower[:,i].mean()*100:.1f} / {upper[:,i].mean()*100:.1f}")
    print("Optimizer success counts (arm/fingers):",successes.sum(axis=0),"/",count)
    print("Saved",args.h5)


if __name__ == "__main__":
    main()
