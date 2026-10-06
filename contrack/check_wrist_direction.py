"""Validate all three stable palm axes using targets/actual FK saved by retargeting.

This checks palm direction, lateral direction and normal independently. Object
origin and curling fingertips are deliberately not used as orientation references.
"""
import argparse
import os
import h5py
import numpy as np
from palm_geometry import robot_palm_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--num-frames", type=int, default=8)
    args = ap.parse_args()
    if args.num_frames < 1:
        ap.error("--num-frames must be positive")
    from dex_retargeting.robot_wrapper import RobotWrapper
    urdf = os.path.join(args.assets_dir, "urdf", "xhand_right.urdf")
    basis, _ = robot_palm_frame(RobotWrapper(urdf), urdf)
    with h5py.File(args.h5, "r") as f:
        if "grab_source/keypoints/palm_rotmat" not in f or "grab_source/wrist_targets" not in f:
            raise SystemExit("Run compute_hand_keypoints.py then estimate_calib.py --write-targets or retarget_xarm_xhand.py first")
        human = f["grab_source/keypoints/palm_rotmat"][:]
        g = f["grab_source/wrist_targets"]
        target = g["rotations"][:]
        actual = g["actual_rotations"][:] if "actual_rotations" in g else None
        positions = g["positions"][:]
        actual_positions = g["actual_positions"][:] if "actual_positions" in g else None
    if len(human) == 0:
        raise SystemExit("Empty clip")
    def axis_errors(rotations):
        robot_axes = rotations @ basis
        dots = np.sum(robot_axes * human, axis=1)
        return np.degrees(np.arccos(np.clip(dots, -1., 1.)))
    target_error = axis_errors(target)
    print("Target palm axis errors (forward / across / normal), degrees:")
    for t in np.unique(np.linspace(0, len(human)-1, args.num_frames).astype(int)):
        print(t, target_error[t])
    if actual is not None:
        actual_error = axis_errors(actual)
        print("Actual FK axis error mean (deg):", actual_error.mean(axis=0))
        print("Actual FK axis error max (deg):", actual_error.max(axis=0))
        error = np.linalg.norm(actual_positions - positions, axis=1) * 1000
        print("Actual FK position error mean/max (mm):", error.mean(), error.max())
    else:
        print("No actual FK stored yet; run retargeting to check achieved wrist pose.")


if __name__ == "__main__":
    main()
