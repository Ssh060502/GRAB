"""Inspect stable palm calibration and optionally save robot wrist targets.

Run after mano_keypoints.py. No fingertip/contact averaging is used.
Offset is explicit: MCP alignment alone cannot identify anatomical origin offsets
between differently proportioned hands. Default keeps the MANO wrist center.
"""
import argparse
import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from palm_geometry import robot_palm_frame, wrist_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--wrist-offset-palm", type=float, nargs=3, default=(0., 0., 0.),
                    help="robot origin minus MANO wrist, in palm axes, meters")
    ap.add_argument("--write-targets", action="store_true")
    args = ap.parse_args()
    from dex_retargeting.robot_wrapper import RobotWrapper
    urdf = os.path.join(args.assets_dir, "urdf", "xhand_right.urdf")
    basis, anchors = robot_palm_frame(RobotWrapper(urdf), urdf)
    with h5py.File(args.h5, "r+" if args.write_targets else "r") as f:
        k = f["grab_source/keypoints"]
        if "palm_rotmat" not in k:
            raise SystemExit("Missing palm geometry: rerun mano_keypoints.py")
        if not k.attrs.get("is_rhand", 1):
            raise SystemExit("XHand right calibration requires right-hand keypoints")
        palm, raw = k["palm_rotmat"][:], k["wrist_rotmat"][:]
        pos, rot = wrist_targets(k["wrist_pos"][:], palm, basis, args.wrist_offset_palm)
        corrections = raw.transpose(0, 2, 1) @ rot
        mean = R.from_matrix(corrections).mean()
        spread = (mean.inv() * R.from_matrix(corrections)).magnitude() * 180 / np.pi
        print("Robot MCP anchors in right_hand_link (m):", anchors)
        print("MANO-local correction variation: mean %.4f deg, max %.4f deg" %
              (spread.mean(), spread.max()))
        print("Use retarget_xarm_xhand.py to solve arm and finger joints.")
        print("Offset in palm axes (m):", args.wrist_offset_palm)
        if args.write_targets:
            if "grab_source/wrist_targets" in f:
                del f["grab_source/wrist_targets"]
            g = f.create_group("grab_source/wrist_targets")
            g.create_dataset("positions", data=pos)
            g.create_dataset("rotations", data=rot)
            g.create_dataset("robot_palm_basis", data=basis)
            g.attrs["wrist_offset_palm_m"] = args.wrist_offset_palm
            g.attrs["method"] = "wrist_mcp_palm"


if __name__ == "__main__":
    main()
