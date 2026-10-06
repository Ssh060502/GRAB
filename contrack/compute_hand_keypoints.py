"""Step 0 (runs in the `grab` conda env, needs smplx/torch/chumpy): compute GRAB hand
keypoints (wrist + 5 fingertips, ConTrack world frame) and write them into the converted
h5's grab_source group, for retarget_xarm_xhand.py to consume later.

Split into its own script/env on purpose: pinocchio (needed by retarget_xarm_xhand.py) has no
pip wheel for Python 3.9 past version 2.6.18, so it needs a separate Python >=3.10 env; keeping
smplx/torch/chumpy (and the numpy<1.24 they require) out of that env avoids a real, unresolvable
version conflict with pinocchio/dex_retargeting there (which want numpy>=2.0). See the comment
in retarget_xarm_xhand.py.

Example:
    python contrack/compute_hand_keypoints.py --h5 out/grab-s1_teapot_pour_1.h5 \
        --grab-root /path/to/GRAB_dataset --model-path /path/to/GRAB_models
"""

import argparse

import h5py

from mano_keypoints import get_hand_keypoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--grab-root", required=True)
    ap.add_argument("--model-path", required=True)
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--is-rhand", dest="is_rhand", action="store_true")
    side.add_argument("--is-lhand", dest="is_rhand", action="store_false")
    ap.set_defaults(is_rhand=True)
    args = ap.parse_args()

    with h5py.File(args.h5, "r+") as f:
        g = f["grab_source"]
        seq, start, end = g.attrs["sequence"], int(g.attrs["start"]), int(g.attrs["end"])
        Rm, tm = g["Rm"][:], g["tm"][:]

        kp = get_hand_keypoints(args.grab_root, seq, args.is_rhand, start, end, args.model_path, Rm, tm)

        if "keypoints" in g:
            del g["keypoints"]
        k = g.create_group("keypoints")
        k.create_dataset("wrist_pos", data=kp["wrist_pos"])
        k.create_dataset("wrist_rotmat", data=kp["wrist_rotmat"])
        k.create_dataset("palm_rotmat", data=kp["palm_rotmat"])
        k.attrs["is_rhand"] = int(args.is_rhand)
        k.attrs["palm_axes"] = "forward,pinky_to_index,right_handed_normal"
        m = k.create_group("mcp")
        for name, arr in kp["mcp"].items():
            m.create_dataset(name, data=arr)
        t = k.create_group("tips")
        for name, arr in kp["tips"].items():
            t.create_dataset(name, data=arr)
    print(f"wrote keypoints for {seq} [{start}:{end}] into {args.h5}:grab_source/keypoints")


if __name__ == "__main__":
    main()
