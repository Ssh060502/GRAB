"""Extract GRAB hand targets for Sharpa, including PIP/IP and DIP centers.

Run in the grab environment. Reuses MANO extraction without changing XHand code.
An additional MANO forward pass exposes intermediate joints absent from its H5.

--side right (default) writes grab_source/keypoints, as before.
--side left writes grab_source/keypoints_left and leaves the right group untouched.
--side both writes both. Left and right use the same MANO joint/vertex indices:
MANO_LEFT and MANO_RIGHT share one topology and joint order.
"""
import argparse
from pathlib import Path

GROUP = {"right": "keypoints", "left": "keypoints_left"}
PIP_INDEX = {"thumb": 14, "index": 2, "middle": 5, "ring": 11, "pinky": 8}
DIP_INDEX = {"index": 3, "middle": 6, "ring": 12, "pinky": 9}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--grab-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--side", choices=("right", "left", "both"), default="right")
    args = ap.parse_args()
    import h5py
    import numpy as np
    import smplx
    import torch
    import trimesh
    from mano_keypoints import get_hand_keypoints

    sides = ("right", "left") if args.side == "both" else (args.side,)
    with h5py.File(args.h5, "r+") as f:
        if f["qpos"].shape[1:] != (2, 29):
            raise ValueError("Use grab_to_contrack_sharpa.py to create a two-side Sharpa H5")
        g = f["grab_source"]
        seq = g.attrs["sequence"]
        start, end = int(g.attrs["start"]), int(g.attrs["end"])
        rm, tm = g["Rm"][:], g["tm"][:]
        with np.load(Path(args.grab_root) / "grab" / (seq + ".npz"), allow_pickle=True) as data:
            hands = {side: data["rhand" if side == "right" else "lhand"].item() for side in sides}
        # Compute every requested side before touching the file, so a failure leaves it unchanged.
        computed = {}
        for side in sides:
            is_rhand = side == "right"
            hand = hands[side]
            kp = get_hand_keypoints(args.grab_root, seq, is_rhand, start, end, args.model_path, rm, tm)
            template = np.asarray(trimesh.load(Path(args.grab_root) / hand["vtemp"],
                                               process=False).vertices, np.float32)
            model = smplx.create(args.model_path, model_type="mano", is_rhand=is_rhand,
                                 use_pca=False, flat_hand_mean=True, v_template=template,
                                 batch_size=end-start)
            params = hand["params"]
            with torch.no_grad():
                out = model(global_orient=torch.tensor(params["global_orient"][start:end], dtype=torch.float32),
                            hand_pose=torch.tensor(params["fullpose"][start:end], dtype=torch.float32),
                            transl=torch.tensor(params["transl"][start:end], dtype=torch.float32))
            joints = out.joints.detach().cpu().numpy()
            if joints.shape[1] != 16:
                raise ValueError("Expected MANO 16-joint ordering")
            kp["pip"] = {name: (joints[:, index] @ rm.T + tm).astype(np.float32)
                         for name, index in PIP_INDEX.items()}
            kp["dip"] = {name: (joints[:, index] @ rm.T + tm).astype(np.float32)
                         for name, index in DIP_INDEX.items()}
            computed[side] = kp
        for side, kp in computed.items():
            if GROUP[side] in g:
                del g[GROUP[side]]
            k = g.create_group(GROUP[side])
            for name, value in kp.items():
                if isinstance(value, dict):
                    group = k.create_group(name)
                    for finger, arr in value.items():
                        group.create_dataset(finger, data=arr)
                else:
                    k.create_dataset(name, data=value)
            k.attrs["is_rhand"] = int(side == "right")
            k.attrs["palm_axes"] = "forward,pinky_to_index,right_handed_normal"
            k.attrs["intermediate_joint_mapping"] = "thumb IP=14; PIP=2,5,11,8; DIP=3,6,12,9"
    print(f"Wrote {' and '.join(sides)} MANO wrist/MCP/PIP/IP/DIP/tips for {seq} [{start}:{end}]")


if __name__ == "__main__":
    main()
