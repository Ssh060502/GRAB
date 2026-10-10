"""Extract right GRAB hand targets for Sharpa, including PIP/IP and DIP centers.

Run in the grab environment. Reuses MANO extraction without changing XHand code.
An additional MANO forward pass exposes intermediate joints absent from its H5.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--grab-root", required=True)
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()
    import h5py
    import numpy as np
    import smplx
    import torch
    import trimesh
    from mano_keypoints import get_hand_keypoints

    with h5py.File(args.h5, "r+") as f:
        if f["qpos"].shape[1:] != (2, 29):
            raise ValueError("Use grab_to_contrack_sharpa.py to create a two-side Sharpa H5")
        g = f["grab_source"]
        seq = g.attrs["sequence"]
        start, end = int(g.attrs["start"]), int(g.attrs["end"])
        rm, tm = g["Rm"][:], g["tm"][:]
        kp = get_hand_keypoints(args.grab_root, seq, True, start, end, args.model_path, rm, tm)
        with np.load(Path(args.grab_root) / "grab" / (seq + ".npz"), allow_pickle=True) as data:
            hand = data["rhand"].item()
        template = np.asarray(trimesh.load(Path(args.grab_root) / hand["vtemp"],
                                           process=False).vertices, np.float32)
        model = smplx.create(args.model_path, model_type="mano", is_rhand=True,
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
                     for name, index in {"thumb": 14, "index": 2, "middle": 5, "ring": 11, "pinky": 8}.items()}
        kp["dip"] = {name: (joints[:, index] @ rm.T + tm).astype(np.float32)
                     for name, index in {"index": 3, "middle": 6, "ring": 12, "pinky": 9}.items()}
        if "keypoints" in g:
            del g["keypoints"]
        k = g.create_group("keypoints")
        for name, value in kp.items():
            if isinstance(value, dict):
                group = k.create_group(name)
                for finger, arr in value.items():
                    group.create_dataset(finger, data=arr)
            else:
                k.create_dataset(name, data=value)
        k.attrs["is_rhand"] = 1
        k.attrs["palm_axes"] = "forward,pinky_to_index,right_handed_normal"
        k.attrs["intermediate_joint_mapping"] = "thumb IP=14; PIP=2,5,11,8; DIP=3,6,12,9"
    print(f"Wrote right MANO wrist/MCP/PIP/IP/DIP/tips for {seq} [{start}:{end}]")


if __name__ == "__main__":
    main()
