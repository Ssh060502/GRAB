"""Compare a raw GRAB sequence with the matching ConTrack reference clip (.h5).

Purpose: find out exactly how GRAB -> ConTrack was converted (fixed world transform,
quaternion convention, contact definition) by checking a clip the ConTrack authors shipped.

Example (file name encodes source sequence and frame range):
    python contrack/compare_grab_h5.py \
        --h5 ConTrack/data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5 \
        --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset

Needs only numpy, scipy, h5py, trimesh.
"""

import argparse
import os
import re

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

# GRAB contact ids for finger joints (tools/utils.py contact_ids): 3 per finger,
# finger order Index, Middle, Pinky, Ring, Thumb -> same order as ConTrack segment_names.
GRAB_FIRST_ID = {"left": 26, "right": 41}


def kabsch(P, Q):
    """Rigid transform (Rm, t) minimising ||Rm @ P_i + t - Q_i||."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, qc - Rm @ pc


def parse_name(path):
    m = re.match(r"grab-(s\d+)_(.+)-(?:xarm7_xhand|rby1a_sharpa)-(\d+)_(\d+)\.h5$", os.path.basename(path))
    if m is None:
        raise ValueError("cannot parse sequence and frame range from file name; pass --seq/--start/--end")
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--grab-root", required=True, help="folder containing grab/ and tools/")
    ap.add_argument("--seq", default=None, help="e.g. s1/hammer_use_1 (default: parsed from file name)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--n-verts", type=int, default=3000, help="vertices used for the pose fit")
    args = ap.parse_args()

    sbj, name, s0, s1 = parse_name(args.h5) if args.seq is None else (None, None, None, None)
    seq = args.seq or f"{sbj}/{name}"
    start = args.start if args.start is not None else s0
    end = args.end if args.end is not None else s1

    d = np.load(os.path.join(args.grab_root, "grab", seq + ".npz"), allow_pickle=True)
    obj = d["object"].item()
    rv = obj["params"]["global_orient"][start:end].astype(np.float64)
    tr = obj["params"]["transl"][start:end].astype(np.float64)
    T = end - start
    print(f"sequence {seq}  frames [{start}:{end}] -> {T} frames, GRAB framerate {float(d['framerate'])}")

    f = h5py.File(args.h5, "r")
    hv = f["object_tracks/0/vertices"][:].astype(np.float64)
    hf = f["object_tracks/0/faces"][:]
    hq = f["object_tracks/0/orientations_xyzw"][:].astype(np.float64)
    ht = f["object_tracks/0/translations"][:].astype(np.float64)
    assert hq.shape[0] == T, f"h5 has {hq.shape[0]} frames, GRAB slice has {T}"

    # ---------------- 1. mesh -----------------------------------------------------------
    mesh = trimesh.load(os.path.join(args.grab_root, obj["object_mesh"]), process=False)
    gv = np.asarray(mesh.vertices, np.float64)
    gf = np.asarray(mesh.faces)
    print("\n[1] mesh")
    print("  same vertex count:", gv.shape == hv.shape, " max |dv|:", np.abs(gv - hv).max() if gv.shape == hv.shape else "n/a")
    print("  same faces       :", gf.shape == hf.shape and bool((gf == hf).all()))
    if gv.shape != hv.shape:
        print("  meshes differ -> stop, the vertex correspondence below would be invalid")
        return

    # ---------------- 2. world vertices and fixed transform -----------------------------
    idx = np.random.RandomState(0).choice(len(gv), min(args.n_verts, len(gv)), replace=False)
    v = gv[idx]
    Rg = R.from_rotvec(rv).as_matrix()                       # (T,3,3) = batch_rodrigues(rv)
    Vg = np.einsum("vi,tij->tvj", v, Rg) + tr[:, None]       # GRAB ObjectModel: v @ R + t
    Rh = R.from_quat(hq).as_matrix()                         # column-vector convention
    Vh = np.einsum("tij,vj->tvi", Rh, v) + ht[:, None]       # ConTrack world vertices

    Rm, tm = kabsch(Vg[0], Vh[0])
    res = np.linalg.norm(np.einsum("ij,tvj->tvi", Rm, Vg) + tm - Vh, axis=-1)
    print("\n[2] fit  ConTrack_world = Rm @ GRAB_world + tm   (fitted on frame 0, checked on all frames)")
    print("  Rm =\n", np.round(Rm, 5))
    print("  tm =", np.round(tm, 5))
    print(f"  residual over all frames: mean {res.mean():.2e} m, max {res.max():.2e} m")
    print("  (near 0 => pure fixed rigid transform; large => something else, e.g. time offset/other convention)")

    # ---------------- 3. quaternion convention -------------------------------------------
    print("\n[3] quaternion convention (angle error between predicted and stored orientation, degrees)")
    for label, Rcand in (("R(rv)   ", Rg), ("R(rv)^T ", Rg.transpose(0, 2, 1))):
        pred = np.einsum("ij,tjk->tik", Rm, Rcand)
        err = R.from_matrix(np.einsum("tij,tkj->tik", pred, Rh)).magnitude() * 180 / np.pi
        print(f"  Rm @ {label} : mean {err.mean():.3f}  max {err.max():.3f}")
    print("  the candidate with ~0 error is the right convention for  quat = from_matrix(Rm @ cand)")

    # ---------------- 4. contacts ----------------------------------------------------------
    print("\n[4] contacts: GRAB labels (contact.object, ids 26-40 left / 41-55 right) vs h5 is_contact")
    lab = d["contact"].item()["object"][start:end]
    seg = [s.decode() for s in f["contacts/segment_names"][:]]
    hc = f["contacts/0/is_contact"][:].astype(bool)          # (T, 2, 15)
    hp = f["contacts/0/points"][:]
    is_r = f["is_rhand"][:].astype(bool)
    print(f"  h5 contact_threshold {float(f['contacts/contact_threshold'][()])}, GRAB label threshold {d['contact'].item()['threshold']}")
    for h in range(hc.shape[1]):
        side = "right" if is_r[h] else "left"
        base = GRAB_FIRST_ID[side]
        agree, tot, pdist = [], [], []
        for s in range(15):
            g = (lab == base + s).any(1)
            agree.append(int((g == hc[:, h, s]).sum()))
            tot.append(T)
            both = np.where(g & hc[:, h, s])[0]
            if len(both):
                loc = np.array([gv[lab[t] == base + s].mean(0) for t in both[:50]])
                pdist.append(np.linalg.norm(loc - hp[both[:50], h, s], axis=-1).mean())
        print(f"  {side:5s} hand: frame agreement {sum(agree) / sum(tot):.3f}; "
              f"h5 contact-frames {int(hc[:, h].sum())}, GRAB-label contact-frames "
              f"{int(sum((lab == base + s).any(1).sum() for s in range(15)))}")
        if pdist:
            print(f"        mean |mean(labelled verts) - h5 point| = {np.mean(pdist):.4f} m (object local frame check)")
    print("  segments:", seg)


if __name__ == "__main__":
    main()
