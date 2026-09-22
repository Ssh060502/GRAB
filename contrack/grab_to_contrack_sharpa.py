"""Convert one raw GRAB sequence into a ConTrack reference clip (.h5) for the
RB-Y1A + Sharpa task (mimic_rby1a_sharpa), object + contact side only.

This is the RB-Y1A+Sharpa twin of grab_to_contrack.py (which targets xArm7+XHand).
The object/contact conversion is identical between the two tasks -- verified by
diffing the shipped xhand and sharpa versions of the same GRAB clip
(grab-s1_hammer_use_1-*-120_840.h5): translations and is_contact are byte-identical,
contact points differ by <= 1e-6 m (float rounding). Only the robot-specific fields
differ, see ROBOT DIFFERENCES below.

    ConTrack_world = Rm @ GRAB_world + tm            (Rm = +90 deg about z, verified on shipped clips)
    quat_xyzw      = from_matrix(Rm @ R(rotvec)^T)   (GRAB applies v @ R + t, i.e. R^T on column vectors)

Contacts use GRAB's own per-vertex labels (contact.object, ids 26-40 left / 41-55 right hand
finger joints, 3 per finger in the order Index, Middle, Pinky, Ring, Thumb == ConTrack segments).
Nothing about GRAB's contact labels is re-optimised; the 0.005 m contact_threshold field is never
read by any ConTrack code (grepped the whole repo) -- it's informational only.

ROBOT DIFFERENCES vs the xArm7+XHand converter (from the shipped sharpa reference clip):
  - qpos           : (T, 2, 29) instead of (T, 2, 19)   -- still written as NaN, see below
  - urdf_name      : both hands "rby1a_sharpa.urdf" (one bimanual robot) instead of two separate URDFs
  - base_translation: both hands (0, 0, -0.85) instead of (0, +-0.4, 0)

qpos (robot joint angles) is NOT part of GRAB. It is written as NaN so the file cannot be
trained on by accident; the file gets a root attribute qpos_is_placeholder=1.

Example:
    python contrack/grab_to_contrack_sharpa.py --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset \
        --seq s1/teapot_pour_1 --out out/grab-s1_teapot_pour_1-rby1a_sharpa.h5
"""

import argparse
import os

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

RM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
GRAB_FIRST_ID = (26, 41)  # (left, right) first finger-joint contact id in GRAB
SEGMENTS = [f"{f}_{p}" for f in ("index", "middle", "pinky", "ring", "thumb") for p in ("proximal", "middle", "distal")]
CONTACT_THRESHOLD = 0.005  # matches the ConTrack reference clips; never actually read by any ConTrack code

# --- RB-Y1A + Sharpa specifics (from data/sharpa/grab-s1_hammer_use_1-rby1a_sharpa-120_840.h5) ---
URDF_NAMES = ("rby1a_sharpa.urdf", "rby1a_sharpa.urdf")
BASE_TRANSLATION = np.array([[0.0, 0.0, -0.85], [0.0, 0.0, -0.85]], np.float32)
QPOS_JOINTS = 29


def pick_range(lab, margin, max_frames):
    """Frames from (first contact - margin) to (last contact + margin), optionally capped."""
    T = lab.shape[0]
    has = (lab > 0).any(1)
    if not has.any():
        raise SystemExit("this sequence has no object contact at all")
    first, last = np.where(has)[0][[0, -1]]
    start, end = max(0, first - margin), min(T, last + 1 + margin)
    if max_frames is not None and end - start > max_frames:
        end = start + max_frames
    return int(start), int(end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-root", required=True, help="folder that contains grab/ and tools/")
    ap.add_argument("--seq", required=True, help="e.g. s1/teapot_pour_1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--margin", type=int, default=60, help="frames kept before/after the contact span")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--obj-xy", type=float, nargs=2, default=(0.5, 0.0),
                    help="where the object starts in ConTrack xy (inferred convention: x 0.5-0.7, y ~ 0)")
    ap.add_argument("--z-offset", type=float, default=0.005,
                    help="tm_z = -table_height - z_offset (inferred from the shipped clips, 3-6 mm)")
    ap.add_argument("--tm", type=float, nargs=3, default=None, help="override the translation tm completely")
    args = ap.parse_args()

    d = np.load(os.path.join(args.grab_root, "grab", args.seq + ".npz"), allow_pickle=True)
    obj = d["object"].item()
    contact = d["contact"].item()
    lab_all = contact["object"]
    T_all = int(d["n_frames"])

    start, end = args.start, args.end
    if start is None or end is None:
        s0, e0 = pick_range(lab_all, args.margin, args.max_frames)
        start = s0 if start is None else start
        end = e0 if end is None else end
    T = end - start
    print(f"{args.seq}: {T_all} frames total, cropping [{start}:{end}] -> {T} frames ({T / float(d['framerate']):.2f} s)")

    rv = obj["params"]["global_orient"][start:end].astype(np.float64)
    tr = obj["params"]["transl"][start:end].astype(np.float64)
    table_z = float(d["table"].item()["params"]["transl"][start][2])

    if args.tm is not None:
        tm = np.array(args.tm, np.float64)
    else:
        g0 = RM @ tr[0]
        tm = np.array([args.obj_xy[0] - g0[0], args.obj_xy[1] - g0[1], -table_z - args.z_offset])
    print(f"table height {table_z:.4f} m, tm = {np.round(tm, 4)}")

    # ---- object ---------------------------------------------------------------------------
    mesh = trimesh.load(os.path.join(args.grab_root, obj["object_mesh"]), process=False)
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.uint32)
    Rg = R.from_rotvec(rv).as_matrix()                              # (T,3,3)
    quat = R.from_matrix(RM[None] @ Rg.transpose(0, 2, 1)).as_quat()  # xyzw
    flip = np.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1.0, 1.0)  # keep the sign continuous
    quat = quat * np.concatenate([[1.0], np.cumprod(flip)])[:, None]
    trans = tr @ RM.T + tm

    # ---- contacts (GRAB labels, unchanged) ---------------------------------------------------
    lab = lab_all[start:end]
    is_contact = np.zeros((T, 2, 15), np.uint8)
    points = np.full((T, 2, 15, 3), np.nan, np.float32)
    for h, base in enumerate(GRAB_FIRST_ID):
        for s in range(15):
            m = lab == base + s                                     # (T, V)
            cnt = m.sum(1)
            is_contact[:, h, s] = cnt > 0
            ok = cnt > 0
            if ok.any():
                points[ok, h, s] = (m[ok].astype(np.float32) @ verts) / cnt[ok, None]
    print("frames with contact per hand (left, right):", (is_contact.any(2)).sum(0).tolist())
    print("frames per segment  left :", is_contact[:, 0].sum(0).tolist())
    print("frames per segment  right:", is_contact[:, 1].sum(0).tolist())

    # ---- write -------------------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("base_translation", data=BASE_TRANSLATION)
        f.create_dataset("fps", data=np.array([d["framerate"]] * 2, np.float32))
        f.create_dataset("is_rhand", data=np.array([0, 1], np.uint8))
        f.create_dataset("urdf_name", data=np.array([s.encode() for s in URDF_NAMES], dtype="S17"))
        f.create_dataset("qpos", data=np.full((T, 2, QPOS_JOINTS), np.nan, np.float32))
        f.attrs["qpos_is_placeholder"] = 1

        g = f.create_group("object_tracks/0")
        g.create_dataset("vertices", data=verts)
        g.create_dataset("faces", data=faces)
        g.create_dataset("orientations_xyzw", data=quat.astype(np.float32))
        g.create_dataset("translations", data=trans.astype(np.float32))
        g.create_dataset("log_path", data=np.bytes_(args.seq))

        c = f.create_group("contacts")
        c.create_dataset("contact_threshold", data=np.float32(CONTACT_THRESHOLD))
        c.create_dataset("segment_names", data=np.array([s.encode() for s in SEGMENTS], dtype="S15"))
        c0 = c.create_group("0")
        c0.create_dataset("is_contact", data=is_contact)
        c0.create_dataset("points", data=points)
        c0.create_dataset("log_path", data=np.bytes_(args.seq))

        # raw GRAB hand parameters, cropped (for the later retargeting stage; ConTrack ignores this group)
        s = f.create_group("grab_source")
        s.attrs.update({"sequence": args.seq, "start": start, "end": end, "n_frames_total": T_all})
        s.create_dataset("Rm", data=RM)
        s.create_dataset("tm", data=tm)
        for hand in ("lhand", "rhand"):
            for k, v in d[hand].item()["params"].items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T_all:
                    s.create_dataset(f"{hand}/{k}", data=v[start:end])
    print("saved", args.out)


if __name__ == "__main__":
    main()
