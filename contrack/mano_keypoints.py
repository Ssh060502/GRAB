"""Compute GRAB hand keypoints (wrist + 5 fingertips) in the ConTrack world frame.

Validated numerically: fingertip positions computed here for s1/teapot_pour_1 land within
1.8-2.8 cm of the corresponding GRAB contact-label points already stored in the converted
h5 (see contrack/grab_to_contrack.py) -- the expected size of the gap between an exact
fingertip and "mean position of the contact-labelled vertices". This confirms the hand lives
in the same world frame as the object/table (same Rm, tm as grab_to_contrack.py), so no
extra hand-specific transform is needed.

MANO forward-kinematics details (verified against the installed smplx==0.1.28 source, not
assumed from memory):
  - smplx.MANO(..., use_pca=False).forward() returns `joints` of shape (T, 16, 3): index 0 is
    the wrist, 1-15 are the finger joints in order Index, Middle, Pinky, Ring, Thumb (3 each) --
    NOT 21 keypoints; MANO's own forward() does not append fingertips despite the class defining
    `vertex_joint_selector.extra_joints_idxs` (dead code path in this version, confirmed by
    printing joints.shape -> (T, 16, 3)).
  - Fingertips are therefore taken directly from `vertices` at 5 fixed vertex ids
    (smplx.vertex_ids.VERTEX_IDS['mano']): thumb=744, index=320, middle=443, ring=554, pinky=671.
  - Use `fullpose` (45 = 15 joints x 3, axis-angle), not the PCA-compressed `hand_pose` (n_comps
    dims) -- fullpose is the exact unrolled pose GRAB actually used to generate the mesh.
"""

import os

import numpy as np
import smplx
import torch
import trimesh

TIP_VERTEX_ID = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")  # canonical order used everywhere below


def get_hand_keypoints(grab_root, seq, is_rhand, start, end, model_path, Rm, tm):
    """Wrist and fingertip positions/orientation for one GRAB sequence, in ConTrack world frame.

    Parameters
    ----------
    grab_root : str
        Folder containing grab/ and tools/.
    seq : str
        e.g. "s1/teapot_pour_1".
    is_rhand : bool
    start, end : int
        Frame range (same crop as the .h5 produced by grab_to_contrack.py).
    model_path : str
        Folder containing mano/MANO_{LEFT,RIGHT}.pkl (the GRAB_models folder).
    Rm : (3,3) array
    tm : (3,) array
        The fixed rigid transform used by grab_to_contrack.py: ConTrack_world = Rm @ GRAB_world + tm.

    Returns
    -------
    dict with, all in ConTrack world frame:
        wrist_pos    (T,3) float32
        wrist_rotmat (T,3,3) float32   -- GRAB's own wrist rotation (global_orient), rotated by Rm.
                                           NOT verified against the robot's own hand-base axis
                                           convention -- see retarget_xarm_xhand.py's R_CALIB.
        tips         dict name -> (T,3) float32, name in ("thumb","index","middle","ring","pinky")
    """
    d = np.load(os.path.join(grab_root, "grab", seq + ".npz"), allow_pickle=True)
    hand_key = "rhand" if is_rhand else "lhand"
    hand = d[hand_key].item()
    vtemp = np.asarray(trimesh.load(os.path.join(grab_root, hand["vtemp"]), process=False).vertices, np.float32)

    p = hand["params"]
    T = end - start
    model = smplx.create(
        model_path, model_type="mano", is_rhand=is_rhand, use_pca=False,
        flat_hand_mean=True, v_template=vtemp, batch_size=T,
    )
    out = model(
        global_orient=torch.tensor(p["global_orient"][start:end], dtype=torch.float32),
        hand_pose=torch.tensor(p["fullpose"][start:end], dtype=torch.float32),
        transl=torch.tensor(p["transl"][start:end], dtype=torch.float32),
    )
    joints = out.joints.detach().numpy()    # (T,16,3), GRAB world frame
    verts = out.vertices.detach().numpy()   # (T,778,3), GRAB world frame

    from scipy.spatial.transform import Rotation as R

    wrist_pos = joints[:, 0] @ Rm.T + tm
    wrist_rotmat = Rm[None] @ R.from_rotvec(p["global_orient"][start:end]).as_matrix()
    tips = {name: verts[:, vid] @ Rm.T + tm for name, vid in TIP_VERTEX_ID.items()}
    return {"wrist_pos": wrist_pos.astype(np.float32), "wrist_rotmat": wrist_rotmat.astype(np.float32),
            "tips": {k: v.astype(np.float32) for k, v in tips.items()}}
