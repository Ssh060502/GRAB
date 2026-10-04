"""Independent, from-scratch check of "does the retargeted qpos actually get near the object" --
written without reusing any code from retarget_xarm_xhand.py or visualize_retarget.py, so it
doesn't share whatever bug either of those might have. Answers one question with a plain number:
at a given frame, what is the true nearest distance between the fingertip's actual COLLISION MESH
(not just the single reference point used for IK) and the object's mesh surface?

This is the number that should be trusted over eyeballing a rendered picture, and over the IK
residual (which only measures distance between two abstract reference points, not real geometry).

Runs in the `retarget` env (needs pinocchio via dex_retargeting).

Example:
    python contrack/verify_contact_gap.py --h5 out/grab-s1_teapot_pour_1_xhand.h5 \
        --assets-dir /home/shshao/ConTrack/assets --frame 400
"""

import argparse
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]
FINGER_COLLISION_LINKS = {
    "thumb": "right_hand_thumb_rota_tip", "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip", "ring": "right_hand_ring_tip", "pinky": "right_hand_pinky_tip",
}


def collision_meshes_for_links(urdf_path, link_names):
    """link name -> (trimesh.Trimesh in link-local frame, local_origin 4x4), using <collision> if
    present else <visual> (fingertip links in this URDF only define <visual>, checked directly)."""
    base_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    out = {}
    for link in tree.findall("link"):
        name = link.get("name")
        if name not in link_names:
            continue
        geoms = link.findall("collision") or link.findall("visual")
        meshes = []
        for g in geoms:
            mesh_el = g.find("geometry/mesh")
            if mesh_el is None:
                continue
            fn = os.path.normpath(os.path.join(base_dir, mesh_el.get("filename")))
            if not os.path.isfile(fn):
                continue
            origin_el = g.find("origin")
            xyz = np.array([float(v) for v in origin_el.get("xyz", "0 0 0").split()]) if origin_el is not None else np.zeros(3)
            rpy = np.array([float(v) for v in origin_el.get("rpy", "0 0 0").split()]) if origin_el is not None else np.zeros(3)
            origin = np.eye(4)
            origin[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
            origin[:3, 3] = xyz
            meshes.append((trimesh.load(fn, process=False, force="mesh"), origin))
        out[name] = meshes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--frame", type=int, required=True)
    args = ap.parse_args()

    from dex_retargeting.robot_wrapper import RobotWrapper

    urdf_path = os.path.join(args.assets_dir, "urdf", "xarm_xhand_right.urdf")
    robot = RobotWrapper(urdf_path)
    joint_idx = np.array([robot.get_joint_index(n) for n in XARM_JOINT_NAMES + HAND_JOINT_NAMES])
    tip_meshes = collision_meshes_for_links(urdf_path, set(FINGER_COLLISION_LINKS.values()))

    with h5py.File(args.h5, "r") as f:
        t = args.frame
        qpos = f["qpos"][t, 1, :]  # right hand
        right_base = f["base_translation"][1]
        obj_v = f["object_tracks/0/vertices"][:]
        obj_f = f["object_tracks/0/faces"][:]
        obj_t = f["object_tracks/0/translations"][t]
        obj_q = f["object_tracks/0/orientations_xyzw"][t]
        seg_names = [s.decode() for s in f["contacts/segment_names"][:]]
        is_contact = f["contacts/0/is_contact"][t, 1, :]

    obj_R = R.from_quat(obj_q).as_matrix()
    obj_world = obj_v @ obj_R.T + obj_t
    obj_mesh = trimesh.Trimesh(vertices=obj_world, faces=obj_f, process=False)

    full = np.zeros(robot.dof)
    full[joint_idx] = qpos
    robot.compute_forward_kinematics(full)

    print(f"frame {t}: GRAB-recorded distal contact this frame -> "
          + ", ".join(f"{n.replace('_distal','')}={bool(is_contact[seg_names.index(n)])}"
                       for n in seg_names if n.endswith("_distal")))
    for finger, link_name in FINGER_COLLISION_LINKS.items():
        pose = robot.get_link_pose(robot.get_link_index(link_name))
        min_d = None
        for mesh, origin in tip_meshes.get(link_name, []):
            M = pose @ origin
            v_world = mesh.vertices @ M[:3, :3].T + M[:3, 3] + right_base
            fingertip_mesh = trimesh.Trimesh(vertices=v_world, faces=mesh.faces, process=False)
            closest, dist, _ = trimesh.proximity.closest_point(obj_mesh, fingertip_mesh.vertices)
            min_d = dist.min() if min_d is None else min(min_d, dist.min())
        if min_d is None:
            print(f"  {finger:7s} no collision/visual mesh found for {link_name}")
        else:
            print(f"  {finger:7s} nearest distance between {link_name}'s mesh and the object's mesh: {min_d*1000:.2f} mm")


if __name__ == "__main__":
    main()
