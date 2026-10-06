"""Stable palm frames. Columns are forward, pinky-to-index, right-handed normal.

The normal is a geometric convention, not an assumed anatomical palm-facing axis.
Both human and robot must use the same ordered landmarks.
"""

import numpy as np
import xml.etree.ElementTree as ET


def palm_frame(wrist, index, middle, pinky):
    def unit(v):
        length = np.linalg.norm(v, axis=-1, keepdims=True)
        if not np.isfinite(v).all() or np.any(length < 1e-8):
            raise ValueError("Invalid or degenerate palm landmarks")
        return v / length
    forward = unit(np.asarray(middle) - wrist)
    across = np.asarray(index) - pinky
    across = unit(across - np.sum(across * forward, axis=-1, keepdims=True) * forward)
    return np.stack([forward, across, np.cross(forward, across)], axis=-1)


def robot_palm_frame(robot, urdf_path, origin_link="right_hand_link"):
    """Use the first actuated joint on each fingertip chain as a fixed MCP anchor.

    Its child origin is at the joint pivot, so its position is independent of
    that joint's rotation. Reject chains whose anchor is not fixed to the palm.
    """
    tips = {"index": "right_hand_index_rota_tip", "middle": "right_hand_mid_tip",
            "pinky": "right_hand_pinky_tip"}
    parents = {j.find("child").get("link"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    origin = robot.get_link_pose(robot.get_link_index(origin_link))
    anchors = {}
    for name, tip in tips.items():
        chain, link = [], tip
        while link != origin_link:
            if link not in parents:
                raise ValueError(f"{tip} is not a descendant of {origin_link}")
            joint = parents[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        moving = [j for j in reversed(chain) if j.get("type") != "fixed"]
        if not moving or moving[0].get("type") not in ("revolute", "continuous"):
            raise ValueError(f"Cannot identify a fixed MCP pivot for {name}")
        link = moving[0].find("child").get("link")
        point = robot.get_link_pose(robot.get_link_index(link))[:3, 3]
        anchors[name] = origin[:3, :3].T @ (point - origin[:3, 3])
    return palm_frame(np.zeros(3), anchors["index"], anchors["middle"], anchors["pinky"]), anchors


def wrist_targets(wrist_pos, human_palm_rotmat, robot_palm_basis, offset):
    """offset is robot origin minus human wrist, expressed in palm coordinates (m)."""
    rotations = human_palm_rotmat @ robot_palm_basis.T
    positions = wrist_pos + np.einsum("tij,j->ti", human_palm_rotmat, offset)
    return positions, rotations
