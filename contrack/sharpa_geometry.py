"""RB-Y1A/Sharpa joint order and URDF kinematics, independent of XHand.

Names/order follow ConTrack robots/rby1a_sharpa.py. Runtime needs numpy/scipy,
not Isaac Sim, Pinocchio or meshes. Use the full assembly URDF consistently.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ARM_NAMES = [f"right_arm_{i}" for i in range(7)]
HAND_NAMES = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE",
    "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
    "right_pinky_PIP", "right_pinky_DIP",
]
DEFAULT_ARM = np.array([0., 0., 0., -1.57, 0., 0., 0.])
PALM_LINK = "right_hand_C_MC"
TIP_LINKS = [f"right_{f}_fingertip" for f in FINGERS]
# Child origins of PIP/IP and DIP are anatomical joint pivots, not tip proxies.
PIP_LINKS = ["right_thumb_DP"] + [f"right_{f}_MP" for f in FINGERS[1:]]
DIP_LINKS = [f"right_{f}_DP" for f in FINGERS[1:]]
REFERENCE_COMMIT = "2542f40cf7af94ed7ebe7b378cce4dbff0d2121f"


class SharpaKinematics:
    """Fixed-root URDF FK: T_child = T_parent @ T_origin @ T_motion."""
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        links = {e.get("name") for e in root.findall("link")}
        joints, children = [], set()
        for element in root.findall("joint"):
            kind = element.get("type")
            if kind not in ("fixed", "revolute", "continuous") or element.find("mimic") is not None:
                raise ValueError(f"Unsupported joint {element.get('name')}: {kind}/mimic")
            parent = element.find("parent").get("link")
            child = element.find("child").get("link")
            origin = element.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(
                    origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis_el = element.find("axis")
            axis = np.fromstring(axis_el.get("xyz", "1 0 0") if axis_el is not None else "1 0 0", sep=" ")
            if kind == "fixed":
                axis = np.array([1., 0., 0.])  # fixed joints may legally declare a zero axis
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("Zero joint axis")
            axis /= np.linalg.norm(axis)
            limits = element.find("limit")
            bounds = (-np.pi, np.pi) if kind == "continuous" else (
                (float(limits.get("lower")), float(limits.get("upper"))) if kind != "fixed" else (0., 0.))
            joints.append(dict(name=element.get("name"), parent=parent, child=child,
                               origin=transform, axis=axis, kind=kind, bounds=bounds))
            children.add(child)
        roots = links - children
        if len(roots) != 1:
            raise ValueError("Expected one URDF root")
        self.root = roots.pop()
        self.joints, reached = [], {self.root}
        while joints:
            ready = [j for j in joints if j["parent"] in reached]
            if not ready:
                raise ValueError("Disconnected/cyclic URDF")
            for joint in ready:
                self.joints.append(joint)
                reached.add(joint["child"])
                joints.remove(joint)
        self.by_name = {j["name"]: j for j in self.joints}
        missing = set(ARM_NAMES + HAND_NAMES) - self.by_name.keys()
        if missing:
            raise ValueError(f"Full RB-Y1A+Sharpa URDF required; missing {sorted(missing)}")
        self.active = {j["name"] for j in self.joints if j["kind"] != "fixed"}
        expected = set(ARM_NAMES + HAND_NAMES + [n.replace("right_", "left_") for n in ARM_NAMES + HAND_NAMES])
        if self.active != expected:
            raise ValueError("Unexpected movable joints; confirm fixed torso/head and 58 arm/hand joints")

    def limits(self, names):
        return np.array([self.by_name[n]["bounds"] for n in names])

    def fk(self, q=None):
        q = {} if q is None else q
        poses = {self.root: np.eye(4)}
        for joint in self.joints:
            transform = joint["origin"]
            angle = q.get(joint["name"], 0.)
            if joint["kind"] != "fixed" and angle != 0.:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(joint["axis"] * angle).as_matrix()
                transform = transform @ motion
            poses[joint["child"]] = poses[joint["parent"]] @ transform
        return poses

    def hand_points(self, finger_q):
        poses = self.fk(dict(zip(HAND_NAMES, finger_q)))
        origin = poses[PALM_LINK]
        def local(names):
            points = np.stack([poses[n][:3, 3] for n in names])
            return (points - origin[:3, 3]) @ origin[:3, :3]
        return local(TIP_LINKS), local(PIP_LINKS), local(DIP_LINKS)


def palm_basis(model):
    """MCP locations at zero reference pose; pinky CMC is NOT an MCP proxy.

    Freeze this basis once. Allowing its pinky MCP to move with CMC during
    retargeting would couple finger motion back into the arm target.
    """
    poses = model.fk()
    origin = poses[PALM_LINK]
    anchors = {f: origin[:3, :3].T @ (poses[f"right_{f}_MCP_VL"][:3, 3] - origin[:3, 3])
               for f in ("index", "middle", "pinky")}
    forward = anchors["middle"] / np.linalg.norm(anchors["middle"])
    across = anchors["index"] - anchors["pinky"]
    across -= (across @ forward) * forward
    if np.linalg.norm(across) < 1e-8:
        raise ValueError("Degenerate Sharpa palm reference")
    across /= np.linalg.norm(across)
    return np.column_stack([forward, across, np.cross(forward, across)]), anchors


def wrist_targets(wrist, human_basis, robot_basis, offset):
    return (wrist + np.einsum("tij,j->ti", human_basis, offset),
            human_basis @ robot_basis.T)
