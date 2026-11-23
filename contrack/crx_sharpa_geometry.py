"""FANUC CRX-5iA + Sharpa Wave joint order and URDF kinematics (one side).

Unlike RB-Y1A (one URDF, two arms, one shared base translation), the CRX setup
is two independent single-arm URDFs, each with its own base pose. Joint order
per side is J1..J6 followed by the 22 Sharpa joints, matching dual_crx_control /
dual_sharpa_wave_ros2 and the Isaac scene bundle. Needs numpy/scipy only.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ARM_NAMES = [f"J{i}" for i in range(1, 7)]
HAND_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
ARM_DOF, HAND_DOF = 6, 22
SIDE_DOF = ARM_DOF + HAND_DOF
URDF_NAMES = ("crx5ia_sharpa_wave_left.urdf", "crx5ia_sharpa_wave_right.urdf")
# ROS/URDF degrees from dual_crx_control config/initial_pose.yaml (not pendant J3).
REST_ARM_DEG = {"left": (0., 30., -30., 0., 60., 0.), "right": (-90., -30., 210., 0., -60., 0.)}


def hand_names(side):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return [f"{side}_{s}" for s in HAND_SUFFIXES]


def pose_matrix(xyz_rpy):
    """xyz + fixed-axis roll/pitch/yaw (URDF origin convention) -> 4x4."""
    values = np.asarray(xyz_rpy, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected six finite values: x y z roll pitch yaw")
    matrix = np.eye(4)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    return matrix


def scene_matrix(x, y, z, yaw):
    """Rigid scene placement that keeps the table horizontal (yaw only)."""
    return pose_matrix([x, y, z, 0., 0., yaw])


def _rodrigues(axis, angle):
    k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3) + np.sin(angle) * k + (1. - np.cos(angle)) * (k @ k)


class CrxSharpaKinematics:
    """Fixed-root URDF FK for one CRX+Sharpa side: T_child = T_parent @ T_origin @ T_motion.

    Poses are expressed in the URDF root (the robot base; the bundled world ->
    base_link joint is identity). The base pose in the scene is applied outside.
    """
    def __init__(self, path, side):
        self.path, self.side = Path(path), side
        self.arm_names = list(ARM_NAMES)
        self.hand_names = hand_names(side)
        self.palm_link = f"{side}_hand_C_MC"
        self.tip_links = [f"{side}_{f}_fingertip" for f in FINGERS]
        # Child origins of PIP/IP and DIP are anatomical joint pivots, not tip proxies.
        self.pip_links = [f"{side}_thumb_DP"] + [f"{side}_{f}_MP" for f in FINGERS[1:]]
        self.dip_links = [f"{side}_{f}_DP" for f in FINGERS[1:]]
        root = ET.parse(path).getroot()
        links = {e.get("name") for e in root.findall("link")}
        joints, children = [], set()
        for element in root.findall("joint"):
            kind = element.get("type")
            if kind not in ("fixed", "revolute") or element.find("mimic") is not None:
                raise ValueError(f"Unsupported joint {element.get('name')}: {kind}/mimic")
            origin = element.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.array(origin.get("xyz", "0 0 0").split(), dtype=float)
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.array(origin.get("rpy", "0 0 0").split(), dtype=float)).as_matrix()
            axis, bounds = np.array([1., 0., 0.]), (0., 0.)
            if kind != "fixed":
                axis = np.array(element.find("axis").get("xyz").split(), dtype=float)
                if np.linalg.norm(axis) < 1e-12:
                    raise ValueError("Zero joint axis")
                axis = axis / np.linalg.norm(axis)
                limit = element.find("limit")
                bounds = (float(limit.get("lower")), float(limit.get("upper")))
            child = element.find("child").get("link")
            joints.append(dict(name=element.get("name"), parent=element.find("parent").get("link"),
                               child=child, origin=transform, axis=axis, kind=kind, bounds=bounds))
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
        active = {j["name"] for j in self.joints if j["kind"] != "fixed"}
        if active != set(self.arm_names + self.hand_names):
            raise ValueError(f"Expected exactly 6 CRX + 22 {side} Sharpa joints in {self.path.name}; "
                             f"unexpected {sorted(active ^ set(self.arm_names + self.hand_names))}")
        needed = [self.palm_link] + self.tip_links + self.pip_links + self.dip_links
        missing = [n for n in needed if n not in links]
        if missing:
            raise ValueError(f"URDF lacks hand links {missing}")

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
                motion[:3, :3] = _rodrigues(joint["axis"], angle)
                transform = transform @ motion
            poses[joint["child"]] = poses[joint["parent"]] @ transform
        return poses

    def palm_pose(self, arm_q):
        return self.fk(dict(zip(self.arm_names, arm_q)))[self.palm_link]

    def hand_points(self, finger_q):
        """Tip, PIP/IP and DIP positions in the palm-link frame."""
        poses = self.fk(dict(zip(self.hand_names, finger_q)))
        origin = poses[self.palm_link]
        def local(names):
            points = np.stack([poses[n][:3, 3] for n in names])
            return (points - origin[:3, 3]) @ origin[:3, :3]
        return local(self.tip_links), local(self.pip_links), local(self.dip_links)

    def palm_basis(self):
        """Same definition as the RB-Y1A solver: MCP pivots at the zero pose, frozen once."""
        poses = self.fk()
        origin = poses[self.palm_link]
        anchors = {f: origin[:3, :3].T @ (poses[f"{self.side}_{f}_MCP_VL"][:3, 3] - origin[:3, 3])
                   for f in ("index", "middle", "pinky")}
        forward = anchors["middle"] / np.linalg.norm(anchors["middle"])
        across = anchors["index"] - anchors["pinky"]
        across = across - (across @ forward) * forward
        if np.linalg.norm(across) < 1e-8:
            raise ValueError("Degenerate Sharpa palm reference")
        across /= np.linalg.norm(across)
        return np.column_stack([forward, across, np.cross(forward, across)]), anchors
