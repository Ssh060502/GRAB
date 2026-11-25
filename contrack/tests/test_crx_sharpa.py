"""CRX-5iA + Sharpa FK/IK tests on the real right-side URDF. No GRAB models or Isaac Sim.

The H5 round-trip test needs h5py and is skipped where it cannot be imported.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

CONTRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRACK))
from crx_sharpa_geometry import (CrxSharpaKinematics, ARM_NAMES, HAND_SUFFIXES, SIDE_DOF,
                                 REST_ARM_DEG, pose_matrix, scene_matrix)
from retarget_crx_sharpa import (retarget_side, solve_arm, choose_turn, transform_points,
                                 DEFAULT_LEFT_BASE, DEFAULT_RIGHT_BASE, ARM_SEEDS_DEG)
URDF = Path(__file__).parent / "crx_assets" / "crx5ia_sharpa_wave_right.urdf"
# Copy of ConTrack assets/urdf/crx_sharpa_left.urdf (calibrated hand mount); kinematics only.
LEFT_URDF = Path(__file__).parent / "crx_assets" / "crx_sharpa_left_calibrated.urdf"
try:
    import h5py
except Exception:  # missing, or a broken binary install
    h5py = None


def synthetic_targets(model, base, arm_path, finger_path):
    """World-frame keypoints produced by the robot itself, so zero error is attainable."""
    basis, _ = model.palm_basis()
    arrays = {k: [] for k in ("wrist", "palm", "tips", "pip", "dip")}
    for arm, fingers in zip(arm_path, finger_path):
        palm = base @ model.palm_pose(arm)
        tip, pip, dip = model.hand_points(fingers)
        arrays["wrist"].append(palm[:3, 3])
        arrays["palm"].append(palm[:3, :3] @ basis)
        for name, value in (("tips", tip), ("pip", pip), ("dip", dip)):
            arrays[name].append(palm[:3, 3] + value @ palm[:3, :3].T)
    return {k: np.array(v) for k, v in arrays.items()}


class CrxSharpaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = CrxSharpaKinematics(URDF, "right")
        cls.base = pose_matrix(DEFAULT_RIGHT_BASE)
        rest = np.deg2rad(REST_ARM_DEG["right"])
        steps = np.linspace(0., 1., 8)[:, None]
        cls.arm_path = rest + steps * np.array([.25, -.2, .15, -.4, .3, .35])
        low, high = cls.model.limits(cls.model.hand_names).T
        cls.finger_path = low + (.15 + .35 * steps) * (high - low)

    def test_joint_order_and_count(self):
        self.assertEqual(self.model.arm_names, ARM_NAMES)
        self.assertEqual(self.model.hand_names, ["right_" + s for s in HAND_SUFFIXES])
        self.assertEqual(len(self.model.arm_names + self.model.hand_names), SIDE_DOF)
        with self.assertRaises(ValueError):
            CrxSharpaKinematics(URDF, "left")  # right URDF has no left_* joints

    def test_zero_pose_matches_crx5ia_dimensions(self):
        poses = self.model.fk()
        np.testing.assert_allclose(poses["base_link"], np.eye(4), atol=1e-12)
        # 0.185 base height + 0.41 upper arm; 0.43 forearm; 0.13 wrist offset; 0.145 flange.
        np.testing.assert_allclose(poses["flange"][:3, 3], [.575, -.13, .595], atol=1e-9)

    def test_palm_basis_is_right_handed_orthonormal(self):
        basis, anchors = self.model.palm_basis()
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1., places=12)
        np.testing.assert_allclose(anchors["middle"], [0, .01, .0989], atol=1e-9)

    def test_arm_ik_respects_rotated_base_and_limits(self):
        target = self.model.palm_pose(self.arm_path[3])
        result = solve_arm(self.model, target[:3, 3], target[:3, :3],
                           [np.deg2rad(s) for s in ARM_SEEDS_DEG])
        achieved = self.model.palm_pose(result.x)
        np.testing.assert_allclose(achieved, target, atol=1e-6)
        bounds = self.model.limits(self.model.arm_names)
        self.assertTrue(np.all(result.x >= bounds[:, 0]) and np.all(result.x <= bounds[:, 1]))

    def test_choose_turn_keeps_pose_and_prefers_room_or_continuity(self):
        bounds = self.model.limits(self.model.arm_names)
        q = np.deg2rad([190., 50., -34., -159., 62., 225.])
        first = choose_turn(q, bounds)
        # J1 190 -> -170 and J6 225 (at its limit) -> -135; J2/J3/J5 have no full-turn choice.
        np.testing.assert_allclose(np.rad2deg(first), [-170., 50., -34., -159., 62., -135.], atol=1e-9)
        np.testing.assert_allclose(self.model.palm_pose(first), self.model.palm_pose(q), atol=1e-9)
        # With a previous frame the nearest equivalent angle wins, even near a limit.
        near = choose_turn(np.deg2rad([-165., 50., -34., -159., 62., -142.]), bounds,
                           previous=np.deg2rad([195., 50., -34., -159., 62., 220.]))
        np.testing.assert_allclose(np.rad2deg(near), [195., 50., -34., -159., 62., 218.], atol=1e-9)
        # 230 would be outside J6's +-225 limit, so -130 is the only option and is kept.
        stuck = choose_turn(np.deg2rad([0., 0., 0., 0., 0., -130.]), bounds,
                            previous=np.deg2rad([0., 0., 0., 0., 0., 224.]))
        self.assertAlmostEqual(np.rad2deg(stuck[5]), -130., places=9)

    def test_first_frame_seed_near_a_limit_is_turned_away_from_it(self):
        q = np.deg2rad([-90., -30., 210., 0., -60., 215.])  # J6 10 deg from its +225 limit
        target = self.model.palm_pose(q)
        result = solve_arm(self.model, target[:3, 3], target[:3, :3], [q])
        np.testing.assert_allclose(self.model.palm_pose(result.x), target, atol=1e-6)
        self.assertAlmostEqual(np.rad2deg(result.x[5]), 215. - 360., places=4)

    def test_retarget_recovers_reachable_motion_on_the_rest_branch(self):
        arrays = synthetic_targets(self.model, self.base, self.arm_path, self.finger_path)
        result = retarget_side(self.model, arrays, self.base)
        self.assertLess(result["pos_err"].max(), 1e-6)
        self.assertLess(result["rot_err"].max(), 1e-4)
        self.assertLess(result["finger_err"].max(), 5e-4)
        # Same IK branch as the lab rest pose, and continuous along the path.
        np.testing.assert_allclose(result["qpos"][:, :6], self.arm_path, atol=1e-5)
        np.testing.assert_allclose(result["actual_pos"], arrays["wrist"], atol=1e-6)

    def left_setup(self):
        model = CrxSharpaKinematics(LEFT_URDF, "left")
        steps = np.linspace(0., 1., 8)[:, None]
        arm_path = np.deg2rad(REST_ARM_DEG["left"]) + steps * np.array([-.3, .2, -.15, .4, -.3, .35])
        low, high = model.limits(model.hand_names).T
        return model, pose_matrix(DEFAULT_LEFT_BASE), arm_path, low + (.15 + .35 * steps) * (high - low)

    def test_left_palm_basis_is_right_handed_orthonormal(self):
        model, _, _, _ = self.left_setup()
        self.assertEqual(model.hand_names, ["left_" + s for s in HAND_SUFFIXES])
        basis, _ = model.palm_basis()
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1., places=12)

    def test_retarget_recovers_reachable_left_motion(self):
        model, base, arm_path, finger_path = self.left_setup()
        arrays = synthetic_targets(model, base, arm_path, finger_path)
        result = retarget_side(model, arrays, base)
        self.assertLess(result["pos_err"].max(), 1e-6)
        self.assertLess(result["rot_err"].max(), 1e-4)
        self.assertLess(result["finger_err"].max(), 5e-4)
        np.testing.assert_allclose(result["qpos"][:, :6], arm_path, atol=1e-5)

    def test_velocity_limits_come_from_the_urdf(self):
        np.testing.assert_allclose(self.model.velocity_limits(self.model.arm_names),
                                   np.deg2rad([150., 150., 180., 225., 225., 225.]), atol=1e-9)

    def test_scene_matrix_keeps_table_horizontal(self):
        scene = scene_matrix(.15, -.4, -.175, np.deg2rad(90.))
        np.testing.assert_allclose(scene[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(transform_points(scene, np.array([[1., 0., 0.]])),
                                   [[.15, .6, -.175]], atol=1e-12)

    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_command_line_writes_new_h5_and_leaves_input_untouched(self):
        scene = scene_matrix(.15, -.4, -.175, 0.)
        crx = synthetic_targets(self.model, self.base, self.arm_path, self.finger_path)
        inverse = np.linalg.inv(scene)
        count = len(self.arm_path)
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "in.h5", Path(folder) / "out.h5"
            with h5py.File(source, "w") as f:
                f.create_dataset("qpos", data=np.full((count, 2, 29), np.nan, np.float32))
                f.create_dataset("is_rhand", data=np.array([0, 1], np.uint8))
                f.create_dataset("fps", data=np.array([120., 120.], np.float32))
                f.create_dataset("urdf_name", data=np.array([b"rby1a_sharpa.urdf"] * 2))
                f.create_dataset("base_translation", data=np.zeros((2, 3), np.float32))
                g = f.create_group("object_tracks/0")
                g.create_dataset("translations", data=np.tile([.5, 0., .01], (count, 1)).astype(np.float32))
                g.create_dataset("orientations_xyzw", data=np.tile([0., 0., 0., 1.], (count, 1)).astype(np.float32))
                s = f.create_group("grab_source")
                s.create_dataset("Rm", data=np.eye(3))
                s.create_dataset("tm", data=np.zeros(3))
                k = s.create_group("keypoints")
                k.attrs["is_rhand"] = 1
                k.create_dataset("wrist_pos", data=transform_points(inverse, crx["wrist"]))
                k.create_dataset("palm_rotmat", data=inverse[:3, :3] @ crx["palm"])
                for name, fingers in (("tips", 5), ("pip", 5), ("dip", 4)):
                    group = k.create_group(name)
                    order = ("thumb", "index", "middle", "ring", "pinky")[5 - fingers:]
                    for i, finger in enumerate(order):
                        group.create_dataset(finger, data=transform_points(inverse, crx[name][:, i]))
            before = source.read_bytes()
            subprocess.run([sys.executable, str(CONTRACK / "retarget_crx_sharpa.py"), "--h5", str(source),
                            "--out", str(out), "--right-urdf", str(URDF), "--time-scale", "3"],
                           check=True, cwd=CONTRACK)
            self.assertEqual(before, source.read_bytes())
            with h5py.File(out, "r") as f:
                self.assertEqual(f["qpos"].shape, (count, 2, 28))
                np.testing.assert_allclose(f["fps"][:], [40., 40.])
                self.assertEqual(f.attrs["time_scale"], 3.)
                self.assertEqual(len(f.attrs["right_urdf_sha256"]), 64)
                np.testing.assert_allclose(f["qpos"][:, 1, :6], self.arm_path, atol=1e-4)
                np.testing.assert_allclose(f["qpos"][:, 0, :6],
                                           np.tile(np.deg2rad(REST_ARM_DEG["left"]), (count, 1)), atol=1e-6)
                self.assertEqual([n.decode() for n in f["urdf_name"][:]],
                                 ["crx5ia_sharpa_wave_left.urdf", "crx5ia_sharpa_wave_right.urdf"])
                np.testing.assert_allclose(f["object_tracks/0/translations"][0], [.65, -.4, -.165], atol=1e-6)
                np.testing.assert_allclose(f["base_translation"][1], DEFAULT_RIGHT_BASE[:3], atol=1e-6)
                quat = f["base_quat_wxyz"][1]
                np.testing.assert_allclose(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(),
                                           self.base[:3, :3], atol=1e-6)
                self.assertLess(f["grab_source/sharpa_diagnostics/wrist_position_error_m"][:].max(), 1e-5)

            # The optional whole-trajectory mode must work through the CLI/H5
            # interface while preserving the original input and default path.
            searched = Path(folder) / "searched.h5"
            subprocess.run([sys.executable, str(CONTRACK / "retarget_crx_sharpa.py"), "--h5", str(source),
                            "--out", str(searched), "--right-urdf", str(URDF), "--time-scale", "3",
                            "--arm-search"], check=True, cwd=CONTRACK, capture_output=True)
            self.assertEqual(before, source.read_bytes())
            with h5py.File(searched, "r") as f:
                self.assertEqual(f.attrs["arm_solver"], "trajectory_search")
                search = f["grab_source/sharpa_diagnostics/arm_search"]
                selected = int(search.attrs["selected_candidate"])
                self.assertEqual(search["candidate_summary"][selected, 0], -1)
                self.assertLess(f["grab_source/sharpa_diagnostics/wrist_position_error_m"][:].max(), .001)
                self.assertLess(np.abs(np.diff(f["qpos"][:,1,:6], axis=0)).max(), .3)


    @unittest.skipIf(h5py is None, "h5py unavailable")
    def test_command_line_retargets_both_hands(self):
        scene = scene_matrix(.15, -.4, -.175, 0.)
        inverse = np.linalg.inv(scene)
        left_model, left_base, left_arm, left_fingers = self.left_setup()
        targets = {"keypoints": synthetic_targets(self.model, self.base, self.arm_path, self.finger_path),
                   "keypoints_left": synthetic_targets(left_model, left_base, left_arm, left_fingers)}
        count = len(self.arm_path)
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "in.h5", Path(folder) / "out.h5"
            with h5py.File(source, "w") as f:
                f.create_dataset("qpos", data=np.full((count, 2, 29), np.nan, np.float32))
                f.create_dataset("is_rhand", data=np.array([0, 1], np.uint8))
                f.create_dataset("fps", data=np.array([120., 120.], np.float32))
                g = f.create_group("object_tracks/0")
                g.create_dataset("translations", data=np.tile([.5, 0., .01], (count, 1)).astype(np.float32))
                g.create_dataset("orientations_xyzw", data=np.tile([0., 0., 0., 1.], (count, 1)).astype(np.float32))
                s = f.create_group("grab_source")
                s.create_dataset("Rm", data=np.eye(3))
                s.create_dataset("tm", data=np.zeros(3))
                for group_name, crx in targets.items():
                    k = s.create_group(group_name)
                    k.attrs["is_rhand"] = int(group_name == "keypoints")
                    k.create_dataset("wrist_pos", data=transform_points(inverse, crx["wrist"]))
                    k.create_dataset("palm_rotmat", data=inverse[:3, :3] @ crx["palm"])
                    for name, fingers in (("tips", 5), ("pip", 5), ("dip", 4)):
                        group = k.create_group(name)
                        order = ("thumb", "index", "middle", "ring", "pinky")[5 - fingers:]
                        for i, finger in enumerate(order):
                            group.create_dataset(finger, data=transform_points(inverse, crx[name][:, i]))
            subprocess.run([sys.executable, str(CONTRACK / "retarget_crx_sharpa.py"), "--h5", str(source),
                            "--out", str(out), "--right-urdf", str(URDF), "--left-urdf", str(LEFT_URDF),
                            "--hands", "both"], check=True, cwd=CONTRACK)
            with h5py.File(out, "r") as f:
                np.testing.assert_allclose(f["qpos"][:, 0, :6], left_arm, atol=1e-4)
                np.testing.assert_allclose(f["qpos"][:, 1, :6], self.arm_path, atol=1e-4)
                self.assertEqual(f.attrs["retargeted_hands"], "both")
                self.assertEqual(len(f.attrs["left_urdf_sha256"]), 64)
                for suffix in ("", "_left"):
                    self.assertLess(
                        f[f"grab_source/sharpa_diagnostics{suffix}/wrist_position_error_m"][:].max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
