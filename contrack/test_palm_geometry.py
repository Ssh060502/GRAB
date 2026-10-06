"""Dependency-light regression checks for palm frames and wrist coordinates."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from palm_geometry import palm_frame, robot_palm_frame, wrist_targets


class PalmGeometryTests(unittest.TestCase):
    def test_rigid_equivariance_and_offset(self):
        points = np.array([[0., 0., 0.], [1., .3, 0.], [1., 0., 0.], [1., -.3, 0.]])
        basis = palm_frame(*points)
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        moved = points @ rotation.T + [2., 3., 4.]
        world_basis = palm_frame(*moved)
        np.testing.assert_allclose(world_basis, rotation @ basis)
        np.testing.assert_allclose(world_basis.T @ world_basis, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(world_basis), 1.)
        pos, rot = wrist_targets(moved[:1], world_basis[None], basis, [.02, .03, .04])
        np.testing.assert_allclose(rot[0] @ basis, world_basis)
        np.testing.assert_allclose(pos[0], moved[0] + world_basis @ [.02, .03, .04])

    def test_degenerate_landmarks_fail(self):
        with self.assertRaises(ValueError):
            palm_frame(np.zeros(3), [1., 0., 0.], [2., 0., 0.], [3., 0., 0.])

    def test_urdf_anchor_uses_first_pivot_not_tip(self):
        origin = np.eye(4)
        origin[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        origin[:3, 3] = [3., 4., 5.]
        poses = {"right_hand_link": origin}
        xml = "<robot name='test'>"
        for name, tip, point in (("index", "right_hand_index_rota_tip", [1., .3, 0.]),
                                 ("middle", "right_hand_mid_tip", [1., 0., 0.]),
                                 ("pinky", "right_hand_pinky_tip", [1., -.3, 0.])):
            pose = origin.copy()
            pose[:3, 3] += origin[:3, :3] @ point
            poses[name] = pose
            xml += f"<joint name='{name}' type='revolute'><parent link='right_hand_link'/><child link='{name}'/></joint>"
            xml += f"<joint name='{name}_tip' type='fixed'><parent link='{name}'/><child link='{tip}'/></joint>"
        xml += "</robot>"
        robot = SimpleNamespace(dof=3, compute_forward_kinematics=lambda q: None,
                                get_link_index=lambda name: name, get_link_pose=lambda name: poses[name])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hand.urdf'
            path.write_text(xml)
            basis, _ = robot_palm_frame(robot, path)
        np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)

    def test_fingers_use_actual_wrist_and_rotated_hand_root(self):
        # Exercise the real solver residual without importing unavailable scipy/pin.
        source = ast.parse(Path(__file__).with_name('retarget_xarm_xhand.py').read_text(encoding='utf-8'))
        fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'solve_fingers')
        root_rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        wrist_rot = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        wrist = np.array([2., 3., 4.])
        local = np.array([[.1, .01 * i, .02] for i in range(5)])
        root = np.eye(4)
        root[:3, :3], root[:3, 3] = root_rot, [7., 8., 9.]
        poses = [root]
        for vector in local:
            pose = root.copy()
            pose[:3, 3] += root_rot @ vector
            poses.append(pose)
        robot = SimpleNamespace(dof=1, compute_forward_kinematics=lambda q: None,
                                get_link_pose=lambda index: poses[index])
        names = ('thumb', 'index', 'middle', 'ring', 'pinky')
        targets = dict(zip(names, wrist + local @ wrist_rot.T))
        contact = np.full((5, 3), np.nan)
        contact[0] = targets['thumb'].copy()
        targets['thumb'] = targets['thumb'] + 1.  # override must restore the zero residual
        def optimizer(residual, x0, **kwargs):
            np.testing.assert_allclose(residual(x0), np.zeros(15), atol=1e-12)
            return SimpleNamespace(x=x0)
        namespace = {'np': np, 'least_squares': optimizer}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<solver>', 'exec'), namespace)
        namespace['solve_fingers'](robot, wrist, wrist_rot, np.eye(3), targets,
                                   [0], 0, [1, 2, 3, 4, 5], np.array([[-1., 1.]]),
                                   np.zeros(1), contact_targets=contact)


if __name__ == '__main__':
    unittest.main()
