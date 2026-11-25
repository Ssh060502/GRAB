"""Regression checks for complete arm tracking using the real CRX URDF."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

CONTRACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CONTRACK))
from crx_sharpa_geometry import CrxSharpaKinematics, REST_ARM_DEG, pose_matrix
from crx_arm_search import search_arm_trajectory, turn_variants
from retarget_crx_sharpa import retarget_side, DEFAULT_RIGHT_BASE
from test_crx_sharpa import synthetic_targets

MODEL=Path(__file__).parent/"crx_assets"/"crx5ia_sharpa_wave_right.urdf"


class TrajectorySearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=CrxSharpaKinematics(MODEL,"right")

    def test_variants_keep_distinct_turns_and_identical_fk(self):
        q=np.deg2rad([-90,-30,210,0,-60,215])
        variants=turn_variants(q,self.model.limits(self.model.arm_names))
        self.assertTrue(any(abs(v[5]-np.deg2rad(-145))<1e-9 for v in variants))
        self.assertTrue(any(abs(v[5]-q[5])<1e-9 for v in variants))
        for v in variants:
            np.testing.assert_allclose(self.model.palm_pose(v),self.model.palm_pose(q),atol=1e-9)

    def test_alternate_initial_turn_finishes_when_positive_turn_hits_limit(self):
        q=np.deg2rad([-90,-30,210,0,-60,215])
        path=np.tile(q,(9,1));path[:,5]+=np.deg2rad(np.linspace(0,30,9))
        poses=np.array([self.model.palm_pose(v) for v in path])
        result=search_arm_trajectory(self.model,poses[:,:3,3],poses[:,:3,:3],[q],subdivisions=1)
        self.assertTrue(np.all(result["path"][:,5]<0))
        self.assertLess(np.abs(np.diff(result["path"],axis=0)).max(),.1)
        self.assertTrue(np.any(result["summary"][:,0]>=0))
        self.assertTrue(np.any(result["summary"][:,0]<0))
        for actual,target in zip(result["path"],poses):
            np.testing.assert_allclose(self.model.palm_pose(actual),target,atol=1e-5)

    def test_unreachable_frame_fails_without_fitting_fingers(self):
        q=np.deg2rad(REST_ARM_DEG["right"])
        arm_path=np.tile(q,(2,1))
        fingers=np.tile(np.zeros(22),(2,1))
        base=pose_matrix(DEFAULT_RIGHT_BASE)
        arrays=synthetic_targets(self.model,base,arm_path,fingers)
        arrays["wrist"][1]+=[10,0,0]
        with patch("retarget_crx_sharpa.solve_fingers") as finger_solver:
            with self.assertRaisesRegex(RuntimeError,"NOT a proof"):
                retarget_side(self.model,arrays,base,arm_search=True,
                              search_options=dict(subdivisions=0))
            finger_solver.assert_not_called()

    def test_search_complete_before_unchanged_finger_solver_runs(self):
        q=np.deg2rad(REST_ARM_DEG["right"])
        arm_path=q+np.linspace(0,1,4)[:,None]*np.array([.02,-.02,.01,-.03,.02,.03])
        low,high=self.model.limits(self.model.hand_names).T
        hand_path=np.tile(low+.3*(high-low),(4,1))
        base=pose_matrix(DEFAULT_RIGHT_BASE)
        arrays=synthetic_targets(self.model,base,arm_path,hand_path)
        result=retarget_side(self.model,arrays,base,arm_search=True,search_options=dict(subdivisions=1))
        self.assertIsNotNone(result["arm_search"])
        self.assertLess(result["pos_err"].max(),.001)
        self.assertLess(result["rot_err"].max(),.5)
        self.assertLess(result["finger_err"].max(),.001)

    def test_single_frame_and_invalid_options(self):
        q=np.deg2rad(REST_ARM_DEG["right"])
        pose=self.model.palm_pose(q)
        result=search_arm_trajectory(self.model,pose[None,:3,3],pose[None,:3,:3],[q])
        self.assertEqual(result["path"].shape,(1,6))
        with self.assertRaises(ValueError):
            search_arm_trajectory(self.model,pose[None,:3,3],pose[None,:3,:3],[q],max_step=0)

    def test_midpoint_retries_do_not_bypass_stored_frame_step_guard(self):
        def pose(q):
            matrix=np.eye(4);matrix[0,3]=q[0]
            return matrix
        model=SimpleNamespace(side="test",arm_names=["J1"],palm_pose=pose,
                              limits=lambda names: np.array([[-2.,2.]]),
                              velocity_limits=lambda names: np.ones(1))
        # Controlled optimizer that cannot converge across more than .5 rad
        # in one call; subdivision must bridge the gap, but not bypass max_step.
        def optimizer(residual,initial,bounds,**kwargs):
            desired=-residual(np.zeros(1))[0]
            value=desired if abs(desired-initial[0])<=.5 else initial[0]
            return SimpleNamespace(x=np.clip([value],*bounds),success=True)
        positions=np.array([[0.,0,0],[.8,0,0]])
        rotations=np.tile(np.eye(3),(2,1,1))
        with patch("crx_arm_search.least_squares",side_effect=optimizer):
            with self.assertRaises(RuntimeError):
                search_arm_trajectory(model,positions,rotations,[np.zeros(1)],max_step=1.,subdivisions=0)
            result=search_arm_trajectory(model,positions,rotations,[np.zeros(1)],max_step=1.,subdivisions=1)
            np.testing.assert_allclose(result["path"][:,0],[0,.8])
            with self.assertRaises(RuntimeError):
                search_arm_trajectory(model,positions,rotations,[np.zeros(1)],max_step=.3,subdivisions=3)


if __name__=="__main__":
    unittest.main()
