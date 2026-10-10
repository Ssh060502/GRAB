"""Real-URDF FK/IK and H5 integration tests. No GRAB models/Isaac Sim needed."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import h5py
from scipy.spatial.transform import Rotation

CONTRACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CONTRACK))
from sharpa_geometry import (SharpaKinematics, ARM_NAMES, HAND_NAMES, DEFAULT_ARM,
                             PALM_LINK, FINGERS, palm_basis, wrist_targets)
from retarget_rby1a_sharpa import solve_arm, solve_fingers, validate_input
ASSETS=Path(__file__).parent/"sharpa_assets"


class SharpaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=SharpaKinematics(ASSETS/"rby1a_sharpa.urdf")

    def test_joint_order_matches_upstream(self):
        tree=ast.parse((ASSETS/"robot_config.py").read_text())
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=="SHARPA_RIGHT_JOINT_NAMES" for t in n.targets))
        self.assertEqual(HAND_NAMES,ast.literal_eval(assignment.value))
        self.assertEqual(len(self.model.active),58)

    def test_fixed_mount_and_reference_mcp(self):
        poses=self.model.fk()
        arm=poses["link_right_arm_6"]
        relative=np.linalg.inv(arm)@poses[PALM_LINK]
        np.testing.assert_allclose(relative[:3,3],[0,0,-.1261],atol=1e-12)
        np.testing.assert_allclose(relative[:3,:3],Rotation.from_euler("y",np.pi).as_matrix(),atol=1e-9)
        basis,anchors=palm_basis(self.model)
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(anchors["middle"],[0,.01,.0989],atol=1e-12)
        # MCP, not CMC: it includes the downstream .00836/.0048 offset.
        self.assertGreater(np.linalg.norm(anchors["pinky"]-np.array([.01136,-.0263,.0869])),.005)

    def test_fk_matches_independent_chain_composition(self):
        q=dict(zip(ARM_NAMES+HAND_NAMES,np.linspace(-.1,.2,29)))
        poses=self.model.fk(q)
        # Independently traverse only the thumb chain in reverse parent order.
        import xml.etree.ElementTree as ET
        elements=ET.parse(self.model.path).getroot().findall("joint")
        parents={e.find("child").get("link"):e for e in elements}
        chain=[]; link="right_thumb_fingertip"
        while link!=self.model.root:
            element=parents[link];chain.append(element);link=element.find("parent").get("link")
        result=np.eye(4)
        for element in reversed(chain):
            o=element.find("origin"); transform=np.eye(4)
            transform[:3,3]=np.fromstring(o.get("xyz"),sep=" ")
            transform[:3,:3]=Rotation.from_euler("xyz",np.fromstring(o.get("rpy"),sep=" ")).as_matrix()
            result=result@transform
            if element.get("type")!="fixed":
                axis=np.fromstring(element.find("axis").get("xyz"),sep=" ")
                motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(axis*q[element.get("name")]).as_matrix()
                result=result@motion
        np.testing.assert_allclose(result,poses["right_thumb_fingertip"],atol=1e-12)

    def test_reachable_arm_and_fingers(self):
        arm_q=DEFAULT_ARM+np.array([.1,-.3,.2,-.2,.1,.2,-.1])
        target=self.model.fk(dict(zip(ARM_NAMES,arm_q)))[PALM_LINK]
        arm=solve_arm(self.model,target[:3,3],target[:3,:3],DEFAULT_ARM)
        actual=self.model.fk(dict(zip(ARM_NAMES,arm.x)))[PALM_LINK]
        self.assertLess(np.linalg.norm(actual[:3,3]-target[:3,3]),1e-4)
        self.assertLess(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude(),1e-3)
        bounds=self.model.limits(HAND_NAMES)
        fingers_q=bounds[:,0]*.25+bounds[:,1]*.75
        tips,pip,dip=self.model.hand_points(fingers_q)
        fingers=solve_fingers(self.model,dict(tips=tips,pip=pip,dip=dip),np.zeros(22),regularization=0.)
        achieved=self.model.hand_points(fingers.x)
        for a,b in zip(achieved,(tips,pip,dip)):
            self.assertLess(np.max(np.linalg.norm(a-b,axis=1)),.001)

    def test_cli_and_h5_world_frame_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); file=directory/"clip.h5"
            assets=directory/"assets"; (assets/"urdf").mkdir(parents=True)
            import shutil
            shutil.copyfile(self.model.path,assets/"urdf"/"rby1a_sharpa.urdf")
            basis,_=palm_basis(self.model)
            base=np.array([0,0,-.85])
            arm_q=DEFAULT_ARM+np.array([.1,-.3,.2,-.2,.1,.2,-.1])
            pose=self.model.fk(dict(zip(ARM_NAMES,arm_q)))[PALM_LINK]
            bounds=self.model.limits(HAND_NAMES)
            finger_q=bounds[:,0]*.4+bounds[:,1]*.6
            local=self.model.hand_points(finger_q)
            with h5py.File(file,"w") as f:
                f.create_dataset("qpos",data=np.full((2,2,29),np.nan,dtype=np.float32))
                f.create_dataset("is_rhand",data=[0,1])
                f.create_dataset("urdf_name",data=np.array(["rby1a_sharpa.urdf"]*2,dtype="S17"))
                f.create_dataset("base_translation",data=np.stack([base,base]))
                f.create_dataset("fps",data=[120.,120.])
                # Object/contact datasets must survive retargeting unchanged.
                f.create_dataset("contacts/sentinel",data=[41,42])
                k=f.create_group("grab_source/keypoints"); k.attrs["is_rhand"]=1
                wrist=pose[:3,3]+base
                k.create_dataset("wrist_pos",data=np.tile(wrist,(2,1)))
                k.create_dataset("palm_rotmat",data=np.tile(pose[:3,:3]@basis,(2,1,1)))
                for group,points,names in zip(("tips","pip","dip"),local,(FINGERS,FINGERS,FINGERS[1:])):
                    g=k.create_group(group)
                    for name,point in zip(names,points):
                        g.create_dataset(name,data=np.tile(wrist+pose[:3,:3]@point,(2,1)))
                validate_input(f)
            run=subprocess.run([sys.executable,"-B",str(CONTRACK/"retarget_rby1a_sharpa.py"),
                "--h5",str(file),"--assets-dir",str(assets),"--regularization","0"],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            with h5py.File(file) as f:
                self.assertEqual(f["qpos"].shape,(2,2,29))
                self.assertTrue(np.isfinite(f["qpos"][:]).all())
                np.testing.assert_allclose(f["qpos"][:,0,:7],np.tile(DEFAULT_ARM,(2,1)),atol=1e-7)
                np.testing.assert_array_equal(f["contacts/sentinel"][:],[41,42])
                self.assertLess(f["grab_source/sharpa_diagnostics/tip_error_m"][:].max(),.002)
                self.assertEqual(f.attrs["qpos_is_placeholder"],0)

    def test_converter_preserves_contact_and_object_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"grab"/"s1").mkdir(parents=True)
            mesh="v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
            (root/"object.obj").write_text(mesh)
            translations=np.array([[.1,.2,.8],[.11,.2,.8],[.12,.2,.8]])
            rotvec=np.tile([0,0,.2],(3,1))
            labels=np.array([[41,0,0,0],[43,0,0,0],[55,0,0,0]])
            np.savez(root/"grab"/"s1"/"synthetic.npz",n_frames=3,framerate=120.,
                     object=dict(object_mesh="object.obj",params=dict(global_orient=rotvec,transl=translations)),
                     contact=dict(object=labels,threshold=.005),
                     table=dict(params=dict(transl=np.tile([0,0,.8],(3,1)))),
                     lhand=dict(params={}),rhand=dict(params={}))
            file=root/"converted.h5"
            run=subprocess.run([sys.executable,"-B",str(CONTRACK/"grab_to_contrack_sharpa.py"),
                "--grab-root",str(root),"--seq","s1/synthetic","--out",str(file),
                "--start","0","--end","3","--tm","0","0","0"],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            with h5py.File(file) as f:
                self.assertEqual(f["qpos"].shape,(3,2,29))
                self.assertTrue(np.isnan(f["qpos"][:]).all())
                rm=f["grab_source/Rm"][:]
                np.testing.assert_allclose(f["object_tracks/0/translations"][:],translations@rm.T,atol=1e-7)
                expected=rm@Rotation.from_rotvec(rotvec).as_matrix().transpose(0,2,1)
                np.testing.assert_allclose(Rotation.from_quat(f["object_tracks/0/orientations_xyzw"][:]).as_matrix(),expected,atol=1e-7)
                contact=f["contacts/0/is_contact"][:]
                self.assertEqual(contact[0,1,0],1)
                self.assertEqual(contact[1,1,2],1)
                self.assertEqual(contact[2,1,14],1)
                self.assertEqual(contact[:,0].sum(),0)


if __name__=="__main__":
    unittest.main()
