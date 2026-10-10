# GRAB to ConTrack: RB-Y1A + Sharpa (right side)

This is independent of the XArm/XHand solver. Existing XHand scripts are unchanged.
The first implementation retargets the right arm and hand. Left arm uses ConTrack's
`[0,0,0,-1.57,0,0,0]` pose and left fingers remain at zero. Left-contact GRAB clips
are therefore not fully reproduced; their contact labels are preserved unchanged.

## Run on the server

Use a **new Sharpa H5**, not your existing XHand file. Replace the paths below.

```bash
# grab environment: object and contact conversion
conda activate grab
python contrack/grab_to_contrack_sharpa.py --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset --seq s1/teapot_pour_1 --out /mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s1_teapot_pour_1_sharpa.h5

# grab environment: MANO wrist/MCP/tips plus intermediate PIP/IP and DIP targets
python contrack/sharpa_keypoints.py --h5 /mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s1_teapot_pour_1_sharpa.h5 --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset --model-path /mnt/ssd1/shenghe/datasets/GRAB_models

# retarget environment: right arm and 22 finger joints
conda activate retarget
python contrack/retarget_rby1a_sharpa.py --h5 /mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s1_teapot_pour_1_sharpa.h5 --assets-dir ~/ConTrack/assets

# skeleton + object MP4; add --frame 300 and use .png for one image
python contrack/visualize_rby1a_sharpa.py --h5 /mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s1_teapot_pour_1_sharpa.h5 --assets-dir ~/ConTrack/assets --out sharpa_teapot.mp4 --stride 4
```

Sharpa extraction reuses the existing MANO extraction function and makes a second
MANO forward pass to expose intermediate joints. Existing XHand code/data need
not be changed. Old keypoints containing only fingertips/MCPs are insufficient:
run `sharpa_keypoints.py` for this route.

Retargeting needs NumPy, SciPy and h5py; visualization additionally needs
matplotlib, imageio and imageio-ffmpeg. Conversion/extraction also needs trimesh
and the existing grab environment's MANO/smplx/torch dependencies. The Sharpa
solver does **not** require Pinocchio, dex_retargeting or Isaac Sim.

```bash
pip install -r contrack/requirements_sharpa.txt
```

## Geometry and optimization

- All robot geometry comes from the **full `rby1a_sharpa.urdf`**, including fixed
  torso and the hand mounting transform. Its hand geometry differs slightly from
  the standalone `sharpa_right.urdf`; do not mix the two in the solver.
- Controlled palm link: `right_hand_C_MC`. Fingertips use the explicit
  `right_{thumb,index,middle,ring,pinky}_fingertip` links, including their fixed offsets.
- Reference palm axes use index/middle/pinky MCP pivot locations at zero pose.
  Pinky MCP is downstream of a movable CMC: freeze the reference once so finger
  motion does not change the arm target. It is not replaced by the CMC pivot.
- Target rotation is human palm basis times robot reference basis transpose.
  Default palm origin is the human wrist center. An explicit
  `--wrist-offset-palm forward across normal` offset is in meters in human palm axes.
  Human/robot anatomical origin offsets are not inferred automatically.
- Whole-robot base translation comes from the H5 (reference: `[0,0,-0.85]`).
  Both sides must share it. Base rotation is assumed identity.
- Arm IK fits position and rotation-vector error, default scale `0.05 m/rad`.
  Poor tracking retries other seeds. The actual FK wrist is used for finger targets.
- Finger objective fits 5 tips, 5 PIP/IP centers and 4 DIP centers, with intermediate
  weight `0.35`; previous finger angle penalty defaults to `0.001 m/rad`.
  MANO thumb joint 14 maps to the robot IP pivot, not a nonexistent thumb PIP.
  Fingers start slightly inside the bounds; poor fits also try a midpoint seed.
- Regularization selects among redundant hand poses, but does not guarantee
  smoothness or a unique anatomical solution. Different hand sizes can still
  leave significant errors. There is no contact replacement, collision avoidance,
  force optimization or physics grasp validation.

## Output and diagnostics

`qpos` remains `(T,2,29)` in left/right order: 7 arm joints then 22 hand joints,
in the exact order used by ConTrack's robot configuration and state loader.
The input is validated before solving; qpos is updated after all frames are solved.
Object and contact datasets are not altered by retargeting. Converter reruns overwrite
the output H5, so use a distinct filename and preserve experiments as needed.

Target and achieved wrists are saved in `grab_source/wrist_targets`. Per-frame
position/rotation/tip errors, optimizer success flags and per-joint lower/upper
saturation are saved in `grab_source/sharpa_diagnostics`, with joint/finger names.
Optimizer success is convergence status, not a guarantee of physical reachability.
Printed tip errors use four decimal places in mm, avoiding misleading rounded zeros.

The renderer shows the full URDF **skeleton**, object mesh and MANO tip targets,
not shaded robot meshes. It uses the H5 frame rate and streams MP4 frames rather
than retaining the entire video in memory. It validates geometry placement but
cannot assess hand-surface contact or collision. Use ConTrack/Isaac Sim for that.

## Verified scope

Repository source: https://github.com/Ssh060502/ConTrack at commit
`2542f40cf7af94ed7ebe7b378cce4dbff0d2121f`.

```bash
python contrack/tests/test_sharpa.py -v
```

Tests use the repository's real URDF and joint configuration: joint order, fixed
mount/reference landmarks, independent chain FK, reachable arm/finger inverse
problems, command-line H5 writeback, synthetic GRAB object/contact conversion.
The repository's real hammer reference H5 was read (`720,2,29`, shared -0.85 m
base), and frame 300 rendered and visually inspected. Actual raw GRAB/MANO model
files are not present locally, so real GRAB extraction/retargeting must still be
checked on the server. These tests establish implementation consistency, not
successful physical grasping or training reward.
