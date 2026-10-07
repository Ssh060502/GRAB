# Stable palm wrist retargeting (right XArm7 / XHand)

Run keypoint extraction in the GRAB environment, then robot IK in the
retarget environment. Replace the paths below with your dataset/model/assets paths.
Existing H5 files need fresh keypoint extraction because the old files lack MCPs.

```bash
python contrack/mano_keypoints.py --h5 out/clip.h5 --grab-root /path/to/GRAB_dataset --model-path /path/to/GRAB_models
python contrack/retarget_xarm_xhand.py --h5 out/clip.h5 --assets-dir /path/to/ConTrack/assets
```

`retarget_xarm_xhand.py` computes the robot palm basis and target wrist poses
directly using `palm_geometry.py`, then solves the arm and fingers. No separate
calibration command or previously saved wrist targets are required.

The human palm basis has columns:

1. wrist to middle MCP (forward);
2. pinky MCP to index MCP, orthogonalized against forward (across);
3. forward cross across (normal).

The normal is a right-handed geometric convention. No anatomical palm-facing
sign is inferred from a thumb tip or from the object's origin.
MANO MCP indices are index=1, middle=4, pinky=7, ring=10, thumb=13 in its 16-joint
output. Their centers stay attached to the palm when the fingers rotate.

Robot MCP anchors are the first revolute/continuous joint pivots along each
index/middle/pinky chain below `right_hand_link`, expressed in that link's frame.
Check the selected joint pivots against the actual robot geometry. An
unusual URDF whose first movable pivot is not the corresponding anatomical MCP
requires a robot-specific landmark mapping. Missing/degenerate anchors fail.

If `B_h(t)` is the human palm basis in world coordinates and `B_r` is the robot
palm basis in hand-link coordinates, the target is:

```text
R_target(t) = B_h(t) @ B_r.T
p_target(t) = p_MANO_wrist(t) + B_h(t) @ offset_palm
```

`--wrist-offset-palm forward across normal` specifies the robot origin minus
MANO wrist in meters, using the above palm axes. Default is `0 0 0`, preserving
wrist-center alignment. This anatomical origin correspondence is not determined
automatically from hands with different dimensions. Supply the offset to
`retarget_xarm_xhand.py` when an origin adjustment is needed.

Arm IK matches position and rotation-vector error (rotation scale 0.05 m/rad).
After IK, finger world targets are transformed using the **actual** wrist FK.
The standalone hand FK vectors are expressed in `right_hand_link` coordinates,
including when that link is rotated relative to the standalone URDF root.
Contact-priority remains opt-in; its targets also use the actual wrist pose.
All robot-base rotations are assumed identity, as in the original converter.

H5 additions:

- `grab_source/keypoints/mcp/{finger}`: world MCP positions;
- `grab_source/keypoints/palm_rotmat`: human world palm basis;
- `grab_source/wrist_targets/positions`, `rotations`: target robot wrist pose;
- `grab_source/wrist_targets/actual_positions`, `actual_rotations`: achieved FK,
  written by retargeting.

Retargeting automatically reports wrist position and full rotation tracking errors,
world fingertip errors, joint-limit saturation and sampled reach. These provide
the routine checks without a separate checker command. The optional
`check_wrist_direction.py` additionally reports forward/across/normal errors
separately; removing it does not affect extraction or retargeting.
No Sharpa IK implementation is added here.

Dependency-light regression checks:

```bash
python contrack/test_palm_geometry.py
```
