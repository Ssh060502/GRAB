# Stable palm wrist retargeting (right XArm7 / XHand)

Run keypoint extraction in the GRAB environment, then calibration/IK in the
retarget environment. Replace the paths below with your dataset/model/assets paths.
Existing H5 files need fresh keypoint extraction because the old files lack MCPs.

```bash
python contrack/compute_hand_keypoints.py --h5 out/clip.h5 --grab-root /path/to/GRAB_dataset --model-path /path/to/GRAB_models
python contrack/estimate_calib.py --h5 out/clip.h5 --assets-dir /path/to/ConTrack/assets --write-targets
python contrack/retarget_xarm_xhand.py --h5 out/clip.h5 --assets-dir /path/to/ConTrack/assets
python contrack/check_wrist_direction.py --h5 out/clip.h5 --assets-dir /path/to/ConTrack/assets
```

`estimate_calib.py` is an inspection step; retargeting also computes the palm
mapping itself, so it does not require saved calibration targets. It prints a
legacy RPY equivalent for comparison, not a parameter needed in default palm mode.

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
Inspect the printed anchor positions against the actual robot geometry. An
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
automatically from hands with different dimensions. Supply the same offset to
calibration and retargeting if you use both commands.

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

The direction checker compares forward/across/normal separately. Near-zero target
axis error verifies the mapping construction; it does not independently verify
that the chosen URDF landmarks match the physical anatomy. Actual FK errors
measure the arm's ability to track the target. Finger diagnostics now report
world position errors, including the arm's remaining tracking error.

Legacy root-rotation mapping is available with `--wrist-mode mano --calib-rpy X Y Z`.
The previous direction-checker's `--calib-rpy` option has been replaced by checks
of saved wrist targets and actual FK. No Sharpa IK implementation is added here.

Dependency-light regression checks:

```bash
python contrack/test_palm_geometry.py
```
