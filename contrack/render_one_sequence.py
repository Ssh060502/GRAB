"""Render ONE GRAB sequence (real human body/hand + object) to a video, for a direct visual
comparison against our retargeted robot animation. Adapted from examples/render_grab.py, which
only works as a random-10-sequences demo hard-coded to "*eat*" filenames -- this version takes
an explicit --seq (and optional --start/--end, matching the crop used for the ConTrack h5) and
assembles the rendered frames into an mp4 instead of leaving a folder of loose PNGs.

Renders in GRAB's own native coordinate frame -- NOT converted through Rm/tm into ConTrack world
-- since this is a rendering of the ORIGINAL source data, standalone, not meant to be spatially
composited with the robot video. Treat the two videos as a side-by-side "does the grip look
similar in spirit" comparison, not a literal overlay (different camera, different coordinate
frame, no simulator in common).

Camera is placed automatically from the object's own position range during the rendered frames
(a fixed 3/4-view offset, not a tracking/following camera), instead of render_grab.py's hard-coded
camera_pose (tuned for a different scene).

Needs pyrender, which needs a working OpenGL context even offscreen -- if rendering fails or
produces a black/garbled image on a headless server, try:
    export PYOPENGL_PLATFORM=egl
before running (a common fix for headless pyrender; not verified against this specific server).

Runs in the `grab` env (needs smplx/torch, same as compute_hand_keypoints.py), plus one more:
    pip install pyrender

Example:
    python contrack/render_one_sequence.py --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset \
        --model-path /mnt/ssd1/shenghe/datasets/GRAB_models --seq s1/teapot_pour_1 \
        --start 99 --end 819 --out out/teapot_grab_source.mp4
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for tools./grab.

import numpy as np
import smplx
import torch

from tools.meshviewer import Mesh, MeshViewer, colors
from tools.objectmodel import ObjectModel
from tools.utils import parse_npz, params2torch, to_cpu


def lookat_pose(eye, target, world_up=(0, 0, 1)):
    """4x4 camera-to-world matrix, OpenGL/pyrender convention (camera looks down its own -z)."""
    eye, target, world_up = (np.asarray(v, float) for v in (eye, target, world_up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -forward, eye
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--seq", required=True, help="e.g. s1/teapot_pour_1")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--azim", type=float, default=-45,
                    help="camera azimuth in degrees, rotation around the vertical axis (0=facing +x, "
                         "90=facing +y, ...). Default -45 matches the original fixed view. Try adding/"
                         "subtracting 90 at a time to see the object from a different side.")
    ap.add_argument("--elev", type=float, default=26,
                    help="camera elevation in degrees above the horizontal plane (0=straight on, 90=top-down)")
    ap.add_argument("--dist-scale", type=float, default=1.0, help="multiply the auto camera distance (zoom)")
    args = ap.parse_args()

    seq_data = parse_npz(os.path.join(args.grab_root, "grab", args.seq + ".npz"))
    n_comps, gender = seq_data["n_comps"], seq_data["gender"]
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else seq_data.n_frames
    T = end - start
    print(f"{args.seq}: {seq_data.n_frames} frames total, rendering [{start}:{end}] -> {T} frames")

    def crop(params):
        return {k: v[start:end] for k, v in params.items()}

    sbj_mesh = os.path.join(args.grab_root, seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
    sbj_m = smplx.create(model_path=args.model_path, model_type="smplx", gender=gender,
                         num_pca_comps=n_comps, v_template=sbj_vtemp, batch_size=T)
    verts_sbj = to_cpu(sbj_m(**params2torch(crop(seq_data.body.params))).vertices)

    obj_mesh_path = os.path.join(args.grab_root, seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh_path)
    obj_m = ObjectModel(v_template=np.array(obj_mesh.vertices), batch_size=T)
    verts_obj = to_cpu(obj_m(**params2torch(crop(seq_data.object.params))).vertices)

    table_mesh_path = os.path.join(args.grab_root, seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh_path)
    table_m = ObjectModel(v_template=np.array(table_mesh.vertices), batch_size=T)
    verts_table = to_cpu(table_m(**params2torch(crop(seq_data.table.params))).vertices)

    contact_obj = seq_data["contact"]["object"][start:end]
    contact_body = seq_data["contact"]["body"][start:end]

    mv = MeshViewer(width=args.width, height=args.height, offscreen=True)
    center = verts_obj.reshape(-1, 3).mean(0)
    radius = (float(np.linalg.norm(verts_obj.reshape(-1, 3) - center, axis=-1).max()) + 0.5) * args.dist_scale
    azim, elev = np.radians(args.azim), np.radians(args.elev)
    direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    eye = center + radius * direction
    mv.update_camera_pose(lookat_pose(eye, center))

    frame_ids = list(range(0, T, args.stride))
    frames = []
    for i, frame in enumerate(frame_ids):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors["yellow"])
        o_mesh.set_vertex_colors(vc=colors["red"], vertex_ids=contact_obj[frame] > 0)
        s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors["pink"], smooth=True)
        s_mesh.set_vertex_colors(vc=colors["red"], vertex_ids=contact_body[frame] > 0)
        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors["white"])
        mv.set_static_meshes([o_mesh, s_mesh, t_mesh])

        tmp_path = args.out + f".tmp_{i:05d}.png"
        mv.save_snapshot(tmp_path)
        import imageio.v2 as imageio
        frames.append(imageio.imread(tmp_path))
        os.remove(tmp_path)
        if i % 20 == 0:
            print(f"rendered {frame}/{T}")
    # not calling mv.close_viewer() here: tools/meshviewer.py's close_viewer() reads
    # self.viewer.is_active, an attribute pyrender.OffscreenRenderer doesn't have (only the
    # interactive pyrender.Viewer does) -- a pre-existing bug in that shared GRAB helper, not
    # something introduced here. Skipping it is harmless: nothing else needs explicit cleanup
    # before the process exits.

    import imageio.v2 as imageio
    fps = 120.0 / args.stride
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        imageio.mimsave(args.out, frames, fps=fps)
        print("saved", args.out)
    except Exception as exc:
        print(f"[warn] mp4 encode failed ({exc}); saving PNG frames instead")
        frame_dir = args.out + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        for i, img in enumerate(frames):
            imageio.imwrite(os.path.join(frame_dir, f"frame_{i:05d}.png"), img)
        print("saved", frame_dir)


if __name__ == "__main__":
    main()
