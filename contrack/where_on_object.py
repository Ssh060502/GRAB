"""Where on the object's own mesh does each finger segment actually touch, across a whole GRAB
sequence -- straight from GRAB's own raw contact labels, nothing to do with our retargeting
pipeline, calib, or the rendered qpos at all. Answers: is a given finger's contact pattern a
"tip touch" (one small spot) or a "wrap" (a long smear of contacted vertices along the object),
and roughly where that spot/smear sits on the mesh.

Example:
    python contrack/where_on_object.py --grab-root /mnt/ssd1/shenghe/datasets/GRAB_dataset \
        --seq s1/teapot_pour_1 --out out/teapot_contact_where.png
"""

import argparse
import os

import numpy as np
import trimesh

GRAB_FIRST_ID_RIGHT = 41
SEG_NAMES = [f"{f}_{p}" for f in ("index", "middle", "pinky", "ring", "thumb") for p in ("proximal", "middle", "distal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fingers", nargs="+", default=["index", "thumb"],
                    help="which fingers to highlight (default: index, thumb -- the ones in question)")
    args = ap.parse_args()

    d = np.load(os.path.join(args.grab_root, "grab", args.seq + ".npz"), allow_pickle=True)
    obj = d["object"].item()
    lab = d["contact"].item()["object"]  # (T, V), GRAB's own per-vertex per-frame labels
    mesh = trimesh.load(os.path.join(args.grab_root, obj["object_mesh"]), process=False)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    print(f"{args.seq}: object mesh has {len(verts)} vertices, {len(faces)} faces")
    print(f"mesh bounding box: min {verts.min(0).round(3)}  max {verts.max(0).round(3)}  (object's own rest frame)")

    colors = {"index": (0.9, 0.1, 0.1), "middle": (0.1, 0.6, 0.1), "pinky": (0.1, 0.1, 0.9),
              "ring": (0.8, 0.6, 0.0), "thumb": (0.7, 0.0, 0.8)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    for col, (elev, azim) in enumerate([(20, 30), (20, 120)]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color=(0.85, 0.85, 0.85), alpha=0.5, linewidth=0)
        for finger in args.fingers:
            base = GRAB_FIRST_ID_RIGHT + SEG_NAMES.index(f"{finger}_proximal")  # proximal id; +1,+2 = middle,distal
            # "ever touched at any frame" -> union across the whole clip, so a wrap shows as a cluster/smear
            ever_touched = np.zeros(len(verts), bool)
            for s in range(3):
                ever_touched |= (lab == base + s).any(0)
            n = int(ever_touched.sum())
            print(f"  {finger:7s}: {n} distinct vertices ever labelled as contact (right hand), "
                  f"across proximal+middle+distal")
            if n > 0:
                pts = verts[ever_touched]
                spread = np.linalg.norm(pts - pts.mean(0), axis=1)
                print(f"           centroid {pts.mean(0).round(3)}  spread (mean/max dist from centroid) "
                      f"{spread.mean():.4f} / {spread.max():.4f} m  <- small spread = tip touch, large = wrap/smear")
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(finger, "black"),
                          s=15, label=f"{finger} (n={n})", depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {col+1}: elev={elev} azim={azim}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{args.seq}: which object vertices each finger ever touches (GRAB raw labels, object rest frame)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
