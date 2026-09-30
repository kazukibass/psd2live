"""Pictures of the fitted 3D head proxy (tools/fit3d_to_drawn.py).

mesh3d_views.png  the face-base mesh lifted to its depth on the dome, seen from the front, the side, the
                  top and three-quarters; dome grid, centre line and the pinned eyes / nose / mouth.
mesh3d_9dir.png   the face-base mesh turned to every fitted direction, triangles coloured by stretch
                  against the front pose (grey <= 1.45, then green -> red).

Usage: python3 render_mesh3d.py <rest-geometry.json> <face3d_fit.json> <out-dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import face_turns as F
import fit3d_head as HH
import fit3d_to_drawn as T
import head3d as H
import nose_check as N

# which head model the fit file came from: fit3d_head.py (ellipsoid, has "angles") or fit3d_to_drawn.py (dome)
MODEL = {"depth": lambda P, x, y: H.depth(P, x, y, "face"),
         "project": lambda P, pts, yaw, pitch: T.project(P, pts[:, 0], pts[:, 1], H.depth(P, pts[:, 0], pts[:, 1], "face"), yaw, pitch)[0]}


def use_ellipsoid():
    MODEL["depth"] = HH.depth
    MODEL["project"] = lambda P, pts, yaw, pitch: HH.project(P, pts, yaw, pitch)[0]


def lifted(P, pts):
    z = MODEL["depth"](P, pts[:, 0], pts[:, 1])
    return np.column_stack([pts[:, 0] - P["cx"], z, -(pts[:, 1] - P["cy"])])   # x right, y toward viewer, z up


def views(P, rest, tris, out):
    fig = plt.figure(figsize=(14, 12), dpi=110)
    xyz = lifted(P, rest)
    gx, gy = np.meshgrid(np.linspace(P["cx"] - 430, P["cx"] + 430, 29), np.linspace(20, 900, 30))
    gz = MODEL["depth"](P, gx, gy)
    mids = np.column_stack([np.full(len(T.MID_Y), 900.0), T.MID_Y])
    mid = lifted(P, mids)
    marks = {k: lifted(P, np.array([v], float))[0] for k, v in N.CANVAS.items()}
    norm = plt.Normalize(xyz[:, 1].min(), xyz[:, 1].max())
    cols = plt.cm.viridis(norm(xyz[tris].mean(1)[:, 1]))
    for i, (title, elev, azim) in enumerate((("front", 0, -90), ("side (from the right)", 0, 0),
                                             ("top", 89, -90), ("three-quarter", 18, -55))):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.plot_wireframe(gx - P["cx"], gz, -(gy - P["cy"]), color=(0.6, 0.6, 0.6), linewidth=0.3, alpha=0.5)
        pc = Poly3DCollection(xyz[tris], facecolors=cols, edgecolors=(0.2, 0.2, 0.2, 0.35), linewidths=0.3)
        ax.add_collection3d(pc)
        ax.plot(mid[:, 0], mid[:, 1], mid[:, 2], color=(1, 0.5, 0), linewidth=2.5)
        for k, p in marks.items():
            ax.scatter(*p, color=(0.9, 0, 0.6), s=30, depthshade=False)
            ax.text(p[0] + 8, p[1] + 8, p[2] + 8, k, fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((np.ptp(gx), np.ptp(gz) + 1, np.ptp(gy)))
        ax.set_title(title)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("depth (px)")
        ax.set_zlabel("up (px)")
    fig.suptitle("fitted head proxy: " + "  ".join(f"{k} {v:.2f}" for k, v in P.items() if not k.endswith("ratio")))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def turned(P, rest, tris, dirs, out):
    fig, axes = plt.subplots(3, 3, figsize=(13, 14), dpi=110)
    R = np.stack([rest[tris[:, 1]] - rest[tris[:, 0]], rest[tris[:, 2]] - rest[tris[:, 0]]], axis=2)
    ok = np.abs(np.linalg.det(R)) / 2 >= 4.0
    tris, R = tris[ok], R[ok]
    for i, (name, _, _) in enumerate(F.DIRECTIONS):
        ax = axes[i // 3][i % 3]
        yaw, pitch = (0.0, 0.0) if name == "front" else (dirs[name]["yaw"], dirs[name]["pitch"])
        cur = MODEL["project"](P, rest, yaw, pitch)
        Q = np.stack([cur[tris[:, 1]] - cur[tris[:, 0]], cur[tris[:, 2]] - cur[tris[:, 0]]], axis=2)
        sv = np.linalg.svd(Q @ np.linalg.inv(R), compute_uv=False)
        an = sv[:, 0] / sv[:, 1]
        t = np.clip((an - 1.45) / 1.6, 0, 1)
        colors = np.where((an <= 1.45)[:, None], [[0.80, 0.86, 0.80]],
                          np.column_stack([0.35 + 0.65 * t, 0.55 * (1 - t), 0.2 * np.ones_like(t)]))
        ax.add_collection(PolyCollection(cur[tris], facecolors=colors, edgecolors=(0.45, 0.45, 0.45), linewidths=0.25))
        mids = np.column_stack([np.full(len(T.MID_Y), 900.0), T.MID_Y])
        m = MODEL["project"](P, mids, yaw, pitch)
        ax.plot(m[:, 0], m[:, 1], color=(1, 0.5, 0), linewidth=2)
        ax.set_xlim(640, 1160)
        ax.set_ylim(830, 280)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}  yaw {yaw:+.1f}  pitch {pitch:+.1f}  stretch p95 {np.percentile(an, 95):.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    geo_path, fit_path, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    fit = json.load(open(fit_path))
    prefix = "mesh3d"
    if "angles" in fit:
        use_ellipsoid()
        prefix = "head3d_mesh"
    views(fit["shape"], rest, tris, os.path.join(out_dir, f"{prefix}_views.png"))
    turned(fit["shape"], rest, tris, fit["directions"], os.path.join(out_dir, f"{prefix}_9dir.png"))
