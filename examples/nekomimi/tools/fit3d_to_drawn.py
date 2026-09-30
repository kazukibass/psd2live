"""Fit the 3D head proxy to the hand-drawn centre line and face region.

The black line drawn on each reference cell is a rough midline, guessed from where the nose is. Instead
of forcing the face texture onto it in 2D (tools/fit_to_region.py, which has to squeeze one half of the
face 2.5x), the dome itself is fitted: its shape is shared by all directions, and every direction gets
its own yaw, pitch and a small shift. The projected midline of the face is pulled onto the drawn line
(strong), the projected face outline towards the outer edge of the blue marker (weak, since it is
drawn roughly and partly hidden by hair). What is left is decided by the 3D shape, so the far side of
the face narrows by foreshortening rather than by a 2D squeeze.

Usage: python3 fit3d_to_drawn.py <drawn.jpg> <rest-geometry.json> <nekomimi.psd> <out-dir>
Writes <out-dir>/face3d_<direction>.png, face3d_overlay_9dir.png, face3d_fit.json.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import face_region_compare as C
import face_turns as F
import fit_to_region as R
import head3d as H
import nose_check as N

# head shape: the harness-free fit (face_turns.FREE_FIT). FIX_SHAPE keeps it; otherwise it may move.
FIX_SHAPE = "--free-shape" not in sys.argv
_E = 1e-6 if FIX_SHAPE else None
SHAPE = [(k, F.FREE_FIT[k], F.FREE_FIT[k] - (_E or d), F.FREE_FIT[k] + (_E or d))
         for k, d in (("cx", 60), ("cy", 120), ("rx", 150), ("ry", 190), ("rz", 170), ("k", 0.7), ("py", 150), ("pz", 200))]
W_MID, W_EDGE, W_SHIFT = 1.0, 0.35, 0.4
W_SHAPE = 15.0     # per quarter of a parameter's allowed range away from the harness-free head shape
W_AWAY = 20.0      # per face vertex that turns past the edge of the head
MAX_STRETCH = 3.0  # harness: 98th percentile of per-triangle stretch allowed on the face base
W_STRETCH = 60.0   # per unit of stretch above MAX_STRETCH
MID_Y = np.arange(330, 771, 15)


def project(P, x, y, z, yaw, pitch):
    """Same model as head3d.project, with explicit angles in degrees."""
    yaw, pitch = np.radians(yaw), np.radians(pitch)
    dx = x - P["cx"]
    x1 = P["cx"] + dx * np.cos(yaw) + z * np.sin(yaw)
    z1 = -dx * np.sin(yaw) + z * np.cos(yaw)
    y2 = P["py"] + (y - P["py"]) * np.cos(pitch) - (z1 - P["pz"]) * np.sin(pitch)
    return np.column_stack([x1, y2]), z1


def main(drawn_path, geo_path, psd_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    drawn = np.array(Image.open(drawn_path).convert("RGB"))
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    loop = R.boundary_loop(tris)
    nose = np.array(N.CANVAS["nose"], float)
    mids = np.column_stack([np.full(len(MID_Y), C.MID_X, float), MID_Y])
    s = F.SHEET_SCALE

    # drawn targets per direction (cell coordinates)
    targets = {}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        if (ax, ay) == (0, 0):
            continue
        r, c = divmod(i, 3)
        cell = drawn[r * F.CELL_H:(r + 1) * F.CELL_H, c * F.CELL_W:(c + 1) * F.CELL_W]
        region, _ = C.drawn_region(cell)
        mid = C.drawn_midline(cell, region)
        line = np.polyfit(mid[:, 1], mid[:, 0], 2)
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        ys = np.where(region)[0]
        targets[name] = dict(ax=ax, ay=ay, line=line, y_span=(ys.min(), ys.max()), contour=contour,
                             tree=cKDTree(contour), region=region, ref_nose=np.array(N.REFERENCE[name]["nose"][:2], float))
    names = list(targets)

    Rm = np.stack([rest[tris[:, 1]] - rest[tris[:, 0]], rest[tris[:, 2]] - rest[tris[:, 0]]], axis=2)
    keep = np.abs(np.linalg.det(Rm)) / 2 >= 4.0
    tris = tris[keep]
    rest_inv = np.linalg.inv(Rm[keep]) * s    # cell px per canvas px folded in
    x0 = [p[1] for p in SHAPE]
    lo = [p[2] for p in SHAPE]
    hi = [p[3] for p in SHAPE]
    for n in names:
        t = targets[n]
        x0 += [F.FREE_FIT["yaw"] * np.sign(t["ax"]), (F.FREE_FIT["up"] if t["ay"] > 0 else F.FREE_FIT["down"]) * np.sign(t["ay"]), 0, 0]
        lo += [-60 if t["ax"] < 0 else (-5 if t["ax"] == 0 else 0), -45 if t["ay"] < 0 else (-5 if t["ay"] == 0 else 0), -40, -40]
        hi += [0 if t["ax"] < 0 else (5 if t["ax"] == 0 else 60), 0 if t["ay"] < 0 else (5 if t["ay"] == 0 else 45), 40, 40]
    x0, lo, hi = np.array(x0, float), np.array(lo, float), np.array(hi, float)

    def unpack(v):
        P = dict(zip([p[0] for p in SHAPE], v[:len(SHAPE)]))
        per = {n: v[len(SHAPE) + 4 * k: len(SHAPE) + 4 * k + 4] for k, n in enumerate(names)}
        return P, per

    def place(P, yaw, pitch, shift, pts, n):
        z = H.depth(P, pts[:, 0], pts[:, 1], "face")
        proj, zf = project(P, pts[:, 0], pts[:, 1], z, yaw, pitch)
        zn = H.depth(P, nose[:1], nose[1:], "face")
        pn, _ = project(P, nose[:1], nose[1:], zn, yaw, pitch)
        off = targets[n]["ref_nose"] - pn[0] * s + shift
        return proj * s + off, zf, off

    def residuals(v):
        P, per = unpack(v)
        res = []
        for n in names:
            yaw, pitch, tx, ty = per[n]
            t = targets[n]
            m, _, _ = place(P, yaw, pitch, np.array([tx, ty]), mids, n)
            inside = (m[:, 1] >= t["y_span"][0]) & (m[:, 1] <= t["y_span"][1])
            res += list(W_MID * np.where(inside, m[:, 0] - np.polyval(t["line"], m[:, 1]), 0.0))
            b, _, _ = place(P, yaw, pitch, np.array([tx, ty]), rest[loop], n)
            res += list(W_EDGE * t["tree"].query(b)[0][::3])
            res += list(W_EDGE * cKDTree(b).query(t["contour"][::6])[0])
            res += [W_SHIFT * tx, W_SHIFT * ty]
            _, zf, _ = place(P, yaw, pitch, np.array([tx, ty]), rest[loop], n)
            res.append(W_AWAY * np.sqrt(np.clip(-zf, 0, None).sum()))
            allp, _, _ = place(P, yaw, pitch, np.array([tx, ty]), rest, n)
            M = np.stack([allp[tris[:, 1]] - allp[tris[:, 0]], allp[tris[:, 2]] - allp[tris[:, 0]]], axis=2) @ rest_inv
            sv = np.linalg.svd(M, compute_uv=False)
            an = np.where(np.linalg.det(M) > 0, sv[:, 0] / np.maximum(sv[:, 1], 1e-6), 50.0)   # a flip counts as huge
            res.append(W_STRETCH * max(0.0, float(np.percentile(an, 98)) - MAX_STRETCH))
        span = np.array([p[3] - p[2] for p in SHAPE]) / 4
        res += list(W_SHAPE * (v[:len(SHAPE)] - np.array([p[1] for p in SHAPE])) / span)
        return np.array(res)

    fit = least_squares(residuals, x0, bounds=(lo, hi), diff_step=1e-3)
    P, per = unpack(fit.x)
    print("shape: " + ", ".join(f"{k}={v:.1f}" for k, v in P.items()))

    psd = PSDImage.open(psd_path)
    layer = next(l for l in psd if l.name == "face")
    tex = np.zeros((psd.height, psd.width, 4), np.uint8)
    tex[layer.top:layer.bottom, layer.left:layer.right] = np.array(layer.topil().convert("RGBA"))
    canvas = Image.fromarray(drawn).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    report = {"shape": {k: round(float(v), 2) for k, v in P.items()}, "directions": {}}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        r, c = divmod(i, 3)
        x0c, y0c = c * F.CELL_W, r * F.CELL_H
        if name == "front":
            continue
        yaw, pitch, tx, ty = per[name]
        cell_pts, zf, off = place(P, yaw, pitch, np.array([tx, ty]), rest, name)
        canvas_pts = (cell_pts - off) / s
        part = F.warp(tex, rest, canvas_pts, tris)
        Image.fromarray(part).save(os.path.join(out_dir, f"face3d_{name}.png"))
        small = Image.fromarray(part).resize((round(part.shape[1] * s), round(part.shape[0] * s)), Image.LANCZOS)
        a = np.array(small)
        a[..., 3] = (a[..., 3] * 0.5).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(a), (round(x0c + off[0]), round(y0c + off[1])))
        mask = np.zeros((F.CELL_H, F.CELL_W), np.uint8)
        for tri in cell_pts[tris]:
            cv2.fillConvexPoly(mask, np.round(tri * 4).astype(np.int32), 1, shift=2)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            pts = [(p[0][0] + x0c, p[0][1] + y0c) for p in cnt]
            d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)
        m, _, _ = place(P, yaw, pitch, np.array([tx, ty]), mids, name)
        d.line([(p[0] + x0c, p[1] + y0c) for p in m], fill=(255, 140, 0, 255), width=3)

        t = targets[name]
        mk = mask > 0
        iou = float((t["region"] & mk).sum() / (t["region"] | mk).sum())
        yc = int(sum(t["y_span"]) / 2)
        cols = np.where(mk[yc])[0]
        share = (np.interp(yc, m[:, 1], m[:, 0]) - cols.min()) / max(np.ptp(cols), 1)
        cols_d = np.where(t["region"][yc])[0]
        share_d = (np.polyval(t["line"], yc) - cols_d.min()) / max(np.ptp(cols_d), 1)
        p95, mx, flips = R.stretch(rest, canvas_pts, tris)
        away = int((zf < 0).sum())
        row = dict(yaw=round(float(yaw), 1), pitch=round(float(pitch), 1), shift=[round(float(tx), 1), round(float(ty), 1)],
                   iou=round(iou, 3), midline_share_drawn=round(float(share_d), 3), midline_share_model=round(float(share), 3),
                   stretch_p95=round(p95, 2), stretch_max=round(mx, 2), flips=flips, vertices_facing_away=away)
        report["directions"][name] = row
        d.rectangle((x0c + 330, y0c + 8, x0c + 505, y0c + 30), fill=(255, 255, 255, 230))
        d.text((x0c + 336, y0c + 12), f"yaw {row['yaw']} pitch {row['pitch']} IoU {row['iou']}", fill=(0, 0, 0, 255))
    canvas.convert("RGB").save(os.path.join(out_dir, "face3d_overlay_9dir.png"))
    json.dump(report, open(os.path.join(out_dir, "face3d_fit.json"), "w"), indent=1)
    for k, v in report["directions"].items():
        print(f"{k:11s} {v}")


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("--")][:4])
