"""Fit a human-like 3D head to the hand-drawn centre lines and face regions.

Differences from tools/fit3d_to_drawn.py, whose free dome collapsed into a narrow ridge along the nose:

  head shape   a rounded box (superellipsoid, z = rz * (1 - |u|^n - |v|^n)^(1/n), n 2..3) with human
               proportions enforced by the bounds: skull half-width rx 250..360 px (the face is 420 px
               wide), depth rz 0.75..1.0 rx, height ry 1.1..1.35 rx. It cannot turn into a ridge.
  angles       four angles only: yaw to the left, yaw to the right, pitch up, pitch down. The corners are
               built from them (yaw, then pitch) - they are not fitted on their own.
  shift        at most +-8 sheet px per direction, so up/down has to come from the pitch.

Targets and harness are as before: the projected face midline onto the drawn centre line (strong), the
face outline towards the drawn region (weak), 98th-percentile triangle stretch <= 3 and no flips.

Usage: python3 fit3d_head.py <drawn.jpg> <rest-geometry.json> <nekomimi.psd> <out-dir>
Writes <out-dir>/head3d_<direction>.png, head3d_overlay_9dir.png, head3d_fit.json.
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
import nose_check as N

# name, initial, lower, upper  (ratios are relative to rx)
SHAPE = [("cx", 900, 870, 930), ("cy", 500, 420, 580), ("rx", 300, 250, 360),
         ("rz_ratio", 0.85, 0.75, 1.0), ("ry_ratio", 1.2, 1.1, 1.35),
         ("n", 2.4, 2.0, 3.0), ("py", 760, 680, 860), ("pz", -60, -200, 60)]
ANGLES = [("yaw_left", -28, -45, -10), ("yaw_right", 28, 10, 45), ("pitch_up", 18, 5, 40), ("pitch_down", -18, -40, -5)]
MAX_SHIFT = 8.0
W_MID, W_EDGE, W_SHIFT, W_STRETCH, MAX_STRETCH = 1.0, 0.35, 0.3, 60.0, 3.0
MID_Y = np.arange(330, 771, 15)


def shape_of(v):
    P = dict(zip([p[0] for p in SHAPE], v))
    P["rz"] = P["rz_ratio"] * P["rx"]
    P["ry"] = P["ry_ratio"] * P["rx"]
    return P


def depth(P, x, y):
    u = (x - P["cx"]) / P["rx"]
    w = (y - P["cy"]) / P["ry"]
    # superellipse cross-section: n = 2 is an ellipsoid, larger n a rounded box (flatter front, steeper sides)
    return P["rz"] * np.clip(1 - np.abs(u) ** P["n"] - np.abs(w) ** P["n"], 0.001, None) ** (1 / P["n"])


def project(P, pts, yaw, pitch):
    x, y = pts[:, 0], pts[:, 1]
    z = depth(P, x, y)
    yr, pr = np.radians(yaw), np.radians(pitch)
    dx = x - P["cx"]
    x1 = P["cx"] + dx * np.cos(yr) + z * np.sin(yr)
    z1 = -dx * np.sin(yr) + z * np.cos(yr)
    y2 = P["py"] + (y - P["py"]) * np.cos(pr) - (z1 - P["pz"]) * np.sin(pr)
    return np.column_stack([x1, y2]), z1


def angles_for(name, A):
    yaw = A["yaw_left"] if "left" in name else A["yaw_right"] if "right" in name else 0.0
    pitch = A["pitch_up"] if name.startswith("up") else A["pitch_down"] if name.startswith("down") else 0.0
    return yaw, pitch


def main(drawn_path, geo_path, psd_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    drawn = np.array(Image.open(drawn_path).convert("RGB"))
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris_all = np.array(face["indices"]).reshape(-1, 3)
    loop = R.boundary_loop(tris_all)
    Rm = np.stack([rest[tris_all[:, 1]] - rest[tris_all[:, 0]], rest[tris_all[:, 2]] - rest[tris_all[:, 0]]], axis=2)
    keep = np.abs(np.linalg.det(Rm)) / 2 >= 4.0
    tris, rest_inv = tris_all[keep], np.linalg.inv(Rm[keep])
    nose = np.array([N.CANVAS["nose"]], float)
    mids = np.column_stack([np.full(len(MID_Y), C.MID_X, float), MID_Y])
    s = F.SHEET_SCALE

    targets = {}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        if name == "front":
            continue
        r, c = divmod(i, 3)
        cell = drawn[r * F.CELL_H:(r + 1) * F.CELL_H, c * F.CELL_W:(c + 1) * F.CELL_W]
        region, _ = C.drawn_region(cell)
        mid = C.drawn_midline(cell, region)
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        ys = np.where(region)[0]
        targets[name] = dict(line=np.polyfit(mid[:, 1], mid[:, 0], 2), y_span=(ys.min(), ys.max()),
                             contour=contour, tree=cKDTree(contour), region=region,
                             ref_nose=np.array(N.REFERENCE[name]["nose"][:2], float))
    names = list(targets)
    nS, nA = len(SHAPE), len(ANGLES)

    def unpack(v):
        P = shape_of(v[:nS])
        A = dict(zip([a[0] for a in ANGLES], v[nS:nS + nA]))
        shifts = {n: v[nS + nA + 2 * k: nS + nA + 2 * k + 2] for k, n in enumerate(names)}
        return P, A, shifts

    def place(P, A, shift, pts, n):
        yaw, pitch = angles_for(n, A)
        proj, zf = project(P, pts, yaw, pitch)
        pn, _ = project(P, nose, yaw, pitch)
        off = targets[n]["ref_nose"] - pn[0] * s + shift
        return proj * s + off, zf, off

    def residuals(v):
        P, A, shifts = unpack(v)
        res = []
        for n in names:
            t, sh = targets[n], shifts[n]
            m, _, _ = place(P, A, sh, mids, n)
            inside = (m[:, 1] >= t["y_span"][0]) & (m[:, 1] <= t["y_span"][1])
            res += list(W_MID * np.where(inside, m[:, 0] - np.polyval(t["line"], m[:, 1]), 0.0))
            allp, _, _ = place(P, A, sh, rest, n)
            b = allp[loop]
            res += list(W_EDGE * t["tree"].query(b)[0][::3])
            res += list(W_EDGE * cKDTree(b).query(t["contour"][::6])[0])
            res += list(W_SHIFT * sh)
            M = np.stack([allp[tris[:, 1]] - allp[tris[:, 0]], allp[tris[:, 2]] - allp[tris[:, 0]]], axis=2) @ rest_inv / s
            sv = np.linalg.svd(M, compute_uv=False)
            an = np.where(np.linalg.det(M) > 0, sv[:, 0] / np.maximum(sv[:, 1], 1e-6), 50.0)
            res.append(W_STRETCH * max(0.0, float(np.percentile(an, 98)) - MAX_STRETCH))
        return np.array(res)

    x0 = np.array([p[1] for p in SHAPE] + [a[1] for a in ANGLES] + [0.0] * (2 * len(names)))
    lo = np.array([p[2] for p in SHAPE] + [a[2] for a in ANGLES] + [-MAX_SHIFT] * (2 * len(names)))
    hi = np.array([p[3] for p in SHAPE] + [a[3] for a in ANGLES] + [MAX_SHIFT] * (2 * len(names)))
    fit = least_squares(residuals, x0, bounds=(lo, hi), diff_step=1e-3)
    P, A, shifts = unpack(fit.x)
    print("shape: " + ", ".join(f"{k}={v:.2f}" for k, v in P.items()))
    print("angles: " + ", ".join(f"{k}={v:.1f}" for k, v in A.items()))

    psd = PSDImage.open(psd_path)
    layer = next(l for l in psd if l.name == "face")
    tex = np.zeros((psd.height, psd.width, 4), np.uint8)
    tex[layer.top:layer.bottom, layer.left:layer.right] = np.array(layer.topil().convert("RGBA"))
    canvas = Image.fromarray(drawn).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    report = {"shape": {k: round(float(v), 3) for k, v in P.items()}, "angles": {k: round(float(v), 2) for k, v in A.items()},
              "directions": {}}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        if name == "front":
            continue
        r, c = divmod(i, 3)
        x0c, y0c = c * F.CELL_W, r * F.CELL_H
        sh = shifts[name]
        cell_pts, zf, off = place(P, A, sh, rest, name)
        canvas_pts = (cell_pts - off) / s
        part = F.warp(tex, rest, canvas_pts, tris_all)
        Image.fromarray(part).save(os.path.join(out_dir, f"head3d_{name}.png"))
        small = Image.fromarray(part).resize((round(part.shape[1] * s), round(part.shape[0] * s)), Image.LANCZOS)
        a = np.array(small)
        a[..., 3] = (a[..., 3] * 0.5).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(a), (round(x0c + off[0]), round(y0c + off[1])))
        mask = np.zeros((F.CELL_H, F.CELL_W), np.uint8)
        for tri in cell_pts[tris_all]:
            cv2.fillConvexPoly(mask, np.round(tri * 4).astype(np.int32), 1, shift=2)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            pts = [(p[0][0] + x0c, p[0][1] + y0c) for p in cnt]
            d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)
        m, _, _ = place(P, A, sh, mids, name)
        d.line([(p[0] + x0c, p[1] + y0c) for p in m], fill=(255, 140, 0, 255), width=3)

        t = targets[name]
        mk = mask > 0
        yc = int(sum(t["y_span"]) / 2)
        cols, cols_d = np.where(mk[yc])[0], np.where(t["region"][yc])[0]
        yaw, pitch = angles_for(name, A)
        p95, mx, flips = R.stretch(rest, canvas_pts, tris_all)
        row = dict(yaw=round(float(yaw), 1), pitch=round(float(pitch), 1), shift=[round(float(x), 1) for x in sh],
                   iou=round(float((t["region"] & mk).sum() / (t["region"] | mk).sum()), 3),
                   midline_share_drawn=round(float((np.polyval(t["line"], yc) - cols_d.min()) / max(np.ptp(cols_d), 1)), 3),
                   midline_share_model=round(float((np.interp(yc, m[:, 1], m[:, 0]) - cols.min()) / max(np.ptp(cols), 1)), 3),
                   stretch_p95=round(p95, 2), stretch_max=round(mx, 2), flips=flips)
        report["directions"][name] = row
        d.rectangle((x0c + 330, y0c + 8, x0c + 505, y0c + 30), fill=(255, 255, 255, 230))
        d.text((x0c + 336, y0c + 12), f"yaw {row['yaw']} pitch {row['pitch']} IoU {row['iou']}", fill=(0, 0, 0, 255))
    canvas.convert("RGB").save(os.path.join(out_dir, "head3d_overlay_9dir.png"))
    json.dump(report, open(os.path.join(out_dir, "head3d_fit.json"), "w"), indent=1)
    for k, v in report["directions"].items():
        print(f"{k:11s} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
