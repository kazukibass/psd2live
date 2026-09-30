"""Fit the turned face base to the outer edge of the hand-drawn face region.

Starting from the 3D-turned face base (placed nose-on-nose, as in tools/nose_check.py), the outline of
the face mesh is moved onto the outer edge of the blue marker region (tools/face_region_compare.py).
The outline is split where the drawn centre line meets it and each half is matched by arc length; the
model's centre line goes onto the drawn one. The bulk of the move is one affine map and only the
remainder is spread into the inside (thin-plate spline). Every triangle's
stretch against the front pose is reported, so the cost of the fit is visible.

Usage: python3 fit_to_region.py <drawn.jpg> <rest-geometry.json> <nekomimi.psd> <out-dir>
Writes <out-dir>/fitted_face_<direction>.png (canvas size), fitted_overlay_9dir.png, fitted.json.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from scipy.interpolate import RBFInterpolator

import face_region_compare as C
import face_turns as F
import head3d as H
import nose_check as N


def boundary_loop(tris):
    """Ordered outer boundary vertex loop of a triangle mesh."""
    edges = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    nbr = {}
    for (a, b), n in edges.items():
        if n == 1:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
    start = next(iter(nbr))
    loop, prev, cur = [start], None, start
    while True:
        nxt = [v for v in nbr[cur] if v != prev][0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return np.array(loop)


def arc_params(pts):
    """Normalised arc length (0..1) of each point along a closed polyline."""
    seg = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    return np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()


def start_at_bottom(pts, centre, clockwise):
    """Rotate a closed polyline to start at its lowest point below `centre`, in a fixed orientation."""
    area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if (area > 0) != clockwise:
        pts = pts[::-1]
    below = np.abs(pts[:, 0] - centre[0]) < 0.25 * np.ptp(pts[:, 0])
    k = np.argmax(np.where(below, pts[:, 1], -np.inf))
    return np.roll(pts, -k, 0), k


def stretch(rest, cur, tris):
    a, b, c = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    A, B, C2 = cur[tris[:, 0]], cur[tris[:, 1]], cur[tris[:, 2]]
    R = np.stack([b - a, c - a], axis=2)
    P = np.stack([B - A, C2 - A], axis=2)
    ok = np.abs(np.linalg.det(R)) / 2 >= 4.0
    M = P[ok] @ np.linalg.inv(R[ok])
    sv = np.linalg.svd(M, compute_uv=False)
    an = sv[:, 0] / sv[:, 1]
    return float(np.percentile(an, 95)), float(an.max()), int((np.linalg.det(M) < 0).sum())


if __name__ == "__main__":
    drawn_path, geo_path, psd_path, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    drawn = np.array(Image.open(drawn_path).convert("RGB"))
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    loop = boundary_loop(tris)
    nose_pin = N.pin(rest, tris, N.CANVAS["nose"])
    mid_pins = [N.pin(rest, tris, (C.MID_X, y)) for y in C.MID_Y]
    z = H.depth(F.FREE_FIT, rest[:, 0], rest[:, 1], "face")
    s = F.SHEET_SCALE

    psd = PSDImage.open(psd_path)
    layer = next(l for l in psd if l.name == "face")
    tex = np.zeros((psd.height, psd.width, 4), np.uint8)
    tex[layer.top:layer.bottom, layer.left:layer.right] = np.array(layer.topil().convert("RGBA"))

    canvas = Image.fromarray(drawn).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    report = {}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        r, c = divmod(i, 3)
        x0, y0 = c * F.CELL_W, r * F.CELL_H
        region, _ = C.drawn_region(drawn[y0:y0 + F.CELL_H, x0:x0 + F.CELL_W])
        px, py = H.project(F.FREE_FIT, rest[:, 0], rest[:, 1], z, ax, ay) if (ax, ay) != (0, 0) else (rest[:, 0], rest[:, 1])
        cur = np.column_stack([px, py])
        at = lambda pts, pinned: (pts[tris[pinned[0]]] * pinned[1][:, None]).sum(0)
        off = np.array(N.REFERENCE[name]["nose"][:2], float) - at(cur, nose_pin) * s
        cell = cur * s + off                                     # model vertices in cell coordinates

        if region is not None:
            cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
            # 1. correspondences anchored on the centre line: the outline is split where the centre line
            #    meets it (top and chin) and each half is matched by arc length; the model's centre line
            #    goes onto the drawn one
            mid_model = np.array([at(cell, p) for p in mid_pins])
            mid_drawn = C.drawn_midline(drawn[y0:y0 + F.CELL_H, x0:x0 + F.CELL_W], region)
            src, dst = [], []
            if mid_drawn is not None and len(mid_drawn) > 10:
                line = np.polyfit(mid_drawn[:, 1], mid_drawn[:, 0], 2)      # x as a smooth function of y
                ys = np.arange(0, F.CELL_H)
                inside = region[ys, np.clip(np.polyval(line, ys).round().astype(int), 0, F.CELL_W - 1)]
                ys = ys[inside]
                d_top = np.array([np.polyval(line, ys.min()), ys.min()])
                d_bot = np.array([np.polyval(line, ys.max()), ys.max()])
                # model: outline vertices nearest to the top and the bottom of its centre line
                m_top_i = np.argmin(np.linalg.norm(cell[loop] - mid_model[0], axis=1))
                m_bot_i = np.argmin(np.linalg.norm(cell[loop] - mid_model[-1], axis=1))
                c_top_i = np.argmin(np.linalg.norm(contour - d_top, axis=1))
                c_bot_i = np.argmin(np.linalg.norm(contour - d_bot, axis=1))

                def halves(pts, i_top, i_bot):
                    n = len(pts)
                    a1 = [(i_top + k) % n for k in range((i_bot - i_top) % n + 1)]
                    a2 = [(i_bot + k) % n for k in range((i_top - i_bot) % n + 1)]
                    left_first = pts[a1].mean(0)[0] < pts[a2].mean(0)[0]
                    return (a1, a2) if left_first else (a2[::-1], a1[::-1])

                m_left, m_right = halves(cell[loop], m_top_i, m_bot_i)
                c_left, c_right = halves(contour, c_top_i, c_bot_i)
                for mi, ci in ((m_left, c_left), (m_right, c_right)):
                    mp, cp = cell[loop[mi]], contour[ci]
                    # both halves run top -> bottom
                    if mp[0][1] > mp[-1][1]:
                        mi, mp = mi[::-1], mp[::-1]
                    if cp[0][1] > cp[-1][1]:
                        cp = cp[::-1]
                    tm = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(mp, axis=0), axis=1))])
                    tcv = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cp, axis=0), axis=1))])
                    tm, tcv = tm / tm[-1], tcv / tcv[-1]
                    src += list(mp)
                    dst += list(np.column_stack([np.interp(tm, tcv, cp[:, 0]), np.interp(tm, tcv, cp[:, 1])]))
                # centre line: same relative height between top and bottom
                tmid = (mid_model[:, 1] - mid_model[0, 1]) / (mid_model[-1, 1] - mid_model[0, 1])
                ty = d_top[1] + tmid * (d_bot[1] - d_top[1])
                src += list(mid_model)
                dst += list(np.column_stack([np.polyval(line, ty), ty]))
            src, dst = np.array(src), np.array(dst)
            ids = None
            # 2. the bulk of the move as one affine map (no local distortion) ...
            X = np.column_stack([src, np.ones(len(src))])
            A, *_ = np.linalg.lstsq(X, dst, rcond=None)
            moved = np.column_stack([cell, np.ones(len(cell))]) @ A
            moved_src = np.column_stack([src, np.ones(len(src))]) @ A
            # 3. ... and only the remainder spread smoothly from the controls into the inside
            rbf = RBFInterpolator(moved_src, dst - moved_src, kernel="thin_plate_spline", smoothing=5.0)
            fitted = moved + rbf(moved)
        else:
            fitted = cell
        fitted_canvas = (fitted - off) / s

        part = F.warp(tex, rest, fitted_canvas, tris)
        Image.fromarray(part).save(os.path.join(out_dir, f"fitted_face_{name}.png"))
        small = Image.fromarray(part).resize((round(part.shape[1] * s), round(part.shape[0] * s)), Image.LANCZOS)
        a = np.array(small)
        a[..., 3] = (a[..., 3] * 0.5).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(a), (round(x0 + off[0]), round(y0 + off[1])))

        mask = np.zeros((F.CELL_H, F.CELL_W), np.uint8)
        for t in fitted[tris]:
            cv2.fillConvexPoly(mask, np.round(t * 4).astype(np.int32), 1, shift=2)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            pts = [(p[0][0] + x0, p[0][1] + y0) for p in cnt]
            d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)
        mid = np.array([at(fitted, p) for p in mid_pins])
        d.line([(p[0] + x0, p[1] + y0) for p in mid], fill=(255, 140, 0, 255), width=3)

        row = {}
        p95, mx, flips = stretch(rest, fitted_canvas, tris)
        row.update(stretch_p95=round(p95, 2), stretch_max=round(mx, 2), flips=flips)
        if region is not None:
            mk = mask > 0
            row["iou"] = round(float((region & mk).sum() / (region | mk).sum()), 3)
            yc = int((np.where(region)[0].min() + np.where(region)[0].max()) / 2)
            cols = np.where(mk[yc])[0]
            xl = C.x_at(mid, yc)
            if xl is not None and len(cols):
                row["model_midline_share"] = round((xl - cols.min()) / max(cols.max() - cols.min(), 1), 3)
        report[name] = row
        d.rectangle((x0 + 360, y0 + 8, x0 + 505, y0 + 30), fill=(255, 255, 255, 230))
        d.text((x0 + 366, y0 + 12), f"IoU {row.get('iou', '-')}  p95 {row['stretch_p95']}", fill=(0, 0, 0, 255))
    canvas.convert("RGB").save(os.path.join(out_dir, "fitted_overlay_9dir.png"))
    json.dump(report, open(os.path.join(out_dir, "fitted.json"), "w"), indent=1)
    for k, v in report.items():
        print(f"{k:11s} {v}")
