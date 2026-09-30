"""Head turns matched to the reference sheet, for every head drawable (face, eyes, mouth, hair, ears).

For each of the 8 key directions:
  1. every head vertex is turned by the human-like 3D head of tools/fit3d_head.py (corners = yaw then
     pitch);
  2. a 2D correction moves the face outline onto the outer edge of the hand-drawn face region and the
     face midline onto the drawn centre line (affine part + thin-plate spline, as tools/fit_to_region.py).
     The same correction is applied to hair and ears, so they stay together with the face; a ring of
     fixed anchors 120 sheet px outside the face keeps it from running away at the far hair tips;
  3. the whole head is placed so that the nose sits where the reference nose sits relative to the collar
     buckle of that cell (the body does not move).
Physics and motions are left as psd2live generates them.

Usage: python3 fit_full_head.py <drawn.jpg> <standard-rig-geometry.json> <head3d_fit.json> <angles-9dir.png> <out.json>
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

import face_region_compare as C
import face_turns as F
import fit3d_head as HH
import fit_to_region as R
import nose_check as N

HEAD_ROOT = "DeformHeadContainer"
FRONT_NOSE_CELL = np.array(N.REFERENCE["front"]["nose"][:2], float)
FRONT_NOSE_CANVAS = np.array(N.CANVAS["nose"], float)
ANCHOR_RING = 120.0   # sheet px outside the face centre beyond which the correction fades to zero
# variants for diagnosis: --no-correction, --no-placement
USE_CORRECTION = "--no-correction" not in sys.argv
USE_PLACEMENT = "--no-placement" not in sys.argv


def halves(pts, i_top, i_bot):
    n = len(pts)
    a1 = [(i_top + k) % n for k in range((i_bot - i_top) % n + 1)]
    a2 = [(i_bot + k) % n for k in range((i_top - i_bot) % n + 1)]
    return (a1, a2) if pts[a1].mean(0)[0] < pts[a2].mean(0)[0] else (a2[::-1], a1[::-1])


def controls(face_cell, loop, mid_model, region, mid_drawn):
    """Source -> target control points: outline halves matched by arc length, midline onto the drawn line."""
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    line = np.polyfit(mid_drawn[:, 1], mid_drawn[:, 0], 2)
    ys = np.arange(0, F.CELL_H)
    ys = ys[region[ys, np.clip(np.polyval(line, ys).round().astype(int), 0, F.CELL_W - 1)]]
    d_top = np.array([np.polyval(line, ys.min()), ys.min()])
    d_bot = np.array([np.polyval(line, ys.max()), ys.max()])
    b = face_cell[loop]
    m_left, m_right = halves(b, np.argmin(np.linalg.norm(b - mid_model[0], axis=1)),
                             np.argmin(np.linalg.norm(b - mid_model[-1], axis=1)))
    c_left, c_right = halves(contour, np.argmin(np.linalg.norm(contour - d_top, axis=1)),
                             np.argmin(np.linalg.norm(contour - d_bot, axis=1)))
    src, dst = [], []
    for mi, ci in ((m_left, c_left), (m_right, c_right)):
        mp, cp = b[mi], contour[ci]
        if mp[0][1] > mp[-1][1]:
            mp = mp[::-1]
        if cp[0][1] > cp[-1][1]:
            cp = cp[::-1]
        tm = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(mp, axis=0), axis=1))])
        tc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cp, axis=0), axis=1))])
        tm, tc = tm / tm[-1], tc / tc[-1]
        src += list(mp)
        dst += list(np.column_stack([np.interp(tm, tc, cp[:, 0]), np.interp(tm, tc, cp[:, 1])]))
    tmid = (mid_model[:, 1] - mid_model[0, 1]) / (mid_model[-1, 1] - mid_model[0, 1])
    ty = d_top[1] + tmid * (d_bot[1] - d_top[1])
    src += list(mid_model)
    dst += list(np.column_stack([np.polyval(line, ty), ty]))
    return np.array(src), np.array(dst)


def main(drawn_path, geo_path, fit_path, sheet_path, out_path):
    drawn = np.array(Image.open(drawn_path).convert("RGB"))
    geo = json.load(open(geo_path))
    fit = json.load(open(fit_path))
    P, A = fit["shape"], fit["angles"]
    heads = [d for d in geo["drawables"] if HEAD_ROOT in d["chain"]]
    face = next(d for d in heads if d["id"] == "ArtMeshFace")
    face_rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    loop = R.boundary_loop(tris)
    mids = np.column_stack([np.full(len(C.MID_Y), C.MID_X, float), C.MID_Y])
    nose = np.array([N.CANVAS["nose"]], float)
    buckles = F.buckle_offsets(np.array(Image.open(sheet_path).convert("RGB")))
    s = F.SHEET_SCALE

    out = {"source": "fit_full_head", "poses": []}
    for i, (name, _, _) in enumerate(F.DIRECTIONS):
        if name == "front":
            continue
        ax = -45 if "left" in name else 45 if "right" in name else 0
        ay = 30 if name.startswith("up") else -30 if name.startswith("down") else 0
        yaw, pitch = HH.angles_for(name, A)
        r, c = divmod(i, 3)
        cell_img = drawn[r * F.CELL_H:(r + 1) * F.CELL_H, c * F.CELL_W:(c + 1) * F.CELL_W]
        region, _ = C.drawn_region(cell_img)
        mid_drawn = C.drawn_midline(cell_img, region)

        # 1. 3D turn, placed nose-on-nose in cell coordinates
        pn, _ = HH.project(P, nose, yaw, pitch)
        ref_nose = np.array(N.REFERENCE[name]["nose"][:2], float)
        to_cell = lambda pts: pts * s + (ref_nose - pn[0] * s)
        face_cell = to_cell(HH.project(P, face_rest, yaw, pitch)[0])
        mid_cell = to_cell(HH.project(P, mids, yaw, pitch)[0])

        # 2. 2D correction onto the drawn region and centre line, faded out beyond the anchor ring
        src, dst = controls(face_cell, loop, mid_cell, region, mid_drawn)
        X = np.column_stack([src, np.ones(len(src))])
        aff, *_ = np.linalg.lstsq(X, dst, rcond=None)
        centre = face_cell.mean(0)
        ring = centre + ANCHOR_RING * np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 36, endpoint=False)),
                                                       np.sin(np.linspace(0, 2 * np.pi, 36, endpoint=False))])
        ring_src = ring
        ring_dst = ring   # no correction far from the face: hair tips keep the plain 3D turn
        # the affine part is blended out with distance so that it too vanishes at the ring
        def corrected(pts):
            moved = np.column_stack([pts, np.ones(len(pts))]) @ aff
            w = np.clip(1 - np.linalg.norm(pts - centre, axis=1) / ANCHOR_RING, 0, 1)
            w = w * w * (3 - 2 * w)                       # smoothstep
            base = pts + (moved - pts) * w[:, None]
            return base
        base_src = corrected(src)
        rbf = RBFInterpolator(np.vstack([base_src, ring_src]), np.vstack([dst - base_src, ring_dst - ring_src]),
                              kernel="thin_plate_spline", smoothing=5.0)

        # 3. absolute placement: the nose goes where it sits relative to this cell's buckle
        shift, _ = buckles[name]
        nose_canvas = FRONT_NOSE_CANVAS + (ref_nose - shift - FRONT_NOSE_CELL) / s
        drawables = {}
        for d in heads:
            rest = np.array(d["world"]["rest"]).reshape(-1, 2)
            cell = to_cell(HH.project(P, rest, yaw, pitch)[0])
            if USE_CORRECTION:
                b = corrected(cell)
                fitted = b + rbf(b)
            else:
                fitted = cell
            anchor = nose_canvas if USE_PLACEMENT else pn[0]
            canvas = anchor + (fitted - ref_nose) / s
            drawables[d["id"]] = np.round((canvas - rest).ravel(), 2).tolist()
        out["poses"].append({"angleX": ax, "angleY": ay, "yaw": yaw, "pitch": pitch, "drawables": drawables})
        print(f"{name:11s} yaw {yaw:+.1f} pitch {pitch:+.1f} nose -> ({nose_canvas[0]:.0f}, {nose_canvas[1]:.0f})")
    json.dump(out, open(out_path, "w"))


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("--")][:5])
