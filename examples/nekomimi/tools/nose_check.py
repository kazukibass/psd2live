"""Distortion of the turned face base, measured from the nose.

The nose tip (and the eyes, mouth and chin) are pinned to the face mesh by barycentric coordinates, so
they move with the mesh. For every direction the turned face is placed on its reference cell with the
two noses on top of each other (translation only, sheet scale), and each landmark is compared as a
vector from the nose: length ratio (model / reference) and angle difference.

Usage: python3 nose_check.py <rest-geometry.json> <nekomimi.psd> <angles-9dir.png> <out-dir>
Writes <out-dir>/nose_overlay_9dir.png and <out-dir>/nose_check.json, prints the table.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage

import face_turns as F
import head3d as H

# landmarks on the canvas (front illustration)
CANVAS = {"nose": (900, 604), "eye_r": (785, 504), "eye_l": (1024, 504), "mouth": (903, 657), "chin": (900, 771)}

# measured on the reference sheet, in cell coordinates (512 x 341 per cell); "est" = nose hidden, estimated
# from the nose offset to the eye mid-point in the neighbouring cells
REFERENCE = {
    "up-left":    dict(nose=(220, 128, "est"), eye_r=(205, 128), eye_l=(278, 108), mouth=(222, 160), chin=(217, 215)),
    "up":         dict(nose=(255, 122), eye_r=(212, 112), eye_l=(298, 112), mouth=(255, 150), chin=(255, 200)),
    "up-right":   dict(nose=(320, 128, "est"), eye_r=(268, 108), eye_l=(332, 128), mouth=(313, 160), chin=(317, 215)),
    "left":       dict(nose=(219, 174), eye_r=(207, 147), eye_l=(275, 140), mouth=(235, 195), chin=(250, 233)),
    "front":      dict(nose=(255, 187), eye_r=(215, 152), eye_l=(297, 152), mouth=(257, 206), chin=(252, 242)),
    "right":      dict(nose=(318, 172), eye_r=(265, 140), eye_l=(330, 147), mouth=(303, 195), chin=(297, 233)),
    "down-left":  dict(nose=(210, 198), eye_r=(192, 167), eye_l=(258, 166), mouth=(232, 217), chin=(250, 245)),
    "down":       dict(nose=(255, 217), eye_r=(212, 180), eye_l=(298, 183), mouth=(257, 231), chin=(252, 247)),
    "down-right": dict(nose=(321, 202, "est"), eye_r=(272, 170), eye_l=(340, 170), mouth=(300, 221), chin=(313, 245)),
}


def pin(rest, tris, p):
    """(triangle index, barycentric weights) of point p on the mesh."""
    p = np.asarray(p, float)
    for i, t in enumerate(tris):
        a, b, c = rest[t]
        m = np.column_stack([b - a, c - a])
        if abs(np.linalg.det(m)) < 1e-6:
            continue
        u, v = np.linalg.solve(m, p - a)
        if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6:
            return i, np.array([1 - u - v, u, v])
    raise ValueError(f"point {p} is not on the mesh")


if __name__ == "__main__":
    geo_path, psd_path, sheet_path, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    pins = {k: pin(rest, tris, p) for k, p in CANVAS.items()}

    psd = PSDImage.open(psd_path)
    layer = next(l for l in psd if l.name == "face")
    tex = np.zeros((psd.height, psd.width, 4), np.uint8)
    tex[layer.top:layer.bottom, layer.left:layer.right] = np.array(layer.topil().convert("RGBA"))
    sheet = Image.open(sheet_path).convert("RGBA")
    d = ImageDraw.Draw(sheet)

    z = H.depth(F.FREE_FIT, rest[:, 0], rest[:, 1], "face")
    results = {}
    s = F.SHEET_SCALE
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        px, py = H.project(F.FREE_FIT, rest[:, 0], rest[:, 1], z, ax, ay) if (ax, ay) != (0, 0) else (rest[:, 0], rest[:, 1])
        cur = np.column_stack([px, py])
        model = {k: (cur[tris[ti]] * w[:, None]).sum(0) for k, (ti, w) in pins.items()}
        ref = REFERENCE[name]
        r, c = divmod(i, 3)
        cell0 = np.array([c * F.CELL_W, r * F.CELL_H], float)
        ref_nose = np.array(ref["nose"][:2], float)
        # translation that puts the model nose on the reference nose
        offset = cell0 + ref_nose - model["nose"] * s

        part = F.warp(tex, rest, cur, tris)
        small = Image.fromarray(part).resize((round(part.shape[1] * s), round(part.shape[0] * s)), Image.LANCZOS)
        a = np.array(small)
        a[..., 3] = (a[..., 3] * 0.5).astype(np.uint8)
        sheet.alpha_composite(Image.fromarray(a), (round(offset[0]), round(offset[1])))
        m = (np.array(small)[..., 3] > 128).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            pts = [(p[0][0] + round(offset[0]), p[0][1] + round(offset[1])) for p in cnt]
            d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)

        rows = {}
        for k in ("eye_r", "eye_l", "mouth", "chin"):
            vr = np.array(ref[k][:2], float) - ref_nose
            vm = (model[k] - model["nose"]) * s
            ratio = np.linalg.norm(vm) / max(np.linalg.norm(vr), 1e-6)
            ang = np.degrees(np.arctan2(vm[1], vm[0]) - np.arctan2(vr[1], vr[0]))
            ang = (ang + 180) % 360 - 180
            miss = np.linalg.norm(vm - vr)
            rows[k] = dict(ratio=round(float(ratio), 3), angle=round(float(ang), 1), miss_px=round(float(miss), 1))
            pr = cell0 + np.array(ref[k][:2]); pm = offset + model[k] * s
            d.ellipse((pr[0] - 4, pr[1] - 4, pr[0] + 4, pr[1] + 4), outline=(0, 170, 0, 255), width=2)
            d.ellipse((pm[0] - 3, pm[1] - 3, pm[0] + 3, pm[1] + 3), fill=(230, 0, 180, 255))
            d.line([tuple(pr), tuple(pm)], fill=(255, 120, 0, 255), width=1)
        pn = cell0 + ref_nose
        d.line([(pn[0] - 7, pn[1]), (pn[0] + 7, pn[1])], fill=(0, 90, 255, 255), width=2)
        d.line([(pn[0], pn[1] - 7), (pn[0], pn[1] + 7)], fill=(0, 90, 255, 255), width=2)
        mean_dev = float(np.mean([abs(v["ratio"] - 1) for v in rows.values()]))
        results[name] = dict(nose_estimated=len(ref["nose"]) > 2, landmarks=rows, mean_length_error=round(mean_dev, 3))
    sheet.convert("RGB").save(os.path.join(out_dir, "nose_overlay_9dir.png"))
    json.dump(results, open(os.path.join(out_dir, "nose_check.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{'direction':11s} {'nose':4s} " + "  ".join(f"{k:>18s}" for k in ("eye_r", "eye_l", "mouth", "chin")) + "   mean")
    for name, r in results.items():
        cells = "  ".join(f"x{v['ratio']:.2f} {v['angle']:+5.1f}° {v['miss_px']:4.1f}px" for v in r["landmarks"].values())
        print(f"{name:11s} {'est' if r['nose_estimated'] else 'meas':4s} {cells}   {r['mean_length_error']:.0%}")
