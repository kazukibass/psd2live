"""Compare a hand-drawn face region / centre line with the turned face base.

Input is the nose overlay (tools/nose_check.py) with the face region drawn over it by hand: blue =
face region outline, black = face centre line. Both are extracted by colour per cell and compared
with the model's face base (placed nose-on-nose exactly as in the overlay) and its centre line (the
canvas column x = 900 pinned to the face mesh).

Usage: python3 face_region_compare.py <drawn.jpg> <rest-geometry.json> <out-dir>
Writes <out-dir>/region_compare_9dir.png and region_compare.json, prints the table.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import face_turns as F
import head3d as H
import nose_check as N

MID_X = 900                       # face centre column on the canvas
MID_Y = np.arange(330, 771, 10)   # forehead to chin


def drawn_region(cell_rgb):
    r, g, b = [cell_rgb[..., i].astype(int) for i in range(3)]
    blue = ((b > 150) & (r < 90) & (g < 110) & (b - r > 90)).astype(np.uint8)
    if blue.sum() < 50:
        return None, None
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    cs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    region = np.zeros_like(blue)
    cv2.drawContours(region, [max(cs, key=cv2.contourArea)], -1, 1, thickness=-1)
    return region > 0, blue > 0


def drawn_midline(cell_rgb, region):
    r, g, b = [cell_rgb[..., i].astype(int) for i in range(3)]
    black = (r < 50) & (g < 50) & (b < 50)
    if region is not None:
        black &= cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    else:
        black[250:] = False
    pts = []
    for y in range(black.shape[0]):
        xs = np.where(black[y])[0]
        if len(xs):
            pts.append((float(np.median(xs)), y))
    return np.array(pts) if pts else None


def x_at(line, y):
    """x of a polyline (sorted by y) at height y, or None outside its span."""
    if line is None or len(line) < 2 or y < line[:, 1].min() or y > line[:, 1].max():
        return None
    return float(np.interp(y, line[:, 1], line[:, 0]))


if __name__ == "__main__":
    drawn_path, geo_path, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    drawn = np.array(Image.open(drawn_path).convert("RGB"))
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)
    nose_pin = N.pin(rest, tris, N.CANVAS["nose"])
    mid_pins = [N.pin(rest, tris, (MID_X, y)) for y in MID_Y]
    z = H.depth(F.FREE_FIT, rest[:, 0], rest[:, 1], "face")
    s = F.SHEET_SCALE

    canvas = Image.fromarray(drawn).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    results = {}
    for i, (name, ax, ay) in enumerate(F.DIRECTIONS):
        r, c = divmod(i, 3)
        x0, y0 = c * F.CELL_W, r * F.CELL_H
        cell = drawn[y0:y0 + F.CELL_H, x0:x0 + F.CELL_W]
        region, _ = drawn_region(cell)
        mid_drawn = drawn_midline(cell, region)

        px, py = H.project(F.FREE_FIT, rest[:, 0], rest[:, 1], z, ax, ay) if (ax, ay) != (0, 0) else (rest[:, 0], rest[:, 1])
        cur = np.column_stack([px, py])
        at = lambda pinned: (cur[tris[pinned[0]]] * pinned[1][:, None]).sum(0)
        off = np.array(N.REFERENCE[name]["nose"][:2], float) - at(nose_pin) * s   # cell coords of canvas origin
        model_mid = np.array([at(p) * s + off for p in mid_pins])
        tri_px = (cur[tris] * s + off).astype(np.float32)
        model = np.zeros((F.CELL_H, F.CELL_W), np.uint8)
        for t in tri_px:
            cv2.fillConvexPoly(model, np.round(t * 4).astype(np.int32), 1, shift=2)
        model = model > 0

        row = {}
        if region is not None:
            inter, union = (region & model).sum(), (region | model).sum()
            row["iou"] = round(float(inter / union), 3)
            ys, xs = np.where(region)
            ym, xm = np.where(model)
            row["drawn_box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            row["model_box"] = [int(xm.min()), int(ym.min()), int(xm.max()), int(ym.max())]
            # where the centre line sits across the face width, at the height of the drawn region's middle
            yc = (ys.min() + ys.max()) / 2
            for key, line, mask in (("drawn", mid_drawn, region), ("model", model_mid, model)):
                xl = x_at(line, yc)
                cols = np.where(mask[int(yc)])[0]
                if xl is not None and len(cols):
                    row[f"{key}_midline_share"] = round((xl - cols.min()) / max(cols.max() - cols.min(), 1), 3)
        results[name] = row

        # drawing: drawn region (blue fill), model outline (magenta), both centre lines
        overlay = np.zeros((F.CELL_H, F.CELL_W, 4), np.uint8)
        if region is not None:
            overlay[region] = (40, 80, 255, 60)
        canvas.alpha_composite(Image.fromarray(overlay), (x0, y0))
        cs, _ = cv2.findContours(model.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            pts = [(p[0][0] + x0, p[0][1] + y0) for p in cnt]
            d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)
        d.line([(p[0] + x0, p[1] + y0) for p in model_mid], fill=(255, 140, 0, 255), width=3)
        label = f"IoU {row.get('iou', '-')}"
        d.rectangle((x0 + 380, y0 + 8, x0 + 505, y0 + 30), fill=(255, 255, 255, 230))
        d.text((x0 + 386, y0 + 12), label, fill=(0, 0, 0, 255))
    canvas.convert("RGB").save(os.path.join(out_dir, "region_compare_9dir.png"))
    json.dump(results, open(os.path.join(out_dir, "region_compare.json"), "w"), indent=1)
    print(f"{'direction':11s} {'IoU':>5s}  {'drawn box (x0,y0,x1,y1)':>24s}  {'model box':>24s}  midline share drawn/model")
    for name, r in results.items():
        print(f"{name:11s} {str(r.get('iou', '-')):>5s}  {str(r.get('drawn_box', '-')):>24s}  {str(r.get('model_box', '-')):>24s}  "
              f"{r.get('drawn_midline_share', '-')} / {r.get('model_midline_share', '-')}")
