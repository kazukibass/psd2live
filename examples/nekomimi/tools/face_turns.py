"""Face-base part turned by the unconstrained 3D proxy fit, one PNG per direction, overlaid on the sheet.

The face layer is warped triangle by triangle with its psd2live mesh; each vertex moves as in
tools/head3d.py with the first (harness-free) fit, which turns the head about 30 deg left/right,
30 deg up and 16 deg down. Each result is then placed on its reference cell, aligned by the collar
buckle found in that cell, at the sheet scale (87 px between the eyes vs 239 px on the canvas).

Usage: python3 face_turns.py <rest-geometry.json> <nekomimi.psd> <angles-9dir.png> <out-dir>
Writes <out-dir>/face_<direction>.png (canvas size, transparent) and <out-dir>/overlay_9dir.png.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from psd_tools import PSDImage

import head3d as H

# the harness-free fit (tools/head3d.py before the harness penalties were added)
FREE_FIT = dict(cx=902.1, cy=522.3, rx=357.0, ry=365.9, rz=327.6, k=0.9, nose=83.1, hair=0.0, back=0.0,
                ear=0.0, py=723.7, pz=23.8, yaw=29.6, up=30.2, down=16.0)
DIRECTIONS = [("up-left", -45, 30), ("up", 0, 30), ("up-right", 45, 30),
              ("left", -45, 0), ("front", 0, 0), ("right", 45, 0),
              ("down-left", -45, -30), ("down", 0, -30), ("down-right", 45, -30)]
CELL_W, CELL_H = 512, 341
SHEET_SCALE = 87 / 239             # reference px per canvas px
FRONT_EYES_CANVAS = (904.5, 504)   # mid-point between the eyes on the canvas
FRONT_EYES_CELL = (253.5, 152)     # the same point in the front cell


def warp(tex, rest, cur, tris):
    """Piecewise-affine warp of an RGBA canvas-size texture from rest to cur mesh positions."""
    h, w = tex.shape[:2]
    out = np.zeros_like(tex, dtype=np.float32)
    for t in tris:
        src = rest[t].astype(np.float32)
        dst = cur[t].astype(np.float32)
        x0, y0 = np.floor(dst.min(0)).astype(int) - 1
        x1, y1 = np.ceil(dst.max(0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        m = cv2.getAffineTransform(src, dst - np.float32([x0, y0]))
        patch = cv2.warpAffine(tex, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round((dst - [x0, y0]) * 4).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=2)
        sel = mask > 0
        out[y0:y1, x0:x1][sel] = patch[sel]
    return out.astype(np.uint8)


def buckle_offsets(sheet):
    """Collar-buckle shift of every cell relative to the front cell (template match)."""
    front = sheet[CELL_H:2 * CELL_H, CELL_W:2 * CELL_W]
    tpl = cv2.cvtColor(front[235:290, 225:295], cv2.COLOR_RGB2GRAY)
    ref_pos = None
    offsets = {}
    for i, (name, _, _) in enumerate(DIRECTIONS):
        r, c = divmod(i, 3)
        cell = cv2.cvtColor(sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W], cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(cell[150:, :], tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        pos = np.array([loc[0], loc[1] + 150])
        offsets[name] = (pos, score)
    ref_pos = offsets["front"][0]
    return {k: (p - ref_pos, s) for k, (p, s) in offsets.items()}


if __name__ == "__main__":
    geo_path, psd_path, sheet_path, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    geo = json.load(open(geo_path))
    face = next(d for d in geo["drawables"] if d["id"] == "ArtMeshFace")
    rest = np.array(face["world"]["rest"]).reshape(-1, 2)
    tris = np.array(face["indices"]).reshape(-1, 3)

    psd = PSDImage.open(psd_path)
    layer = next(l for l in psd if l.name == "face")
    tex = np.zeros((psd.height, psd.width, 4), np.uint8)
    tex[layer.top:layer.bottom, layer.left:layer.right] = np.array(layer.topil().convert("RGBA"))

    sheet = np.array(Image.open(sheet_path).convert("RGB"))
    offsets = buckle_offsets(sheet)
    overlay = Image.fromarray(sheet).convert("RGBA")

    z = H.depth(FREE_FIT, rest[:, 0], rest[:, 1], "face")
    for i, (name, ax, ay) in enumerate(DIRECTIONS):
        px, py = H.project(FREE_FIT, rest[:, 0], rest[:, 1], z, ax, ay) if (ax, ay) != (0, 0) else (rest[:, 0], rest[:, 1])
        cur = np.column_stack([px, py])
        part = warp(tex, rest, cur, tris)
        Image.fromarray(part).save(os.path.join(out_dir, f"face_{name}.png"))

        # place on the reference cell: canvas -> cell coordinates, shifted by that cell's buckle offset
        shift, score = offsets[name]
        r, c = divmod(i, 3)
        scale = SHEET_SCALE
        ox = c * CELL_W + FRONT_EYES_CELL[0] - FRONT_EYES_CANVAS[0] * scale + shift[0]
        oy = r * CELL_H + FRONT_EYES_CELL[1] - FRONT_EYES_CANVAS[1] * scale + shift[1]
        small = Image.fromarray(part).resize((round(part.shape[1] * scale), round(part.shape[0] * scale)), Image.LANCZOS)
        a = np.array(small)
        a[..., 3] = (a[..., 3] * 0.55).astype(np.uint8)   # translucent so the drawing shows through
        overlay.alpha_composite(Image.fromarray(a), (round(ox), round(oy)))
        # outline of the part in magenta
        m = (np.array(small)[..., 3] > 128).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        d = ImageDraw.Draw(overlay)
        for cnt in cs:
            pts = [(p[0][0] + round(ox), p[0][1] + round(oy)) for p in cnt]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=(230, 0, 180, 255), width=2)
        print(f"{name:10s} buckle shift {tuple(int(v) for v in shift)} match {score:.2f}")
    overlay.convert("RGB").save(os.path.join(out_dir, "overlay_9dir.png"))
