"""Head-turn targets for the 9-direction keyforms, measured from reference/angles-9dir.png.

Each reference cell (512x341) was measured by hand against the collar buckle, which is the body anchor
(the framing of the cells differs slightly). Offsets below are in reference-cell pixels relative to the
front cell; they are scaled onto the canvas with the eye distance (canvas 239 px / reference 87 px).

Usage: python3 angle_targets.py <out.json>
Writes one dense displacement field per ParamAngleX/ParamAngleY key (thin-plate spline through the
landmarks, zero on the body), which tools/headless applies to every head mesh as vertex keyforms.
"""
import json
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator

SCALE = 239 / 87  # canvas px per reference px

# Rest positions on the canvas (front illustration). "r" = screen-left (character right), "l" = screen-right.
LANDMARKS = {
    "eye_r": (785, 504), "eye_l": (1024, 504),
    "eye_out_r": (700, 485), "eye_out_l": (1102, 480), "eye_in_r": (840, 500), "eye_in_l": (975, 500),
    "cheek_r": (730, 600), "cheek_l": (1070, 600),
    "nose": (900, 600), "mouth": (903, 657), "chin": (900, 771),
    "jaw_r": (770, 690), "jaw_l": (1030, 690),
    "brow_mid": (900, 410), "forehead": (900, 250),
    "top": (900, 40),
    "ear_r": (540, 15), "ear_l": (1262, 8),
    "ear_base_r": (640, 230), "ear_base_l": (1160, 230),
    "hair_r": (500, 560), "hair_l": (1320, 560),
    "hair_tip_r": (600, 740), "hair_tip_l": (1200, 740),
}
MIRROR = {"eye_r": "eye_l", "eye_out_r": "eye_out_l", "eye_in_r": "eye_in_l", "cheek_r": "cheek_l", "jaw_r": "jaw_l", "ear_r": "ear_l", "ear_base_r": "ear_base_l",
          "hair_r": "hair_l", "hair_tip_r": "hair_tip_l"}
MIRROR.update({v: k for k, v in list(MIRROR.items())})

# Turning toward screen-right (ParamAngleX = +45). Left/right cells were averaged and mirrored.
TURN_RIGHT = {
    "eye_r": (61, -14), "eye_l": (41, -14),
    "eye_out_r": (58, -14), "eye_out_l": (36, -14), "eye_in_r": (63, -14), "eye_in_l": (45, -14),
    "cheek_r": (58, -13), "cheek_l": (33, -13),
    "nose": (73, -18), "mouth": (59, -13), "chin": (48, -12),
    "jaw_r": (55, -12), "jaw_l": (30, -12),
    "brow_mid": (52, -14), "forehead": (35, -10),
    "top": (15, -4),
    "ear_r": (28, -5), "ear_l": (-11, -5),
    "ear_base_r": (25, -8), "ear_base_l": (-5, -8),
    "hair_r": (-8, -12), "hair_l": (-8, -12),
    "hair_tip_r": (10, -12), "hair_tip_l": (5, -12),
}
# Looking up (ParamAngleY = +30): the lower face folds up toward the eyes, the ears drop behind the skull.
LOOK_UP = {
    "eye_r": (0, -40), "eye_l": (0, -40),
    "eye_out_r": (0, -40), "eye_out_l": (0, -40), "eye_in_r": (0, -40), "eye_in_l": (0, -40),
    "cheek_r": (-3, -38), "cheek_l": (3, -38),
    "nose": (0, -64), "mouth": (0, -53), "chin": (0, -44),
    "jaw_r": (-6, -25), "jaw_l": (6, -25),
    "brow_mid": (0, -30), "forehead": (0, -12),
    "top": (0, 5),
    "ear_r": (-12, 53), "ear_l": (12, 53),
    "ear_base_r": (-6, 30), "ear_base_l": (6, 30),
    "hair_r": (-5, 3), "hair_l": (5, 3),
    "hair_tip_r": (0, -10), "hair_tip_l": (0, -10),
}
# Looking down (ParamAngleY = -30): features sink, the chin tucks onto the collar, the crown grows.
LOOK_DOWN = {
    "eye_r": (0, 33), "eye_l": (0, 33),
    "eye_out_r": (0, 33), "eye_out_l": (0, 33), "eye_in_r": (0, 33), "eye_in_l": (0, 33),
    "cheek_r": (3, 28), "cheek_l": (-3, 28),
    "nose": (0, 31), "mouth": (0, 28), "chin": (0, 11),
    "jaw_r": (6, 22), "jaw_l": (-6, 22),
    "brow_mid": (0, 30), "forehead": (0, 22),
    "top": (0, -8),
    "ear_r": (-10, 8), "ear_l": (10, 8),
    "ear_base_r": (-6, 12), "ear_base_l": (6, 12),
    "hair_r": (-6, -3), "hair_l": (6, -3),
    "hair_tip_r": (0, 12), "hair_tip_l": (0, 12),
}
# Body points never move with the head.
BODY_ANCHORS = [(x, y) for x in range(150, 1700, 150) for y in (1000, 1300, 1600)] + [(820, 900), (980, 900)]


def turn_left():
    return {k: (-TURN_RIGHT[MIRROR.get(k, k)][0], TURN_RIGHT[MIRROR.get(k, k)][1]) for k in LANDMARKS}


# Diagonal cells rise/sink less than the straight up/down cells (up-left eyes: -27/-45 vs up -40 plus
# turn -14): the vertical part of a diagonal is scaled by this factor.
DIAGONAL_VERTICAL = 0.7


def offsets(ax, ay):
    """Reference-pixel offsets for one key; diagonals combine the horizontal and vertical offsets."""
    hx = TURN_RIGHT if ax > 0 else turn_left() if ax < 0 else None
    vy = LOOK_UP if ay > 0 else LOOK_DOWN if ay < 0 else None
    out = {}
    for k in LANDMARKS:
        dx = dy = 0.0
        if hx:
            dx += hx[k][0]
            dy += hx[k][1]
        if vy:
            f = DIAGONAL_VERTICAL if hx else 1.0
            dx += vy[k][0] * f
            dy += vy[k][1] * f
        out[k] = (dx * SCALE, dy * SCALE)
    return out


def field(ax, ay, x0=250, y0=-100, x1=1550, y1=1100, step=10):
    names = list(LANDMARKS)
    src = np.array([LANDMARKS[k] for k in names] + BODY_ANCHORS, float)
    off = offsets(ax, ay)
    dst = np.array([off[k] for k in names] + [(0, 0)] * len(BODY_ANCHORS), float)
    rbf = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=1.0)
    xs = np.arange(x0, x1 + 1, step)
    ys = np.arange(y0, y1 + 1, step)
    gx, gy = np.meshgrid(xs, ys)
    d = rbf(np.column_stack([gx.ravel(), gy.ravel()]))
    return dict(angleX=ax, angleY=ay, x0=x0, y0=y0, step=step, nx=len(xs), ny=len(ys),
                dx=np.round(d[:, 0], 2).tolist(), dy=np.round(d[:, 1], 2).tolist())


if __name__ == "__main__":
    keys = [(ax, ay) for ay in (-30, 0, 30) for ax in (-45, 0, 45) if (ax, ay) != (0, 0)]
    with open(sys.argv[1], "w") as f:
        json.dump({"poses": [field(ax, ay) for ax, ay in keys]}, f)
    print("poses:", len(keys))
