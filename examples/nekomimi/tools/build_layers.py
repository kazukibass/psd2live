"""Split the flat character illustration into psd2live semantic layers and write a PSD.

Usage: python3 build_layers.py <front.png> <out.psd>
All coordinates below are hand-measured on the 1800x1600 front illustration (reference/front.png).
"""
import struct
import sys
import numpy as np
from psd_tools.compression import Compression, compress
import cv2
from PIL import Image, ImageDraw

SRC_PATH = sys.argv[1] if len(sys.argv) > 1 else "src.png"
OUT_PSD = sys.argv[2] if len(sys.argv) > 2 else "character.psd"
SRC = np.array(Image.open(SRC_PATH).convert("RGBA")).astype(np.uint8)
H, W = SRC.shape[:2]
RGB = SRC[..., :3].astype(int)
ALPHA = SRC[..., 3]
R, G, B = RGB[..., 0], RGB[..., 1], RGB[..., 2]
LUM = 0.3 * R + 0.59 * G + 0.11 * B
OPAQUE = ALPHA > 8
YY, XX = np.mgrid[0:H, 0:W]


def poly(points):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(points, fill=255)
    return np.array(img) > 0


def ellipse(cx, cy, rx, ry):
    return ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 <= 1.0


def dilate(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def inpaint(rgb, hole, radius=6):
    """Fill `hole` pixels of an RGB image from their surroundings, on a bbox crop for speed."""
    ys, xs = np.where(hole)
    if len(ys) == 0:
        return rgb
    pad = 20
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    crop = rgb[y0:y1, x0:x1].astype(np.uint8).copy()
    mask = hole[y0:y1, x0:x1].astype(np.uint8) * 255
    out = rgb.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(crop, mask, radius, cv2.INPAINT_TELEA)
    return out


def layer(rgb, mask, soft_alpha=None):
    """RGBA layer: colour from rgb, alpha from mask (optionally keeping the source's soft edge)."""
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    a = mask.astype(np.uint8) * 255
    if soft_alpha is not None:
        a = np.minimum(a, soft_alpha)
    out[..., 3] = a
    return out


# ---------------------------------------------------------------- skin regions
line = ((LUM < 215) & (ALPHA > 128)).astype(np.uint8)
line = cv2.dilate(line, np.ones((2, 2), np.uint8))
free = ((line == 0) & (ALPHA > 200)).astype(np.uint8)
_, lab = cv2.connectedComponents(free, connectivity=4)
skin_comp = lab == lab[620, 900]
# grow skin over the thin anti-aliased edge that the line mask ate, but never into dark lines
skin = skin_comp | (dilate(skin_comp, 2) & (LUM > 200) & (R - B > 8))

FACE_POLY = poly([(690, 430), (715, 360), (790, 322), (900, 312), (1010, 322), (1085, 360), (1110, 430), (1110, 640), (1040, 700), (900, 776), (765, 695), (690, 640)])
face_skin = skin & FACE_POLY
neck_box = poly([(805, 690), (1000, 690), (1000, 770), (805, 770)])
neck_skin = neck_box & ~FACE_POLY & (LUM > 150) & (R - B > 8) & OPAQUE

# ---------------------------------------------------------------- eyes
EYES = {
    # screen-left eye = character right
    "r": dict(
        region=poly([(695, 470), (730, 461), (790, 457), (842, 468), (846, 490), (830, 522), (810, 540),
                     (760, 542), (728, 530), (695, 482)]),
        white=poly([(714, 482), (750, 479), (810, 477), (827, 487), (831, 500), (826, 515), (810, 527),
                    (780, 532), (750, 529), (730, 521), (717, 500)]),
        iris=(785, 504, 34, 30),
        cut=490,
    ),
    # screen-right eye = character left
    "l": dict(
        region=poly([(960, 478), (990, 459), (1025, 457), (1070, 459), (1101, 464), (1103, 482), (1085, 522),
                     (1070, 538), (1030, 542), (995, 537), (976, 516), (963, 492)]),
        white=poly([(981, 490), (1000, 482), (1050, 480), (1080, 484), (1092, 492), (1090, 505), (1080, 520),
                    (1060, 530), (1030, 533), (1000, 530), (988, 520), (981, 505)]),
        iris=(1024, 504, 32, 29),
        cut=490,
    ),
}
SCLERA = np.array([250, 246, 248])

eye_layers = {}
face_holes = np.zeros((H, W), bool)
face_detail = np.zeros((H, W), bool)
for side, e in EYES.items():
    cx, cy, rx, ry = e["iris"]
    iris_full = ellipse(cx, cy, rx, ry)
    dark = e["region"] & (LUM < 140) & ~(iris_full & e["white"])
    lash = dark & (YY <= e["cut"])
    lash = dilate(lash, 1) & e["region"] & (LUM < 200) & ~(iris_full & e["white"])
    lower = e["region"] & ~e["white"] & ~lash & (YY > e["cut"] - 4) & (LUM < 205)

    iris_visible = iris_full & e["white"] & ~lash
    iris_rgb = inpaint(RGB, iris_full & ~iris_visible, radius=5)
    white_rgb = RGB.copy()
    white_fill = e["white"] & (iris_full | lash | (LUM < 200))
    white_rgb[white_fill] = SCLERA
    white_rgb = cv2.GaussianBlur(white_rgb.astype(np.float32), (0, 0), 1.0).astype(int)
    white_rgb[e["white"] & ~white_fill] = RGB[e["white"] & ~white_fill]

    eye_layers[side] = dict(
        eyewhite=layer(np.where(lower[..., None], RGB, white_rgb), e["white"] | lower,
                       np.where(lower, ALPHA, 255).astype(np.uint8)),
        irides=layer(iris_rgb, iris_full & ~lower),
        eyelash=layer(RGB, lash, ALPHA),
    )
    # repaint the whole (slightly grown) eye area with skin so nothing ghosts when the eye parts move
    face_holes |= dilate(e["region"], 4) & ~(~skin & (LUM > 150) & ~e["region"])

# ---------------------------------------------------------------- brows / nose / mouth
brows = {
    "r": poly([(770, 402), (830, 402), (830, 426), (770, 426)]) & (LUM < 125),
    "l": poly([(985, 398), (1066, 398), (1066, 426), (985, 426)]) & (LUM < 125),
}
for side in brows:
    brows[side] = dilate(brows[side], 1) & (LUM < 190) & ~(~skin & (LUM > 150))
    face_holes |= brows[side]

skin_med = np.median(RGB[face_skin], axis=0)
nose_box = poly([(884, 548), (918, 548), (918, 612), (884, 612)])
nose = nose_box & (np.abs(RGB - skin_med).sum(-1) > 14)
face_holes |= nose

mouth_line = poly([(845, 644), (962, 644), (962, 668), (845, 668)]) & (LUM < 200)
face_holes |= dilate(mouth_line, 1)

# Synthesised open mouth: psd2live expects the maximally-open drawing and compresses it to a line.
SS = 4  # draw supersampled, then downscale for anti-aliased edges
MOUTH_BOX = (840, 640, 970, 700)
top = [(854, 656), (870, 654), (903, 653), (935, 654), (953, 657)]
bottom = [(940, 670), (920, 679), (903, 681), (885, 679), (866, 670)]
outline = [((x - MOUTH_BOX[0]) * SS, (y - MOUTH_BOX[1]) * SS) for x, y in top + bottom]
big_size = ((MOUTH_BOX[2] - MOUTH_BOX[0]) * SS, (MOUTH_BOX[3] - MOUTH_BOX[1]) * SS)
big = Image.new("RGBA", big_size, (0, 0, 0, 0))
ImageDraw.Draw(big).polygon(outline, fill=(122, 48, 56, 255))
tongue = Image.new("RGBA", big_size, (0, 0, 0, 0))
ImageDraw.Draw(tongue).ellipse(((878 - 840) * SS, (668 - 640) * SS, (930 - 840) * SS, (688 - 640) * SS),
                               fill=(214, 116, 124, 255))
clip = Image.new("L", big_size, 0)
ImageDraw.Draw(clip).polygon(outline, fill=255)
tongue.putalpha(Image.fromarray(np.minimum(np.array(tongue)[..., 3], np.array(clip))))
big.alpha_composite(tongue)
ImageDraw.Draw(big).line(outline + [outline[0]], fill=(72, 42, 42, 255), width=4 * SS, joint="curve")
mouth_pil = Image.new("RGBA", (W, H), (0, 0, 0, 0))
mouth_pil.paste(big.resize((MOUTH_BOX[2] - MOUTH_BOX[0], MOUTH_BOX[3] - MOUTH_BOX[1]), Image.LANCZOS), MOUTH_BOX[:2])
mouth_layer = np.array(mouth_pil)

# ---------------------------------------------------------------- face / neck
body_region_pre = OPAQUE & (((YY >= 745) & (XX >= 770) & (XX <= 1030)) | (YY >= 880))
face_mask = (FACE_POLY & ~skin) | face_skin | face_holes
face_mask &= FACE_POLY
face_rgb = RGB.copy()
hidden = FACE_POLY & ~face_skin
face_rgb[hidden] = skin_med
face_rgb = inpaint(face_rgb, face_holes & face_skin | face_holes, radius=4)
face_rgb[face_skin & ~face_holes] = RGB[face_skin & ~face_holes]
below_jaw = FACE_POLY & (YY > 640) & ~face_skin & ~face_holes & ((LUM > 150) | (ALPHA < 128)) & ~(dilate(face_skin, 3) & (LUM <= 215))
face_layer = layer(face_rgb, FACE_POLY & ~below_jaw & ~(body_region_pre & ~face_skin & (LUM < 120)))

neck_med = np.median(RGB[neck_skin], axis=0) if neck_skin.any() else skin_med
neck_poly = poly([(815, 620), (995, 620), (998, 775), (812, 775)])
neck_rgb = RGB.copy()
neck_rgb[neck_poly & ~neck_skin] = neck_med
neck_layer = layer(neck_rgb, neck_poly)

# ---------------------------------------------------------------- ears
EARS = {
    "r": dict(
        visible=poly([(533, 10), (560, 12), (600, 44), (660, 78), (703, 99), (717, 132), (700, 147), (663, 187),
                      (647, 207), (617, 233), (600, 253), (600, 318), (580, 287), (550, 240), (537, 195),
                      (525, 100)]),
        base=poly([(717, 132), (775, 170), (745, 300), (600, 318), (600, 253), (663, 187)]),
    ),
    "l": dict(
        visible=poly([(1250, 6), (1266, 6), (1282, 25), (1278, 100), (1271, 150), (1258, 200), (1241, 262),
                      (1218, 301), (1207, 318), (1177, 253), (1157, 203), (1150, 187), (1117, 147),
                      (1087, 117), (1117, 96), (1183, 46), (1223, 26)]),
        base=poly([(1087, 117), (1030, 160), (1060, 300), (1207, 318), (1177, 253), (1150, 187)]),
    ),
}
ear_all = np.zeros((H, W), bool)
for e in EARS.values():
    ear_all |= e["visible"] & OPAQUE

# ---------------------------------------------------------------- body / ponytail / hair
PONY_POLY = poly([(650, 683), (550, 792), (600, 833), (500, 897), (583, 917), (600, 1033), (650, 1117),
                  (708, 1183), (735, 1270), (767, 1183), (783, 1117), (800, 1050), (790, 983), (785, 900),
                  (790, 800), (800, 717), (767, 683)])
neutral = (np.max(RGB, -1) - np.min(RGB, -1)) < 7
pony = PONY_POLY & OPAQUE & (YY >= 695) & ~(neutral & (ALPHA > 250)) & ~neck_skin
pony = cv2.morphologyEx(pony.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0

body_region = OPAQUE & (((YY >= 745) & (XX >= 770) & (XX <= 1030)) | (YY >= 880))
body = body_region & ~pony & ~neck_skin & ~FACE_POLY
body_rgb = inpaint(RGB, (pony & body_region) | (FACE_POLY & body_region), radius=7)
body_layer = layer(body_rgb, body | (pony & body_region) | (FACE_POLY & body_region & (YY >= 745)),
                   np.where(pony | FACE_POLY, 255, ALPHA).astype(np.uint8))
pony_layer = layer(RGB, pony, ALPHA)

hair = OPAQUE & ~body_region & ~pony & ~face_skin & ~neck_skin & ~ear_all & ~face_holes
hair &= ~(FACE_POLY & ~(LUM < 250) & skin)
root = PONY_POLY & (YY >= 695) & (YY < 740)
hair |= root & OPAQUE & ~ (neutral & (ALPHA > 250))
fade = np.clip((740 - YY) / 45.0, 0, 1)
front_alpha = np.where(root, (ALPHA * fade).astype(np.uint8), ALPHA).astype(np.uint8)
front_hair_layer = layer(RGB, hair & ~(PONY_POLY & (YY >= 740)), front_alpha)
hair_core = cv2.erode((hair & (ALPHA > 250)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0

ear_layers = {}
for side, e in EARS.items():
    vis = e["visible"] & OPAQUE
    e["base"] = e["base"] & hair_core & dilate(vis, 14)
    tan = vis & (R - B > 25) & (LUM > 120)
    ear_rgb = RGB.copy()
    ear_rgb[e["base"] & ~vis] = np.median(RGB[tan], axis=0)
    ear_layers[side] = layer(ear_rgb, vis | e["base"], np.where(e["base"] & ~vis, 255, ALPHA).astype(np.uint8))


# back hair: the hair, with the hole behind the face filled with a darker hair tone
# (the fill stays inside the face outline so its straight edges are always covered by the face)
hair_med = np.median(RGB[hair & (LUM > 200)], axis=0)
back_rgb = RGB.copy()
back_fill = FACE_POLY & (YY < 690) & ~hair
back_rgb[back_fill] = (hair_med * 0.93).astype(int)
back_hair_layer = layer(back_rgb, back_fill | hair & (YY < 740),
                        np.where(back_fill, 255, ALPHA).astype(np.uint8))

# ---------------------------------------------------------------- facedetail (lower lids)
face_detail_layer = layer(RGB, face_detail, ALPHA)

# ---------------------------------------------------------------- assemble (bottom -> top)
LAYERS = [
    ("back hair", back_hair_layer),
    ("ears-r", ear_layers["r"]),
    ("ears-l", ear_layers["l"]),
    ("neck", neck_layer),
    ("topwear", body_layer),
    ("ponytail", pony_layer),
    ("face", face_layer),
    ("eyewhite-r", eye_layers["r"]["eyewhite"]),
    ("eyewhite-l", eye_layers["l"]["eyewhite"]),
    ("irides-r", eye_layers["r"]["irides"]),
    ("irides-l", eye_layers["l"]["irides"]),
    ("eyelash-r", eye_layers["r"]["eyelash"]),
    ("eyelash-l", eye_layers["l"]["eyelash"]),
    ("eyebrow-r", layer(RGB, brows["r"], ALPHA)),
    ("eyebrow-l", layer(RGB, brows["l"], ALPHA)),
    ("nose", layer(RGB, nose, ALPHA)),
    ("mouth", mouth_layer),
    ("front hair", front_hair_layer),
]


# ---------------------------------------------------------------- PSD writer (RGB 8bit, raw channels)
def pascal(name):
    b = name.encode("ascii", "replace")[:255]
    s = bytes([len(b)]) + b
    return s + b"\0" * ((-len(s)) % 4)


def rle(channel):
    """PackBits per PSD spec: row byte counts followed by the packed rows."""
    h, w = channel.shape
    return compress(np.ascontiguousarray(channel).tobytes(), Compression.RLE, w, h, 8, 1)


def write_psd(path, layers):
    records, channel_data = b"", b""
    for name, rgba in layers:
        ys, xs = np.where(rgba[..., 3] > 0)
        if len(ys) == 0:
            raise SystemExit(f"empty layer: {name}")
        t, l, b, r = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        crop = rgba[t:b, l:r]
        chans = [(-1, crop[..., 3]), (0, crop[..., 0]), (1, crop[..., 1]), (2, crop[..., 2])]
        rec = struct.pack(">iiiiH", t, l, b, r, len(chans))
        for cid, data in chans:
            packed = struct.pack(">H", 1) + rle(data)
            rec += struct.pack(">hI", cid, len(packed))
            channel_data += packed
        uname = name.encode("utf-16-be")
        luni = struct.pack(">I", len(name)) + uname
        luni += b"\0" * ((-len(luni)) % 4)
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + pascal(name)
        extra += b"8BIMluni" + struct.pack(">I", len(luni)) + luni
        rec += b"8BIMnorm" + struct.pack(">BBBB", 255, 0, 0, 0) + struct.pack(">I", len(extra)) + extra
        records += rec
    layer_info = struct.pack(">h", len(layers)) + records + channel_data
    if len(layer_info) % 2:
        layer_info += b"\0"
    layer_info = struct.pack(">I", len(layer_info)) + layer_info
    lmi = layer_info + struct.pack(">I", 0)
    comp = Image.new("RGBA", (W, H))
    for _, rgba in layers:
        comp.alpha_composite(Image.fromarray(rgba))
    carr = np.array(comp)
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, H, W, 8, 3)
    body = struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", len(lmi)) + lmi
    packed = [rle(carr[..., c]) for c in (0, 1, 2, 3)]
    img = struct.pack(">H", 1) + b"".join(p[:2 * H] for p in packed) + b"".join(p[2 * H:] for p in packed)
    with open(path, "wb") as f:
        f.write(header + body + img)
    return comp


if __name__ == "__main__":
    comp = write_psd(OUT_PSD, LAYERS)
    print("layers:", len(LAYERS))
