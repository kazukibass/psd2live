"""Head turns from a 3D proxy: every head vertex gets a depth, the head is rotated, and projected back.

The previous approach pushed ~20 landmarks independently and interpolated between them, which bent the
face like rubber. Here the only free quantities are a handful of shape and angle parameters, so every
pose is (up to foreshortening) a rigid turn of one smooth head surface and cannot bend the face locally.

  depth z(x, y)   a smooth dome over the head: rz * exp(-k * r^2), r = ellipse radius around (cx, cy)
                  + part offsets: nose tip forward, front hair a shell just in front of the skull,
                  back hair and ears behind it
  yaw             rotation about the vertical axis through cx          (ParamAngleX = +-45 key)
  pitch           rotation about a horizontal axis at the neck pivot   (ParamAngleY = +30 / -30 keys)
  projection      orthographic

The parameters are fitted by least squares to the landmark offsets measured on the reference sheet
(tools/angle_targets.py). Diagonal keys are the yaw followed by the pitch - no hand-tuned mixing.

Usage: python3 head3d.py <rest-geometry.json> <out-displacements.json> [--report fit.json]
"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares

import angle_targets as T

HEAD_ROOT = "DeformHeadContainer"
KEYS = [(ax, ay) for ay in (-30, 0, 30) for ax in (-45, 0, 45) if (ax, ay) != (0, 0)]

# which depth rule each drawable follows
FACE_SURFACE = {"ArtMeshFace", "ArtMeshEyewhiteL", "ArtMeshEyewhiteR", "ArtMeshIridesL", "ArtMeshIridesR",
                "ArtMeshEyelashL", "ArtMeshEyelashR", "ArtMeshEyebrowL", "ArtMeshEyebrowR",
                "ArtMeshMouth", "ArtMeshMouth_lip_0", "ArtMeshMouth_lip_1"}
NOSE = {"ArtMeshNose"}
FRONT_HAIR = {"ArtMeshFrontHair"}
BACK_HAIR = {"ArtMeshBackHair"}
EARS = {"ArtMeshEarsL", "ArtMeshEarsR"}

# parameter vector: name, initial value, lower, upper
PARAMS = [
    ("cx", 900, 860, 940), ("cy", 470, 380, 560),
    ("rx", 330, 220, 480), ("ry", 380, 260, 560), ("rz", 330, 150, 520), ("k", 0.9, 0.05, 2.5),
    ("nose", 40, 0, 120),          # nose tip in front of the surface (px)
    ("hair", 25, 0, 80),           # front-hair shell in front of the surface
    ("back", 4, 0, 8),             # back hair: small step behind the front-hair shell
    ("ear", 10, 0, 20),            # ears: small step behind the hair shell
    ("py", 800, 650, 950), ("pz", -120, -350, 100),   # pitch pivot (neck): canvas y, depth
    ("yaw", 22, 5, 45), ("up", 22, 5, 50), ("down", 20, 5, 50),   # degrees at the keys
]
NAMES = [p[0] for p in PARAMS]


def depth(P, x, y, kind):
    r2 = ((x - P["cx"]) / P["rx"]) ** 2 + ((y - P["cy"]) / P["ry"]) ** 2
    z = P["rz"] * np.exp(-P["k"] * r2)
    if kind == "nose":
        z = z + P["nose"]
    elif kind == "front":
        z = z + P["hair"]
    elif kind == "back":
        # the back-hair layer repeats the visible hair (plus the fill behind the face), so it has to sit in
        # the same shell as the front hair; only a small step back is allowed
        z = z + P["hair"] - P["back"]
    elif kind == "ear":
        # ears grow out of the hair: same shell as the hair, only a small step back, or they come loose
        z = z + P["hair"] - P["ear"]
    return z


def project(P, x, y, z, ax, ay):
    yaw = np.radians(P["yaw"] * ax / 45.0)
    pitch = np.radians((P["up"] if ay > 0 else P["down"]) * abs(ay) / 30.0) * (1 if ay > 0 else -1)
    dx = x - P["cx"]
    x1 = P["cx"] + dx * np.cos(yaw) + z * np.sin(yaw)
    z1 = -dx * np.sin(yaw) + z * np.cos(yaw)
    dy = y - P["py"]
    dz = z1 - P["pz"]
    y2 = P["py"] + dy * np.cos(pitch) - dz * np.sin(pitch)   # looking up moves front points up
    return x1, y2


LANDMARK_KIND = {"nose": "nose", "ear_r": "ear", "ear_l": "ear", "ear_base_r": "ear", "ear_base_l": "ear",
                 "hair_r": "back", "hair_l": "back", "hair_tip_r": "front", "hair_tip_l": "front", "top": "front"}


def targets():
    """(landmark, key, observed offset in canvas px) for the four straight directions."""
    out = []
    for key, table in (((45, 0), T.TURN_RIGHT), ((-45, 0), T.turn_left()), ((0, 30), T.LOOK_UP), ((0, -30), T.LOOK_DOWN)):
        for name, (dx, dy) in table.items():
            out.append((name, key, np.array([dx, dy]) * T.SCALE))
    return out


def residuals(v, obs):
    P = dict(zip(NAMES, v))
    res = []
    for name, (ax, ay), off in obs:
        x, y = T.LANDMARKS[name]
        z = depth(P, x, y, LANDMARK_KIND.get(name, "face"))
        px, py = project(P, x, y, z, ax, ay)
        # ears and hair outlines are the least certain measurements: half weight
        w = 0.5 if LANDMARK_KIND.get(name) in ("ear", "back", "front") else 1.0
        res += [w * (px - x - off[0]), w * (py - y - off[1])]
    return np.array(res)


# Harness limits enforced inside the fit (a little tighter than tools/harness.py, for margin)
FIT_ANISO = 1.38        # per-triangle stretch s1/s2, face-surface parts
FIT_HAIR_ANISO = 1.52   # same for hair and ears
FIT_EYE_ASPECT = 0.26   # |eye aspect ratio change|
PENALTY = 400.0         # px of landmark error that one unit of limit violation costs


def mesh_constraints(geo):
    """Rest triangles of the face-surface parts and the eye-white vertices, for the in-fit harness."""
    tris, eyes = [], []
    for d in geo["drawables"]:
        rest = np.array(d["world"]["rest"]).reshape(-1, 2)
        if d["id"] in FACE_SURFACE | NOSE | FRONT_HAIR | BACK_HAIR | EARS and not d["id"].startswith("ArtMeshMouth"):
            t = np.array(d["indices"]).reshape(-1, 3)
            a, b, c = rest[t[:, 0]], rest[t[:, 1]], rest[t[:, 2]]
            area = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) / 2
            t = t[area >= 4.0]
            tris.append((rest, t, kind_of(d["id"])))
        if d["id"].startswith("ArtMeshEyewhite"):
            eyes.append(rest)
    return tris, eyes


def violations(P, cons):
    tris, eyes = cons
    out = []
    for ax, ay in KEYS:
        worst = 0.0
        worst_hair = 0.0
        for rest, t, kind in tris:
            z = depth(P, rest[:, 0], rest[:, 1], kind)
            px, py = project(P, rest[:, 0], rest[:, 1], z, ax, ay)
            cur = np.column_stack([px, py])
            R = np.stack([rest[t[:, 1]] - rest[t[:, 0]], rest[t[:, 2]] - rest[t[:, 0]]], axis=2)
            Q = np.stack([cur[t[:, 1]] - cur[t[:, 0]], cur[t[:, 2]] - cur[t[:, 0]]], axis=2)
            sv = np.linalg.svd(Q @ np.linalg.inv(R), compute_uv=False)
            an = sv[:, 0] / np.maximum(sv[:, 1], 1e-6)
            if kind in ("front", "back", "ear"):
                worst_hair = max(worst_hair, float(np.percentile(an, 96)))
            else:
                worst = max(worst, float(np.percentile(an, 98)))
        out.append(max(0.0, worst - FIT_ANISO))
        out.append(max(0.0, worst_hair - FIT_HAIR_ANISO))
        for rest in eyes:
            z = depth(P, rest[:, 0], rest[:, 1], "face")
            px, py = project(P, rest[:, 0], rest[:, 1], z, ax, ay)
            w0, h0 = rest.max(0) - rest.min(0)
            w1, h1 = px.max() - px.min(), py.max() - py.min()
            out.append(max(0.0, abs((h1 / w1) / (h0 / w0) - 1) - FIT_EYE_ASPECT))
    return np.array(out)


def fit(geo):
    obs = targets()
    cons = mesh_constraints(geo)
    x0 = np.array([p[1] for p in PARAMS], float)
    lo = np.array([p[2] for p in PARAMS], float)
    hi = np.array([p[3] for p in PARAMS], float)

    def total(v):
        return np.concatenate([residuals(v, obs), PENALTY * violations(dict(zip(NAMES, v)), cons)])

    r = least_squares(total, x0, bounds=(lo, hi), diff_step=1e-3)
    print("harness violations after fit:", np.round(violations(dict(zip(NAMES, r.x)), cons), 3).tolist())
    r.fun = residuals(r.x, obs)
    P = dict(zip(NAMES, r.x))
    per = np.abs(residuals(r.x, obs)).reshape(-1, 2)
    detail = [dict(landmark=n, key=list(k), observed=[round(float(o[0]), 1), round(float(o[1]), 1)],
                   error=[round(float(e[0]), 1), round(float(e[1]), 1)]) for (n, k, o), e in zip(obs, per)]
    return P, float(np.sqrt((r.fun ** 2).mean())), detail


def kind_of(did):
    if did in NOSE:
        return "nose"
    if did in FRONT_HAIR:
        return "front"
    if did in BACK_HAIR:
        return "back"
    if did in EARS:
        return "ear"
    return "face"


if __name__ == "__main__":
    geo = json.load(open(sys.argv[1]))
    P, rms, detail = fit(geo)
    print("fit rms %.1f px" % rms)
    print("  " + ", ".join(f"{k}={v:.1f}" for k, v in P.items()))
    out = {"params": P, "poses": []}
    for ax, ay in KEYS:
        drawables = {}
        for d in geo["drawables"]:
            if HEAD_ROOT not in d["chain"]:
                continue
            rest = np.array(d["world"]["rest"]).reshape(-1, 2)
            z = depth(P, rest[:, 0], rest[:, 1], kind_of(d["id"]))
            px, py = project(P, rest[:, 0], rest[:, 1], z, ax, ay)
            disp = np.column_stack([px - rest[:, 0], py - rest[:, 1]])
            drawables[d["id"]] = np.round(disp.ravel(), 2).tolist()
        out["poses"].append({"angleX": ax, "angleY": ay, "drawables": drawables})
    json.dump(out, open(sys.argv[2], "w"))
    if "--report" in sys.argv:
        json.dump({"params": P, "rms": rms, "landmarks": detail},
                  open(sys.argv[sys.argv.index("--report") + 1], "w"), indent=1)
