"""Per-vertex head-turn displacements: face parts from the fitted 3D head, everything else unchanged.

Face parts (face base, eyes, brows, nose, mouth) are moved by the human-like 3D head of
tools/fit3d_head.py: ParamAngleX -45 / +45 use its left / right yaw, ParamAngleY +30 / -30 its up /
down pitch, and the corners are the yaw followed by the pitch. No sheet alignment shift is applied (the
neck stays where it is). All other head drawables (hair, ears) keep psd2live's own motion: their
displacement is read from the standard rig at the same key.

Usage: python3 export_face3d.py <standard-rig-geometry.json> <head3d_fit.json> <out.json>
The output is the input of `psd2live-headless --retarget-vertices`.
"""
import json
import sys

import numpy as np

import fit3d_head as HH

FACE_PARTS = {"ArtMeshFace", "ArtMeshNose", "ArtMeshMouth", "ArtMeshMouth_lip_0", "ArtMeshMouth_lip_1",
              "ArtMeshEyewhiteL", "ArtMeshEyewhiteR", "ArtMeshIridesL", "ArtMeshIridesR",
              "ArtMeshEyelashL", "ArtMeshEyelashR", "ArtMeshEyebrowL", "ArtMeshEyebrowR"}
KEYS = [(ax, ay) for ay in (-30, 0, 30) for ax in (-45, 0, 45) if (ax, ay) != (0, 0)]

if __name__ == "__main__":
    geo = json.load(open(sys.argv[1]))
    fit = json.load(open(sys.argv[2]))
    P, A = fit["shape"], fit["angles"]
    out = {"source": "fit3d_head", "angles": A, "poses": []}
    for ax, ay in KEYS:
        yaw = A["yaw_left"] if ax < 0 else A["yaw_right"] if ax > 0 else 0.0
        pitch = A["pitch_up"] if ay > 0 else A["pitch_down"] if ay < 0 else 0.0
        pose = "x%+d_y%+d" % (ax, ay)
        drawables = {}
        for d in geo["drawables"]:
            if "DeformHeadContainer" not in d["chain"]:
                continue
            rest = np.array(d["world"]["rest"]).reshape(-1, 2)
            if d["id"] in FACE_PARTS:
                cur, _ = HH.project(P, rest, yaw, pitch)
            else:
                cur = np.array(d["world"][pose]).reshape(-1, 2)
            drawables[d["id"]] = np.round((cur - rest).ravel(), 2).tolist()
        out["poses"].append({"angleX": ax, "angleY": ay, "yaw": yaw, "pitch": pitch, "drawables": drawables})
        print(f"X {ax:+d} Y {ay:+d}: yaw {yaw:+.1f} pitch {pitch:+.1f}")
    json.dump(out, open(sys.argv[3], "w"))
