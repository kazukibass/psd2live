"""Deformation harness: numeric pass/fail checks on a rig's geometry at every head pose.

Input: the JSON written by `psd2live-headless --export-geometry` (canvas pixels, y down).
Usage: python3 harness.py <geometry.json> [--report report.json]

Checks (per pose, against the rest pose):
  1. triangle distortion  - every mesh triangle's rest->pose linear map, as singular values s1 >= s2:
                            anisotropy s1/s2 (stretch in one direction vs the other), area ratio s1*s2,
                            and flips (det < 0). Rubber-sheet distortion shows up here first.
  2. part rigidity        - each part is fitted with the best affine map (which already allows turning,
                            foreshortening and shear); what is left over is non-affine bending, as a share
                            of the part's size. A head that turns stays close to affine per part.
  3. eyes                 - eye-white aspect ratio change, and the size ratio between the two eyes.
  4. layer separation     - where two head layers show the same drawing at rest (back hair under front
                            hair, ears under hair), their vertices must move together; a gap between them
                            shows up as a doubled outline.
The limits are in LIMITS below; any violation fails the pose.
"""
import json
import sys

import numpy as np

FACE_PARTS = ("ArtMeshFace", "ArtMeshNose", "ArtMeshMouth", "ArtMeshEyewhiteL", "ArtMeshEyewhiteR",
              "ArtMeshEyelashL", "ArtMeshEyelashR", "ArtMeshIridesL", "ArtMeshIridesR",
              "ArtMeshEyebrowL", "ArtMeshEyebrowR")
HAIR_PARTS = ("ArtMeshFrontHair", "ArtMeshBackHair", "ArtMeshEarsL", "ArtMeshEarsR")

LIMITS = {
    # p95 over the part's triangles (small slivers at mesh edges are ignored by using p95, but a hard max
    # still catches single collapsed triangles)
    "face_aniso_p95": 1.45, "face_aniso_max": 2.2,
    "hair_aniso_p95": 1.6, "hair_aniso_max": 3.0,
    "area_ratio_min": 0.55, "area_ratio_max": 1.6,
    "flips": 0,
    # non-affine residual, RMS as a share of the part's diagonal
    "face_nonaffine": 0.025, "hair_nonaffine": 0.05,
    # eyes
    "eye_aspect_change": 0.30,     # |aspect_pose / aspect_rest - 1|
    "eye_size_ratio_min": 0.55,    # smaller eye area / larger eye area
    "separation_p95": 12.0,        # px, back hair vs front hair drift where they overlap at rest
}
PAIRS = [("ArtMeshBackHair", "ArtMeshFrontHair"), ("ArtMeshEarsL", "ArtMeshFrontHair"), ("ArtMeshEarsR", "ArtMeshFrontHair")]


def separation(a_rest, a_pose, b_rest, b_pose, radius=6.0):
    """p95 drift between vertices of layer a and the nearest vertex of layer b within `radius` at rest."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(b_rest).query(a_rest, distance_upper_bound=radius)
    ok = np.isfinite(dist)
    if not ok.any():
        return 0.0
    drift = np.linalg.norm((a_pose[ok] - a_rest[ok]) - (b_pose[idx[ok]] - b_rest[idx[ok]]), axis=1)
    return float(np.percentile(drift, 95))


def tri_metrics(rest, pose, tris):
    a, b, c = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    A, B, C = pose[tris[:, 0]], pose[tris[:, 1]], pose[tris[:, 2]]
    R = np.stack([b - a, c - a], axis=2)          # 2x2 per triangle (columns = edges)
    P = np.stack([B - A, C - A], axis=2)
    det_r = np.linalg.det(R)
    ok = np.abs(det_r) / 2 >= 4.0                  # skip slivers under 4 px^2: too small to measure
    M = P[ok] @ np.linalg.inv(R[ok])
    s = np.linalg.svd(M, compute_uv=False)
    det = np.linalg.det(M)
    return s[:, 0] / np.maximum(s[:, 1], 1e-6), s[:, 0] * s[:, 1], int((det < 0).sum())


def nonaffine(rest, pose):
    X = np.column_stack([rest, np.ones(len(rest))])
    coef, *_ = np.linalg.lstsq(X, pose, rcond=None)
    resid = pose - X @ coef
    diag = np.linalg.norm(rest.max(0) - rest.min(0))
    return float(np.sqrt((resid ** 2).sum(1).mean()) / max(diag, 1e-6))


def eye_stats(pts):
    w, h = pts.max(0) - pts.min(0)
    return h / max(w, 1e-6), w * h


def evaluate(geo):
    poses = [p["name"] for p in geo["poses"]]
    parts = {d["id"]: d for d in geo["drawables"]}
    report = {}
    for pose in poses:
        if pose == "rest":
            continue
        fails, rows = [], {}
        for pid in FACE_PARTS + HAIR_PARTS:
            d = parts.get(pid)
            if d is None or d["world"].get(pose) is None:
                continue
            rest = np.array(d["world"]["rest"]).reshape(-1, 2)
            cur = np.array(d["world"][pose]).reshape(-1, 2)
            tris = np.array(d["indices"]).reshape(-1, 3)
            aniso, area, flips = tri_metrics(rest, cur, tris)
            if len(aniso) == 0:
                continue
            na = nonaffine(rest, cur)
            face = pid in FACE_PARTS
            row = dict(aniso_p95=round(float(np.percentile(aniso, 95)), 3), aniso_max=round(float(aniso.max()), 3),
                       area_p5=round(float(np.percentile(area, 5)), 3), area_p95=round(float(np.percentile(area, 95)), 3),
                       flips=flips, nonaffine=round(na, 4))
            rows[pid] = row
            kind = "face" if face else "hair"
            if row["aniso_p95"] > LIMITS[f"{kind}_aniso_p95"]:
                fails.append(f"{pid}: stretch p95 {row['aniso_p95']} > {LIMITS[f'{kind}_aniso_p95']}")
            if row["aniso_max"] > LIMITS[f"{kind}_aniso_max"]:
                fails.append(f"{pid}: stretch max {row['aniso_max']} > {LIMITS[f'{kind}_aniso_max']}")
            if row["area_p5"] < LIMITS["area_ratio_min"] or row["area_p95"] > LIMITS["area_ratio_max"]:
                fails.append(f"{pid}: area ratio {row['area_p5']}..{row['area_p95']}")
            if flips > LIMITS["flips"]:
                fails.append(f"{pid}: {flips} flipped triangles")
            if na > LIMITS[f"{kind}_nonaffine"]:
                fails.append(f"{pid}: bending {na:.3f} > {LIMITS[f'{kind}_nonaffine']}")
        eyes = {}
        for side in ("L", "R"):
            d = parts.get(f"ArtMeshEyewhite{side}")
            if d is None:
                continue
            r = eye_stats(np.array(d["world"]["rest"]).reshape(-1, 2))
            c = eye_stats(np.array(d["world"][pose]).reshape(-1, 2))
            eyes[side] = (r, c)
            change = abs(c[0] / r[0] - 1)
            if change > LIMITS["eye_aspect_change"]:
                fails.append(f"eye {side}: aspect changed {change:.0%}")
        if len(eyes) == 2:
            a, b = eyes["L"][1][1], eyes["R"][1][1]
            ratio = min(a, b) / max(a, b)
            rows["eye_size_ratio"] = round(ratio, 3)
            if ratio < LIMITS["eye_size_ratio_min"]:
                fails.append(f"eyes: size ratio {ratio:.2f} < {LIMITS['eye_size_ratio_min']}")
        for a, b in PAIRS:
            if a in parts and b in parts:
                arr = lambda d, k: np.array(d["world"][k]).reshape(-1, 2)
                sep = separation(arr(parts[a], "rest"), arr(parts[a], pose), arr(parts[b], "rest"), arr(parts[b], pose))
                rows[f"separation {a}/{b}"] = round(sep, 1)
                if sep > LIMITS["separation_p95"]:
                    fails.append(f"{a} drifts {sep:.0f}px from {b}")
        report[pose] = dict(ok=not fails, fails=fails, parts=rows)
    return report


if __name__ == "__main__":
    geo = json.load(open(sys.argv[1]))
    rep = evaluate(geo)
    if "--report" in sys.argv:
        json.dump(rep, open(sys.argv[sys.argv.index("--report") + 1], "w"), indent=1)
    bad = 0
    for pose, r in rep.items():
        status = "PASS" if r["ok"] else "FAIL"
        bad += not r["ok"]
        print(f"{pose:12s} {status}  " + ("; ".join(r["fails"][:4]) + (" ..." if len(r["fails"]) > 4 else "")))
    print(f"{len(rep) - bad}/{len(rep)} poses pass")
    sys.exit(1 if bad else 0)
