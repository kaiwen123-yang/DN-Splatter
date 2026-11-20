"""Paired held-out model metrics and separate Metric3D sensor diagnostics."""
from pathlib import Path

import numpy as np

from capture_metric3d_common import (B0_REPORT, HOLDOUT, PRIOR, REPORT, RUN, SCENE, demand, read, selection, write)
from capture_full_report import average, matched, temporal
from capture_s01_eval import depth_stats


def paired_row(a, b):
    demand(a["id"] == b["id"] and a["reference_pixels"] == b["reference_pixels"], "paired source")
    delta = {k: b["rgb"][k] - v for k, v in a["rgb"].items()}
    za, zb = a["depth_full_reference"], b["depth_full_reference"]
    for k in ("MAE_m", "RMSE_m", "AbsRel", "absolute_error_P50_m", "absolute_error_P95_m"):
        delta[k] = zb[k] - za[k] if za is not None and zb is not None else None
    return dict(id=a["id"], reference_pixels=a["reference_pixels"], B0=a, B1=b, delta=delta)


def changes(rows):
    counts = {}
    for key in rows[0]["delta"] if rows else ():
        values = [r["delta"][key] for r in rows if r["delta"][key] is not None]
        signed = np.array(values) * (1 if key in ("psnr","ssim") else -1)
        counts[key] = dict(improved=int((signed>0).sum()), degraded=int((signed<0).sum()),
                           equal=int((signed==0).sum()), unavailable=len(rows)-len(values),
                           mean_delta=float(np.mean(values)) if values else None)
    return counts


def normal_angles(stem, ref, K):
    # Native one-pixel differences evaluated on a fixed stride-8 lattice.
    y, x = np.mgrid[1:ref.shape[0]-1:8, 1:ref.shape[1]-1:8]
    fx, fy, cx, cy = K
    def xyz(dy, dx):
        z = ref[y+dy, x+dx]
        return np.stack(((x+dx+.5-cx)*z/fx, (y+dy+.5-cy)*z/fy, z), -1)
    p, l, r, t, b = (xyz(*d) for d in ((0,0),(0,-1),(0,1),(-1,0),(1,0)))
    z = p[...,2]
    keep = np.isfinite(z) & (z>=.3) & (z<=6)
    for neighbor in (l, r, t, b):
        v = neighbor[...,2]
        keep &= np.isfinite(v) & (v>=.3) & (v<=6) & (np.abs(v-z)<=np.maximum(.05,.03*z))
    n = np.cross(r-l,b-t)
    length = np.linalg.norm(n, axis=-1)
    keep &= length>0
    n = n[keep]/length[keep,None]
    n[np.einsum("ij,ij->i",n,p[keep])>0] *= -1
    prior = np.load(PRIOR/"normal_cv"/(stem+".npy"), mmap_mode="r")[y[keep],x[keep]]
    good = np.isfinite(prior).all(axis=1) & (np.linalg.norm(prior,axis=1)>0)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n[good],prior[good]),-1,1)))


def prior_diagnostics(split, rows, K):
    details, diffs, targets, ang = [], [], [], []
    for row in rows:
        stem = row["id"]
        ref = np.load(SCENE/"depth"/(stem+".npy"), mmap_mode="r")
        pred = np.load(PRIOR/"depth_m"/(stem+".npy"), mmap_mode="r")
        support = np.isfinite(ref) & (ref>=.3) & (ref<=6)
        valid = support & np.isfinite(pred) & (pred>0)
        v = ref[valid].astype(float)
        d = np.abs(pred[valid].astype(float)-v)
        angles = normal_angles(stem, ref, K)
        z = depth_stats(d,v)
        diffs.append(d)
        targets.append(v)
        ang.append(angles)
        details.append(dict(id=stem, reference=int(support.sum()), hit=len(d), invalid=int(support.sum())-len(d),
               depth_hit=z, normal_samples=len(angles), normal_median_deg=float(np.median(angles)) if len(angles) else None))
    d, v, angles = np.concatenate(diffs), np.concatenate(targets), np.concatenate(ang)
    nref = sum(r["reference"] for r in details)
    invalid = nref-len(d)
    conditional = depth_stats(d,v)
    result = dict(reference_pixels=nref, finite_positive=len(d), invalid=invalid, coverage=len(d)/nref if nref else None,
        conditional_pooled=conditional, full_reference_pooled=None if invalid else conditional,
        hit_frame_mean=average([r["depth_hit"] for r in details if r["depth_hit"]]),
        full_reference_frame_mean=None if invalid else average([r["depth_hit"] for r in details if r["depth_hit"]]),
        normal_sensor_samples=len(angles), median_angle_deg=float(np.median(angles)) if len(angles) else None)
    write(REPORT/f"prior_diagnostic_{split}.json", dict(summary=result, frames=details, fitted_scale=False,
          normal_reference="Same-capture sensor finite differences, stride 8, four neighbors within tau, normals oriented toward camera."))
    return result


def main():
    seq = selection()
    by_id = {r["id"]:r for r in seq}
    metrics = {s:read(REPORT/f"metrics_{s}.json") for s in HOLDOUT}
    baseline_windows = read(B0_REPORT/"metrics_by_time.json")["windows"]
    meta = read(SCENE/"transforms.json")
    checkpoint = read(RUN/"training_complete.json")["checkpoint_sha256"]
    matched(metrics, seq)
    s01 = read(REPORT/"comparison_s01_same_sources.json")
    summary = {}
    for split in HOLDOUT:
        a, b = read(B0_REPORT/f"metrics_{split}.json"), metrics[split]
        demand(b["checkpoint_sha256"] == checkpoint and a["frame_count"] == b["frame_count"] == 184, split)
        rows = [paired_row(x,y) for x,y in zip(a["frames"],b["frames"],strict=True)]
        for row in rows:
            row["relative_s"] = by_id[row["id"]]["relative_s"]
        indices = {r["id"]:r["B1"] for r in rows}
        windows = []
        for window in baseline_windows:
            lo, hi = window["start_s"], window["end_exclusive_s"]
            group = [r for r in rows if lo<=r["relative_s"]<hi]
            windows.append(dict(start_s=lo,end_exclusive_s=hi, changes=changes(group),
                B0=window["splits"][split], B1=temporal(group,indices,split)))
        old = {r["id"]:r for r in s01[split]["frames"]}
        subset = [dict(**r,s01=old[r["id"]]["s01_rgb"],s01_depth=old[r["id"]]["s01_depth"]) for r in rows if r["id"] in old]
        demand(len(subset)==24, "same s01 heldout set")
        write(REPORT/f"paired_{split}.json",dict(frames=rows, windows=windows, s01_same_sources=subset))
        summary[split] = dict(frames=len(rows), changes=changes(rows), same_source_24=changes(subset),
            prior_diagnostic=prior_diagnostics(split, rows, [meta[k] for k in ("fl_x","fl_y","cx","cy")]),
            models={label:{k:v[k] for k in ("rgb_frame_means","depth_frame_means","depth_pixel_pooled",
                     "reference_pixels","invalid_prediction_pixels","reference_alpha_zero")} for label,v in (("B0",a),("B1",b))})
    write(REPORT/"paired_summary.json",summary)


if __name__=="__main__":
    main()
