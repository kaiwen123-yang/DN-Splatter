"""Review published Gemini 335 numeric samples without opening the original bag.

This is a deterministic same-pair projection diagnostic, not a pose estimator,
calibration optimizer, dense hole filler or DN training-data exporter.
Dependencies: numpy, Pillow, matplotlib. Array pixel centers are integers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image


def quantiles(values):
    a = np.asarray(values)
    return dict(zip(("p50", "p95", "p99", "max"),
                    np.percentile(a, (50, 95, 99, 100)).tolist())) if a.size else None


def brown(x, y, d):
    k1, k2, k3, k4, k5, k6, p1, p2 = d
    r2 = x*x + y*y
    radial = (1 + k1*r2 + k2*r2**2 + k3*r2**3) / (1 + k4*r2 + k5*r2**2 + k6*r2**3)
    return (x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
            y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y)


def scaled_camera(camera, scale):
    return {
        "width": int(camera["width"]*scale),
        "height": int(camera["height"]*scale),
        "fx": camera["fx"]*scale,
        "fy": camera["fy"]*scale,
        "cx": (camera["cx"] + 0.5)*scale - 0.5,
        "cy": (camera["cy"] + 0.5)*scale - 0.5,
    }


def project(points, camera, distortion=None):
    x, y = points[:, 0]/points[:, 2], points[:, 1]/points[:, 2]
    if distortion is not None:
        x, y = brown(x, y, distortion)
    return np.column_stack((camera["fx"]*x + camera["cx"],
                            camera["fy"]*y + camera["cy"]))


def zbuffer(uv, z, camera):
    xy = np.rint(uv).astype(np.int64)
    inside = ((z > 0) & np.isfinite(z)
              & (xy[:, 0] >= 0) & (xy[:, 0] < camera["width"])
              & (xy[:, 1] >= 0) & (xy[:, 1] < camera["height"]))
    result = np.full(camera["width"]*camera["height"], np.inf, dtype=np.float64)
    flat = xy[inside, 1]*camera["width"] + xy[inside, 0]
    np.minimum.at(result, flat, z[inside])
    result[~np.isfinite(result)] = 0
    return result.reshape(camera["height"], camera["width"]), inside


def remap_rgb(rgb, original, target, distortion):
    vv, uu = np.indices((target["height"], target["width"]), dtype=np.float64)
    x = (uu-target["cx"])/target["fx"]
    y = (vv-target["cy"])/target["fy"]
    xd, yd = brown(x, y, distortion)
    u, v = original["fx"]*xd+original["cx"], original["fy"]*yd+original["cy"]
    good = (u >= 0) & (v >= 0) & (u <= rgb.shape[1]-1) & (v <= rgb.shape[0]-1)
    u = np.clip(u, 0, rgb.shape[1]-1)
    v = np.clip(v, 0, rgb.shape[0]-1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0+1, rgb.shape[1]-1), np.minimum(v0+1, rgb.shape[0]-1)
    du, dv = (u-u0)[..., None], (v-v0)[..., None]
    out = ((1-du)*(1-dv)*rgb[v0, u0] + du*(1-dv)*rgb[v0, u1]
           + (1-du)*dv*rgb[v1, u0] + du*dv*rgb[v1, u1])
    out[~good] = 0
    return np.rint(out).astype(np.uint8), good


def near_discontinuity(raw_m, valid):
    # Mark the nearer side only. This is an overlay aid, not a reference edge.
    edge = np.zeros(raw_m.shape, dtype=bool)
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        za, zb = raw_m[a], raw_m[b]
        changed = (valid[a] & valid[b]
                   & (np.abs(za-zb) > np.maximum(0.05, 0.03*np.minimum(za, zb))))
        edge[a] |= changed & (za < zb)
        edge[b] |= changed & (zb < za)
    return edge & (raw_m >= 0.3) & (raw_m <= 6.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--source-commit", default="a13bb40445c1c049c2931094e533a5ed43472297")
    args = ap.parse_args()
    source, output = args.audit_dir.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("Keep review outputs separate from the immutable input audit")
    output.mkdir(parents=True, exist_ok=True)
    cal = json.loads((source/"calibration.json").read_text(encoding="utf-8"))
    meta = json.loads((source/"samples_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((source/"summary.json").read_text(encoding="utf-8"))
    color, depth = cal["cameras"]["RGB"], cal["cameras"]["depth"]
    d = color["D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"]
    assert color["distortion_model_enum"] == 4
    assert not np.any(depth["D_sdk_k1_k2_k3_k4_k5_k6_p1_p2"])
    ext = next(e for e in cal["static_extrinsics"]
               if e["source_sensor"] == "depth" and e["destination_sensor"] == "color")
    rotation = np.array(ext["R"], dtype=np.float64)
    trans = np.array(ext["translation_mm"], dtype=np.float64)/1000
    target = scaled_camera(color, 0.5)
    chosen = [s for s in meta["samples"] if s.get("depth_file") and s.get("rgb_file")]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(18, 3.8*len(chosen)), layout="constrained")
    rows, all_values, inputs = [], [], []
    for filename in ("calibration.json", "samples_metadata.json", "summary.json"):
        inputs.append({"path": filename, "sha256": hashlib.sha256((source/filename).read_bytes()).hexdigest()})
    for row, sample in enumerate(chosen):
        rgb_path, dep_path = source/sample["rgb_file"], source/sample["depth_file"]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        with np.load(dep_path, allow_pickle=False) as archive:
            raw = archive["depth"].copy()
        assert raw.shape == (depth["height"], depth["width"]) and raw.dtype == np.uint16
        assert hashlib.sha256(raw.tobytes()).hexdigest() == sample["depth_pixels_sha256"]
        assert hashlib.sha256(rgb_path.read_bytes()).hexdigest() == sample["rgb_payload_sha256"]
        scale = sample["metres_per_depth_unit"]
        assert scale == 0.001
        valid = (raw != 0) & (raw != 65535)
        vd, ud = np.nonzero(valid)
        zd = raw[valid].astype(np.float64)*scale
        pd = np.column_stack(((ud-depth["cx"])/depth["fx"]*zd,
                               (vd-depth["cy"])/depth["fy"]*zd, zd))
        pc = pd @ rotation.T + trans
        uv_native = project(pc, color, d)
        uv_pinhole = project(pc, color)
        uv_small = project(pc, target)
        native_z, inside = zbuffer(uv_native, pc[:, 2], color)
        small_z, inside_small = zbuffer(uv_small, pc[:, 2], target)
        rect_rgb, remap_valid = remap_rgb(rgb, color, target, d)
        small_z[~remap_valid] = 0
        naive = np.column_stack(((ud+0.5)*color["width"]/depth["width"]-0.5,
                                  (vd+0.5)*color["height"]/depth["height"]-0.5))
        edge = near_discontinuity(raw.astype(np.float64)*scale, valid)[valid]
        eligible = inside & edge
        before_edge = naive[eligible]
        correct_edge = uv_native[eligible]
        # Limit only visualization density; statistics below use all valid points.
        stride = max(1, len(correct_edge)//18000)
        before_edge, correct_edge = before_edge[::stride], correct_edge[::stride]
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Pair {sample['id']:02d} | t={sample['time_rel_first_depth_s']:.3f}s | original RGB")
        axes[row, 1].imshow(rgb)
        axes[row, 1].scatter(before_edge[:, 0], before_edge[:, 1], s=0.55,
                             c="#ff40ce", alpha=0.55, linewidths=0, label="resize-only")
        axes[row, 1].scatter(correct_edge[:, 0], correct_edge[:, 1], s=0.55,
                             c="#00ff8c", alpha=0.8, linewidths=0, label="calibrated D2C")
        axes[row, 1].set_title("Near depth edges: magenta resize / green calibrated")
        axes[row, 2].imshow(rect_rgb)
        near_z = np.ma.masked_where((small_z < 0.3) | (small_z > 6.0), small_z)
        im = axes[row, 2].imshow(near_z, cmap="turbo", vmin=0.3, vmax=6.0, alpha=0.55)
        axes[row, 2].set_title("640x360 rectified grid | projected Zc, 0.3-6 m")
        for ax in axes[row]:
            ax.set_axis_off()
        vals = raw[valid].astype(np.float64)*scale
        all_values.append(vals)
        bins = [0, 0.3, 3, 6, 10, 20, 30, 65.535]
        counts, _ = np.histogram(vals, bins=bins)
        rows.append({
            "sample_id": sample["id"], "relative_time_s": sample["time_rel_first_depth_s"],
            "depth_minus_rgb_ms": sample["depth_minus_rgb_ns"]/1e6,
            "native_depth_pixels": int(raw.size), "non_sentinel_pixels": int(valid.sum()),
            "non_sentinel_fraction": float(valid.mean()),
            "native_depth_bins_m": bins,
            "native_depth_bin_counts_among_non_sentinel": counts.tolist(),
            "fraction_of_non_sentinel_in_0p3_3m": float(((vals >= 0.3) & (vals <= 3)).mean()),
            "fraction_of_non_sentinel_in_0p3_6m": float(((vals >= 0.3) & (vals <= 6)).mean()),
            "fraction_of_non_sentinel_in_0p3_10m": float(((vals >= 0.3) & (vals <= 10)).mean()),
            "fraction_of_non_sentinel_above_20m": float((vals > 20).mean()),
            "native_depth_quantiles_m": quantiles(vals),
            "projected_points_in_native_rgb_fraction": float(inside.mean()),
            "native_rgb_single_point_zbuffer_coverage": float((native_z > 0).mean()),
            "rectified_640_single_point_zbuffer_coverage": float((small_z > 0).mean()),
            "rectified_640_0p3_6m_fraction_of_all_pixels": float(((small_z >= 0.3) & (small_z <= 6)).mean()),
            "rectified_rgb_valid_mapping_fraction": float(remap_valid.mean()),
            "resize_only_vs_calibrated_displacement_native_px": quantiles(np.linalg.norm(naive[inside]-uv_native[inside], axis=1)),
            "ignore_distortion_displacement_native_px": quantiles(np.linalg.norm(uv_pinhole[inside]-uv_native[inside], axis=1)),
            "abs_color_z_minus_depth_z_m": quantiles(np.abs(pc[inside, 2]-zd[inside])),
            "sample_depth_pixels_sha256_verified": True, "original_jpeg_sha256_verified": True,
        })
        for path in (rgb_path, dep_path):
            inputs.append({"path": path.relative_to(source).as_posix(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.colorbar(im, ax=axes[:, 2].tolist(), shrink=0.6, label="Projected color-axis Z (m); only 0.3-6 m shown")
    fig.suptitle("Gemini 335 same-pair projection review | Numeric samples only | No world poses or depth ground truth", fontsize=14)
    preview = output/"sample_projection_review.jpg"
    fig.savefig(preview, dpi=135, pil_kwargs={"quality": 90})
    plt.close(fig)
    pooled = np.concatenate(all_values)
    report = {
        "schema_version": 1, "status": "executed_on_four_published_samples",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "Pillow": PIL.__version__, "matplotlib": matplotlib.__version__},
        "source_commit": args.source_commit,
        "source_audit_run": summary["run_id"], "source_bag_sha256": summary["source"]["sha256"],
        "original_bag_opened": False, "world_camera_poses_estimated": False, "training_started": False,
        "numeric_sample_count": len(chosen), "inputs": inputs,
        "method": {
            "projection": "Integer array pixel centers; depth Z in m; p_color=R@p_depth+t_m; explicit rational Brown for native RGB; pinhole for rectified grid.",
            "target_rectified_grid_cv": target,
            "target_rectified_grid_colmap_nerfstudio": {**target, "cx": target["cx"]+0.5, "cy": target["cy"]+0.5},
            "zbuffer": "Round each projected point to one nearest target pixel and keep minimum positive color-axis Z. No footprint filling, interpolation or invented depth.",
            "rgb_rectification": "Bilinear inverse remapping with the recorded full Brown model; fixed 640x360 target K; invalid RGB mappings also invalidate projected depth there.",
            "depth_visualization": "Only 0.3-6 m displayed in overlay, not a validated sensor range or deletion of other source depths.",
            "edge_visualization": "Nearer valid side of raw-neighbor depth change > max(0.05 m, 3% of nearer depth), restricted to 0.3-6 m.",
        },
        "pooled_four_sample_non_sentinel_quantiles_m": quantiles(pooled),
        "rows": rows,
        "limits": [
            "Numeric code presence is not depth accuracy; no independent range or calibration ground truth.",
            "Projection displacement compares algorithms, not observed calibration error.",
            "Four samples are not a continuous sequence and cannot establish camera trajectory, drift or full-recording quality.",
            "Single-point z-buffer leaves sampling holes, especially at the larger RGB resolution; no complete training data set was produced.",
            "Edge overlays are qualitative diagnostics, not measured ground-truth edge residuals."
        ],
        "preview": preview.name,
    }
    (output/"numeric_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": len(chosen),
                      "rows": [{k: r[k] for k in ("sample_id", "native_rgb_single_point_zbuffer_coverage",
                                                 "rectified_640_single_point_zbuffer_coverage",
                                                 "resize_only_vs_calibrated_displacement_native_px",
                                                 "fraction_of_non_sentinel_in_0p3_6m",
                                                 "fraction_of_non_sentinel_above_20m")} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
