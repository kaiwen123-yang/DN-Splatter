#!/usr/bin/env python3
"""Recompute count accounting and adjacent-pose evidence from published B1 JSON.

Read-only with respect to baseline inputs. Writes a small review JSON and figure.
This does not run training, recover poses, or assess absolute ground-truth error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.repo / "reports/capture-full-metric3d-30k/20261120-v1"
    output = args.output or args.repo / "reports/capture-quality-review/2026-11-21"
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    event_file = source / "B1_refinement.json"
    pose_file = source / "metric_c2w.json"
    events = json.loads(event_file.read_text())["events"]
    poses = json.loads(pose_file.read_text())["poses"]["full"]
    for e in events:
        assert e["before"] + e["split_children"] + e["duplicates"] - e["cull_union"] == e["after"], e
    names = sorted(poses, key=lambda name: int(name.split("_")[-1]))
    assert names == [f"frame_{i:05d}" for i in range(1846)]
    matrices = np.asarray([poses[name] for name in names], dtype=np.float64)
    rotations = matrices[:, :3, :3]
    relative = np.swapaxes(rotations[:-1], 1, 2) @ rotations[1:]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    translations = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=1)
    first = next(e for e in events if e["step"] == 1600)
    before_stop = next(e for e in events if e["step"] == 14900)
    final_count = events[-1]["after"]
    result = {
        "source_commit": "3728dae350a240d317702d57e560554d529679c6",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [event_file, pose_file]},
        "refinement_events": len(events),
        "all_count_identities_pass": True,
        "first_major_cull": first,
        "first_net_decrease_fraction": (first["before"] - first["after"]) / first["before"],
        "step14900_count": before_stop["after"],
        "final_refinement_count": final_count,
        "post_growth_decrease_fraction": (before_stop["after"] - final_count) / before_stop["after"],
        "largest_rotation_edges": [
            {"from": names[i], "to": names[i + 1], "rotation_deg": float(angles[i]), "translation_m": float(translations[i])}
            for i in np.argsort(angles)[-5:][::-1]
        ],
        "pose_interpretation": "Estimated adjacent poses; visual evidence is separate. Not absolute pose error.",
        "training_interpretation": "Direct event accounting; not isolated causal attribution or a new training run.",
    }
    (output / "recomputed_core.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), layout="constrained")
    ax = axes[0]
    ax.plot([e["step"] for e in events], [e["after"] / 1e6 for e in events], color="#2363ad", linewidth=2)
    ax.axvspan(15000, 30000, color="#f1b66a", alpha=.18)
    ax.annotate("Step 1600: -80.1% net\n2.00M to 0.40M", xy=(1600, .398115), xytext=(4700, 1.45), arrowprops={"arrowstyle": "->", "color": "#333"})
    ax.text(16600, .80, "After 15k: cull only\n0.366M to 0.171M", color="#80420d")
    ax.set(xlabel="Training step (zero-based)", ylabel="Gaussian count (millions)", title="B1: model capacity over training", ylim=(0, 2.15), xlim=(0, 30000))
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    target = np.arange(1289, 1313)
    ax.plot(target, angles[target - 1], marker="o", markersize=4, color="#b74738")
    for i in [1294, 1307]:
        ax.annotate(f"{angles[i-1]:.1f} deg", xy=(i, angles[i-1]), xytext=(i-3, 190), fontsize=10)
    ax.set(xlabel="Target frame index (edge i-1 to i)", ylabel="Estimated relative rotation (degrees)", title="B1 input poses: two abrupt reversals", ylim=(-4, 215))
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "previews/training_pose_evidence.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"output": str(output), "first_drop": result["first_net_decrease_fraction"], "events": len(events)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
