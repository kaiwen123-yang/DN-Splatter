"""Recompute the capture-s01 public-evidence review using only Python's stdlib.

This does not access AutoDL, the bag, RGB-D arrays, or any training process.
It verifies the immutable published evidence and summarizes existing statistics;
it cannot recompute pixel-level residuals from those summaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


SOURCE_COMMIT = "d72f56836028d7a7885f21f02522e4f7cc310429"
RUN = "20261119-s01-5hz-3982cb26"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def summary(values):
    return {"min": min(values), "median": median(values), "max": max(values)}


def verify_evidence(repo, evidence):
    manifest = read(evidence / "artifact_manifest.json")
    verified = []
    for item in manifest["public_in_run_directory"]:
        data = (evidence / item["path"]).read_bytes()
        assert len(data) == item["bytes"], item["path"]
        assert sha(data) == item["sha256"], item["path"]
        verified.append({**item, "match": "exact_bytes"})
    versions = read(evidence / "versions.json")
    scripts = []
    expected = {**versions["files_sha256"], **versions["local_executed_parser_sha256"]}
    for name, expected_sha in expected.items():
        data = (repo / "scripts" / name).read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        if sha(data) == expected_sha:
            match = "exact_bytes"
        elif expected_sha in {sha(normalized), sha(normalized.replace(b"\n", b"\r\n"))}:
            match = "line_endings_only_equivalent"
        else:
            raise ValueError("Published execution script differs: " + name)
        scripts.append({"path": "scripts/" + name, "recorded_sha256": expected_sha,
                        "repository_sha256": sha(data), "match": match})
    return {"public_files": verified, "public_file_count": len(verified),
            "execution_scripts": scripts,
            "execution_scripts_exact": sum(s["match"] == "exact_bytes" for s in scripts),
            "execution_scripts_line_endings_only": sum(s["match"] != "exact_bytes" for s in scripts)}


def build(repo):
    evidence = repo / "reports" / "capture-geometry" / RUN
    provenance = verify_evidence(repo, evidence)
    cross = read(evidence / "cross_view.json")
    scale = read(evidence / "scale_alignment.json")
    dataset = read(evidence / "dataset_manifest.json")
    registration = read(evidence / "registration.json")
    pairs = []
    for p in cross["pairs"]:
        delta = abs(int(p["source"].rsplit("_", 1)[1]) - int(p["target"].rsplit("_", 1)[1]))
        group = "adjacent" if delta == 1 else "about_3_seconds" if delta == 15 else "early_revisit"
        windows = {}
        for key, w in p["windows"].items():
            n = w["overlap"]
            assert 0 < n <= p["shared_pixels"] <= p["target_grid_pixels"]
            counts = ("outside_edge_and_visible", "both_depths_in_window", "projected_beyond_window",
                      "projected_above_20m", "farther_than_target", "nearer_than_target")
            assert all(0 <= w[k] <= n for k in counts)
            windows[key] = {**w, "fractions_of_target_window_overlap": {k: w[k] / n for k in counts},
                            "target_window_overlap_fraction_of_grid": n / p["target_grid_pixels"]}
        pairs.append({**p, "group": group, "index_separation": delta, "windows": windows})
    groups = {}
    for name, count in (("adjacent", 9), ("about_3_seconds", 9), ("early_revisit", 2)):
        selected = [p for p in pairs if p["group"] == name]
        assert len(selected) == count
        w = [p["windows"]["6"] for p in selected]
        groups[name] = {
            "pair_count": len(selected),
            "translation_m": summary([p["translation_m"] for p in selected]),
            "target_window_overlap_count": summary([x["overlap"] for x in w]),
            "conditional_retention_fraction": summary([x["outside_edge_and_visible"] / x["overlap"] for x in w]),
            "pairwise_residual_statistics_m": {
                metric: {q: summary([x[metric][q] for x in w]) for q in ("p50", "p95")}
                for metric in ("unconditional_abs_m", "both_in_window_abs_m", "visible_abs_m")
            },
        }
    special = next(p for p in pairs if (p["source"], p["target"]) == ("frame_00164", "frame_00179"))
    w = special["windows"]["6"]
    negative_bound = max(0.0, -w["unconditional_signed_m"]["min"])
    threshold = w["both_in_window_abs_m"]["p95"]
    assert threshold > negative_bound
    chosen = scale["scale_m_per_sfm_unit"]
    geometry = dataset["geometry"]
    assert not any(geometry[k] for k in ("normal_priors", "training_started", "poisson"))
    coverage640 = [r["target_coverage"] for r in registration["rows"] if r["grid"] == "640"]
    return {
        "schema_version": 1,
        "review_date": "2026-11-19",
        "source_commit": SOURCE_COMMIT,
        "source_directory": evidence.relative_to(repo).as_posix(),
        "decision": "accepted_for_first_training_baseline_with_known_limitations",
        "scope": {
            "executed": "Read published source, images and JSON; verify public byte hashes; recompute ratios and group summaries from reported statistics.",
            "not_executed": ["AutoDL access", "original bag access", "full 241 RGB-D array access",
                             "pixel-level cross-view recomputation", "Omnidata inference", "DN training", "Poisson"],
            "independent_geometry_ground_truth": False,
        },
        "observed_stage_flags": {k: geometry[k] for k in ("normal_priors", "training_started", "poisson")},
        "provenance": {**provenance, "review_script_sha256": sha(Path(__file__).read_bytes())},
        "definitions": {
            "depth": "Colour optical-axis Z in metres, compared in target camera coordinates.",
            "unconditional_abs_m": "All common observed target pixels with measured target Z in [0.3, upper]; projected source Z is unrestricted above zero.",
            "both_in_window_abs_m": "Additionally require projected source Z in target axes in [0.3, upper]; source-camera Z is not range-gated by this test.",
            "visible_abs_m": "Conditional statistic removing target depth-edge dilation, target D2C collision mask, and predicted Z minus measured Z > max(0.05 m, 0.03 * measured Z). This residual-dependent filter is not independently verified visibility.",
            "overlap": "Common observed target pixel, not proof of a corresponding mutually visible surface.",
            "group_summary": "Min, median and max of existing per-pair statistics. Not pooled pixel-level quantiles.",
            "sign": "Positive means projected source is behind the measured target surface.",
        },
        "groups_for_target_window_0_3_to_6_m": groups,
        "pair_164_to_179_positive_tail_inference": {
            "raw_signed_min_m": w["unconditional_signed_m"]["min"],
            "maximum_possible_negative_absolute_residual_m": negative_bound,
            "both_in_window_absolute_p95_m": threshold,
            "all_residuals_at_or_above_this_p95_must_be_positive": True,
            "reason": "The negative-residual bound on the raw superset is below the absolute P95 of the both-in-window subset.",
            "interpretation": "The heaviest approximately 5% tail is source-behind-target. Consistent with occlusion/correspondence contamination, but pose or depth errors can also yield this sign; causation is not identified.",
        },
        "scale_support": {
            "scale_m_per_sfm_unit": chosen,
            "fit_range_m": scale["fit_range_m"],
            "training_fit": scale["windows"]["6"],
            "early_middle_late_scales": [x["scale"] for x in scale["early_middle_late"]],
            "early_middle_late_relative_to_global_percent": [(x["scale"] / chosen - 1) * 100 for x in scale["early_middle_late"]],
            "fit_window_3_to_10_relative_span_percent": (scale["windows"]["10"]["scale"] / scale["windows"]["3"]["scale"] - 1) * 100,
            "disjoint_depth_bands": scale["disjoint_depth_bands"],
            "disjoint_far_band_relative_to_global_percent": (scale["disjoint_depth_bands"]["6-10"]["scale"] / chosen - 1) * 100,
            "fixed_scale_signed_residual_m_by_depth": scale["fixed_scale_signed_residual_m_by_depth"],
            "interpretation": "Internal train-depth/SfM consistency, not external metric accuracy or trajectory ground truth.",
        },
        "dataset_support": {
            "reader_counts": dataset["reader_check"]["counts"],
            "training_input_grid": [1280, 720],
            "depth_range_m": [0.3, 6.0],
            "per_frame_positive_depth_fraction_1280": geometry["depth_positive_fraction"],
            "sampled_untruncated_depth_coverage_640": summary(coverage640),
            "sampled_640_frame_count": len(coverage640),
            "init_points_train_only": geometry["init_points"],
            "dataset_manifest_sha256": dataset["reader_check"]["manifest_sha256"],
        },
        "pairs": pairs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / "reports/capture-geometry-review/2026-11-19/numeric_review.json"
    output = output.resolve()
    evidence = (repo / "reports/capture-geometry" / RUN).resolve()
    if output == evidence or evidence in output.parents:
        raise ValueError("Do not write into the immutable geometry evidence directory")
    result = build(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Reviewed {len(result['pairs'])} pairs; wrote {output}")


if __name__ == "__main__":
    main()
