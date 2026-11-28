#!/usr/bin/env python3
"""Diagnose alignment of published LingBot-Map camera centers using NumPy.

This is a local, read-only numerical reanalysis of published artifacts. It does
not run inference, train a model, update camera poses, or measure GT accuracy.
Neither the 32-center Sim(3) fit nor the all-reference fit is a training-pose
candidate or evidence of an accuracy improvement.

From a repository checkout, use the existing inputs:
  python scripts/diagnose_lingbot_alignment.py --repo-root . \
      --output /tmp/map_alignment_diagnostic.json

For inputs stored elsewhere, use --trajectory and --alignment. Metadata always
records their canonical repository-relative paths, exact byte hashes and Git
blob identities, never local scratch paths. --repo-base, --ref and --input-base
describe provenance; --trajectory-blob and --alignment-blob verify other inputs.

Downloads are opt-in, use the declared ref, and preserve bytes:
  python scripts/diagnose_lingbot_alignment.py --download-missing \
      --output /tmp/map_alignment_diagnostic.json
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import urllib.parse
import urllib.request

import numpy as np

BASELINE_REPOSITORY = "kaiwen123-yang/DN-Splatter"
BASELINE_REF = "ffc157fe30b01796f12e41361c6b2d3ace33521e"
BASELINE_INPUT_BASE = "reports/lingbot-depth-map/20261128-v1/full_train_1478"
BASELINE_BLOBS = {
    "trajectory.json": "d4f3588f720794aede3bdeae3748f00726699e74",
    "alignment.json": "5834fbeb7d9428c10e41156bf068614d276232c4",
}


def repository_name(value):
    value = value.rstrip("/")
    prefix = "https://github.com/"
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.endswith(".git"):
        value = value[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("repo-base must be owner/repo or its GitHub URL")
    return value


def relative_directory(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise argparse.ArgumentTypeError("input-base must be a repository-relative POSIX path")
    return path.as_posix()


def blob_id(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("expected a 40-digit Git blob SHA-1")
    return value.lower()


def byte_identity(content):
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_blob_sha1": hashlib.sha1(header + content).hexdigest(),
    }


def check_blob(content, expected, label):
    identity = byte_identity(content)
    if expected is not None and identity["git_blob_sha1"] != expected:
        raise ValueError(
            f"{label}: exact bytes do not match expected Git blob; "
            f"got {identity['git_blob_sha1']}, expected {expected}. "
            "Do not normalize whitespace or append newlines."
        )
    return identity


def resolve_input(name, explicit, args, source, expected_blob):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_bytes()
    else:
        candidates = [
            Path(args.repo_root) / source["path"],
            Path(args.cache_dir) / name,
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            content = path.read_bytes()
        else:
            if not args.download_missing:
                raise FileNotFoundError(
                    f"{source['path']} is absent. Supply --repo-root or an explicit "
                    "input file; --download-missing is opt-in."
                )
            with urllib.request.urlopen(source["url"], timeout=90) as response:
                content = response.read()
            check_blob(content, expected_blob, name)
            json.loads(content)
            path = Path(args.cache_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    identity = check_blob(content, expected_blob, name)
    source.update(identity)
    source["expected_git_blob_sha1"] = expected_blob
    source["expected_blob_verified"] = expected_blob is not None
    source["hash_basis"] = "exact input bytes, including the original trailing newline"
    return path, json.loads(content)


def distribution(distance):
    distance = np.asarray(distance, dtype=np.float64)
    if len(distance) == 0:
        return None
    return {
        "median": float(np.median(distance)),
        "p95": float(np.percentile(distance, 95)),
        "mean": float(np.mean(distance)),
        "rmse": float(np.sqrt(np.mean(distance * distance))),
    }


def fit_and_compare(x, y, fit_indices, scale=None):
    indices = np.asarray(fit_indices, dtype=np.int64)
    a, b = x[indices], y[indices]
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    u, singular_values, vh = np.linalg.svd(ac.T @ bc)
    reflection = np.eye(3)
    reflection[-1, -1] = np.linalg.det(vh.T @ u.T)
    rotation = vh.T @ reflection @ u.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("Alignment did not yield an orthonormal rotation.")
    if np.linalg.det(rotation) <= 0.999999:
        raise ValueError("Alignment did not yield a proper rotation.")
    denominator = np.sum(ac * ac)
    if denominator <= 1e-12:
        raise ValueError("Calibration camera centers are degenerate.")
    if scale is None:
        scale = np.sum(singular_values * np.diag(reflection)) / denominator
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale.")
    translation = b.mean(axis=0) - scale * rotation @ a.mean(axis=0)
    predicted = scale * x @ rotation.T + translation
    distances = np.linalg.norm(predicted - y, axis=1)
    outside = np.ones(len(x), dtype=bool)
    outside[indices] = False
    # Deliberately return scalar diagnostics only: no camera poses or transform export.
    return {
        "scale": float(scale),
        "all": distribution(distances),
        "outside_fit": distribution(distances[outside]),
    }


def analyze(trajectory, alignment, expected_frames, expected_calibration_frames):
    ids = [row["id"] for row in trajectory]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame IDs in trajectory.")
    if expected_frames and len(ids) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames; got {len(ids)}.")
    id_to_index = {frame_id: i for i, frame_id in enumerate(ids)}
    native = np.asarray([row["native_c2w"] for row in trajectory], dtype=np.float64)
    reference = np.asarray([row["reference_c2w"] for row in trajectory], dtype=np.float64)
    if native.shape != (len(ids), 4, 4) or reference.shape != (len(ids), 4, 4):
        raise ValueError("Expected N x 4 x 4 native_c2w and reference_c2w arrays.")
    x, y = native[:, :3, 3], reference[:, :3, 3]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite camera centers.")
    fit_ids = [
        row["id"] for row in alignment["calibration_frames"]
        if row["ratio"] is not None
    ]
    if len(fit_ids) < 3 or len(set(fit_ids)) != len(fit_ids):
        raise ValueError("Calibration must contain at least 3 distinct valid frame IDs.")
    if expected_calibration_frames and len(fit_ids) != expected_calibration_frames:
        raise ValueError(f"Expected {expected_calibration_frames} calibration IDs; got {len(fit_ids)}.")
    missing = set(fit_ids) - id_to_index.keys()
    if missing:
        raise ValueError(f"Calibration IDs are absent from trajectory: {sorted(missing)}")
    fit_indices = [id_to_index[frame_id] for frame_id in fit_ids]
    result = {
        "n": len(ids),
        "existing_fixed_scale_prefix_rigid": fit_and_compare(
            x, y, fit_indices, scale=alignment["scale_m_per_model_unit"]
        ),
        "prefix_centres_sim3_diagnostic": fit_and_compare(x, y, fit_indices),
        "all_centres_sim3_diagnostic_only": fit_and_compare(x, y, list(range(len(ids)))),
    }
    return result, fit_ids


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-root", default=".", help="Local checkout; defaults to current directory")
    parser.add_argument("--repo-base", type=repository_name, default=BASELINE_REPOSITORY)
    parser.add_argument("--ref", default=BASELINE_REF, help="Source Git ref; immutable commit recommended")
    parser.add_argument("--input-base", type=relative_directory, default=BASELINE_INPUT_BASE)
    parser.add_argument("--trajectory", help="Existing trajectory JSON; location is not published")
    parser.add_argument("--alignment", help="Existing alignment JSON; location is not published")
    parser.add_argument("--trajectory-blob", type=blob_id)
    parser.add_argument("--alignment-blob", type=blob_id)
    parser.add_argument("--expected-frames", type=int, default=1478, help="0 disables the count check")
    parser.add_argument("--expected-calibration-frames", type=int, default=32, help="0 disables the count check")
    parser.add_argument("--cache-dir")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--output", help="JSON output; omitted means stdout")
    args = parser.parse_args()
    if args.expected_frames < 0 or args.expected_calibration_frames < 0:
        parser.error("expected counts must be nonnegative")
    if not args.ref:
        parser.error("ref cannot be empty")
    pinned_baseline = (
        args.repo_base == BASELINE_REPOSITORY
        and args.ref == BASELINE_REF
        and args.input_base == BASELINE_INPUT_BASE
    )
    if args.cache_dir is None:
        cache_key = hashlib.sha256(
            f"{args.repo_base}/{args.ref}/{args.input_base}".encode("utf-8")
        ).hexdigest()[:20]
        args.cache_dir = str(Path(tempfile.gettempdir()) / f"dn-map-alignment-{cache_key}")
    inputs, local_paths, data = {}, {}, {}
    for key in ("trajectory", "alignment"):
        name = key + ".json"
        path_in_repo = (PurePosixPath(args.input_base) / name).as_posix()
        url = (
            f"https://raw.githubusercontent.com/{args.repo_base}/"
            f"{urllib.parse.quote(args.ref, safe='')}/{urllib.parse.quote(path_in_repo, safe='/')}"
        )
        source = {"path": path_in_repo, "url": url}
        expected = getattr(args, key + "_blob")
        if expected is None and pinned_baseline:
            expected = BASELINE_BLOBS[name]
        local_paths[key], data[key] = resolve_input(
            name, getattr(args, key), args, source, expected
        )
        inputs[key] = source
    result, fit_ids = analyze(
        data["trajectory"], data["alignment"],
        args.expected_frames, args.expected_calibration_frames,
    )
    exact_baseline_inputs = all(
        inputs[key]["git_blob_sha1"] == BASELINE_BLOBS[key + ".json"]
        for key in inputs
    )
    reproduced = None
    if pinned_baseline and exact_baseline_inputs:
        values = result["existing_fixed_scale_prefix_rigid"]["all"]
        reproduced = (
            abs(values["median"] - 0.8265402197742225) < 1e-8
            and abs(values["p95"] - 1.6216084544841065) < 1e-8
        )
        if not reproduced:
            raise ValueError("Exact baseline inputs did not reproduce published disagreement.")
    result["audit_metadata"] = {
        "kind": "local_readonly_reanalysis_of_published_artifacts",
        "model_inference_executed": False,
        "model_training_executed": False,
        "server_experiment_executed": False,
        "camera_poses_exported": False,
        "use_for_training_poses": False,
        "independent_accuracy_claim_supported": False,
        "repository": args.repo_base,
        "ref": args.ref,
        "commit": args.ref if re.fullmatch(r"[0-9a-fA-F]{40}", args.ref) else None,
        "reference": "B2 estimated camera centers; not independent ground truth",
        "unit_of_center_disagreement": "metres in B2 reference coordinates",
        "numpy_version": np.__version__,
        "inputs": inputs,
        "published_baseline_disagreement_reproduced": reproduced,
        "fit_frame_ids_for_prefix_diagnostics": fit_ids,
        "fit_and_score_sets": {
            "existing_fixed_scale_prefix_rigid": {
                "scale": "published sensor/Map depth ratio on frozen calibration support",
                "rotation_translation_fit_count": len(fit_ids),
                "all_score_count": result["n"],
                "outside_fit_score_count": result["n"] - len(fit_ids),
            },
            "prefix_centres_sim3_diagnostic": {
                "scale_rotation_translation_fit_count": len(fit_ids),
                "all_score_count": result["n"],
                "outside_fit_score_count": result["n"] - len(fit_ids),
            },
            "all_centres_sim3_diagnostic_only": {
                "scale_rotation_translation_fit_count": result["n"],
                "all_score_count": result["n"],
                "outside_fit_score_count": 0,
            },
        },
        "prefix_sim3_warning": (
            "Fits scale, rotation and translation to the frozen calibration B2 centers. "
            "Reference-agreement diagnosis only; not independent accuracy validation, "
            "an observed accuracy improvement, or a candidate for training camera poses."
        ),
        "full_sequence_sim3_warning": (
            "Fits all B2 centers and evaluates the same centers. Optimistic "
            "reference-fitting diagnosis only; not an observed accuracy improvement, "
            "new trajectory estimate, independent validation, GT error, or training poses."
        ),
        "outside_fit_warning": (
            "Outside the frozen calibration IDs, but still the same correlated training "
            "sequence and estimated B2 reference; not a held-out benchmark."
        ),
        "method": "SVD proper-rotation rigid/similarity alignment of camera centers",
        "reproduction_command": "python scripts/diagnose_lingbot_alignment.py --repo-root .",
        "reproduction_command_scope": "default audited baseline; reuse source metadata for other inputs",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        protected = [path.resolve() for path in local_paths.values()] + [Path(__file__).resolve()]
        if output.resolve() in protected:
            raise ValueError("Output must not overwrite an input or this script.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()

