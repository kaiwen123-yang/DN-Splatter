"""Measure how much B2's RGB residual can be explained by global per-channel gain and offset.

Run from the repository with Python, NumPy and OpenCV:
    python scripts/capture_video_affine_check.py PATH/TO/reference_B1_B2_public.mp4
"""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from capture_quality_eval import digest, read, write


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/capture-full-quality-v2/20261121-v1"
OUTPUT = ROOT / "reports/capture-quality-review/2026-11-22/video_affine_check.json"
TITLE_HEIGHT = 42
PANEL_WIDTH = 1280
STRIDE = 3


def fitted_error(frame):
    if frame.shape != (762, 3840, 3):
        raise ValueError(f"Unexpected decoded B2 video shape {frame.shape}")
    # Columns are reference, B1, B2. The channel-wise fit is invariant to BGR/RGB order.
    pixels = frame[TITLE_HEIGHT::STRIDE]
    reference = pixels[:, :PANEL_WIDTH:STRIDE].reshape(-1, 3).astype(np.float64) / 255
    prediction = pixels[:, 2 * PANEL_WIDTH::STRIDE].reshape(-1, 3).astype(np.float64) / 255
    x = prediction - prediction.mean(axis=0)
    y = reference - reference.mean(axis=0)
    var = (x * x).sum(axis=0)
    gains = np.divide((x * y).sum(axis=0), var, out=np.zeros(3), where=var > 0)
    before = np.mean((reference - prediction) ** 2)
    after = np.mean((y - x * gains) ** 2)
    if before <= 0 or after <= 0:
        raise ValueError("Undefined relative improvement on an exactly matched frame")
    return {"residual_mse_fraction": float(after / before),
            "psnr_gain_db": float(10 * np.log10(before / after))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = read(EVIDENCE / "video_manifest.json")
    video = manifest["video"]
    if args.video.stat().st_size != video["bytes"] or digest(args.video) != video["sha256"]:
        raise ValueError("Video differs from the published B2 manifest")

    checkpoint = Path(manifest["selected_checkpoint"]["path"]).stem
    splits = {}
    frame_counts = {}
    for split in ("val", "test"):
        with gzip.open(EVIDENCE / f"{split}_{checkpoint}.json.gz", "rt", encoding="utf-8") as stream:
            frames = json.load(stream)["frames"]
        frame_counts[split] = len(frames)
        for row in frames:
            index = int(row["id"].removeprefix("frame_"))
            if index in splits or not 0 <= index < manifest["frames"]:
                raise ValueError(f"Invalid or duplicate frame index {index}")
            splits[index] = split

    redacted = set(manifest["redacted_frames"])
    examples = {}
    observations = {split: [] for split in frame_counts}
    capture = cv2.VideoCapture(str(args.video))
    try:
        for index in range(manifest["frames"]):
            if not capture.grab():
                raise ValueError(f"Video ended at frame {index}")
            if index not in splits or index in redacted:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {index}")
            fit = fitted_error(frame)
            observations[splits[index]].append(fit)
            stem = f"frame_{index:05d}"
            if stem in manifest["worst_frames"]:
                examples[stem] = {"split": splits[index], **fit}
        if capture.grab():
            raise ValueError("Unexpected extra video frames")
    finally:
        capture.release()

    summary = {}
    for split, rows in observations.items():
        fractions = np.array([r["residual_mse_fraction"] for r in rows])
        gains = np.array([r["psnr_gain_db"] for r in rows])
        summary[split] = {
            "official_split_frames": frame_counts[split],
            "public_frames_analyzed": len(rows),
            "public_redactions_excluded": frame_counts[split] - len(rows),
            "median_residual_mse_fraction": float(np.median(fractions)),
            "residual_mse_fraction_p25_p75": np.percentile(fractions, [25, 75]).tolist(),
            "median_psnr_gain_db": float(np.median(gains)),
            "max_psnr_gain_db": float(np.max(gains)),
            "frames_above_1db_gain": int(np.count_nonzero(gains > 1)),
        }
    write(args.output, {
        "video": {"filename": args.video.name, "sha256": video["sha256"], "frames": manifest["frames"]},
        "source_split_files": [f"{split}_{checkpoint}.json.gz" for split in frame_counts],
        "sampling": "Display-coded pixels of the reference and B2 panels. Skip 42 title rows, use the top-left pixel and every third pixel in x and y of each 1280x720 panel (102480 pixel pairs per frame).",
        "fit": "Per frame and channel fit reference = gain * prediction + offset by least squares on the sampled pixels. Gains and predicted values are unconstrained. Divide fitted MSE by original MSE on the same sampled pixels, averaging squared errors over all pixels and channels. Frame summaries weight each included frame equally.",
        "scope": "Compressed public video diagnostic. Excludes its manifest redactions from the fit and summaries. Official frozen evaluation uses all 184 val and 184 development test frames.",
        "summary": summary,
        "published_example_frames": examples,
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
    })


if __name__ == "__main__":
    main()
