"""Frozen paths and atomic experiment records for the one B1 comparison."""
import json
from pathlib import Path

from capture_rectify import read, sha

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/capture_metric3d_protocol.json"
P = read(PROTOCOL)
B0 = ROOT / P["baseline"]["training_inputs"]
BASE = B0.parent
B0_REPORT = (ROOT / P["baseline"]["configuration"]).parent
RUN = ROOT / P["trial"]["run_directory"]
SCENE = RUN / "inputs"
REPORT = ROOT / P["trial"]["report_directory"]
STATE = RUN / "state"
PRIOR = RUN / "metric3d"
SOURCE = ROOT / ".local/upstream/Metric3D"
WEIGHT = ROOT / ".local/data/downloads" / P["prior"]["weight_filename"]
GEOMETRY = ROOT / P["baseline"]["geometry_dir"]
SPLITS = ("train", "val", "test")
HOLDOUT = ("val", "test")
RAW_PREFIX = "DepthAndNormalMapsPoisson_"


def demand(condition, location):
    if not condition:
        raise ValueError(f"Metric3D experiment identity or result mismatch: {location}")


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def member(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def selection():
    return read(B0_REPORT / "selection.json")["rows"]


def model_paths(label):
    demand(label in ("B0", "B1"), label)
    return (BASE, B0) if label == "B0" else (RUN, SCENE)
