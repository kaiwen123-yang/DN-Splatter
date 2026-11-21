"""Locations and identities for the joint capture-full B2 adaptation."""
from pathlib import Path

from capture_rectify import read, sha
from capture_metric3d_common import write

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/capture_quality_protocol.json"
P = read(SPEC)
B1 = ROOT / read(ROOT / "configs/capture_metric3d_protocol.json")["trial"]["run_directory"]
GEORUN = ROOT / read(ROOT / "configs/capture_full_test_protocol.json")["paths"]["geometry_run"]
RUN = ROOT / ".local/runs" / P["run_id"]
SCENE = RUN / "inputs"
STATE = RUN / "state"
GEOM = RUN / "geometry"
REPORT = ROOT / "reports/capture-full-quality-v2" / "20261121-v1"


def check(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
