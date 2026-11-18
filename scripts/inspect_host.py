#!/usr/bin/env python3
"""Read selected host facts; write reports only under this project's .local/.

No installation, CUDA workload, environment activation, process signalling, or
system configuration change is performed. Raw host reports are local-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timedelta, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_PACKAGES = {
    "python", "pip", "setuptools", "torch", "pytorch", "torchvision",
    "torchaudio", "torchmetrics", "nerfstudio", "gsplat", "numpy", "open3d",
    "pymeshlab", "cudatoolkit", "cuda-toolkit", "cuda-nvcc", "ninja",
}


def is_within(path: Path, parent: Path) -> bool:
    """Containment check that also works before creating the Python 3.10 env."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def probe(argv: list[str], timeout: int = 15) -> dict:
    """Run a short read-only command, without a shell."""
    executable = shutil.which(argv[0])
    if executable is None:
        return {"status": "unavailable"}
    try:
        result = subprocess.run(
            [executable, *argv[1:]], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout", "timeout_seconds": timeout}
    return {
        "status": "ok" if result.returncode == 0 else "command_failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def read_memory() -> dict:
    selected = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    fields = {}
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text().splitlines():
            name, value = line.split(":", 1)
            if name in selected:
                fields[name + "_KiB"] = int(value.split()[0])
    # AutoDL is containerized: host MemTotal alone may exceed the usable quota.
    cgroup = {}
    for name in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    ):
        path = Path(name)
        if path.is_file():
            cgroup[name] = path.read_text().strip()
    return {"proc_meminfo": fields, "cgroup_limits_and_usage": cgroup}


def inspect_conda(prefixes: list[str]) -> dict:
    # CONDA_EXE is a single explicitly selected variable, not an environment dump.
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    # Read metadata directly. Even `conda env list` can clean a user's environment
    # registry; avoiding Conda commands makes the inventory strictly read-only.
    candidates = set(prefixes)
    active_prefix = os.environ.get("CONDA_PREFIX")
    if active_prefix:
        candidates.add(active_prefix)
    registry = Path.home() / ".conda/environments.txt"
    if registry.is_file():
        candidates.update(line.strip() for line in registry.read_text().splitlines() if line.strip())
    if conda and Path(conda).is_file():
        base = Path(conda).resolve().parents[1]
        candidates.add(str(base))
        env_dir = base / "envs"
        if env_dir.is_dir():
            candidates.update(str(p) for p in env_dir.iterdir() if p.is_dir())
    envs = sorted(p for p in candidates if (Path(p) / "conda-meta").is_dir())
    result = {
        "status": "ok" if envs or conda else "unavailable",
        "executable": conda,
        "envs": envs,
        "discovery": "read_only_registry_and_known_prefixes; custom unregistered envs may be omitted",
    }
    versions = []
    for prefix in prefixes:
        env_path = Path(prefix)
        item = {"prefix": prefix, "status": "ok" if (env_path / "conda-meta").is_dir() else "not_a_conda_prefix"}
        packages = {}
        for metadata in (env_path / "conda-meta").glob("*.json"):
            package = json.loads(metadata.read_text())
            name = package.get("name", "").lower()
            if name in SELECTED_PACKAGES:
                packages[name] = {"name": name, "version": package.get("version"), "source": "conda-meta"}
        for metadata in env_path.glob("lib/python*/site-packages/*.dist-info/METADATA"):
            name, version = None, None
            with metadata.open(errors="replace") as stream:
                for line in stream:
                    if line.startswith("Name: "):
                        name = line[6:].strip().lower()
                    elif line.startswith("Version: "):
                        version = line[9:].strip()
                    if name is not None and version is not None:
                        break
            if name in SELECTED_PACKAGES:
                packages[name] = {"name": name, "version": version, "source": "dist-info"}
        item["selected_packages"] = sorted(packages.values(), key=lambda p: p["name"])
        versions.append(item)
    result["inspected_prefixes"] = versions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Report directory under this repository's .local/ (relative or absolute).",
    )
    parser.add_argument(
        "--inspect-env", action="append", default=[], metavar="CONDA_PREFIX",
        help="Optionally read selected package versions in an existing env; repeatable.",
    )
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    local_root = (PROJECT_ROOT / ".local").resolve()
    # A symlinked .local must not redirect writes outside the repository.
    if not is_within(local_root, PROJECT_ROOT):
        parser.error(".local resolves outside the repository; no report was written")
    requested = args.output_dir or Path(
        ".local/reports/host_" + now.strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output_dir = (
        requested if requested.is_absolute() else PROJECT_ROOT / requested
    ).resolve()
    if not is_within(output_dir, local_root):
        parser.error("--output-dir must stay inside this repository's .local/")

    processes = probe(["ps", "-eo", "pid,ppid,comm,pcpu,pmem", "--sort=-pcpu"])
    if processes["status"] == "ok":
        processes["stdout"] = "\n".join(processes["stdout"].splitlines()[:31])
    disks = {}
    for name, path in (("project_filesystem", PROJECT_ROOT), ("root_filesystem", Path("/"))):
        usage = shutil.disk_usage(path)
        disks[name] = {
            "total_bytes": usage.total, "used_bytes": usage.used,
            "free_bytes": usage.free,
        }

    report = {
        "schema_version": 1,
        "inspection_type": "read_only_host_inventory",
        "created_at_utc": now.isoformat(),
        "created_at_asia_shanghai": now.astimezone(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "cpu": {
            "os_cpu_count": os.cpu_count(),
            "allowed_affinity_count": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
        },
        "memory": read_memory(),
        "disk": disks,
        "gpu_inventory": probe([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,compute_mode",
            "--format=csv",
        ]),
        "gpu_processes": probe([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv",
        ]),
        "top_process_names_without_arguments": processes,
        "nvcc": {"path": shutil.which("nvcc"), "version_probe": probe(["nvcc", "--version"])},
        "conda": inspect_conda(args.inspect_env),
        "autodl_network_turbo": {
            "path": "/etc/network_turbo",
            "readable": os.access("/etc/network_turbo", os.R_OK),
            "activated_by_inspector": False,
        },
        "interpretation": [
            "A snapshot does not reserve resources or establish a safe training budget.",
            "No CUDA computation, installation, activation, or process modification was performed.",
            "Raw reports contain local paths and process names; keep them under ignored .local/.",
            "Missing utilities or incomplete container visibility are unknowns, not evidence of an idle GPU.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "host_inventory.json"
    markdown_path = output_dir / "host_inventory.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(
        "# AutoDL read-only host inventory\n\n"
        f"Time (Asia/Shanghai): {report['created_at_asia_shanghai']}\n\n"
        "This is a host snapshot, not a training admission decision. "
        "Inspect active jobs and observe resource trends before launching work.\n\n"
        "```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n"
    )
    print(f"Wrote local report: {json_path}")
    print(f"Wrote local report: {markdown_path}")
    print("No environment, GPU workload, or other process was changed.")


if __name__ == "__main__":
    main()
