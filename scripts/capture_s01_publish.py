"""Assemble small public evidence and index the retained bulk files by SHA256."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from capture_rectify import read, save, sha, verify
from capture_s01_inputs import GEO, P, PROTOCOL, REPORT, WEIGHT, ensure
from capture_s01_overview import write_overview
from vr_room_publish import patch_matches, records
from vr_room_run import CPU, ROOT, RUN, SCENE, SRC, STATE


def timings():
    samples = records(STATE / "resources.jsonl")
    processes = records(STATE / "process_resources.jsonl")
    result = []
    for record in records(STATE / "stages.jsonl"):
        row = {k: v for k, v in record.items() if k != "command"}
        group = [
            s
            for s in samples
            if s["stage"] == row["stage"] and row["started_utc"] <= s["utc"] <= row["ended_utc"]
        ]
        row["resource_samples"] = len(group)
        proc = [
            s
            for s in processes
            if s["stage"] == row["stage"] and row["started_utc"] <= s["utc"] <= row["ended_utc"]
        ]
        row["observed_child_pids"] = sorted({s["child_pid"] for s in proc})
        row["process_peak_rss_kib"] = max((s["VmHWM"] for s in proc if "VmHWM" in s), default=None)
        row["process_cpu_seconds_last_observed"] = max((s["cpu_seconds"] for s in proc), default=None)
        for k in ("gpu_used_mib_device", "gpu_util_pct_device", "cgroup_memory_bytes"):
            row["peak_sampled_" + k] = max((s[k] for s in group), default=None)
        cpus = []
        for a, b in zip(group, group[1:]):
            interval = (
                datetime.fromisoformat(b["utc"]) - datetime.fromisoformat(a["utc"])
            ).total_seconds()
            if interval > 0:
                cpus.append((b["cgroup_cpu_usage_usec"] - a["cgroup_cpu_usage_usec"]) / 1e6 / interval)
        row["peak_interval_mean_cgroup_cores"] = max(cpus, default=None)
        result.append(row)
    return result


def train_stats(trained):
    cfg = Path(trained["config"])
    reader = EventAccumulator(str(cfg.parent), size_guidance={"scalars": 0})
    reader.Reload()
    tags = reader.Tags()["scalars"]
    series = {}
    for tag in tags:
        if "loss" in tag.lower() or "gaussian" in tag.lower() or "psnr" in tag.lower():
            values = reader.Scalars(tag)
            finite = [x for x in values if np.isfinite(x.value)]
            series[tag] = dict(
                count=len(values),
                nonfinite=len(values) - len(finite),
                first=[finite[0].step, finite[0].value] if finite else None,
                last=[finite[-1].step, finite[-1].value] if finite else None,
            )
    ensure(any("loss" in k.lower() for k in series), "training loss records")
    save(
        REPORT / "training.json",
        dict(
            **trained,
            tensorboard_scalars=series,
            formal_first_batch=[read(p) for p in sorted(STATE.glob("first_batch_*.json"))],
            segments=records(STATE / "training_segments.jsonl"),
            seed=42
        ),
    )
    shutil.copyfile(cfg, REPORT / "config.yml")


def inventory():
    paths = [
        p
        for p in RUN.rglob("*")
        if p.is_file() and not p.relative_to(RUN).parts[0] == "state" and p not in (RUN / "logs/publish.log", RUN / "logs/runner.log")
    ]
    paths.extend(
        [
            p
            for p in STATE.glob("*")
            if p.name.startswith("first_batch_") or p.name == "training_segments.jsonl"
        ]
    )
    index = [
        dict(path=str(p.relative_to(RUN)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(paths)
    ]
    save(
        REPORT / "artifact_manifest.json",
        dict(
            root=str(RUN),
            files=index,
            file_count=len(index),
            total_bytes=sum(row["bytes"] for row in index),
            external_files=[
                dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f))
                for f in (WEIGHT.resolve(), GEO / "manifest.json")
            ],
        ),
    )
    return index


def main():
    verify(GEO)
    solved = read(RUN / "training_complete.json")
    ensure(solved["step"] == 29999 and solved["cumulative_updates"] == 30000, "final step")
    ensure(
        solved["training_manifest_sha256"] == sha(SCENE / "training_manifest.json"), "training identity"
    )
    metrics = {s: read(REPORT / ("metrics_" + s + ".json")) for s in ("val", "test")}
    for s, m in metrics.items():
        ensure(m["frame_count"] == 24 and m["checkpoint_sha256"] == solved["checkpoint_sha256"], s)
    mesh = read(REPORT / "mesh_validation.json")
    audit = read(RUN / "mesh/sampling.json")
    ensure(
        audit["checkpoint_sha256"] == solved["checkpoint_sha256"]
        and mesh["views"] == 193
        and mesh["geometry_finite"],
        "mesh",
    )
    train_stats(solved)
    stages = timings()
    compare = {
        "dn_splatter": patch_matches(SRC, "dn-811f5cd-*.patch"),
        "gsplat": patch_matches(ROOT / ".local/upstream/gsplat", "gsplat-1.0.0-*.patch"),
    }
    ensure(all(compare.values()), "unrecorded code in source checkout")
    parsed = read(REPORT / "reader_check.json")
    save(
        REPORT / "run_manifest.json",
        dict(
            run=RUN.name,
            base_commit="fb6ec1bae800c0f44b01649dd7d5523ed862ec16",
            geometry_commit=P["input"]["geometry_commit"],
            identity=read(STATE / "identity.json"),
            stage_records=stages,
            cpu_budget=CPU,
            resource_scope="Device-wide GPU and whole cgroup RAM and CPU, sampled every 15s. Short stages may have no sample; null means unobserved. Includes other activity and filesystem cache.",
            actual_cpu_threads=dict(
                image_cache_workers=2, blas_threads_per_worker=1, poisson_threads=CPU
            ),
            historical_checkout_matches_patches=compare,
            formal_training=solved,
            reader_counts={k: v["count"] for k, v in parsed["splits"].items()},
            observed_utc=datetime.now(timezone.utc).isoformat(),
            geometry_reverified_at_publication=True,
        ),
    )
    commands = REPORT / "commands.sh"
    commands.write_text(
        "#!/usr/bin/env bash\nset -eo pipefail\ncd /root/autodl-tmp/DN-Splatter\n"
        "# Rechecks this run identity and continues from completed stages.\n"
        "bash scripts/launch_capture_s01.sh\n\n# Actual commands (working directory listed for each):\n"
        + "\n".join(
            "# " + row["cwd"] + " $ " + row["command"] for row in records(STATE / "stages.jsonl")
        )
        + "\n"
    )
    index = inventory()
    ensure(
        any(
            row["path"].endswith("step-000029999.ckpt") and row["sha256"] == solved["checkpoint_sha256"]
            for row in index
        ),
        "indexed checkpoint",
    )
    latest_file = REPORT.parent / "latest.json"
    write_overview(metrics, mesh, solved, stages)
    latest = read(latest_file)
    latest.update(
        status="completed",
        execution_observed_in_this_review=True,
        normal_priors_generated=True,
        training_started=True,
        cumulative_updates=30000,
        final_checkpoint=solved["checkpoint"],
        evaluation_completed=True,
        poisson_completed=True,
        report=str(REPORT.relative_to(ROOT) / "overview.md"),
        measured_results={s: m["rgb_frame_means"] for s, m in metrics.items()},
        note="Formal final checkpoint bound to both holdouts and the original train-view Poisson export.",
    )
    save(latest_file, latest)


if __name__ == "__main__":
    main()
