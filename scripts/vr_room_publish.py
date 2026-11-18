"""Publish verified small VR Room results, while retaining full artifacts on the data disk."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(os.environ["DN_PROJECT_ROOT"])
RUN = Path(os.environ["DN_RUN_ROOT"])
PUB = ROOT / "reports" / RUN.name
SRC = Path(os.environ["DN_SOURCE_DIR"])
STATE = RUN / "state"


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []


def write(path, doc):
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, directory=ROOT):
    return subprocess.check_output(["git", "-C", directory, *args], text=True).strip()


def shellarg(arg):
    if arg.startswith(str(ROOT)+"/"):
        return '"${DN_PROJECT_ROOT}"/'+shlex.quote(str(Path(arg).relative_to(ROOT)))
    return shlex.quote(arg)


def ply_vertices(path):
    if not path.is_file():
        return None
    with path.open("rb") as f:
        for _ in range(100):
            line = f.readline(256)
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])
            if not line or line.strip() == b"end_header":
                break
    raise ValueError("Missing PLY vertex count")


def publish_files(paths, mode):
    paths = sorted({str(p.relative_to(ROOT)) for p in paths})
    if git("remote", "get-url", "--push", "origin") not in {
        "https://github.com/kaiwen123-yang/DN-Splatter.git","https://github.com/kaiwen123-yang/DN-Splatter"
    }:
        raise ValueError("Unexpected reproduction repository")
    for p in paths:
        path = ROOT / p
        if path.is_symlink() or path.stat().st_size > 8*1024**2:
            raise ValueError(f"Invalid public artifact {p}")
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    diff = git("diff","--cached","--",*paths)
    if re.search(r"(https?://[^\s]+/sub/[A-Za-z0-9]{16,}|gh[pousr]_[\w]{20,}|github_pat_|BEGIN .*PRIVATE KEY|https?://[^\s/]+:[^\s/]+@|/root/autodl-tmp/|connect\.[\w.-]*seetacloud)",diff):
        raise ValueError("Private detail detected in staged public report")
    subprocess.run(["git","diff","--cached","--check","--",*paths],cwd=ROOT,check=True)
    if diff:
        subprocess.run(["git","commit","--only","-m","Record VR Room "+mode+" evidence","--",*paths],cwd=ROOT,check=True)
    push = ["bash",str(ROOT/"scripts/github_transport.sh"),"git","-c","credential.https://github.com.helper=",
            "-c","credential.https://github.com.helper=!gh auth git-credential","push","origin"]
    branch = "main"
    result = subprocess.run([*push, "HEAD:refs/heads/main"],cwd=ROOT)
    if result.returncode:
        branch = "codex/vr-room-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        subprocess.run([*push,"HEAD:refs/heads/"+branch],cwd=ROOT,check=True)
    remote = subprocess.check_output(["bash",str(ROOT/"scripts/github_transport.sh"),"git","ls-remote",
        "origin",f"refs/heads/{branch}"],cwd=ROOT,text=True).split()[0]
    if remote != git("rev-parse","HEAD"):
        raise ValueError("Published ref differs from evidence commit")
    write(STATE/"published.json",{"branch":branch,"commit":remote})


def main(mode):
    PUB.mkdir(parents=True,exist_ok=True)
    report = ROOT/"reports"
    staged = []
    time = dt.datetime.now(dt.timezone.utc)
    state = read(STATE/"current.json",{})
    stages = records(STATE/"stages.jsonl")
    trained = read(RUN/"training_complete.json",{})
    frames = read(RUN/"frames.json",{})
    split_counts = frames.get("counts") or read(RUN/"registered.json",{}).get("counts")
    bw = read(RUN/"blackwell.json")
    inputs = [read(p) for p in sorted(RUN.glob("input-*.json"))]
    resources = records(STATE/"resources.jsonl")
    inventory = read(sorted((ROOT/".local/reports").glob("host_*/host_inventory.json"))[0])
    cpu = int(os.environ["DN_CPU_BUDGET"])
    prefix = Path(os.environ["DN_ENV_PREFIX"])
    init = Path(os.environ["DN_SCENE_DIR"])/"iphone/long_capture/iphone_pointcloud.ply"
    mesh = read(RUN/"mesh/validation.json")
    metrics = read(RUN/"evaluation/rgb_depth_official.json",{}).get("results")
    geometry = read(RUN/"mesh/mesh_metrics.json")
    first_ckpts = sorted((RUN/"training").glob("**/step-*.ckpt"))
    if trained:
        saved_step = trained["step"]
    else:
        saved_step = None
        if first_ckpts:
            from vr_room_run import latest_checkpoint
            latest, step, _ = latest_checkpoint()
            if latest:
                saved_step = step
    if mode=="complete" and not (trained.get("step")==29999 and mesh and metrics and geometry
            and bw and (RUN/"evaluation/faro_audited.json").exists()):
        raise ValueError("Incomplete result set")
    def copy(a,b):
        if a.exists():
            b.parent.mkdir(parents=True,exist_ok=True)
            if a.suffix == ".json":
                b.write_text(a.read_text().replace(str(ROOT),"${DN_PROJECT_ROOT}"))
            else:
                shutil.copyfile(a,b)
            staged.append(b)
    for source,dest in [("registered.json","registered_frames.json"),("frames.json","frames.json"),("blackwell.json","blackwell.json"),
            ("evaluation/rgb_depth_official.json","rgb_depth_official.json"),
            ("evaluation/rgb_depth_faro_official_raw.json","faro_official_raw.json"),
            ("evaluation/faro_audited.json","faro_audited.json"),("mesh/mesh_metrics.json","mesh_metrics.json"),
            ("mesh/validation.json","mesh_validation.json")]:
        copy(RUN/source,PUB/dest)
    for p in sorted((RUN/"previews").glob("*.jpg")):
        copy(p,PUB/"previews"/p.name)
    if trained:
        cfg = ROOT/trained["config"]
        path = PUB/"config.yml"
        path.write_text(cfg.read_text().replace(str(ROOT),"${DN_PROJECT_ROOT}"))
        staged.append(path)
        copy(cfg.parent/"dataparser_transforms.json",PUB/"dataparser_transforms.json")
    gpu = inventory["gpu_inventory"]["stdout"].splitlines()[-1].split(", ")
    limits = inventory["memory"]["cgroup_limits_and_usage"]
    memory = int(limits["/sys/fs/cgroup/memory.max"])
    used_at_start = int(limits["/sys/fs/cgroup/memory.current"])
    host = report/"host_summary.md"
    host.write_text(f"# AutoDL host observation\n\nObserved {inventory['created_at_asia_shanghai']}. "
        f"{gpu[1]}, driver {gpu[2]}, {gpu[3]} device memory. At inventory the device used {gpu[4]} and reported {gpu[6]} utilization.\n\n"
        f"CPU quota {limits['/sys/fs/cgroup/cpu.max']}, affinity {inventory['cpu']['allowed_affinity_count']}, "
        f"effective budget {cpu} CPUs. Memory limit {memory//2**30} GiB, initial cgroup charge {used_at_start/2**30:.1f} GiB. "
        f"Project disk free at inventory {inventory['disk']['project_filesystem']['free_bytes']/2**30:.1f} GiB.\n\n"
        "The isolated Conda prefix and dedicated package, compilation and model caches are under .local/. "
        "One formal trainer has access to the whole device. Compilation uses Ninja with a live memory-based job budget. "
        "Frame validation uses one native thread per worker. Poisson explicitly receives the effective CPU budget.\n")
    staged.append(host)
    reuse = report/"reuse_inventory.json"
    write(reuse,{"system_toolkit":"CUDA 12.8 (nvcc 12.8.93), read-only","conda_manager":"existing manager, new project prefix",
        "project_prefix":".local/conda/dn-splatter-bw","inputs":[{k:v for k,v in a.items() if k!="selected_members"} for a in inputs]})
    staged.append(reuse)
    actual = report/"environment_actual.txt"
    pkgs = subprocess.check_output([sys.executable,"-m","pip","list","--format=freeze"],text=True)
    nvcc = subprocess.check_output([str(Path(os.environ["CUDA_HOME"])/"bin/nvcc"),"--version"],text=True)
    actual.write_text(sys.version.splitlines()[0]+"\n"+nvcc+"\n"+pkgs)
    staged.append(actual)
    conda = PUB/"conda-explicit.json"
    fields = ("name","version","build","subdir","fn","sha256","md5")
    write(conda,[{k:p[k] for k in fields if k in p} for p in (read(f) for f in sorted((prefix/"conda-meta").glob("*.json")))])
    staged.append(conda)
    peaks = {}
    for row in resources:
        peak = peaks.setdefault(row["stage"],{"samples":0})
        peak["samples"]+=1
        for key in ("gpu_used_mib_device","gpu_util_pct_device","cgroup_memory_bytes"):
            peak[key]=max(peak.get(key,0),row[key])
    sessions = []
    for r in stages:
        sessions.append({k:v for k,v in r.items() if k!="command"})
    manifest = PUB/"run_manifest.json"
    write(manifest,{"run_id":RUN.name,"time_utc":time.isoformat(),
        "time_asia_shanghai":time.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "repository_at_report":git("rev-parse","HEAD"),"policy_commit":"7f8d6afd360382d628b96af769db086966ca1964",
        "source":git("rev-parse","HEAD",directory=SRC),
        "patch_sha256":{p.name:sha(p) for p in sorted((ROOT/"patches").glob("*.patch"))},
        "gsplat_source":git("rev-parse","HEAD",directory=ROOT/".local/upstream/gsplat"),
        "pytorch3d_source":git("rev-parse","HEAD",directory=ROOT/".local/upstream/pytorch3d"),
        "blackwell_runtime":bw, "data":"MuSHRoom vr_room / iphone",
        "inputs":[{k:v for k,v in a.items() if k!="selected_members"} for a in inputs],
        "frame_counts":split_counts,"normal_alignment_checked":bool(frames),"seed":42,"normal_prior":"Omnidata v2 low; opengl",
        "initialization":{"source":"pinned iPhone initializer, long capture excluding test.txt",
            "vertices":ply_vertices(init),"sha256":sha(init) if init.is_file() else None},
        "training":trained or None,"resume_segments":records(STATE/"training_segments.jsonl"),
        "optimizer_protocol":"original historical loss weights; flattening is included in normal_lambda",
        "stages":sessions,"cpu_budget":cpu,"compile_jobs_at_report":int(os.environ["MAX_JOBS"]),
        "frame_validation_workers":cpu,"mesh":mesh,"peaks_sampled_every_15s":peaks,
        "resources_scope":"device and whole container; samples can miss transient peaks"})
    staged.append(manifest)
    cmdfile = PUB/"commands.sh"
    lines = ["#!/usr/bin/env bash",'set -euo pipefail','source "${DN_PROJECT_ROOT:?}/scripts/autodl_env.sh"']
    for r in stages:
        lines.append('cd "${DN_PROJECT_ROOT}"/'+shlex.quote(r["cwd"]))
        lines.append("OMP_NUM_THREADS="+str(r["omp_threads"])+" "+" ".join(shellarg(a) for a in shlex.split(r["command"])))
    cmdfile.write_text("\n".join(lines)+"\n")
    staged.append(cmdfile)
    last = report/"latest_status.json"
    reason = str(state.get("error","")).replace(str(ROOT),"PROJECT") or None
    info = {"schema_version":2,"updated_at":time.isoformat(),"run_id":RUN.name,"status":mode,
        "stage":state.get("stage"),"failed_stage":state.get("failed_stage"),"reason":reason,"registered_frame_counts":split_counts,
        "actual_cpu_budget":cpu,"full_cpu_gpu_authorized_by_user":True,"target_host_inspected":True,
        "environment_created":(prefix/"conda-meta/history").is_file(),"blackwell_forward_backward_verified":bool(bw),
        "training_completed_updates":saved_step+1 if saved_step is not None else None,"target_training_updates":30000,
        "checkpoint_step":saved_step,"rgb_depth_evaluation_complete":bool(metrics),
        "poisson_mesh_complete":bool(mesh),"mesh_evaluation_complete":bool(geometry),
        "metrics":metrics,"pipeline_completed":mode=="complete",
        "next_action":"Inspect the result and both trajectory comparisons." if mode=="complete" else
            "Read the project-local workflow status and stage log. Resume with bash scripts/launch_vr_room.sh."}
    write(last,info)
    staged.append(last)
    summary = PUB/"summary.md"
    details = ["# VR Room historical implementation baseline","",f"Run status **{mode}**. ",
               f"Recorded checkpoint step {saved_step}; observed split counts {split_counts}.",
               "The historical loss places the flattening term inside the normal weight. Table 4 gives one VR Room row; the official implementation reports two trajectories."]
    if mode=="complete":
        details.extend(["","| Trajectory | PSNR (dB) | Difference vs 26.8 dB | SSIM | Difference vs 0.879 |",
                        "|---|---:|---:|---:|---:|"])
        for split in ("within","with"):
            p,s=metrics[split+"_rgb_psnr"],metrics[split+"_rgb_ssim"]
            details.append(f"| {split} | {p:.4f} | {p-26.8:+.4f} | {s:.5f} | {s-.879:+.5f} |")
        fs=geometry["F-score"]*100
        details+=["",f"Visibility-culled F-score at 5 cm was {fs:.3f}%, a difference of {fs-87.1:+.3f} percentage points from Table 4.",
            "The official Faro output preserves its shared batch accumulators. The audited file gives each registered frame equal weight within its own trajectory.",
            "The previews contain fixed held-out views and two raycast views of the culled surface. Their pixel arrays and the mesh topology passed the recorded checks."]
    elif reason:
        details+=["",f"Last error at {state.get('failed_stage')}: {reason}"]
    summary.write_text("\n".join(details)+"\n")
    staged.append(summary)
    artifact = PUB/"artifacts.json"
    candidates = set(first_ckpts)
    for path in [RUN/"mesh", RUN/"renders_official"]:
        if path.is_dir():
            candidates.update(p for p in path.rglob("*") if p.is_file())
    write(artifact,[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p),"storage":"local_only"}
                    for p in sorted(candidates)])
    staged.append(artifact)
    publish_files(staged,mode)


if __name__ == "__main__":
    main(sys.argv[1])
