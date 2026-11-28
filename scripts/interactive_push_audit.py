"""Verify completed PushT files and summarize the real held-out and rope results."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from interactive_push_assets import write_json


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(test, msg):
    if not test:
        raise ValueError(msg)


def member(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def generation(root):
    base = root/"generated"
    manifest = read(base/"manifest.json")
    rows = manifest["episodes_index"]
    bank_hash = digest(base/"initial_states_actions.npz")
    require(manifest["inputs"]["sha256"] == bank_hash, "Paired input bank differs")
    require({(r["version"], r["episode"]) for r in rows} ==
            {(v, i) for v in ("C0", "C2", "C3") for i in range(512)} and len(rows) == 1536,
            "Incomplete or duplicated paired generation")
    sizes = 0
    failures = {v: Counter() for v in ("C0", "C2", "C3")}
    contacts = Counter()
    failed_episodes = Counter()
    measured = Counter()
    paired = {}
    identical_c2_c3 = 0
    with np.load(base/"initial_states_actions.npz", allow_pickle=False) as f:
        cmds, obj, robot = f["commands"], f["object_xytheta"], f["pusher_xy"]
        visual = set(map(int, f["full_visual_ids"]))
    for r in sorted(rows, key=lambda row:(row["episode"], row["version"])):
        v, i = r["version"], r["episode"]
        d = base/v/f"episode_{i:06d}"
        require(r["identity"][1] == bank_hash and r["steps"] == 150 and
                r["observations_during_generation"] == 0 and r["full_visual"] == (i in visual), f"Identity: {d}")
        for f, info in r["files"].items():
            path = d/f
            require(path.stat().st_size == info["bytes"] and digest(path) == info["sha256"], str(path))
            sizes += info["bytes"]
        with np.load(d/"states.npz", allow_pickle=False) as arr:
            require(np.array_equal(cmds[i], arr["action_command"]) and
                    np.array_equal(np.arange(151)/30, arr["time_s"]) and not arr["observed_present"].any(),
                    f"Command, time or observation identity: {d}")
            pose = arr["object_xytheta"]
            delta = pose[0]-obj[i]
            require(np.max(np.abs(delta[:2])) < 1e-9 and abs(np.arctan2(np.sin(delta[2]),np.cos(delta[2])))<1e-9 and
                    np.allclose(robot[i],arr["executed_pusher_xy"][0],atol=1e-9,rtol=0), f"Initial state: {d}")
            require(int(arr["sim_contact"].sum()) == r["contact_steps"], f"Contact count: {d}")
            if v == "C2":
                paired[i] = pose
            if v == "C3":
                identical_c2_c3 += int(np.array_equal(pose,paired.pop(i)))
        with zipfile.ZipFile(d/"visuals.zip") as z:
            names=z.namelist()
            meta=json.loads(z.read("metadata.json"))
            require(len(meta["cameras"])==2 and meta["full_wh"]==[240,240] and meta["preview_wh"]==[96,96],
                    f"Camera geometry: {d}")
            require(len(set(names))==len(names)==(25*2*2+1+(150*2*2 if i in visual else 0)),f"Visual frames: {d}")
        measured[v] += 1
        contacts[v] += r["contact_steps"]
        failures[v].update(r["failure_reasons"])
        failed_episodes[v] += int(bool(r["failure_reasons"]))
    return {"paired_inputs":512,"episodes_per_version":dict(measured),"control_steps":150*len(rows),
            "full_visual_episodes_per_version":len(visual),"preview_frames_per_camera_per_episode":25,
            "verified_files":2*len(rows),"verified_file_bytes":sizes,"zero_new_observations":True,
            "failed_episodes":dict(failed_episodes),"failure_counts":failures,"contact_steps":dict(contacts),
            "C2_C3_bit_identical_object_trajectories":identical_c2_c3,"manifest":member(base/"manifest.json"),
            "bank":member(base/"initial_states_actions.npz")}


def heldout(root):
    split=read(root/"prepare/split.json")["roles"]
    out={}
    for mode in ("open_loop","causal_online"):
        report=root/"calibration"/(mode+".json")
        data=read(report)["results"]
        summary={}
        for v, conditions in data.items():
            for role, rows in conditions.items():
                require(sorted(split[role])==sorted(r["episode"] for r in rows),f"Replay population: {role}")
                for row in rows:
                    path=root/"calibration/replay"/mode/v/f"episode_{row['episode']:06d}.npz"
                    require(digest(path)==row["sha256"],str(path))
            target=conditions["test"]
            summary[v]={"episodes":len(target),"frames":sum(r["frames"] for r in target),
                        "valid":sum(r["valid"] for r in target),"reference":"shared frozen RGB tracker",
                        "mean_episode_position_m":float(np.mean([r["mean_position_m"] for r in target])),
                        "mean_episode_angle_rad":float(np.mean([r["mean_angle_rad"] for r in target]))}
        out[mode]={"results":summary,"source":member(report)}
    return out


def predictor(root):
    mf=root/"probe/manifest.json"
    registry=read(mf)
    require(registry["training_updates"]==300000 and len(registry["results"])==15,"Incomplete probe runs")
    result={}
    for entry in registry["results"]:
        path=Path(entry["path"])
        require(path.resolve().is_relative_to((root/"probe").resolve()) and digest(path)==entry["sha256"],"Probe receipt")
        r=read(path); condition=r["condition"];seed=r["seed"];aux=0 if condition=="RealOnly" else 1280000
        require(r["updates"]==20000 and r["shared_real_examples"]==1280000 and r["aux_examples"]==aux and
                r["checkpoint_sha256"]==digest(path.parent/"selected.pt"),f"Training identity: {path}")
        result.setdefault(condition,{})[seed]={"selected_step":r["selected_on_real_calibration_step"],
                      "heldout":{h:r["test"][f"{h}/all"] for h in ("0.5s","1s","2s")}}
    require(all(set(seeds)=={42,43,44} for seeds in result.values()) and len(result)==5,"Probe matrix")
    base=read(root/"probe/baselines.json")
    return {"conditions":result,"updates_each":20000,"shared_real_examples_each":1280000,"total_updates":300000,
            "additional_examples_per_augmented_run":1280000,"test_windows":base["test_windows"],
            "baselines":{k:{h:base[k][f"{h}/all"] for h in ("0.5s","1s","2s")} for k in ("persistence","constant_velocity")},
            "manifest":member(mf)}


def rope(root):
    result={}
    for case in ("single_push_rope_1","single_push_rope_4"):
        d=root/"phystwin"/case
        found={}
        for tag in ("G-NN","G-LingBot","G-Q1"):
            f=d/(tag+"_metrics.json")
            if not f.is_file():
                found[tag]={"completed":False}
                continue
            row=read(f);future=row["track_and_external_geometry"]
            require(row["horizon_s"]==.5 and row["revision_count"]==1 and len(future)==16 and
                    digest(d/(tag+"_prediction.npy"))==row["prediction_sha256"],f"Rope prediction identity: {f}")
            endpoint=future[-1]
            found[tag]={"completed":True,"initial_frame":future[0]["frame"],"final_frame":endpoint["frame"],
                        "horizon_s":row["horizon_s"],"revisions":row["revision_count"],"endpoint":endpoint,
                        "track_points_over_all_16_timestamps":sum(x["valid_tracks"] for x in future),
                        "prior_identity":row["shared_prior_sha256"],"accepted_hidden_pixels":row["accepted_hidden_object_pixels"],
                        "source":member(f)}
        require(len({v["prior_identity"] for v in found.values() if v["completed"]})<=1,"Different fixed physics priors")
        result[case]=found
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("--rope-only",action="store_true")
    args=cli.parse_args()
    p=args.root.resolve()
    write_json(p/"phystwin/results_summary.json",rope(p))
    if not args.rope_only:
        videos=read(p/"media/manifest.json")
        for row in [*videos["full_episodes"],videos["heldout_preview"]]:
            f=Path(row["path"])
            require(f.resolve().is_relative_to((p/"media").resolve()) and f.stat().st_size==row["bytes"] and digest(f)==row["sha256"],"Playback file identity")
        tr=read(p/"track_v2/tracking_all.json")
        write_json(p/"pusht_result_summary.json",{
            "role":"held-out object response conditioned on pusher command; independent RGB tracker reference",
            "independent_tracks":{"total_frames":tr["frames"],"valid_frames":tr["valid"],"episodes":len(tr["episodes"])},
            "replay":heldout(p),"generation":generation(p),"probe":predictor(p),
            "media":{"full_episodes":len(videos["full_episodes"]),"preview":videos["heldout_preview"],"manifest":member(p/"media/manifest.json")},
            "static_C3_shape":"Shared known T-block and C2 physical coefficients; no persistent geometry revision was identified.",
            "code_sha256":digest(Path(__file__))})
