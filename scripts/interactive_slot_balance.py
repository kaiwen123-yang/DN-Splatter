"""Read-only decision for one checkpoint handoff between the R3 and R5 training jobs."""
from bisect import bisect_right
from datetime import datetime
from itertools import combinations, groupby, product
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np


HALL = ("R0C", "R1", "R2", "R4", "R3", "R5")
TOTAL = 240_000
SPLIT = 60_000
MIN_GAIN_SECONDS = 15 * 60


def read(path):
    return json.loads(Path(path).read_bytes())


def json_lines(path):
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for i,line in enumerate(lines)
            if line.strip() and (i != len(lines)-1 or line.endswith(b"\n"))]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def decision_inputs(run, cp, val):
    group = run/"c_dn/R3"
    check(not (group/"state/slot_handoff.json").exists() and
          not (group/"state/slot_balance_decision.json").exists(), "the R3 handoff was already considered")
    check(cp == read(group/"state/checkpoint.json") and cp["saved_step"] == SPLIT-1 and
          cp["completed_updates"] == SPLIT, "R3 has no matching durable 60k receipt")
    model = Path(cp["path"]).resolve()
    check(model.is_relative_to((group/"training").resolve()) and model.name == "step-000059999.ckpt" and
          model.stat().st_size == cp["bytes"] and sha(model) == cp["sha256"], "R3 checkpoint failed verification")
    check(val.resolve() == (group/"evaluation/val/step-000059999.json").resolve(), "unexpected validation path")
    obs = read(val)
    check(obs["split"] == "val" and obs["step"] == SPLIT-1 and obs["frame_count"] == len(obs["frames"]) == 184 and
          len({r["id"] for r in obs["frames"]}) == 184 and
          obs["depth_pixel_pooled"]["reference_pixels"] == 29_222_080 and
          obs["checkpoint_sha256"] == cp["sha256"] and Path(obs["checkpoint"]).resolve() == model,
          "full R3 validation does not identify this checkpoint")
    recovery = read(run/"f_evaluation/acceleration/recovery_readback.json")
    check(recovery.get("passed") is True and recovery.get("resume_seconds", 0) > 0 and
          recovery.get("optimizer_updates_executed") == 0 and recovery.get("saved_budget") == TOTAL and
          recovery.get("saved_step", -2)+1 == recovery.get("next_step") and
          recovery.get("next_frame") and recovery.get("next_frame")==recovery.get("observed_next_frame") and
          all(recovery.get(k) is True for k in ("grad_scaler_exact","boundary_state_exact","sampler_rng_exact")) and
          set(recovery.get("rng_exact",[]))=={"python","numpy","torch_cpu","torch_cuda_all"},
          "a timed verified recovery readback is required")
    q = read(run/"state/queue.json")
    check(0 <= time.time() - timestamp(q["host"]["time"]) <= 100, "coordinator observation is stale")
    state = q["training"]
    active = [k for k,r in state.items() if r["active"] and r.get("active_phase", r["phase"]) == "train"]
    check(len(active) == 3 and "R3" in active and all(k in HALL and state[k]["first_batch"] for k in active),
          "three formal hall training workers have not stabilized")
    check(not state["R5"]["active"] and not state["R5"]["log"] and not (run/"c_dn/R5/state/worker.pid").exists()
          and state["R5"]["verified_updates_lower_bound"] == 0 and not state["R5"]["training_complete"],
          "R5 already started")
    for f in ("restore/env.verified.json", "restore/hall.verified.json", "restore/metric.verified.json"):
        check((run/f).is_file(), "R5 restoration prerequisites are absent")
    packets = read(run/"a_depth/supervision/R3/manifest.json")
    check(packets["scope"] == "train" and len(packets["frames"]) == 1478 and packets["accepted_pixels"] > 0,
          "R5 lacks the frozen full R3 supervision packet")
    return state, active, recovery, sha(val)


def windows(rows):
    """Disjoint five-minute windows avoid counting 20-second counter quantization as speed."""
    out = []
    for keys, series in groupby(rows, lambda r: tuple(sorted(r["active"]))):
        if not keys or any(k not in HALL for k in keys):
            continue
        first = previous = None
        for row in series:
            t = timestamp(row["host"]["time"])
            if previous is None or t-previous > 90 or t <= previous:
                first = row
            previous = t
            elapsed = t-timestamp(first["host"]["time"])
            if elapsed < 300:
                continue
            source, target = first["observed_updates"], row["observed_updates"]
            rates = {k: (target[k]-source[k])*3600/elapsed for k in keys if target[k]-source[k] >= 200}
            if rates:
                out.append(dict(end=t, rates=rates, size={k:(target[k]+source[k])/2 for k in rates},
                                clean=len(rates) == len(keys)))
            first = row
    return out


def population(run, stage):
    tables = {}
    for k in HALL:
        path = run/"c_dn"/k/"state/refinement.jsonl"
        if path.exists():
            data = {r["step"]:r["after"] for r in json_lines(path) if r.get("after", 0)>0}
            if data:
                steps = sorted(data)
                tables[k] = steps, [data[s] for s in steps]
    donors = [k for k in ("R0C","R1","R2") if stage[k]["verified_updates_lower_bound"] >= 120_000 and
              k in tables and tables[k][0][-1] >= 119_200]
    check(donors and all(k in tables for k in HALL if k!="R5"), "growth history insufficient to normalize rates")
    def measured(k,step):
        steps,counts = tables[k]
        return counts[max(0,bisect_right(steps,step)-1)]
    def ref(step):
        return statistics.median(measured(k,step) for k in donors)
    def estimate(k,step):
        key = "R3" if k=="R5" else k
        last = tables[key][0][-1]
        if step <= last or last >= 119_200:
            return measured(key,step)
        return min(2_071_347,measured(key,last)*ref(step)/ref(last))
    return estimate, ref(240_000)


def rates_model(data, count, reference):
    # Paired workers share their measurement window. The nuisance speed of a
    # particular concurrent combination cancels from the log throughput ratio.
    known = ("R1","R2","R3","R4")
    def feature(k,step):
        return np.array([np.log(count(k,step)/reference),*(float(k==n) for n in known)])
    X, y = [], []
    for w in data:
        for i,j in combinations(w["rates"],2):
            X.append(feature(i,w["size"][i])-feature(j,w["size"][j]))
            y.append(np.log(w["rates"][j]/w["rates"][i]))
    check(len(y)>=20, "not enough paired training intervals")
    X,y = np.asarray(X),np.asarray(y)
    coef,_,rank,_ = np.linalg.lstsq(X,y,rcond=None)
    check(rank==X.shape[1], "paired measurements do not identify all four empirical variant costs")
    # Reweight unusually long setup/validation windows while preserving the
    # same fixed below/above-growth protocol and all matched variant pairs.
    for _ in range(3):
        err = X@coef-y
        scale = max(.025,1.4826*float(np.median(np.abs(err-np.median(err)))))
        weights = np.minimum(1.,2*scale/np.maximum(np.abs(err),1e-10))**.5
        coef = np.linalg.lstsq(X*weights[:,None],y*weights,rcond=None)[0]
    check(0<=coef[0]<=1 and np.isfinite(coef).all(), "population dependence unstable in observed windows")
    factors = dict(zip(known,np.exp(coef[1:]).tolist()))
    factors["R0C"] = 1.
    def unit(k,step):
        return (count(k,step)/reference)**coef[0]
    capacities = {1:[],2:[],3:[]}
    for w in data:
        n = len(w["rates"])
        if w["clean"] and n in capacities:
            capacities[n].append(sum(speed*unit(k,w["size"][k])*factors[k] for k,speed in w["rates"].items()))
    check(len(capacities[2])>=2 and len(capacities[3])>=3, "matched two/three-worker capacities unavailable")
    return unit,factors,capacities,float(coef[0]),len(y)


def phase_costs(run, rows, recovery):
    launches = [e for e in json_lines(run/"state/coordinator-events.jsonl") if e["stage"]=="launch"
                and e["details"].get("exit")==0 and e["details"].get("variant") in HALL]
    durations, startup, evaluating = {},[],[]
    for k in HALL:
        folder = run/"c_dn"/k
        log = [e for e in launches if e["details"]["variant"]==k]
        begins = [timestamp(e["time"]) for e in log if e["details"]["phase"]=="evaluate"]
        complete = folder/"evaluation/complete.json"
        if complete.is_file():
            v=read(complete)
            check(v["variant"]==k and v["artifacts"] and begins, "completed hall evaluation lacks a matching launch")
            # A retry can reuse already-exported material. Include the entire
            # first-launch-to-final-receipt wall span instead of timing only its tail.
            durations[k]=(complete.stat().st_mtime-min(begins))/3600
            check(durations[k]>0, "invalid full evaluation timing")
        elif rows[k]["training_complete"]:
            evaluating.append(max(0,time.time()-min(begins))/3600 if begins else 0.)
        first=folder/"state/first_batch_000000000.json"
        if first.is_file():
            t=timestamp(read(first)["resources"]["utc"])
            starts=[timestamp(e["time"]) for e in log if e["details"]["phase"]=="train" and timestamp(e["time"])<=t]
            if starts:
                startup.append(t-max(starts))
    check(durations, "no measured full hall evaluation with a completion receipt")
    # Allow two complete 20-second coordinator polls beyond measured setup.
    return durations, (40+max([float(recovery["resume_seconds"]),*startup]))/3600, evaluating


def makespan(progress,active,caps,cost,eval_hours,evaluating,slots,restore_hours,yield_slot):
    """Integrate piecewise work across measured aggregate capacities. Times are hours from now."""
    u = progress.copy()
    running=list(active)
    queue=[k for k in HALL if k not in running and u[k]<TOTAL]
    if yield_slot:
        running.remove("R3")
        queue.remove("R5")
        running.append("R5")
        queue.insert(0,"R3")
    completed=[]
    elapsed=0.
    while running:
        n=len(running)
        rates={k:caps[n]/n/cost(k,u[k]) for k in running}
        # Five-thousand update buckets follow the historical population curve.
        targets={k:min(TOTAL,5000*(int(round(u[k],8))//5000+1)) for k in running}
        horizon=min((targets[k]-u[k])/rates[k] for k in running)
        elapsed+=horizon
        for k in list(running):
            u[k]=min(targets[k],u[k]+rates[k]*horizon)
            if abs(u[k]-targets[k])<1e-6:
                u[k]=targets[k]
            if u[k]>=TOTAL:
                running.remove(k)
                completed.append(elapsed)
        while queue and len(running)<3:
            running.append(queue.pop(0))
    # All earlier unfinished evaluations also occupy the common export lanes.
    free=[0.]*slots
    for used in sorted(evaluating,reverse=True):
        i=free.index(min(free))
        free[i]+=max(0.,eval_hours-used)
    for t in completed:
        i=free.index(min(free))
        free[i]=max(free[i],t)+eval_hours
    # Charging the entire extra restoration against the alternative's final
    # completion is conservative even while the other two trainers progress.
    return max(free)+(restore_hours if yield_slot else 0.)


def should_yield(run,variant,step,checkpoint_receipt,val_path):
    """Return a recommendation only. Missing, contradictory or weak evidence keeps R3 continuous."""
    out=dict(yield_slot=False,step=step,variant=variant)
    if variant!="R3" or step!=SPLIT-1:
        return dict(out,reason="outside the single R3 60k decision")
    try:
        run=Path(run).resolve()
        check(run.name=="interactive-world-20261128-v1","unexpected run identity")
        state,active,recovery,val_hash=decision_inputs(run,checkpoint_receipt,Path(val_path))
        row=run/"state/throughput.jsonl"
        data=windows(json_lines(row))
        recent=[w for w in data if time.time()-w["end"]<=1800 and set(w["rates"])==set(active)]
        check(len(recent)>=2,"three current hall workers lack two recent five-minute advancing windows")
        count,reference=population(run,state)
        unit,factors,capacities,beta,pairs=rates_model(data,count,reference)
        durations,restore,evaluating=phase_costs(run,state,recovery)
        progress={k:int(state[k]["verified_updates_lower_bound"]) for k in HALL}
        progress["R3"]=SPLIT
        check(all(0<=v<=TOTAL for v in progress.values()),"invalid cumulative training counts")
        q=lambda k,p:float(np.quantile(capacities[k],p))
        if len(capacities[1])>=3:
            upper_single=1.1*q(1,.95)
            source="95th percentile of full singleton windows plus ten percent"
        else:
            upper_single=q(2,.95)
            source="95th percentile of the entire two-worker service envelope; no established solo rate"
        # R5 is R3 with normal TV disabled. Its speed has not been measured at
        # this decision. Test both the cheaper hall variant and a slower R3 bound.
        r5_bounds=sorted({min(factors["R2"],factors["R3"]),1.1*factors["R3"]})
        population_factors={k:[float(unit(k,s)) for s in range(0,TOTAL,5000)] for k in HALL}
        results=[]
        for q2,q3,r5,ev,slots in product((.25,.75),(.25,.75),r5_bounds,
                                        sorted({min(durations.values()),max(durations.values())}),(1,2)):
            caps={1:upper_single,2:q(2,q2),3:q(3,q3)}
            def cost(k,s):
                return population_factors[k][min(int(s)//5000,len(population_factors[k])-1)]*(r5 if k=="R5" else factors[k])
            args=(progress,active,caps,cost,ev,evaluating,slots,restore)
            fifo=makespan(*args,False)
            swap=makespan(*args,True)
            results.append(dict(two_worker_q=q2,three_worker_q=q3,r5_factor=r5,eval_hours=ev,
                                evaluation_lanes=slots,fifo_h=fifo,handoff_h=swap,
                                gain_seconds=(fifo-swap)*3600))
        gain=min(r["gain_seconds"] for r in results)
        out.update(yield_slot=bool(gain>=MIN_GAIN_SECONDS),
                   reason="all measured-capacity scenarios favor one handoff" if gain>=MIN_GAIN_SECONDS else
                          "the conservative final-delivery advantage is below fifteen minutes",
                   active=active,progress=progress,checkpoint_sha256=checkpoint_receipt["sha256"],
                   validation_sha256=val_hash,recovery_resume_seconds=recovery["resume_seconds"],
                   recovery_readback_sha256=sha(run/"f_evaluation/acceleration/recovery_readback.json"),
                   conservative_extra_start_seconds=restore*3600,first_hall_evaluation_hours=durations,
                   coordinator_poll_allowance_seconds=40,evaluation_elapsed_hours=evaluating,paired_rate_observations=pairs,
                   first_sample_window_end=data[0]["end"],last_sample_window_end=data[-1]["end"],
                   gaussian_cost_by_5000_updates=population_factors,
                   clean_windows={k:len(v) for k,v in capacities.items()},gaussian_exponent=beta,
                   reference_population=reference,relative_step_cost=factors,upper_single_rate=upper_single,
                   upper_single_source=source,two_worker_capacity_bounds=[q(2,.25),q(2,.75)],
                   three_worker_capacity_bounds=[q(3,.25),q(3,.75)],scenarios=results,
                   worst_gain_seconds=gain,best_gain_seconds=max(r["gain_seconds"] for r in results),
                   minimum_gain_seconds=MIN_GAIN_SECONDS)
    except (OSError,ValueError,TypeError,KeyError,IndexError,ArithmeticError,np.linalg.LinAlgError) as error:
        out.update(yield_slot=False,reason=f"evidence unavailable: {type(error).__name__}: {error}")
    return out
