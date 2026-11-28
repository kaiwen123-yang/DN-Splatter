"""Summarize observed training throughput and the requested noon planning target."""
import argparse
from datetime import datetime,timedelta
import hashlib
import json
import os
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/".local/runs/interactive-world-20261128-v1"
HALL=("R0C","R1","R2","R3","R4","R5")


def read(file):
    return json.loads(file.read_bytes())


def timestamp(row):
    return datetime.fromisoformat(row["host"]["time"])


def interval(rows):
    if len(rows)<2:
        return None
    start,end=rows[0],rows[-1]
    seconds=(timestamp(end)-timestamp(start)).total_seconds()
    if seconds<=0:
        return None
    updates={k:end["observed_updates"][k]-start["observed_updates"][k] for k in HALL}
    return dict(start=start["host"]["time"],end=end["host"]["time"],samples=len(rows),seconds=seconds,
                net_updates=updates,hall_updates_per_hour=round(3600*sum(updates.values())/seconds,1),
                mean_gpu_util_percent=round(sum(r["host"]["gpu_util_percent"] for r in rows)/len(rows),2))


def receipt(path):
    with path.open("rb") as file:
        h=hashlib.file_digest(file,"sha256").hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h)


def main(target,verbose=True):
    if target.tzinfo is None:
        raise ValueError("Specify a timestamp with its UTC offset")
    rows=[json.loads(line) for line in (RUN/"state/throughput.jsonl").read_text().splitlines() if line]
    events=[json.loads(line) for line in (RUN/"state/coordinator-events.jsonl").read_text().splitlines() if line]
    launch=min(datetime.fromisoformat(r["time"]) for r in events if r["stage"]=="launch" and r["details"].get("variant")=="R2")
    end=max((RUN/"f_vr/VR2/evaluation/complete.json").stat().st_mtime,(RUN/"b_map/pose/accepted.json").stat().st_mtime)
    two=[r for r in rows if timestamp(r).timestamp()>=end and timestamp(r)<launch and set(r["active"])=={"R0C","R1"}]
    # Adjacent raw samples are required: do not bridge a job restart or a missing cohort.
    three=[]
    for row in rows:
        if timestamp(row)>=launch and set(row["active"])=={"R0C","R1","R2"} and row["observed_updates"]["R2"]>=100:
            three.append(row)
        elif three:
            break
    last=timestamp(rows[-1])
    remaining=sum(240000-rows[-1]["observed_updates"][k] for k in HALL)
    hours=(target-last).total_seconds()/3600
    recent={f"last_{m}_minutes":interval([r for r in rows if timestamp(r)>=last-timedelta(minutes=m)]) for m in (10,30,60)}
    core=RUN/"f_evaluation/acceleration_20261128"
    evidence={}
    files=[core/"before_admission.json",core/"after_admission_replay.json"]
    for pat in ("f_evaluation/acceleration/*.json","f_evaluation/eval_performance/*.json"):
        files.extend(sorted(RUN.glob(pat)))
    for p in files:
        if p.is_file():
            evidence[str(p.relative_to(RUN))]=receipt(p)
    result=dict(observed_at=last.isoformat(),run_id=RUN.name,planning_target=target.isoformat(),
                total_hall_updates=1440000,remaining_hall_updates=remaining,
                observed_updates=rows[-1]["observed_updates"],clean_two_hall=interval(two),
                initial_three_hall=interval(three),three_after_first_growth=interval([r for r in three if r['observed_updates']['R2']>=5600]),
                rolling=recent,required_hall_updates_per_hour_with_tail={
                    f"{tail}h":round(remaining/(hours-tail),1) if hours>tail else None for tail in (0,1,2,3)},
                coverage="Application progress is sampled every 100 completed updates. Windows include actual contention and validations.",
                evidence=evidence)
    output=core/"speed_report.json";output.parent.mkdir(exist_ok=True,parents=True)
    temp=output.with_name(output.name+f".{os.getpid()}.pending"); temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(output)
    if verbose:
        print(json.dumps({k:result[k] for k in ("observed_at","remaining_hall_updates","clean_two_hall","initial_three_hall",
                                               "three_after_first_growth","required_hall_updates_per_hour_with_tail")}))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planning-target",type=datetime.fromisoformat,required=True)
    main(p.parse_args().planning_target)
