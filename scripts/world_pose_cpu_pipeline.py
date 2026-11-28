"""Use available CPU to evaluate immutable learned correspondences during GPU matching."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import os
from pathlib import Path
import time

from world_pose_refine import OUT, solve


def work(shard,workers):
    import cv2
    cv2.setNumThreads(1)
    return solve(shard,workers,partial=True)


if __name__=="__main__":
    workers=int(os.environ.get("IW_POSE_CPU_WORKERS","4"))
    if not 1<=workers<=25:
        raise ValueError("Invalid worker count")
    from lingbot_common import save
    with (OUT/"cpu-pipeline.lock").open("a+b") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while not (OUT/"matches_complete.json").exists():
                states=list(pool.map(work,range(workers),[workers]*workers))
                done=sum(a for a,_ in states)
                total=sum(b for _,b in states)
                save(OUT/"cpu_pipeline.json",dict(ready_targets_solved=done,all_targets=total,workers=workers))
                if done==total:
                    break
                time.sleep(45)
