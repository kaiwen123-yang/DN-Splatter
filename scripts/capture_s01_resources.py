"""Observe child PIDs, RSS high-water marks and CPU time from Linux procfs."""
import fcntl
import json
import os
from pathlib import Path
import sys
import threading

from vr_room_run import STATE, now


def observe(stop, attached=False):
    with (STATE / "observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while not stop.is_set():
            try:
                state = json.loads((STATE / "current.json").read_text())
                if attached and state["stage"] in {"failed", "complete", "stage_complete"}:
                    break
                pid = state["child_pid"]
                root = Path("/proc") / str(pid)
                status = (root / "status").read_text().splitlines()
                fields = {
                    line.split(":")[0]: int(line.split()[1])
                    for line in status
                    if line.startswith(("VmHWM:", "VmRSS:", "Threads:"))
                }
                ticks = (root / "stat").read_text().rsplit(")", 1)[1].split()
                record = dict(
                    utc=now(),
                    stage=state["stage"],
                    runner_pid=state["pid"],
                    child_pid=pid,
                    cpu_seconds=(int(ticks[11]) + int(ticks[12])) / os.sysconf("SC_CLK_TCK"),
                    **fields
                )
                with (STATE / "process_resources.jsonl").open("a") as output:
                    print(json.dumps(record), file=output)
            except (FileNotFoundError, ProcessLookupError, KeyError, json.JSONDecodeError):
                pass
            stop.wait(2)


if __name__ == "__main__":
    observe(threading.Event(), attached=True)
