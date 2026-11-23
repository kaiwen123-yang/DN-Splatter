"""Bound decoded training frames in memory while preserving their original order."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os


def prefetch(function, items):
    budget=min(int(os.environ["DN_CPU_BUDGET"]),len(os.sched_getaffinity(0)))
    workers=max(1,budget-1)
    print(f"{function.__name__}: {workers} CPU workers and one ordered consumer",flush=True)
    iterator=iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque(pool.submit(function,item) for item in islice(iterator,workers))
        while pending:
            yield pending.popleft().result()
            for item in islice(iterator,1):
                pending.append(pool.submit(function,item))
