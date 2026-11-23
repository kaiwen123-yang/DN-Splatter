"""Resident native-Z implementation of the training weight reference vote."""
import ctypes as ct
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from capture_surface_device import Evidence
from capture_surface_tiles import RUN, read, sha


class WeightVotes:
    def __init__(self, export):
        self.evidence = Evidence(export)
        groups = read(export / 'identity.json')['groups']
        self.index = {name: i for i, name in enumerate(
            n for group in groups for n in sorted(group['names']))}
        src = Path(__file__).with_name('capture_surface_weight_votes.cu')
        lib = RUN / 'control' / ('weight-votes-' + sha(src)[:16] + '.so')
        if not lib.exists():
            temp = lib.with_suffix('.pending.so')
            subprocess.run([str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc'), '-O3',
                            '-arch=sm_120', '--shared', '--fmad=false', '-Xcompiler',
                            '-fPIC', str(src), '-o', str(temp)], check=True)
            temp.replace(lib)
        self.lib = ct.CDLL(str(lib))
        p, i, d = ct.c_void_p, ct.c_int, ct.c_double
        self.lib.weight_votes.argtypes = [i,p,i,i,p,p,p,p,i,i,d,d,d,d,p,i,p,p]
        self.lib.weight_votes.restype = i

    def score(self, points, name, score_first=False):
        e = self.evidence
        p = torch.as_tensor(np.ascontiguousarray(points, dtype='f8'), device='cuda')
        counts = torch.empty((len(p), 3), dtype=torch.int32, device='cuda')
        errors = torch.empty(len(p), dtype=torch.float64, device='cuda')
        torch.cuda.synchronize()
        rc = self.lib.weight_votes(len(p), p.data_ptr(), self.index[name], len(self.index),
                 e.group.data_ptr(), e.cam.data_ptr(), e.depth.data_ptr(), e.edge.data_ptr(),
                 e.h, e.w, *[e.k[k] for k in ('fx','fy','cx','cy')], e.terms.data_ptr(),
                 int(score_first), counts.data_ptr(), errors.data_ptr())
        if rc:
            raise RuntimeError(f'native weight vote CUDA status {rc}')
        return counts.cpu().numpy(), errors.cpu().numpy()
