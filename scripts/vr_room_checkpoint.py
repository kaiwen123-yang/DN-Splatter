"""Load this run's checkpoints with the NumPy scalars saved by Nerfstudio 1.1.3."""
import os
from pathlib import Path
import sys

import numpy as np
import torch


def numpy_scalar_scope():
    return torch.serialization.safe_globals(
        [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]
    )


def project_checkpoint(path):
    path = Path(path).resolve(strict=True)
    path.relative_to((Path(os.environ["DN_RUN_ROOT"])/"training").resolve())
    if path.parent.name != "nerfstudio_models" or path.suffix != ".ckpt":
        raise ValueError(f"Unexpected training checkpoint: {path.name}")
    return path


def read_training_state(path):
    path = project_checkpoint(path)
    with numpy_scalar_scope():
        return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def run_official_cli(checkpoint, command, args):
    from importlib.metadata import distribution
    from unittest.mock import patch

    package = {"ns-train": "nerfstudio", "ns-eval": "nerfstudio", "gs-mesh": "dn-splatter"}[command]
    entry, = distribution(package).entry_points.select(group="console_scripts", name=command)
    checkpoint = project_checkpoint(checkpoint)
    original_load = torch.load

    def load(file, *positional, **options):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == checkpoint:
            with numpy_scalar_scope():
                options["weights_only"] = True
                return original_load(file, *positional, **options)
        return original_load(file, *positional, **options)

    sys.argv = [command, *args]
    with patch.object(torch, "load", load):
        return entry.load()()


if __name__ == "__main__":
    sys.exit(run_official_cli(sys.argv[1], sys.argv[2], sys.argv[3:]))
