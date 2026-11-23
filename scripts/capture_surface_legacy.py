"""Run the unchanged B2 mesh exporters on a manifest-bound B3 checkpoint."""
import sys
from pathlib import Path
import capture_quality_tsdf as exporter
import capture_quality_eval as evaluator
from capture_surface_tiles import RUN,read


def main(action):
    sel=read(RUN/'state/selection.json')
    if not Path(sel['config']).resolve().is_relative_to(RUN):
        raise ValueError('B3 config path mismatch')
    exporter.RUN=RUN
    exporter.SCENE=evaluator.SCENE=RUN/'inputs'
    exporter.STATE=RUN/'state'
    exporter.GEOM=RUN/'geometry'
    exporter.MESH=RUN/'meshes'
    evaluator.RUN=RUN
    exporter.MESH.mkdir(exist_ok=True)
    if action=='model':
        exporter.model()
    elif action in ('sample','fuse','poisson','crop'):
        exporter.supported(action)
    else:
        raise ValueError(action)


if __name__=='__main__':
    main(sys.argv[1])
