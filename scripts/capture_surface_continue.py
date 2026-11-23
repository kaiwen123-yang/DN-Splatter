"""Durable sequential continuation of A4, conditional B3 training and delivery."""
import json
import os
from pathlib import Path
import sys
import threading

from capture_surface_tiles import PLAN, ROOT, RUN, read, sha, write
from vr_room_run import CPU, monitor, stage, status
from capture_s01_resources import observe

STATE=RUN/'state'
B2=ROOT/'.local/runs'/PLAN['baseline']['run_id']


def main():
    if Path(os.environ['DN_RUN_ROOT'])!=RUN or Path(os.environ['DN_SCENE_DIR'])!=RUN/'inputs':
        raise ValueError('B3 run/scene isolation')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:CPU])
    evidence=read(STATE/'frozen_pair_val.json')
    for branch in ('model','sensor'):
        dest=RUN/'evaluation'/('A4_'+branch)
        if evidence[branch]['views']!=184 or evidence[branch]['metrics_sha256']!=sha(dest/'surfaces_val.json'):
            raise ValueError('invalid A4 receipt '+branch)
    binding=dict(run=RUN.name,protocol=sha(ROOT/'configs/capture_surface_quality_protocol.json'),
                 frozen_validation=sha(STATE/'frozen_pair_val.json'),pose=sha(RUN/'geometry/pose/diagnostics.json'))
    def phase(name, script, args, output, threads=1):
        marker=STATE/'continuation'/ (name+'.json')
        if marker.exists():
            old=read(marker)
            if old['binding']!=binding or old['sha256']!=sha(output):
                raise ValueError('changed completed stage '+name)
            return
        stage(name,[sys.executable,'-u',ROOT/'scripts'/script,*args],cwd=ROOT,threads=threads)
        write(marker,dict(binding=binding,path=str(output.relative_to(RUN)),sha256=sha(output)))
    phase('B_texture','capture_surface_texture.py',[],RUN/'texture/B2/receipt.json')
    phase('B_weights','capture_surface_weights.py',['B2','--device','cuda','--chunk','65536'],
          RUN/'weights/B2/receipt.json')
    # The actual pose solve and all 1,478 weight files precede this decision.
    phase('B_inputs','capture_surface_inputs.py',[],STATE/'training_decision.json')
    decision=read(STATE/'training_decision.json')
    if decision['decision']=='train':
        phase('C_training','capture_surface_train.py',['--train','--training-manifest',RUN/'inputs/training_manifest.json'],
              STATE/'selection.json',2)
        for action in ('model','sample','fuse','poisson','crop'):
            target=RUN/'meshes'/('model_tsdf.json' if action=='model' else 'supported/'+action+'.json')
            phase('C_'+action,'capture_surface_legacy.py',[action],target, CPU if action=='poisson' else 2)
        exp=RUN/'export/B3'
        for label,script,args,output in [
            ('submaps','submaps',['B3'],'submaps.json'),('repeatability','repeatability',['B3'],'tolerances.json'),
            ('index','tiles',['B3'],'tiles/complete.json'),('votes','layers',['B3','model'],'layers_model/complete.json'),
            ('assembly','assemble',['B3','model'],'fusion_model/complete.json')]:
            phase('C_'+label,'capture_surface_'+script+'.py',args,exp/output)
        phase('C_mesh_validation','capture_surface_final.py',['val'],RUN/'evaluation/C_val/complete.json',CPU)
    elif decision['decision']!='skipped_equivalent':
        raise ValueError('unresolved training decision')
    phase('selection_freeze','capture_surface_final.py',['freeze'],STATE/'selection_freeze.json')
    phase('development_test','capture_surface_final.py',['test'],RUN/'evaluation/development_test.json',CPU)
    phase('video','capture_surface_video.py',[],RUN/'video/receipt.json',CPU)
    phase('report','capture_surface_report.py',[],RUN/'delivery/spec.json')
    status('artifacts_ready',run_id=RUN.name,report=str(RUN/'delivery/spec.json'))


if __name__=='__main__':
    stop=threading.Event()
    for fn in (monitor,observe):
        threading.Thread(target=fn,args=(stop,),daemon=True).start()
    try:
        main()
    except Exception as exc:
        current=read(STATE/'current.json')
        status('failed',failed_stage=current.get('stage',current.get('phase')),reason=str(exc))
        raise
    finally:
        stop.set()
