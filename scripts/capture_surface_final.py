"""Freeze only on validation, then evaluate new products on all development frames."""
from pathlib import Path
import sys

from capture_surface_tiles import PLAN, ROOT, RUN, read, sha, write
from capture_surface_select import choose_mesh

B2=ROOT/'.local/runs'/PLAN['baseline']['run_id']
BASE=B2/'meshes/supported/supported.ply'
EV=RUN/'evaluation'
STATE=RUN/'state'


def measured(manifest,split):
    from capture_surface_evaluate import evaluate
    dest=Path(manifest['output'])
    marker=dest/(split+'_receipt.json')
    identity=dict(cameras=sha(Path(manifest['scene'])/'transforms.json'),
                  meshes={k:sha(p) for k,p in manifest['meshes'].items()})
    if split=='test':
        identity['freeze']=sha(manifest['freeze'])
    if marker.exists():
        m=read(marker)
        if m['identity']!=identity or not all(sha(dest/name)==d for name,d in m['outputs'].items()):
            raise ValueError('changed surface evaluation '+str(marker))
        return m
    path=dest/('manifest_'+split+'.json')
    write(path,manifest)
    evaluate(path,split)
    result=read(dest/('surfaces_'+split+'.json'))
    expected=PLAN['data']['val_reference_pixels' if split=='val' else 'test_reference_pixels']
    if result['frames']!=184 or any(r['reference']!=expected for r in result['groups']['all'].values()):
        raise ValueError('incomplete evaluation '+str(dest))
    names=['surfaces_'+split+'.json','surface_frames_'+split+'.json']
    m=dict(identity=identity,outputs={name:sha(dest/name) for name in names},frames=184,reference=expected)
    write(marker,m)
    return m


def c_meshes():
    return dict(B3_supported_poisson=str(RUN/'meshes/supported/supported.ply'),
                B3_model_tsdf=str(RUN/'meshes/model_tsdf.ply'),
                B3_new_model_surface=str(RUN/'export/B3/fusion_model/global.ply'))


def c_val():
    meshes=dict(supported_poisson=str(BASE),**c_meshes())
    dest=EV/'C_val'
    row=measured(dict(meshes=meshes,output=str(dest),scene=str(RUN/'inputs')),'val')
    write(dest/'complete.json',row)


def freeze():
    dec=read(STATE/'training_decision.json')
    trained=dec['decision']=='train'
    if not trained and dec['decision']!='skipped_equivalent':
        raise ValueError('training diagnosis')
    base=read(B2/'evaluation/surfaces_val.json')
    candidates={('B2',k):base['groups']['all'][k] for k in ('supported_poisson','model_tsdf')}
    a=read(EV/'A4_model/surfaces_val.json')
    candidates['B2','new_model_surface']=a['groups']['all']['new_model_surface']
    meshes=dict(supported_poisson=str(BASE),new_model_surface=str(RUN/'export/B2/fusion_model/global.ply'),
                new_sensor_surface=str(RUN/'export/B2/fusion_sensor/global.ply'))
    configs={'B2':read(B2/'state/selection.json')['config']}
    steps={'B2':[PLAN['baseline']['selected_step'],PLAN['baseline']['final_step']]}
    ck={}
    rgb='B2'
    if trained:
        sel=read(STATE/'selection.json')
        configs['B3']=sel['config']
        steps['B3']=[sel['selected_step'],sel['final_step']]
        rgb=sel['recommended_rendering_model']
        meshes.update(c_meshes())
        val=read(EV/'C_val/surfaces_val.json')
        for label in c_meshes():
            candidates['B3',label[3:]]=val['groups']['all'][label]
    for name in configs:
        ck[name]={str(step):sha(Path(configs[name]).parent/'nerfstudio_models'/f'step-{step:09d}.ckpt')
                  for step in set(steps[name])}
    selection=choose_mesh(candidates,trained)
    receipt=dict(run_id=RUN.name,status='frozen',training_decision_frozen=True,
                 checkpoint_selection_frozen=True,mesh_selection_frozen=True,thresholds_frozen=True,
                 protocol_sha256=sha(ROOT/'configs/capture_surface_quality_protocol.json'),
                 training_decision_sha256=sha(STATE/'training_decision.json'),trained=trained,
                 checkpoint_steps=steps,checkpoint_sha256=ck,configs=configs,rgb_model=rgb,
                 mesh_selection=selection,mesh_paths=meshes,mesh_sha256={k:sha(v) for k,v in meshes.items()})
    path=STATE/'selection_freeze.json'
    if path.exists() and read(path)!=receipt:
        raise ValueError('frozen choices changed')
    write(path,receipt)


def development_test():
    from capture_quality_eval import evaluate as rgb_eval
    freeze_path=STATE/'selection_freeze.json'
    f=read(freeze_path)
    if f['status']!='frozen' or f['training_decision_sha256']!=sha(STATE/'training_decision.json'):
        raise ValueError('freeze changed')
    if f['trained']:
        # Existing B3 decision accepted zero pose changes, so all cameras are paired.
        if sha(RUN/'inputs/transforms.json')!=sha(B2/'inputs/transforms.json'):
            raise ValueError('pair each candidate at its own corrected camera')
        for step in sorted(set(f['checkpoint_steps']['B3'])):
            path=EV/'test'/f'step-{step:09d}.json'
            if not path.exists():
                rgb_eval(Path(f['configs']['B3']),step,'test')
            data=read(path)
            if data['checkpoint_sha256']!=f['checkpoint_sha256']['B3'][str(step)] or data['frame_count']!=184 or data['depth_pixel_pooled']['reference_pixels']!=29100252:
                raise ValueError('development RGB/depth identity')
    # Load candidate BVHs one at a time; every new surface is evaluated once.
    pairs={}
    for name,mesh in f['mesh_paths'].items():
        if name=='supported_poisson':
            continue
        dest=EV/('test_'+name)
        subset=dict(supported_poisson=f['mesh_sha256']['supported_poisson'],**{name:f['mesh_sha256'][name]})
        pair_freeze=dest/'frozen_pair.json'
        write(pair_freeze,dict(f,mesh_sha256=subset,global_selection_sha256=sha(freeze_path)))
        manifest=dict(scene=str(B2/'inputs'),output=str(dest),freeze=str(pair_freeze),
                      meshes=dict(supported_poisson=str(BASE),**{name:mesh}))
        pairs[name]=measured(manifest,'test')
    write(EV/'development_test.json',dict(freeze_sha256=sha(freeze_path),pairs=pairs,frames=184,
          reference_pixels=29100252,unchanged_camera_views=184,baseline_results=sha(B2/'evaluation/surfaces_test.json')))


if __name__=='__main__':
    {'val':c_val,'freeze':freeze,'test':development_test}[sys.argv[1]]()
