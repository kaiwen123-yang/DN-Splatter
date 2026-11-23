"""Build the small B3 result tables and inventory from measured receipts."""
import gzip
import json
from pathlib import Path
import shutil

from capture_surface_final import B2,ROOT,RUN,STATE,EV,read,write,sha


def main():
    report=ROOT/'reports/capture-full-quality-v3/20261122-v1'
    report.mkdir(exist_ok=True,parents=True)
    f=read(STATE/'selection_freeze.json'); dec=read(STATE/'training_decision.json')
    test=read(EV/'development_test.json'); video=read(RUN/'video/receipt.json')
    if not test['freeze_sha256']==sha(STATE/'selection_freeze.json')==video['freeze_sha256']:
        raise ValueError('report identity')
    files=set(); refs=set()
    def include(path):
        p=Path(path).resolve()
        name=str(p.relative_to(ROOT))
        (files if p.is_relative_to(RUN) else refs).add(name)
        if not p.is_relative_to(RUN) and not p.is_relative_to(B2):
            raise ValueError('unexpected artifact root '+name)
    def include_model(label):
        cfg=Path(f['configs'][label])
        include(cfg)
        for step in sorted(set(f['checkpoint_steps'][label])):
            include(cfg.parent/'nerfstudio_models'/f'step-{step:09d}.ckpt')
    include_model('B2')
    for p in f['mesh_paths'].values(): include(p)
    for name in ('sensor_tsdf.ply','model_tsdf.ply'): include(B2/'meshes'/name)
    for name in ('surfaces_test.json','surfaces_val.json'): include(B2/'evaluation'/name)
    for path in [STATE/'selection_freeze.json',STATE/'training_decision.json',STATE/'frozen_pair_val.json',
                 RUN/'geometry/pose/transforms.json',RUN/'geometry/pose/diagnostics.json',RUN/'geometry/pose/decision.json',EV/'development_test.json',
                 RUN/'texture/B2/manifest.json',RUN/'texture/B2/receipt.json',RUN/'weights/B2/receipt.json',
                 RUN/'weights/B2/native_equivalence.json',RUN/'video/receipt.json',RUN/'video/timestamps.json']:
        include(path)
    for branch in ('model','sensor'):
        for n in ('complete.json','global.ply'):include(RUN/'export/B2'/('fusion_'+branch)/n)
    include(B2/'video/video_manifest.json')
    for movie in video['movies'].values():
        include(ROOT/movie['video']['path'])
        if 'timeline' in movie: include(ROOT/movie['timeline']['path'])
    for loc in video['fixed_views'].values():
        p=ROOT/loc['image']['path']; include(p)
        (report/'previews').mkdir(exist_ok=True)
        shutil.copy2(p,report/'previews'/p.name)
    rgb={}
    for label in ('B2','B3') if f['trained'] else ('B2',):
        run=B2 if label=='B2' else RUN
        entries={}
        for step in sorted(set(f['checkpoint_steps'][label])):
            for split in ('val','test'):
                path=run/'evaluation'/split/f'step-{step:09d}.json'; m=read(path);include(path)
                entries[f'{step+1}_{split}']=dict(views=m['frame_count'],rgb=m['rgb_frame_means'],
                    depth=m['depth_pixel_pooled'],range=m['distance_strata'],time=m['time_strata'],
                    same_source=m['same_s01_source'])
        rgb[label]=entries
    if f['trained']:
        include_model('B3')
        for p in (RUN/'inputs/training_manifest.json',RUN/'inputs/transforms.json',RUN/'inputs/geometry.json',
                  RUN/'inputs/init_train.ply',STATE/'selection.json',RUN/'training_complete.json',
                  STATE/'trainer/first_batch_000000000.json',STATE/'trainer/first_refinement.json',
                  STATE/'trainer/refinement.jsonl',STATE/'trainer/configuration.diff',STATE/'trainer/loaded_sources.json'):
            include(p)
        for p in (EV/'val').glob('step-*.json'): include(p)
        if dec['depth_weights_enabled']:
            include(STATE/'trainer/first_weight_batch.json')
            include(STATE/'trainer/weight_equivalence.json')
    metrics={}; deltas={}; difficult={}
    for path in sorted(EV.glob('*/surfaces_*.json')):
        m=read(path); split=m['split']; frame_path=path.parent/f'surface_frames_{split}.json'
        rows=read(frame_path); include(path); include(frame_path)
        if len(rows)!=184: raise ValueError('incomplete surface rows')
        for aux in ('native_probes.json','surface_local.json'):
            if (path.parent/aux).is_file():include(path.parent/aux)
        key=path.parent.name
        metrics[key]=dict(frames=184,groups=m['groups'],paired=m['paired_10cm'])
        for variant,totals in m['paired_10cm'].items():
            pairs=[(r['id'],r['paired'][variant]['incorrect_to_correct']-r['paired'][variant]['correct_to_incorrect'])
                   for r in rows]
            deltas[key+'/'+variant]=dict(**totals,improved_frames=sum(v>0 for _,v in pairs),
                            degraded_frames=sum(v<0 for _,v in pairs),unchanged_frames=sum(v==0 for _,v in pairs))
            difficult[key+'/'+variant]=dict(lowest_correct10=sorted(
                 ((r['id'],r['surfaces'][variant]['correct_10_fraction_all_reference'],r['reference_pixels']) for r in rows
                  if r['reference_pixels']),key=lambda x:x[1])[:5],largest_degradation=sorted(pairs,key=lambda x:x[1])[:5])
    def copy_result(rel,p):
        out=report/rel; out.parent.mkdir(exist_ok=True,parents=True)
        if out.suffix=='.gz':
            out.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
        else: shutil.copyfile(p,out)
    for name,p in [('selection_freeze.json',STATE/'selection_freeze.json'),('training_decision.json',STATE/'training_decision.json'),
                   ('texture.json.gz',RUN/'texture/B2/receipt.json'),('weights.json.gz',RUN/'weights/B2/receipt.json'),
                   ('video.json',RUN/'video/receipt.json'),('sensor_mesh.json',RUN/'export/B2/fusion_sensor/complete.json')]:
        copy_result(name,p)
    for path in sorted(EV.glob('*/surface_frames_*.json')):
        copy_result(f'paired/{path.parent.name}_{path.stem}.json.gz',path)
    for path in [*EV.glob('*/surface_local.json'),*EV.glob('*/native_probes.json')]:
        copy_result(f'paired/{path.parent.name}_{path.name}',path)
    write(report/'metrics.json',dict(rgb=rgb,surfaces=metrics,paired_changes=deltas,difficult=difficult,
           unchanged_pose_subset=dict(views=184,why='all B3 cameras byte-identical to B2',same_as='all'),
           development_reference='finite measured sensor Z in [0.3,6] m'))
    mesh=f['mesh_selection']; win=mesh['selected']
    text=['# 大厅 B3 完整结果','',f'验证集选出的 RGB 模型为 **{f["rgb_model"]}**，模型网格为 **{win}**。',
          '开发测试沿用 184 个相机和 29,100,252 个有效参考像素。深度误差参照同次采集传感器光轴 Z。','',
          '| 冻结 B2，开发测试 | 5 cm 正确覆盖 | 10 cm 正确覆盖 | 20 cm 正确覆盖 | 命中 | 前方冲突 | 命中 MAE |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    frozen=metrics['test_new_model_surface']['groups']['all']
    for name,m in frozen.items():
        text.append('| '+name+' | '+' | '.join([f"{m[k+'_fraction_all_reference']:.2%}" for k in
               ('correct_05','correct_10','correct_20','hits','foreground')])+f" | {m['MAE_hit_m']:.4f} m |")
    d=deltas['test_new_model_surface/new_model_surface']
    text+=['',f"新成面使 {d['incorrect_to_correct']:,} 个像素转为 10 cm 内正确，另有 {d['correct_to_incorrect']:,} 个原本正确的像素退化。"
             f"逐帧净改善 {d['improved_frames']} 张、净退化 {d['degraded_frames']} 张、相同 {d['unchanged_frames']} 张。",
           '',f"训练决定为 {dec['decision']}。接受位姿修改 {dec['accepted_pose_changes']} 项，启用深度权重 {dec['depth_weights_enabled']}，额外纹理细分 {dec['texture_split_enabled']}。",
           '', '完整距离档、30 秒时间段、同源子集、共同命中误差和困难相机见 [metrics.json](metrics.json)。'
           '逐帧记录见 [paired](paired)，四处同相机裁剪见 [previews](previews)。所有网格、视频和模型的服务器路径、字节数与 SHA-256 见 [artifacts.json](artifacts.json)。']
    spec=dict(run_id=RUN.name,training_updates=240000 if f['trained'] else 0,
              training_decision='trained' if f['trained'] else 'skipped_equivalent',
              training_switches=[dict(kind=k,evidence=v) for k,v in dec['diagnostic_receipts'].items()],
              selected_step=f['checkpoint_steps']['B3' if f['trained'] else 'B2'][0],
              selection_frozen=True,test_complete=True,privacy_verified=video['privacy_verified'],timeline_frames=video['frames'],
              fixed_views_complete=len(video['fixed_views'])==(8 if f['trained'] else 4),val_frames=184,test_frames=184,
              artifacts=sorted(files),references=sorted(refs),overview='\n'.join(text))
    spec['small_reports']=['metrics.json','selection_freeze.json','training_decision.json','texture.json.gz',
        'weights.json.gz','video.json','sensor_mesh.json']
    spec['small_reports'] += [str(p.relative_to(report)) for p in sorted((report/'paired').glob('*'))]
    spec['small_reports'] += ['previews/'+Path(row['image']['path']).name for row in video['fixed_views'].values()]
    write(RUN/'delivery/spec.json',spec)


if __name__=='__main__':
    main()
