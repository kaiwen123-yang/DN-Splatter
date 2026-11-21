"""Frozen-checkpoint B2 evaluation and validation-based model selection."""
from pathlib import Path
import json
import os
import sys
import hashlib
import numpy as np

ROOT = Path(os.environ.get('DN_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
RUN = Path(os.environ.get('DN_RUN_OVERRIDE', ROOT / '.local/runs/capture-full-quality-v2-20261121-v1')).resolve()
SCENE = Path(os.environ.get('DN_SCENE_OVERRIDE', RUN / 'inputs')).resolve()
EVAL = RUN / 'evaluation'
FINAL = 239999
NODES = [n - 1 for n in range(30000, 240001, 30000)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(config: Path, step: int, split: str):
    """Return the native eval_setup tuple for the requested frozen checkpoint."""
    from nerfstudio.utils.eval_utils import eval_setup
    from vr_room_checkpoint import numpy_scalar_scope
    config = Path(config).resolve()
    if split not in ('val', 'test', 'train'):
        raise ValueError(split)
    def bind(cfg):
        if Path(cfg.pipeline.datamanager.dataparser.data).resolve() != SCENE:
            raise ValueError('config input identity mismatch')
        cfg.load_step = step
        return cfg
    with numpy_scalar_scope():
        result = eval_setup(config, test_mode=split, update_config_callback=bind)
    if result[3] != step or not Path(result[2]).resolve().is_relative_to(RUN):
        raise ValueError('checkpoint identity mismatch')
    return result


def stats(reference, predicted, alpha, mask):
    n = int(mask.sum())
    valid = mask & np.isfinite(predicted) & (predicted > 0)
    signed = predicted[valid].astype(np.float64) - reference[valid].astype(np.float64)
    absolute = np.abs(signed)
    count = len(signed)
    return dict(reference_pixels=n, invalid_prediction_pixels=n-count,
                absolute_error_sum=float(absolute.sum()), signed_error_sum=float(signed.sum()),
                squared_error_sum=float(np.dot(signed,signed)),relative_error_sum=float((absolute/reference[valid]).sum()),
                positive_error_sum=float(signed[signed>0].sum()),negative_error_sum=float(signed[signed<0].sum()),
                MAE_m=float(absolute.mean()) if count and count == n else None,
                conditional_MAE_m=float(absolute.mean()) if count else None,
                alpha_zero=int((mask & (alpha == 0)).sum()),
                alpha_ge_05=int((mask & (alpha >= .5)).sum()),
                correct_5cm=int((absolute <= .05).sum()),
                correct_10cm=int((absolute <= .1).sum()),
                correct_20cm=int((absolute <= .2).sum()),
                foreground_conflict=int((signed < -np.maximum(.05, .03 * reference[valid])).sum()),
                positive_error=int((signed > 0).sum()), negative_error=int((signed < 0).sum()))


def aggregate(rows):
    keys = ('reference_pixels', 'invalid_prediction_pixels', 'absolute_error_sum', 'signed_error_sum',
            'alpha_zero', 'alpha_ge_05', 'correct_5cm', 'correct_10cm', 'correct_20cm',
            'foreground_conflict', 'positive_error', 'negative_error', 'squared_error_sum','relative_error_sum',
            'positive_error_sum','negative_error_sum')
    result = {k: sum(row[k] for row in rows) for k in keys}
    n = result['reference_pixels']
    result['MAE_m'] = result['absolute_error_sum'] / n if n and not result['invalid_prediction_pixels'] else None
    result['RMSE_m'] = float(np.sqrt(result['squared_error_sum']/n)) if n and not result['invalid_prediction_pixels'] else None
    result['AbsRel'] = result['relative_error_sum']/n if n and not result['invalid_prediction_pixels'] else None
    result['frame_mean_MAE_m'] = (float(np.mean([r['MAE_m'] for r in rows if r['reference_pixels']]))
                                  if n and all(r['MAE_m'] is not None for r in rows if r['reference_pixels']) else None)
    for k in ('alpha_zero', 'alpha_ge_05', 'correct_5cm', 'correct_10cm', 'correct_20cm', 'foreground_conflict'):
        result[k + '_fraction'] = result[k] / n if n else None
    return result


def evaluate(config, step, split):
    _, pipe, checkpoint, _ = load_model(config, step, split)
    return evaluate_fixed(config,step,split,pipe,checkpoint)


def evaluate_fixed(config,step,split,pipe,checkpoint):
    """The training callback evaluates the just-saved, immutable parameter state before any next update."""
    import torch
    meta = read(SCENE / 'transforms.json')
    names = meta[split + '_filenames']
    ds = pipe.datamanager.eval_dataset
    if len(names) != 184 or [p.relative_to(SCENE).as_posix() for p in ds.image_filenames] != names:
        raise ValueError('holdout membership mismatch')
    from capture_metric3d_common import REPORT as b1_report, selection as full_selection
    subset=read(b1_report/'comparison_s01_same_sources.json')[split]['frames']
    same={r['id']:r for r in subset}
    seq={r['id']:r for r in full_selection()}
    if len(seq)!=1846 or len(same)!=24 or any(str(seq[r['id']]['actual_rgb_timestamp_ns'])!=str(r['timestamp'])
                                               for r in subset):
        raise ValueError('frozen same-source timestamps')
    times={k:r['relative_s'] for k,r in seq.items()}
    rows = []
    abs_errors=[]
    with torch.no_grad():
        for i, name in enumerate(names):
            stem = Path(name).stem
            batch = ds.get_data(i)
            out = pipe.model.get_outputs_for_camera(camera=ds.cameras[i:i+1].to(pipe.device))
            pred = out['rgb'].reshape(720, 1280, 3).permute(2, 0, 1)[None]
            gt = batch['image'].to(pred.device).permute(2, 0, 1)[None]
            rgb = {k: float(getattr(pipe.model, k)(gt, pred)) for k in ('psnr', 'ssim', 'lpips')}
            ref = np.load(SCENE / 'depth' / (stem + '.npy'))
            z = out['depth'].detach().cpu().numpy().reshape(ref.shape)
            alpha = out['accumulation'].detach().cpu().numpy().reshape(ref.shape)
            mask = np.isfinite(ref) & (ref >= .3) & (ref <= 6)
            valid=mask & np.isfinite(z) & (z>0)
            abs_errors.append(np.abs(z[valid].astype(np.float64)-ref[valid].astype(np.float64)))
            distances = {}
            for lo, hi in ((.3, 1), (1, 2), (2, 4), (4, 6)):
                distances[f'{lo}-{hi}'] = stats(ref, z, alpha, mask & (ref >= lo) & ((ref <= hi) if hi == 6 else (ref < hi)))
            rows.append(dict(id=stem, rgb=rgb, relative_s=times[stem], same_s01_source=stem in same,
                             depth=stats(ref, z, alpha, mask), distance=distances))
            print(f'{split} step={step} {i+1}/{len(names)}', flush=True)
    rgb = {k: float(np.mean([r['rgb'][k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
    by_time = {}
    for r in rows:
        key = str(int(r['relative_s'] // 30) * 30) if r['relative_s'] is not None else 'unknown'
        by_time.setdefault(key, []).append(r['depth'])
    result = dict(split=split, step=step, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  config=str(Path(config).resolve()), config_sha256=digest(config), frame_count=len(rows),
                  rgb_frame_means=rgb, depth_pixel_pooled=aggregate([r['depth'] for r in rows]),
                  distance_strata={key: aggregate([r['distance'][key] for r in rows]) for key in rows[0]['distance']},
                  time_strata={key: aggregate(value) for key, value in by_time.items()},
                  same_s01_source=aggregate([r['depth'] for r in rows if r['same_s01_source']]), frames=rows)
    def mean_rgb(items):
        return {k:float(np.mean([r['rgb'][k] for r in items])) for k in rgb}
    result['same_s01_source']['rgb_frame_means']=mean_rgb([r for r in rows if r['same_s01_source']])
    if result['depth_pixel_pooled']['MAE_m'] is not None:
        q=np.percentile(np.concatenate(abs_errors),[50,95])
        result['depth_pixel_pooled'].update(absolute_error_P50_m=float(q[0]),absolute_error_P95_m=float(q[1]))
    for key in by_time:
        result['time_strata'][key]['rgb_frame_means']=mean_rgb([r for r in rows if int(r['relative_s']//30)*30==int(key)])
    write(EVAL / split / f'step-{step:09d}.json', result)
    return result


def select(config):
    baseline = read(ROOT / 'reports/capture-full-metric3d-30k/20261120-v1/metrics_val.json')['rgb_frame_means']
    rows = []
    for step in NODES:
        m = read(EVAL / 'val' / f'step-{step:09d}.json')
        rgb = m['rgb_frame_means']
        mae = m['depth_pixel_pooled']['MAE_m']
        eligible = mae is not None and rgb['psnr'] >= baseline['psnr'] and rgb['ssim'] >= baseline['ssim'] and rgb['lpips'] <= baseline['lpips']
        rows.append(dict(step=step, rgb=rgb, depth_MAE_m=mae, eligible=eligible))
    candidates = [r for r in rows if r['eligible']]
    selected = min(candidates, key=lambda r: (r['depth_MAE_m'], -r['rgb']['psnr'], r['step']))['step'] if candidates else FINAL
    result = dict(config=str(Path(config).resolve()), selected_step=selected, final_step=FINAL,
                  noninferiority_gate_passed=bool(candidates), baseline=baseline, candidates=rows)
    write(RUN / 'state/selection.json', result)
    return result


def main():
    if len(sys.argv) != 2 or sys.argv[1] != 'test':
        raise SystemExit('usage: capture_quality_eval.py test')
    selection = read(RUN / 'state/selection.json')
    for step in sorted({selection['selected_step'], selection['final_step']}):
        evaluate(Path(selection['config']), step, 'test')
    write(EVAL/'test_complete.json',dict(steps=sorted({selection['selected_step'],selection['final_step']}),frames_per_step=184))


if __name__ == '__main__':
    main()
