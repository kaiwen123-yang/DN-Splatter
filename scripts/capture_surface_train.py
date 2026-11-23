"""B3 training primitives; formal launcher is deliberately gated until inputs are frozen."""
from pathlib import Path
import json

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'configs/capture_surface_quality_protocol.json').read_text(encoding='utf-8'))
RUN = ROOT / '.local' / 'runs' / SPEC['run_id']


def training_decision(pose, weights, texture):
    """Require actual diagnostic results; missing evidence is never an equivalence decision."""
    if pose.get('training_decision') == 'incomplete_not_equivalent':
        raise ValueError('pose diagnostics are incomplete; training decision is unavailable')
    if texture.get('status') != 'diagnosed' or not texture.get('attribution', {}).get('runtime_output_keys'):
        raise ValueError('texture diagnostics are incomplete; training decision is unavailable')
    if 'depth_weights_enabled' not in weights or 'nonidentity_pixels' not in weights:
        raise ValueError('depth weight diagnostics are incomplete')
    accepted = pose['accepted_pose_changes']
    if not isinstance(accepted, (int, list)) or isinstance(accepted, bool):
        raise ValueError('accepted_pose_changes must be a count or list')
    count = accepted if isinstance(accepted, int) else len(accepted)
    if count < 0 or type(weights['depth_weights_enabled']) is not bool or type(texture['texture_split_enabled']) is not bool:
        raise ValueError('invalid training decision')
    if weights['depth_weights_enabled'] and weights['nonidentity_pixels'] <= 0:
        raise ValueError('enabled weights lack nonidentity pixels')
    changed = bool(count or weights['depth_weights_enabled'] or texture['texture_split_enabled'])
    return dict(accepted_pose_changes=count, depth_weights_enabled=weights['depth_weights_enabled'],
                texture_split_enabled=texture['texture_split_enabled'],
                decision='train' if changed else 'skipped_equivalent')


def weighted_edge_aware_log_l1(pred, gt, rgb, mask, weights, empty_counts=None):
    """Match B2's left/top directional pixel convention and independent denominators."""
    if pred.shape != gt.shape or weights.shape != pred.shape:
        raise ValueError('depth and weight shapes differ')
    if mask is not None and mask.shape != pred.shape:
        raise ValueError('mask shape differs')
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError('invalid depth weights')
    logl1 = torch.log(1 + torch.abs(pred - gt))
    dx = torch.mean(torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True)
    dy = torch.mean(torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True)
    terms = (torch.exp(-dx) * logl1[..., :, :-1, :], torch.exp(-dy) * logl1[..., :-1, :, :])
    ws = (weights[..., :, :-1, :], weights[..., :-1, :, :])
    masks = (mask[..., :, :-1, :], mask[..., :-1, :, :]) if mask is not None else (None, None)
    result = pred.sum() * 0
    for direction, values, w, m in zip(('x', 'y'), terms, ws, masks):
        if m is not None:
            values, w = values[m], w[m]
        denominator = w.sum()
        if not values.numel() or denominator.item() == 0:
            if empty_counts is not None:
                empty_counts[direction] = empty_counts.get(direction, 0) + 1
            result = result + values.sum() * 0
        else:
            result = result + (values * w).sum() / denominator
    return result


def check_all_one_equivalence(original):
    """Verify forward and prediction gradients against the installed B2 implementation."""
    generator = torch.Generator().manual_seed(42)
    pred = torch.rand((7, 9, 1), generator=generator, dtype=torch.float64, requires_grad=True)
    gt = torch.rand((7, 9, 1), generator=generator, dtype=torch.float64)
    rgb = torch.rand((7, 9, 3), generator=generator, dtype=torch.float64)
    mask = torch.rand((7, 9, 1), generator=generator) > .3
    expected = original(pred, gt, rgb, mask)
    actual = weighted_edge_aware_log_l1(pred, gt, rgb, mask, torch.ones_like(pred))
    g0, = torch.autograd.grad(expected, pred, retain_graph=True)
    g1, = torch.autograd.grad(actual, pred)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(g1, g0, rtol=1e-12, atol=1e-12)
    return dict(loss_abs_error=float((actual-expected).abs()), gradient_max_abs_error=float((g1-g0).abs().max()))


def validate_recovery_checkpoint(path, expected_manifest, target_updates=240000):
    """Inspect B2-compatible recovery state before computing a B3 resume budget.

    The input manifest binds the checkpoint to the B3 scene; an unbound B2
    checkpoint must never become the starting point for the new experiment.
    """
    from vr_room_checkpoint import numpy_scalar_scope
    with numpy_scalar_scope():
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a mapping")
    required = ("step", "pipeline", "optimizers", "schedulers", "scalers", "b2_recovery")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError("checkpoint missing recovery fields: " + ", ".join(missing))
    step = checkpoint["step"]
    if type(step) is not int or not 0 <= step < target_updates or (step + 1) % 10000:
        raise ValueError("invalid cumulative checkpoint step")
    recovery = checkpoint["b2_recovery"]
    if not isinstance(recovery, dict) or recovery.get("manifest") != expected_manifest:
        raise ValueError("checkpoint is not bound to the B3 input manifest")
    fields = ("sampler", "remaining", "perm", "rng", "growth")
    missing = [key for key in fields if key not in recovery]
    if missing:
        raise ValueError("checkpoint missing sampler or growth state: " + ", ".join(missing))
    if any(key not in recovery["growth"] for key in ("xys_grad_norm", "vis_counts", "max_2Dsize")):
        raise ValueError("checkpoint missing densification state")
    if not isinstance(recovery['sampler'], dict) or not {'epoch', 'order'} <= recovery['sampler'].keys():
        raise ValueError('checkpoint missing sampler epoch or order')
    if not isinstance(recovery['remaining'], (list, tuple)) or not isinstance(recovery['rng'], dict):
        raise ValueError('invalid sampler or RNG recovery state')
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= recovery['rng'].keys():
        raise ValueError('checkpoint missing RNG state')
    return {"saved_step": step, "completed_updates": step + 1,
            "remaining_updates": target_updates - (step + 1), "manifest": expected_manifest}


def run_formal(manifest_path):
    """Run the B2 trainer against a separately prepared, manifest-bound B3 scene.

    The input manifest must be produced only after the three training decisions
    and initialization are complete. No B2 weights are loaded for a fresh run.
    """
    import hashlib
    import os

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(RUN.resolve()):
        raise ValueError('B3 manifest must reside inside the B3 run')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('run_id') != SPEC['run_id']:
        raise ValueError('B3 run identity mismatch')
    decision = training_decision(manifest['pose'], manifest['weights'], manifest['texture'])
    if decision['decision'] != 'train':
        raise ValueError('formal training requires a measured nonidentity change')
    # Decisions must be tied to the diagnostic files, not merely copied into a
    # manifest. This also makes a later change to a receipt invalidate recovery.
    receipts = manifest.get('diagnostic_receipts', {})
    for kind in ('pose', 'weights', 'texture'):
        receipt = receipts.get(kind, {})
        path = (RUN / receipt.get('path', '')).resolve()
        if not path.is_relative_to(RUN.resolve()) or not path.is_file():
            raise ValueError('missing B3 diagnostic receipt: ' + kind)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != receipt.get('sha256'):
            raise ValueError('diagnostic receipt hash mismatch: ' + kind)
        if json.loads(path.read_text(encoding='utf-8')) != manifest[kind]:
            raise ValueError('manifest differs from diagnostic receipt: ' + kind)
    # These switches require explicit integration with the native model. A
    # diagnostic receipt alone is not evidence that the trainer consumes them.
    if decision['texture_split_enabled']:
        raise ValueError('texture subdivision requires verified contributing Gaussian IDs and a resumable native split implementation')
    scene = (RUN / 'inputs').resolve()
    if manifest_path != scene / 'training_manifest.json':
        raise ValueError('training manifest must be the scene training_manifest.json')
    for name, expected in manifest['files'].items():
        path = (scene / name).resolve()
        if not path.is_relative_to(scene) or not path.is_file():
            raise ValueError('invalid B3 input path: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError('B3 input hash mismatch: ' + name)
    if not {'transforms.json', 'geometry.json'}.issubset(manifest['files']):
        raise ValueError('manifest must bind camera and initialization identities')
    meta = json.loads((scene / 'transforms.json').read_text(encoding='utf-8'))
    if (len(meta['train_filenames']) != 1478 or len(set(meta['train_filenames'])) != 1478
            or len(meta['val_filenames']) != 184 or len(meta['test_filenames']) != 184):
        raise ValueError('incorrect training population')
    splits = [set(meta[k]) for k in ('train_filenames', 'val_filenames', 'test_filenames')]
    if any(len(split) != expected for split, expected in zip(splits, (1478, 184, 184))):
        raise ValueError('duplicate filenames within a split')
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('training and held-out splits overlap')
    frame_paths = [frame['file_path'] for frame in meta['frames']]
    if len(frame_paths) != len(set(frame_paths)) or not set.union(*splits).issubset(frame_paths):
        raise ValueError('split frames missing or duplicated in camera metadata')
    os.environ['DN_RUN_OVERRIDE'] = str(RUN)
    os.environ['DN_SCENE_OVERRIDE'] = str(scene)
    import capture_quality_train as trainer
    import capture_quality_eval as evaluator
    from capture_surface_select import choose_checkpoint
    trainer.RUN = RUN
    trainer.SCENE = scene
    trainer.STATE = RUN / 'state'
    trainer.GEOM = RUN / 'geometry'
    trainer.AUDIT = trainer.STATE / 'trainer'
    trainer.legacy.SCENE = scene
    trainer.legacy.STATE = trainer.STATE
    evaluator.RUN = RUN
    evaluator.SCENE = scene
    evaluator.STATE = trainer.STATE
    if decision['depth_weights_enabled']:
        import numpy as np
        from unittest.mock import patch
        from dn_splatter.losses import EdgeAwareLogL1
        from dn_splatter.dn_model import DNSplatterModel
        entries = manifest['weights']['frames']
        names = json.loads((scene / 'transforms.json').read_text(encoding='utf-8'))['train_filenames']
        if len(entries) != len(names) or [e['frame'] for e in entries] != names:
            raise ValueError('weight receipt does not cover the ordered train split')
        paths = []
        for entry in entries:
            path = (ROOT / entry['path']).resolve()
            if not path.is_relative_to(RUN.resolve()) or not path.is_file():
                raise ValueError('weight file outside B3 run or missing')
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('weight file hash mismatch')
            paths.append(path)
        original_forward = EdgeAwareLogL1.forward
        equivalence = check_all_one_equivalence(lambda p, g, rgb, mask: original_forward(
            EdgeAwareLogL1(implementation='scalar'), p, g, rgb, mask))
        trainer.AUDIT.mkdir(parents=True, exist_ok=True)
        (trainer.AUDIT / 'weight_equivalence.json').write_text(json.dumps(equivalence, indent=2))
        original_loss = DNSplatterModel.get_loss_dict
        first_weight_batch = {}
        def weighted_loss(model, outputs, batch, *args, **kwargs):
            if not model.training:
                return original_loss(model, outputs, batch, *args, **kwargs)
            idx = int(batch['image_idx'])
            array = np.load(paths[idx], allow_pickle=False)
            weights = torch.as_tensor(array, device=outputs['depth'].device, dtype=outputs['depth'].dtype)
            if weights.ndim == 2:
                weights = weights.unsqueeze(-1)
            if weights.shape != outputs['depth'].shape:
                raise ValueError('weight and rendered depth dimensions differ for ' + names[idx])
            if not first_weight_batch:
                first_weight_batch.update(frame=names[idx], shape=list(weights.shape),
                    minimum=float(weights.min()), maximum=float(weights.max()),
                    nonidentity_valid_pixels=int(((weights>0)&(weights<1)).sum()), sha256=entries[idx]['sha256'])
            def forward(module, pred, gt, rgb, mask):
                if module.implementation != 'scalar':
                    raise ValueError('B3 weights require scalar EdgeAwareLogL1')
                value = weighted_edge_aware_log_l1(pred, gt, rgb, mask, weights)
                if 'consumed_at_step' not in first_weight_batch:
                    first_weight_batch.update(consumed_at_step=int(model.step), weighted_loss=float(value.detach()),
                        valid_pixels=int(mask.sum()) if mask is not None else weights.numel())
                    target=trainer.AUDIT/'first_weight_batch.json'
                    if not target.exists():
                        evaluator.write(target,first_weight_batch)
                return value
            with patch.object(EdgeAwareLogL1, 'forward', forward):
                return original_loss(model, outputs, batch, *args, **kwargs)
        trainer._b3_weighted_loss = weighted_loss
    original_config = trainer.config
    def b3_config():
        cfg = original_config(experiment='capture-full-b3', timestamp='surface-v3')
        # The B2 trainer discovers its latest checkpoint itself. Validate that
        # exact file before it can restore anything or run a missing val node.
        checkpoints = sorted(cfg.get_checkpoint_dir().glob('step-*.ckpt'))
        if checkpoints:
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            recovery = validate_recovery_checkpoint(checkpoints[-1], digest)
            filename_step = int(checkpoints[-1].stem.removeprefix('step-'))
            if recovery['saved_step'] != filename_step:
                raise ValueError('checkpoint filename and saved step differ')
            print(json.dumps({'b3_recovery_preflight': recovery}, sort_keys=True), flush=True)
        cfg.save_config()
        return cfg
    trainer.config = b3_config
    def select_b3(config):
        files = [RUN / 'evaluation' / 'val' / f'step-{n:09d}.json' for n in evaluator.NODES]
        result = choose_checkpoint(files)
        result['config'] = str(Path(config).resolve())
        evaluator.write(RUN / 'state' / 'selection.json', result)
        return result
    trainer.select = select_b3
    if decision['depth_weights_enabled']:
        with patch.object(DNSplatterModel, 'get_loss_dict', trainer._b3_weighted_loss):
            trainer.main()
    else:
        trainer.main()


def main():
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-checkpoint', type=Path)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train == bool(args.inspect_checkpoint):
        parser.error('choose exactly one of --train and --inspect-checkpoint')
    if args.train:
        run_formal(args.training_manifest)
    else:
        digest = hashlib.sha256(args.training_manifest.read_bytes()).hexdigest()
        result = validate_recovery_checkpoint(args.inspect_checkpoint, digest)
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
