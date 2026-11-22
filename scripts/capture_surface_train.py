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
    if texture.get('status') == 'incomplete':
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
    if not torch.isfinite(weights).all() or (weights < 0).any():
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
    return {"saved_step": step, "completed_updates": step + 1,
            "remaining_updates": target_updates - (step + 1), "manifest": expected_manifest}


def main():
    """Inspect a recovery checkpoint without starting or modifying a training run."""
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-checkpoint', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.training_manifest.read_bytes()).hexdigest()
    result = validate_recovery_checkpoint(args.inspect_checkpoint, digest)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
