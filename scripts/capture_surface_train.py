"""B3 training primitives; formal launcher is deliberately gated until inputs are frozen."""
from pathlib import Path
import json

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'configs/capture_surface_quality_protocol.json').read_text(encoding='utf-8'))
RUN = ROOT / '.local' / 'runs' / SPEC['run_id']


def training_decision(pose, weights, texture):
    """Require actual diagnostic results; missing evidence is never an equivalence decision."""
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
