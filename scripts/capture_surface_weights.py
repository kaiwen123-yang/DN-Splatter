"""Produce B3 train depth weights and held-out-group reliability diagnostics.

Every nonzero frozen RGB-Z pixel is mapped back to its actual native sample.
Groups are partitioned deterministically for scoring and independent validation.
"""
import argparse
import os
from pathlib import Path
import numpy as np

from capture_surface_submaps import ROOT, RUN, PACKET, FLIP, camera_parameters, check, read, sha, write
from capture_surface_votes import NativeVotes
from capture_rectify import grid_setup
from capture_registration_review import project


def mapping(raw, rays, translation, grid, frozen):
    cam, _, _, valid_rgb, _ = grid
    ids = np.flatnonzero((raw.ravel() != 0) & (raw.ravel() != 65535))
    xyz = rays.reshape(-1, 3)[ids] * (raw.ravel()[ids, None] * .001) + translation
    uv = np.rint(project(xyz, cam)).astype(np.int64)
    inside = ((xyz[:, 2] > 0) & np.isfinite(xyz[:, 2]) & (uv[:, 0] >= 0) &
              (uv[:, 0] < cam['width']) & (uv[:, 1] >= 0) & (uv[:, 1] < cam['height']))
    ids, xyz, uv = ids[inside], xyz[inside], uv[inside]
    flat = uv[:, 1] * cam['width'] + uv[:, 0]
    order = np.lexsort((ids, xyz[:, 2], flat))
    order = order[np.r_[True, flat[order][1:] != flat[order][:-1]]]
    flat, ids, xyz = flat[order], ids[order], xyz[order]
    good = valid_rgb.ravel()[flat] & (xyz[:, 2].astype(np.float32) >= .3) & (xyz[:, 2].astype(np.float32) <= 6.)
    flat, ids, xyz = flat[good], ids[good], xyz[good]
    check(np.array_equal(np.sort(flat), np.flatnonzero(frozen.ravel())), 'RGB-Z support differs from frozen input')
    check(np.array_equal(xyz[:, 2].astype(np.float32), frozen.ravel()[flat]), 'RGB-Z values differ from frozen input')
    return flat, ids


def independent_votes(voter, points, source_gid, source_center, heldout):
    """Separate deterministic group sets; reject views near the source or prior views."""
    n = len(points)
    counts = np.zeros((n, 2, 4), np.uint16)
    residual_sum = np.zeros(n, np.float64)
    residual_count = np.zeros(n, np.uint16)
    accepted = [[], []]
    source_ray = source_center - points
    source_norm = np.linalg.norm(source_ray, axis=1)
    source_ray /= np.maximum(source_norm[:, None], 1e-12)
    cosine = np.cos(np.deg2rad(3.))
    for gid, frames in voter.groups:
        if gid == source_gid:
            continue
        partition = int(gid % 5 == heldout)
        cat = np.zeros(n, np.uint8)
        dist = np.full(n, -1.)
        res = np.full(n, np.inf)
        center = np.zeros((n, 3))
        for _, pose, path in frames:
            c, d, r = voter.observe(points, pose, path)
            better = (c > 0) & ((d > dist) | ((d == dist) & (r < res)))
            cat[better], dist[better], res[better] = c[better], d[better], r[better]
            center[better] = pose[:3, 3]
        ray = center - points
        norm = np.linalg.norm(ray, axis=1)
        ray /= np.maximum(norm[:, None], 1e-12)
        eligible = (cat > 0) & (norm > 0) & (source_norm > 0)
        eligible &= (np.linalg.norm(center-source_center, axis=1) >= .1) & ((ray*source_ray).sum(axis=1) <= cosine)
        # Only previously accepted observations can suppress a duplicate.
        for old_center, old_ray, old_mask in accepted[0] + accepted[1]:
            duplicate = (np.linalg.norm(center-old_center, axis=1) < .1) | ((ray*old_ray).sum(axis=1) > cosine)
            eligible &= ~(old_mask & duplicate)
        accepted[partition].append((center, ray, eligible.copy()))
        for category in (1, 2, 3):
            mask = eligible & (cat == category)
            counts[mask, partition, category-1] += 1
        if partition:
            mask = eligible & ((cat == 1) | (cat == 2))
            residual_sum[mask] += res[mask]
            residual_count[mask] += 1
    return counts, residual_sum, residual_count


def run(label, chunk):
    out = RUN / 'weights' / label
    out.mkdir(parents=True, exist_ok=True)
    identity_path = RUN / 'export' / label / 'identity.json'
    identity = read(identity_path)
    voter = NativeVotes(identity_path, RUN / 'export' / label / 'tolerances.json')
    meta = read(ROOT / identity['cameras']['path'])
    cal = read(ROOT / identity['calibration']['path'])
    _, _, rel = camera_parameters(meta, cal)
    rays, translation, grids = grid_setup(cal)
    source = {r['id']: r for r in read(ROOT / identity['selection']['path'])}
    frames = {f['file_path']: f for f in meta['frames']}
    scene = (ROOT / identity['cameras']['path']).parent
    group_of = {name: g['id'] for g in identity['groups'] for name in g['names']}
    check(len(group_of) == len(meta['train_filenames']) == 1478, 'training population')
    stats = [[dict(pixels=0, independent_pixels=0, residual_sum_m=0., observations=0) for _ in range(2)] for _ in range(4)]
    total, changed = 0, 0
    receipts = []
    for name in meta['train_filenames']:
        f = frames[name]; stem = Path(name).stem; gid = group_of[name]
        with np.load(PACKET / source[stem]['depth']['file']) as archive:
            raw = archive['depth']
        frozen = np.load(scene / f['depth_file_path'], mmap_mode='r')
        flat, native = mapping(raw, rays, translation, grids['1280'], frozen)
        dep = cal['cameras']['depth']; y, x = np.divmod(native, dep['width'])
        z = raw.ravel()[native].astype(np.float64) * .001
        p = np.column_stack(((x-dep['cx'])*z/dep['fx'], (y-dep['cy'])*z/dep['fy'], z))
        pose = np.asarray(f['transform_matrix']) @ FLIP @ rel
        p = p @ pose[:3, :3].T + pose[:3, 3]
        weights = np.zeros(frozen.size, np.float32)
        for start in range(0, len(flat), chunk):
            end = min(start+chunk, len(flat))
            counts, rs, rc = independent_votes(voter, p[start:end], gid, pose[:3, 3], gid % 5)
            s, c = counts[:, 0, 0], counts[:, 0, 1]
            w = (1. + s) / (1. + s + c)
            weights[flat[start:end]] = w
            bins = np.minimum(np.searchsorted([1., 2., 4., 6.], frozen.ravel()[flat[start:end]], side='right'), 3)
            for b in range(4):
                for low in range(2):
                    m = (bins == b) & ((w < 1.) == bool(low))
                    v = stats[b][low]
                    v['pixels'] += int(m.sum()); v['independent_pixels'] += int((m & (rc > 0)).sum())
                    v['residual_sum_m'] += float(rs[m].sum()); v['observations'] += int(rc[m].sum())
            changed += int((w < 1.).sum())
        total += len(flat)
        target = out / (stem + '.npy')
        temp = out / (stem + '.pending.npy')
        np.save(temp, weights.reshape(frozen.shape), allow_pickle=False); os.replace(temp, target)
        receipts.append(dict(frame=name, valid=len(flat), path=str(target.relative_to(ROOT)), sha256=sha(target)))
        print(f'{label} weights {len(receipts)}/1478 {stem}', flush=True)
    for row in stats:
        for v in row:
            v['mean_independent_abs_residual_m'] = v['residual_sum_m'] / v['observations'] if v['observations'] else None
    comparisons = [(r[0]['mean_independent_abs_residual_m'], r[1]['mean_independent_abs_residual_m']) for r in stats]
    supported = [(a,b) for a,b in comparisons if a is not None and b is not None]
    enabled = bool(changed and supported and all(b > a for a,b in supported))
    write(out / 'receipt.json', dict(model=label, identity_sha256=sha(identity_path), valid_pixels=total,
          nonidentity_pixels=changed, depth_weights_enabled=enabled, distance_bins_m=[.3,1,2,4,6],
          diagnostics=stats, fold='group id modulo 5 equals source group modulo 5 is held out',
          empty_pixels='zero weight; no evidence on valid pixel gives one', frames=receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('B2', 'B3'))
    parser.add_argument('--chunk', type=int, default=1024)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error('--chunk must be positive')
    run(args.model, args.chunk)
