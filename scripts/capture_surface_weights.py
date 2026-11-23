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
    if not len(ids):
        check(not np.count_nonzero(frozen), 'empty native projection with nonempty frozen Z')
        return ids.copy(), ids
    flat = uv[:, 1] * cam['width'] + uv[:, 0]
    order = np.lexsort((ids, xyz[:, 2], flat))
    order = order[np.r_[True, flat[order][1:] != flat[order][:-1]]]
    flat, ids, xyz = flat[order], ids[order], xyz[order]
    good = valid_rgb.ravel()[flat] & (xyz[:, 2].astype(np.float32) >= .3) & (xyz[:, 2].astype(np.float32) <= 6.)
    flat, ids, xyz = flat[good], ids[good], xyz[good]
    check(np.array_equal(np.sort(flat), np.flatnonzero(frozen.ravel())), 'RGB-Z support differs from frozen input')
    check(np.array_equal(xyz[:, 2].astype(np.float32), frozen.ravel()[flat]), 'RGB-Z values differ from frozen input')
    return flat, ids


def independent_votes(voter, points, source_gid, source_center, heldout, score_first=False):
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
    groups = sorted(voter.groups,key=lambda g: g[0]%5==heldout) if score_first else voter.groups
    for gid, frames in groups:
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


def run(label, chunk, device='cpu'):
    out = RUN / 'weights' / label
    out.mkdir(parents=True, exist_ok=True)
    identity_path = RUN / 'export' / label / 'identity.json'
    identity = read(identity_path)
    export = identity_path.parent
    tolerance_path = export / 'tolerances.json'
    voter = NativeVotes(identity_path, tolerance_path, cache_frames=1478)
    engine = None
    if device == 'cuda':
        from capture_surface_weight_device import WeightVotes
        engine = WeightVotes(export)
    meta = read(ROOT / identity['cameras']['path'])
    cal = read(ROOT / identity['calibration']['path'])
    _, _, rel = camera_parameters(meta, cal)
    rays, translation, grids = grid_setup(cal)
    source = {r['id']: r for r in read(ROOT / identity['selection']['path'])}
    frames = {f['file_path']: f for f in meta['frames']}
    scene = (ROOT / identity['cameras']['path']).parent
    check(sha(RUN/'geometry/pose/transforms.json') == identity['cameras']['sha256'],
          'weights must use the accepted final training poses')
    group_of = {name: g['id'] for g in identity['groups'] for name in g['names']}
    check(len(group_of) == len(meta['train_filenames']) == 1478, 'training population')
    def empty_stats():
        return [[dict(pixels=0, independent_pixels=0, residual_sum_m=0., observations=0,
                      effective_weight_sum=0., crossfit_pixels=0, crossfit_observations=0,
                      crossfit_residual_sum_m=0.) for _ in range(2)] for _ in range(4)]
    stats = empty_stats()
    binding = dict(identity=sha(identity_path), tolerances=sha(tolerance_path),
                   weight_code=sha(Path(__file__)), device=device)
    if engine is not None:
        binding['kernel'] = sha(Path(__file__).with_name('capture_surface_weight_votes.cu'))
    check_path = out/'native_equivalence.json'
    total, changed = 0, 0
    receipts = []
    for name in meta['train_filenames']:
        f = frames[name]; stem = Path(name).stem; gid = group_of[name]
        target = out / (stem + '.npy')
        marker = out/'frames'/(stem+'.json')
        if marker.exists():
            row = read(marker)
            check(row['binding']==binding and row['frame']==name and sha(target)==row['sha256'],
                  'invalid existing depth weight '+name)
            receipts.append(row['entry']); total += row['entry']['valid']; changed += row['changed']
            for b in range(4):
                for low in range(2):
                    for k, v in row['stats'][b][low].items(): stats[b][low][k] += v
            continue
        local = empty_stats(); frame_changed = 0
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
            if engine is None:
                counts, rs, rc = independent_votes(voter, p[start:end], gid, pose[:3, 3], gid % 5)
                s, c = counts[:, 0, 0], counts[:, 0, 1]
                exact, xr, xc=independent_votes(voter,p[start:end],gid,pose[:3,3],gid%5,score_first=True)
                cross_low=exact[:,0,1]>0
            else:
                counts, rs = engine.score(p[start:end], name)
                s, c, rc = counts.T
                exact,xr=engine.score(p[start:end],name,score_first=True)
                cross_low=exact[:,1]>0
                xc=exact[:,2]
                if not check_path.exists():
                    ix = np.linspace(0,end-start-1,min(256,end-start),dtype=int)
                    ref, rr, rn = independent_votes(voter, p[start:end][ix], gid, pose[:3,3], gid%5)
                    np.testing.assert_array_equal(counts[ix], np.column_stack((ref[:,0,0],ref[:,0,1],rn)))
                    np.testing.assert_allclose(rs[ix], rr, atol=1e-9, rtol=1e-9)
                    original_error=float(np.max(np.abs(rs[ix]-rr)))
                    ref,rr,rn=independent_votes(voter,p[start:end][ix],gid,pose[:3,3],gid%5,score_first=True)
                    np.testing.assert_array_equal(exact[ix],np.column_stack((ref[:,0,0],ref[:,0,1],rn)))
                    np.testing.assert_allclose(xr[ix],rr,atol=1e-9,rtol=1e-9)
                    write(check_path, dict(frame=name, measured_pixels=len(ix), counts_equal=True,
                          maximum_residual_sum_difference_m=original_error,
                          crossfit_max_residual_difference_m=float(np.max(np.abs(xr[ix]-rr))),binding=binding))
                    voter.load.cache_clear()
            w = (1. + s) / (1. + s + c)
            weights[flat[start:end]] = w
            bins = np.minimum(np.searchsorted([1., 2., 4., 6.], frozen.ravel()[flat[start:end]], side='right'), 3)
            for b in range(4):
                for low in range(2):
                    m = (bins == b) & ((w < 1.) == bool(low))
                    v = local[b][low]
                    v['pixels'] += int(m.sum()); v['independent_pixels'] += int((m & (rc > 0)).sum())
                    v['residual_sum_m'] += float(rs[m].sum()); v['observations'] += int(rc[m].sum())
                    v['effective_weight_sum'] += float(w[m].sum())
                    independent=(bins==b)&(cross_low==bool(low))
                    v['crossfit_pixels']+=int(independent.sum())
                    v['crossfit_observations']+=int(xc[independent].sum())
                    v['crossfit_residual_sum_m']+=float(xr[independent].sum())
            frame_changed += int((w < 1.).sum())
        total += len(flat)
        temp = out / (stem + '.pending.npy')
        np.save(temp, weights.reshape(frozen.shape), allow_pickle=False); os.replace(temp, target)
        entry = dict(frame=name, valid=len(flat), path=str(target.relative_to(ROOT)), sha256=sha(target))
        write(marker, dict(binding=binding, frame=name, sha256=entry['sha256'], entry=entry,
                           changed=frame_changed, stats=local))
        for b in range(4):
            for low in range(2):
                for k, v in local[b][low].items(): stats[b][low][k] += v
        receipts.append(entry); changed += frame_changed
        print(f'{label} weights {len(receipts)}/1478 {stem}', flush=True)
    for row in stats:
        for v in row:
            v['mean_independent_abs_residual_m'] = v['residual_sum_m'] / v['observations'] if v['observations'] else None
            v['crossfit_MAE_m']=v['crossfit_residual_sum_m']/v['crossfit_observations'] if v['crossfit_observations'] else None
    comparisons = [(r[0]['mean_independent_abs_residual_m'], r[1]['mean_independent_abs_residual_m']) for r in stats]
    supported = [(a,b) for a,b in comparisons if a is not None and b is not None]
    cross=[(r[0]['crossfit_MAE_m'],r[1]['crossfit_MAE_m']) for r in stats]
    tested=[(a,b) for a,b in cross if a is not None and b is not None]
    enabled = bool(changed and supported and tested and all(b>a for a,b in supported+tested))
    write(out / 'receipt.json', dict(model=label, identity_sha256=sha(identity_path), valid_pixels=total,
          nonidentity_pixels=changed, depth_weights_enabled=enabled, distance_bins_m=[.3,1,2,4,6],
          diagnostics=stats, binding=binding, fold='group id modulo 5 equals source group modulo 5 is held out',
          crossfit='Training maps retain the original greedy group order; additionally score all non-held groups first, then validate on independent held groups excluded from score deduplication.',
          empty_pixels='zero weight; no evidence on valid pixel gives one', frames=receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('B2', 'B3'))
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error('--chunk must be positive')
    run(args.model, args.chunk, args.device)
