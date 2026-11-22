"""B3 train-only joint RGB-D relocalization with independent acceptance evidence."""
import copy
import sqlite3
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from capture_quality_paths import GEORUN, ROOT
from capture_quality_pose import pixdepth
from capture_native_neighbor_check import native_observation, count_pair
from capture_surface_submaps import B2, RUN, PACKET, FLIP, camera_parameters, read, sha, write, check

BAD = set(range(1294, 1307))
TARGET = (1304, 1305)


def pose_from_vector(v):
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(v[:3])[0]
    t[:3, 3] = v[3:]
    return np.linalg.inv(t)


def vector_from_pose(t):
    w = np.linalg.inv(t)
    return np.r_[cv2.Rodrigues(w[:3, :3])[0].ravel(), w[:3, 3]]


def project(t, xyz, k):
    p = (xyz - t[:3, 3]) @ t[:3, :3]
    q = p @ k.T
    z = q[:, 2]
    return q[:, :2] / np.where(np.abs(z[:, None]) > 1e-9, z[:, None], np.nan), p[:, 2]


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    path = B2 / 'inputs/transforms.json'
    meta = read(path)
    identity = read(RUN / 'export/B2/identity.json')
    check(identity['cameras']['sha256'] == sha(path), 'Frozen B2 camera identity mismatch')
    train = {int(Path(n).stem[-5:]) for n in meta['train_filenames']}
    frames = {int(Path(f['file_path']).stem[-5:]): f for f in meta['frames']}
    check(set(TARGET) <= train and 1514 in train, 'Required candidates must be training views')
    groups = {int(Path(n).stem[-5:]): g['id'] for g in identity['groups'] for n in g['names']}
    check(set(groups) == train, 'Frozen group population differs from training set')
    k = np.asarray(read(GEORUN / 'rect/grids.json')['1280']['K_cv'], dtype=float)
    invk = np.linalg.inv(k)
    db = sqlite3.connect((GEORUN / 'sfm/train.db').as_uri() + '?mode=ro', uri=True)
    ids = {name: idx for idx, name in db.execute('select image_id,name from images')}
    selection = {r['id']: r for r in read(PACKET / 'selection.json')}
    calibration = read(PACKET / 'calibration.json')
    _, _, extrinsic = camera_parameters(meta, calibration)
    nk = calibration['cameras']['depth']
    yy, xx = np.mgrid[4:nk['height']:8, 4:nk['width']:8]
    yy, xx = yy.ravel(), xx.ravel()

    @lru_cache(maxsize=32)
    def obs(i):
        name = Path(frames[i]['file_path']).name
        idx = ids[name]
        def get(table, dtype):
            rows, cols, blob = db.execute('select rows,cols,data from ' + table + ' where image_id=?', (idx,)).fetchone()
            return np.frombuffer(blob or b'', dtype=dtype).reshape(rows, cols)
        uv = get('keypoints', np.float32)[:, :2].astype(float) - .5
        desc = get('descriptors', np.uint8)
        depth = np.load(GEORUN / 'rect/640/depth' / (name[:-4] + '.npy'))
        return uv, desc, depth

    @lru_cache(maxsize=32)
    def anchor(i):
        uv, desc, depth = obs(i)
        z = np.array([pixdepth(depth, (p + .5) * .5 - .5) for p in uv])
        xyz = (np.c_[uv, np.ones(len(uv))] @ invk.T) * z[:, None]
        t = np.asarray(frames[i]['transform_matrix']) @ FLIP
        return desc, xyz @ t[:3, :3].T + t[:3, 3]

    def match(i, j):
        uv, desc, _ = obs(i)
        descj, xyz = anchor(j)
        if len(desc) < 2 or len(descj) < 2:
            return []
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, descj, k=2)
        return [(m.distance / max(n.distance, 1.), m.queryIdx, j, xyz[m.trainIdx], uv[m.queryIdx])
                for pair in pairs if len(pair) == 2 for m, n in [pair]
                if m.distance < .75 * n.distance and np.isfinite(xyz[m.trainIdx]).all()]

    def collect(i, sources):
        rows = []
        for j in sources:
            rows.extend(match(i, j))
        used, chosen = set(), []
        for row in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            if row[1] not in used:
                used.add(row[1]); chosen.append(row)
        return chosen

    def arrays(rows):
        return (np.array([x[3] for x in rows]).reshape(-1, 3),
                np.array([x[4] for x in rows]).reshape(-1, 2))

    def residual(t, rows):
        xyz, uv = arrays(rows)
        if not len(xyz):
            return {'count': 0, 'median_px': None, 'positive_fraction': None}
        p, z = project(t, xyz, k)
        e = np.linalg.norm(p - uv, axis=1)
        good = np.isfinite(e) & (z > 0)
        return {'count': len(rows), 'positive_count': int(good.sum()),
                'positive_fraction': float(good.mean()),
                'median_px': float(np.median(e[good])) if good.any() else None}

    old = {i: np.asarray(frames[i]['transform_matrix']) @ FLIP for i in TARGET}
    fixed = sorted(train - BAD - {1514})
    # Split by frozen submap, not by individual correspondences, before matching.
    near = {i: sorted(fixed, key=lambda j: (abs(j-i), j))[:100] for i in TARGET}
    fit, hold = {}, {}
    for i in TARGET:
        fit[i] = collect(i, [j for j in near[i] if groups[j] % 2 == 0])
        hold[i] = collect(i, [j for j in near[i] if groups[j] % 2 == 1])
        # Any feature used to fit is excluded from evaluation as well.
        used = {r[1] for r in fit[i]}
        hold[i] = [r for r in hold[i] if r[1] not in used]
    initial, inliers = {}, {}
    for i in TARGET:
        xyz, uv = arrays(fit[i])
        if len(xyz) < 6:
            continue
        ok, rv, tv, idx = cv2.solvePnPRansac(xyz, uv, k, None, iterationsCount=3000,
                reprojectionError=4., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if ok and idx is not None:
            inliers[i] = idx.ravel()
            initial[i] = np.r_[rv.ravel(), tv.ravel()]
    # Joint optimization shares only real mutual image matches; anchors remain fixed.
    proposal = {}
    joint_pairs = []
    if len(initial) == 2:
        u0, d0, _ = obs(TARGET[0]); u1, d1, _ = obs(TARGET[1])
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
        held0 = {r[1] for r in hold[TARGET[0]]}
        held1 = {r[1] for r in hold[TARGET[1]]}
        joint_pairs = [(u0[m.queryIdx], u1[m.trainIdx]) for pair in pairs if len(pair) == 2
                       for m, n in [pair] if m.distance < .75*n.distance
                       and m.queryIdx not in held0 and m.trainIdx not in held1]
        # Symmetric epipolar residual associates the two estimates without creating metric points.
        def objective(v):
            ts = [pose_from_vector(v[:6]), pose_from_vector(v[6:])]
            out = []
            for t, i in zip(ts, TARGET):
                xyz, uv = arrays(fit[i]); xyz, uv = xyz[inliers[i]], uv[inliers[i]]
                p, z = project(t, xyz, k)
                out.extend(np.nan_to_num((p-uv).ravel(), nan=1e4, posinf=1e4, neginf=-1e4))
            if joint_pairs:
                rel = np.linalg.solve(ts[1], ts[0]); tx = rel[:3, 3]
                skew = np.array([[0,-tx[2],tx[1]], [tx[2],0,-tx[0]], [-tx[1],tx[0],0]])
                f = invk.T @ skew @ rel[:3, :3] @ invk
                a = np.c_[np.array([x[0] for x in joint_pairs]), np.ones(len(joint_pairs))]
                b = np.c_[np.array([x[1] for x in joint_pairs]), np.ones(len(joint_pairs))]
                l1, l0 = a @ f.T, b @ f
                denom = np.sqrt(l1[:,0]**2 + l1[:,1]**2 + l0[:,0]**2 + l0[:,1]**2 + 1e-12)
                out.extend(np.sum(b*l1, axis=1)/denom)
            return np.asarray(out)
        result = least_squares(objective, np.r_[initial[TARGET[0]], initial[TARGET[1]]], loss='huber', f_scale=2.5, max_nfev=100)
        proposal = {TARGET[0]: pose_from_vector(result.x[:6]), TARGET[1]: pose_from_vector(result.x[6:])}
    # Evaluate both directions using native raw Z on boundary edges; retain counts for auditing.
    def pair(a, b, t):
        fa, fb = copy.deepcopy(frames[a]), copy.deepcopy(frames[b])
        if a in t: fa['transform_matrix'] = (t[a] @ FLIP).tolist()
        if b in t: fb['transform_matrix'] = (t[b] @ FLIP).tolist()
        x = native_observation(PACKET, fa, selection, extrinsic)
        y = native_observation(PACKET, fb, selection, extrinsic)
        return [count_pair(x, y, nk, yy, xx), count_pair(y, x, nk, yy, xx)]
    boundary = [(1303,1304), (1305,1306)]
    checks = []
    if proposal and all(a in train and b in train for a,b in boundary):
        for a,b in boundary:
            before, after = pair(a,b,old), pair(a,b,proposal)
            checks.append({'edge':[a,b], 'old':before, 'proposed':after})
    rows = {}
    gates = []
    for i in TARGET:
        ins = inliers.get(i, np.array([], dtype=int))
        fitted = [fit[i][j] for j in ins]
        before = residual(old[i], hold[i]); after = residual(proposal[i], hold[i]) if i in proposal else None
        fstat = residual(proposal[i], fitted) if i in proposal else None
        gs = sorted({groups[r[2]] for r in fitted})
        gate = (len(ins) >= 20 and len(gs) >= 2 and fstat and fstat['median_px'] is not None
                and fstat['median_px'] <= 2.5 and fstat['positive_fraction'] == 1
                and before['median_px'] is not None and after['median_px'] is not None
                and after['median_px'] < before['median_px'] and after['positive_fraction'] == 1)
        gates.append(bool(gate))
        delta = np.linalg.solve(old[i], proposal[i]) if i in proposal else np.eye(4)
        rows[str(i)] = dict(fit_correspondences=len(fit[i]), fit_inliers=len(ins), fit_groups=gs,
            heldout_correspondences=len(hold[i]), heldout_groups=sorted({groups[r[2]] for r in hold[i]}),
            fit=fstat, heldout_old=before, heldout_proposed=after,
            old_c2w_cv=old[i].tolist(), proposed_c2w_cv=proposal[i].tolist() if i in proposal else None,
            translation_m=float(np.linalg.norm(delta[:3,3])),
            rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))))
    def projected_errors(source, target):
        depth, source_pose = source
        ref, target_pose = target
        z = depth[yy, xx]
        valid = np.isfinite(z) & (z >= .3) & (z <= 6)
        ids = np.flatnonzero(valid)
        p = np.column_stack(((xx[valid]-nk['cx'])*z[valid]/nk['fx'],
                             (yy[valid]-nk['cy'])*z[valid]/nk['fy'], z[valid]))
        rel = np.linalg.solve(target_pose, source_pose)
        p = p @ rel[:3,:3].T + rel[:3,3]
        valid = np.isfinite(p).all(axis=1) & (p[:,2] > 0)
        p, ids = p[valid], ids[valid]
        u = np.rint(nk['fx']*p[:,0]/p[:,2]+nk['cx']).astype(int)
        v = np.rint(nk['fy']*p[:,1]/p[:,2]+nk['cy']).astype(int)
        valid = (u>=0)&(u<nk['width'])&(v>=0)&(v<nk['height'])
        p, ids, u, v = p[valid], ids[valid], u[valid], v[valid]
        pixel = v*nk['width']+u
        order = np.lexsort((p[:,2], pixel))
        idx = order[np.unique(pixel[order], return_index=True)[1]]
        p, ids, u, v = p[idx], ids[idx], u[idx], v[idx]
        measured = ref[v,u]
        valid = np.isfinite(measured)&(measured>=.3)&(measured<=6)
        return dict(zip(ids[valid].tolist(), (p[valid,2]-measured[valid]).tolist()))

    def paired_depth(a, b):
        results = []
        for src, dst in ((a,b),(b,a)):
            def load(i, transforms):
                f = copy.deepcopy(frames[i])
                if i in transforms:
                    f['transform_matrix'] = (transforms[i] @ FLIP).tolist()
                return native_observation(PACKET, f, selection, extrinsic)
            x0, y0 = load(src, old), load(dst, old)
            x1, y1 = load(src, proposal), load(dst, proposal)
            before = projected_errors(x0, y0); after = projected_errors(x1, y1)
            common = sorted(before.keys() & after.keys())
            e0 = np.array([before[x] for x in common]); e1 = np.array([after[x] for x in common])
            results.append(dict(source=src, target=dst, old_valid=len(before), proposed_valid=len(after),
                common=len(common), old_within10=int((abs(e0)<=.1).sum()),
                proposed_within10=int((abs(e1)<=.1).sum()),
                old_rate=float(np.mean(abs(e0)<=.1)) if len(common) else None,
                proposed_rate=float(np.mean(abs(e1)<=.1)) if len(common) else None))
        return results
    common_depth = [r for a,b in boundary for r in paired_depth(a,b)] if checks else []
    depth_gate = bool(len(common_depth)==4 and all(r['common'] > 0 and
        r['proposed_valid'] >= r['old_valid'] and r['proposed_rate'] > r['old_rate']
        for r in common_depth))
    # Test actual pairs of fixed-map camera centres at each fitted 3-D point.
    viewpoint = {}
    for i in TARGET:
        fitted = [fit[i][j] for j in inliers.get(i, [])]
        pairs = []
        for a in range(len(fitted)):
            for b in range(a + 1, len(fitted)):
                ra, rb = fitted[a], fitted[b]
                if groups[ra[2]] == groups[rb[2]] or np.linalg.norm(ra[3]-rb[3]) > .05:
                    continue
                ca = (np.asarray(frames[ra[2]]['transform_matrix']) @ FLIP)[:3, 3]
                cb = (np.asarray(frames[rb[2]]['transform_matrix']) @ FLIP)[:3, 3]
                point = (ra[3] + rb[3]) * .5
                va, vb = ca - point, cb - point
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                if denom == 0:
                    continue
                distance = float(np.linalg.norm(ca-cb))
                angle = float(np.degrees(np.arccos(np.clip(np.dot(va,vb)/denom,-1,1))))
                if distance >= .1 and angle >= 3:
                    pairs.append({'frames':[ra[2],rb[2]], 'groups':[groups[ra[2]],groups[rb[2]]],
                                  'baseline_m':distance, 'ray_angle_deg':angle})
                    break
            if pairs:
                break
        viewpoint[str(i)] = {'independent_pair': pairs[0] if pairs else None}
    independent_viewpoint_gate = all(viewpoint[str(i)]['independent_pair'] is not None for i in TARGET)
    accepted = bool(all(gates) and depth_gate and independent_viewpoint_gate)
    output = copy.deepcopy(meta)
    if accepted:
        for i in TARGET: frames_out = next(f for f in output['frames'] if int(Path(f['file_path']).stem[-5:]) == i); frames_out['transform_matrix'] = (proposal[i] @ FLIP).tolist()
    out = RUN / 'geometry/pose'
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / 'diagnostics.json').exists() and not (out / 'transforms.json').exists(),
          'Pose output exists; preserve prior evidence and use an explicit recovery procedure')
    write(out / 'transforms.json', output)
    ordered = sorted(train)
    pos = ordered.index(1514)
    adjacent = ordered[pos-1:pos+2]
    diagnostic_edges = []
    for a,b in zip(adjacent, adjacent[1:]):
        diagnostic_edges.append({'edge':[a,b], 'directed':pair(a,b,{})})
    image = cv2.imread(str(B2 / 'inputs' / frames[1514]['file_path']), cv2.IMREAD_GRAYSCALE)
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var()) if image is not None else None
    # Extract real native-depth planes independently; report RGB correspondences to
    # adjacent training frames without treating them as a validated rigid correction.
    depth1514, depth_pose = native_observation(PACKET, frames[1514], selection, extrinsic)
    zz = depth1514[yy, xx]
    valid = np.isfinite(zz) & (zz >= .3) & (zz <= 6)
    points = np.c_[(xx[valid]-nk['cx'])*zz[valid]/nk['fx'],
                   (yy[valid]-nk['cy'])*zz[valid]/nk['fy'], zz[valid]]
    rng = np.random.default_rng(42)
    planes = []
    for _ in range(6):
        if len(points) < 100: break
        best = np.zeros(len(points), dtype=bool)
        for __ in range(300):
            sample = points[rng.choice(len(points), 3, replace=False)]
            normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            normal /= norm
            mask = np.abs((points-sample[0]) @ normal) <= .02
            if mask.sum() > best.sum(): best = mask
        if best.sum() < 100: break
        selected = points[best]; center = selected.mean(axis=0)
        _, singular, vh = np.linalg.svd(selected-center, full_matrices=False)
        normal = vh[-1]
        planes.append({'count':int(best.sum()), 'center_depth_camera_m':center.tolist(),
                       'normal_depth_camera':normal.tolist(), 'singular_values':singular.tolist(),
                       'median_abs_residual_m':float(np.median(np.abs((selected-center) @ normal)))})
        points = points[~best]
    rgb_checks = []
    for neighbour in (adjacent[0], adjacent[2]):
        matches = match(1514, neighbour)
        stat = residual(np.asarray(frames[1514]['transform_matrix']) @ FLIP, matches)
        rgb_checks.append({'frame':neighbour, 'matches':len(matches), 'old_pose_reprojection':stat})
    write(out / 'diagnostics.json', dict(run_id=RUN.name, input_sha256=sha(path), joint_matches=len(joint_pairs),
        candidates=rows, boundary_native_depth=checks, common_support_depth=common_depth, independent_reprojection_gates=gates,
        common_support_depth_gate=depth_gate, independent_viewpoint_gate=independent_viewpoint_gate, viewpoint_evidence=viewpoint, accepted_pose_changes=2 if accepted else 0,
        status='accepted' if accepted else 'original_retained_unresolved',
        failure=None if accepted else 'joint proposal did not satisfy all measured acceptance gates',
        separate_1514={'status':'original_retained_unresolved', 'accepted':False,
                       'adjacent_edges':diagnostic_edges, 'laplacian_variance':blur, 'native_planes':planes, 'rgb_neighbour_checks':rgb_checks,
                       'reason':'rigid submap correction lacks validated multi-plane evidence'},
        training_decision='pending_other_evidence'))
    db.close()


if __name__ == '__main__':
    main()
