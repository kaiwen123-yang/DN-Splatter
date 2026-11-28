"""Fit and audit a bounded tabletop from frozen training RGB-D for the local hall bridge."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from interactive_physics import functional, proxy_scene
from interactive_push_assets import file_hash, write_json


# Interior of the table in the historical training crop (774,406)-(1180,598).
TABLE = np.array([[839, 441], [886, 444], [1098, 488], [951, 491]], np.int32)


def backproject(z, mask, k, t):
    v, u = np.nonzero(mask)
    d = z[v, u]
    cam = np.column_stack(((u-k[0, 2])*d/k[0, 0], (v-k[1, 2])*d/k[1, 1], d))
    return cam@t[:3, :3].T+t[:3, 3]


def fit_plane(points, camera):
    rng = np.random.default_rng(42)
    best = np.zeros(len(points), bool)
    for _ in range(600):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(a-b, a-c)
        distance = np.abs((points-a)@normal)/max(np.linalg.norm(normal), 1e-9)
        accepted = distance < .012
        if accepted.sum() > best.sum():
            best = accepted
    center = points[best].mean(0)
    _, _, basis = np.linalg.svd(points[best]-center, full_matrices=False)
    x, z = basis[0], basis[-1]
    if z@(camera[:3, 3]-center) < 0:
        z = -z
    if x@camera[:3, 0] < 0:
        x = -x
    t = np.eye(4)
    t[:3, :3] = np.stack([x, np.cross(z, x), z])
    t[:3, 3] = -t[:3, :3]@center
    local = points@t[:3, :3].T+t[:3, 3]
    return t, local, best


def supported_rect(local):
    xy = np.ascontiguousarray(local[:, :2], dtype=np.float32)
    hull = cv2.convexHull(xy)
    tree = cKDTree(xy)
    centers = np.stack(np.meshgrid(np.linspace(*np.percentile(xy[:,0],[10,90]),15),
                                   np.linspace(*np.percentile(xy[:,1],[10,90]),9)), axis=-1).reshape(-1,2)
    for scale in np.linspace(1., .20, 33):
        half = np.array([.19, .105])*scale
        grid = np.stack(np.meshgrid(np.linspace(-half[0], half[0], 15), np.linspace(-half[1], half[1], 11)), axis=-1).reshape(-1, 2)
        for center in centers:
            grid_at = grid + center
            if tree.query(grid_at)[0].max() < .035 and all(cv2.pointPolygonTest(hull, tuple(p), False) >= 0 for p in grid_at):
                return half, center
    raise RuntimeError("The observed tabletop patch does not support a bounded rectangle")


def audit_plane(train, anchor, root, k, trans, half):
    real2phys = trans
    uv = np.stack(np.meshgrid(np.linspace(-half[0], half[0], 28), np.linspace(-half[1], half[1], 20)), axis=-1).reshape(-1, 2)
    local = np.column_stack((uv, np.zeros(len(uv))))
    scene = (local-real2phys[:3, 3])@real2phys[:3, :3]
    group0 = anchor["rgb_timestamp_ns"]//10_000_000_000
    emitted = {}
    for row in train:
        group = row["rgb_timestamp_ns"]//10_000_000_000
        camera = np.array(row["c2w_cv"])
        if group == group0 or np.linalg.norm(camera[:3, 3]-np.array(anchor["c2w_cv"])[:3, 3])<.08:
            continue
        cam = (scene-camera[:3, 3])@camera[:3, :3]
        if np.min(cam[:, 2]) < .3 or np.max(cam[:, 2]) > 6:
            continue
        pix = cam@k.T
        pix = np.rint(pix[:, :2]/pix[:, 2:]).astype(int)
        if np.min(pix) < 2 or pix[:, 0].max() >= 1278 or pix[:, 1].max() >= 718:
            continue
        # One independently selected view per 10-s group. Selection uses visibility extent only.
        area = float(cv2.contourArea(cv2.convexHull(pix.astype(np.float32))))
        if area >= 80 and (group not in emitted or area > emitted[group][0]):
            emitted[group] = (area, row, camera, pix)
    results = []
    for group, (_, row, camera, pix) in sorted(emitted.items()):
        path = root/row["depth"]
        if file_hash(path)[0] != row["depth_sha256"]:
            raise ValueError("Audit source depth hash mismatch")
        z = np.load(path, allow_pickle=False)
        mask = np.zeros(z.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pix), 1)
        cloud = backproject(z, (mask>0)&np.isfinite(z)&(z>.3)&(z<6), k, camera)
        p = cloud@trans[:3, :3].T+trans[:3, 3]
        dist = np.abs(p[:, 2])
        results.append({"id": row["id"], "group_10s": int(group), "depth_sha256": row["depth_sha256"],
                        "valid_depths": len(dist), "within_35mm": int((dist<=.035).sum()),
                        "median_abs_plane_m": float(np.median(dist)) if len(dist) else None})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--gs-background", type=Path, help="Fixed-camera RGB and metric optical Z with the same source camera")
    a = parser.parse_args()
    m = json.loads(a.manifest.read_text())
    root = a.source_root or Path(m["source_root"])
    train = [r for r in m["frames"] if r["split"]=="train"]
    anchor = next(r for r in train if r["id"]=="frame_00120")
    k, camera = np.array(m["K_cv"]), np.array(anchor["c2w_cv"])
    src = root/anchor["depth"]
    if file_hash(src)[0] != anchor["depth_sha256"] or file_hash(root/anchor["rgb"])[0] != anchor["rgb_sha256"]:
        raise ValueError("Frozen anchor source mismatched")
    z = np.load(src, allow_pickle=False)
    mask = np.zeros(z.shape, np.uint8)
    cv2.fillConvexPoly(mask, TABLE, 1)
    pts = backproject(z, (mask>0)&np.isfinite(z)&(z>.3)&(z<6), k, camera)
    if len(pts)<200:
        raise RuntimeError("Insufficient measured tabletop footprint")
    trans, local, best = fit_plane(pts, camera)
    half, center = supported_rect(local[best])
    trans[:2, 3] -= center
    inside = np.all(np.abs(local[:, :2]-center) <= half, axis=1)
    audit = audit_plane(train, anchor, root, k, trans, half)
    witnessed = [r for r in audit if r["valid_depths"]>=30 and r["within_35mm"]/r["valid_depths"]>=.70]
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"fit_id": anchor["id"], "polygon_xy": TABLE.tolist(), "fit_depths": len(pts),
              "fit_inliers_12mm": int(best.sum()), "fit_rms_m": float(np.mean(local[best, 2]**2)**.5),
              "inside_bounded_support": int(inside.sum()), "within_35mm_inside_support": int((np.abs(local[inside,2])<=.035).sum()),
              "source_rgb": anchor["rgb_sha256"], "source_z": anchor["depth_sha256"], "train_only": True,
              "support_half_m": half.tolist(), "T_physics_reconstruction": trans.tolist(),
              "audit": audit, "verified_group_count": len(witnessed),
              "assumption": "gravity perpendicular to measured tabletop and facing the camera; no gravity sensor label"}
    write_json(a.out/"support_audit.json", report)
    if not inside.any() or (np.abs(local[inside,2])<=.035).mean()<.70 or len(witnessed)<2:
        raise RuntimeError("Too little independent support for this observed tabletop")
    scene = proxy_scene()
    scene["scene_id"] = "gemini335_table120_with_synthetic_object"
    scene["frames"].update(T_physics_reconstruction=trans.tolist(), gravity_source=report["assumption"])
    scene["background"].update(kind="captured", support_half_xy=half.tolist(), source="measured table polygon + independent training RGB-D groups")
    obj = scene["objects"][0]
    obj["support_groups"] = [r["group_10s"] for r in witnessed]
    # Use the proxy only within measured space; the local world has no added object in its original background.
    scene["physics"]["initial_pusher_xy"] = [-min(.115, half[0]*.85), -.008]
    if a.gs_background:
        with np.load(a.gs_background, allow_pickle=False) as b:
            if not np.allclose(b["K"], k) or not np.allclose(b["T_reconstruction_camera_cv"], camera):
                raise ValueError("GS cache camera differs from observed support camera")
            rgb, render_z = b["rgb"], b["depth_m"]
        bg_type = "frozen Gaussian appearance and metric optical Z"
    else:
        img = cv2.imread(str(root/anchor["rgb"]))
        rgb, render_z = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), z
        bg_type = "captured RGB with measured optical Z; sparse-depth occlusion conservative"
    cam = trans@camera
    bg = (a.out/"fixed_frame120.npz").resolve()
    np.savez_compressed(bg, rgb=rgb, depth_m=render_z, K=k, T_world_camera_cv=cam)
    scene["cameras"]["front"] = {"K": k.tolist(), "T_world_camera_cv": cam.tolist(), "size_wh": [1280,720],
                                 "background_rgbd_npz": str(bg), "source": bg_type,
                                 "sha256": file_hash(bg)[0]}
    functional(a.out, scene)


if __name__ == "__main__":
    main()
