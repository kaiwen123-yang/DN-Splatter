"""Carry the winning source-pixel ID through the original all-depth z-buffer."""
import cv2
import numpy as np
from capture_metric import remap_points
from capture_rectify import describe, edges, imwrite, project, read, save, zbuffer
from capture_s01_inputs import FULL, GEO_RUN, REPORT, ensure

GRID = GEO_RUN / "rect/640"


def frame(stem):
    z = np.load(GRID / "depth" / (stem + ".npy"))
    occ = cv2.imread(str(GRID / "occlusion" / (stem + ".png")), 0) > 0
    unsafe = edges(z) | (z <= 0) | occ
    unsafe = cv2.dilate(unsafe.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
    return z, unsafe


def panel(image, title):
    image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.putText(
        cv2.copyMakeBorder(image, 24, 0, 0, 0, cv2.BORDER_CONSTANT),
        title,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def compare(src, dst, poses, cam, visual):
    zs, src_bad = frame(src)
    zt, dst_bad = frame(dst)
    tf = np.linalg.inv(poses[dst]) @ poses[src]
    src_idx = np.flatnonzero(zs > 0)
    pc = remap_points(zs, zs > 0, cam) @ tf[:3, :3].T + tf[:3, 3]
    positive = pc[:, 2] > 0
    pc, src_idx = pc[positive], src_idx[positive]
    uv = project(pc, cam)
    pred, inside = zbuffer(uv, pc[:, 2], cam)
    xy = np.rint(uv[inside]).astype(int)
    target = xy[:, 1] * cam["width"] + xy[:, 0]
    wins = pc[inside, 2] == pred.ravel()[target]
    winner = np.full(pred.size, zs.size, dtype=int)
    np.minimum.at(winner, target[wins], src_idx[inside][wins])
    winner = np.minimum(winner.reshape(pred.shape), zs.size - 1)
    base = (pred > 0) & (zt >= 0.3) & (zt <= 6)
    both = base & (pred >= 0.3) & (pred <= 6)
    source_range = both & (zs.ravel()[winner] >= 0.3) & (zs.ravel()[winner] <= 6)
    smooth = source_range & ~src_bad.ravel()[winner] & ~dst_bad
    diff = pred - zt
    threshold = np.maximum(0.05, 0.03 * zt)
    masks = dict(
        target_range=base, both_target_axis=both, also_source_range=source_range, local_smooth=smooth
    )
    stats = {}
    for key, mask in masks.items():
        count = int(mask.sum())
        pos = int((diff[mask] > threshold[mask]).sum())
        neg = int((diff[mask] < -threshold[mask]).sum())
        stats[key] = dict(
            pixels=count,
            fraction_of_original=count / int(base.sum()) if base.any() else None,
            signed_m=describe(diff[mask]),
            absolute_m=describe(abs(diff[mask])),
            positive_candidates=pos,
            negative_candidates=neg,
            positive_fraction=pos / count if count else None,
            negative_fraction=neg / count if count else None,
        )
    if visual:

        def depth(z):
            color = cv2.applyColorMap(
                np.uint8(np.clip((z - 0.3) / 5.7, 0, 1) * 255), cv2.COLORMAP_TURBO
            )
            color[z == 0] = 0
            return color

        rgb = lambda stem: cv2.imread(str(GRID / "images" / (stem + ".png")))
        signed = np.uint8(np.clip((diff + 0.5) / 1.0, 0, 1) * 255)
        heat = cv2.applyColorMap(signed, cv2.COLORMAP_COOL)
        heat[~base] = 0
        imgs = [
            panel(rgb(src), src + " source"),
            panel(rgb(dst), dst + " target"),
            panel(depth(zt), "Target Z 0.3 to 6 m"),
            panel(depth(pred), "Projected Z 0.3 to 6 m"),
            panel(heat, "dZ -0.5 cyan to +0.5 magenta"),
            *[panel(np.uint8(m) * 255, k) for k, m in masks.items()],
        ]
        imwrite(
            REPORT / "previews" / (src + "_to_" + dst + ".jpg"),
            np.vstack([np.hstack(imgs[i : i + 3]) for i in (0, 3, 6)]),
        )
    return dict(source=src, target=dst, total_pixels=int(zt.size), subsets=stats)


def main():
    pairs = read(GEO_RUN / "metric/cross_view.json")["pairs"]
    if FULL:
        pairs=[p for p in pairs if "source" in p]
    else:
        ensure(len(pairs) == 20, "pair count")
    poses = {k: np.array(v) for k, v in read(GEO_RUN / "metric/poses.json")["poses"].items()}
    k = np.array(read(GEO_RUN / "rect/grids.json")["640"]["K_cv"])
    cam = dict(width=640, height=360, fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])
    rows = []
    for pair in pairs:
        src, dst = pair["source"], pair["target"]
        visual = (int(src[-5:]), int(dst[-5:])) in {(164, 179), (134, 149), (224, 239)}
        forward = compare(src, dst, poses, cam, visual)
        orig = pair["windows"]["6"]
        ensure(
            forward["subsets"]["target_range"]["pixels"] == orig["overlap"]
            and forward["subsets"]["both_target_axis"]["pixels"] == orig["both_depths_in_window"],
            "forward overlap",
        )
        rows.extend([forward, compare(dst, src, poses, cam, visual)])
    save(
        REPORT / "cross_view_diagnostic.json",
        dict(
            grid=[640, 360],
            directed_pairs=len(rows),
            pairs=rows,
            local_mask="Both native frames: exclude near-side edges, holes and D2C competition, each dilated by square Chebyshev radius 2.",
            threshold_m="max(0.05, 0.03 * target Z); signed difference is projected minus observed. Candidate signs only.",
            visibility="All positive source Z participate before range checks. Nearest target-axis Z wins; ties use lowest source index.",
            preview="Z clips at 0.3 and 6 m; dZ at +/-0.5 m; black for missing Z/dZ, white masks indicate inclusion.",
        ),
    )


if __name__ == "__main__":
    main()
