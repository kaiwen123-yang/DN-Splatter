"""Stream the official ViT-Large prior over the frozen capture, with atomic frame receipts."""
import importlib.metadata as md
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from capture_metric3d_common import P, PRIOR, REPORT, ROOT, SCENE, SOURCE, STATE, WEIGHT, demand, member, read, sha, write
from capture_metric3d_weight import DIGEST, REVISION, SIZE, URL


def main():
    sys.path.insert(0, str(SOURCE))
    from hubconf import metric3d_vit_large
    from mmengine import Config

    weight = member(WEIGHT)
    demand(weight["bytes"] == SIZE and weight["sha256"] == DIGEST, "downloaded ViT-Large weights")
    cfg = Config.fromfile(str(SOURCE / P["prior"]["config"]))
    demand(cfg.model.decode_head.iters == 8, "RAFT iterations")
    model = metric3d_vit_large(pretrain=False)
    # The official state dict is pinned by two independent identifiers before deserialization.
    payload = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    weights = payload["model_state_dict"]
    keys = model.load_state_dict(weights, strict=False)
    demand(not (set(dict(model.named_parameters())) & set(keys.missing_keys)), "missing learned parameters")
    del weights, payload
    model.cuda().eval()
    cv2.setNumThreads(1)
    meta = read(SCENE / "transforms.json")
    h, w = meta["h"], meta["w"]
    s = min(616 / h, 1064 / w)
    hh, ww = int(h * s), int(w * s)
    top, left = (616 - hh) // 2, (1064 - ww) // 2
    K = np.array([meta[k] for k in ("fl_x", "fl_y", "cx", "cy")], dtype=float)
    mean = np.array(P["prior"]["mean_rgb"], np.float32)
    std = np.array(P["prior"]["std_rgb"], np.float32)
    demand((h, w, hh, ww, top, left) == (720, 1280, 598, 1064, 9, 0), "image geometry")
    info = dict(weight=weight, weight_url=URL, hf_revision=REVISION, source_commit=P["prior"]["commit"],
          model="metric3d_vit_large", raft_iterations=8, dtype=str(next(model.parameters()).dtype),
          autocast=False, tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
          packages={k: md.version(k) for k in ("torch", "torchvision", "mmengine", "opencv-python")},
          missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys),
          K_fx_fy_cx_cy=K.tolist(), resize_scale=s, resized_hw=[hh, ww],
          K_resized=(K * s).tolist(), K_padded=(K * s + [0, 0, left, top]).tolist(),
          pad_tblr=[top, 616-hh-top, left, 1064-ww-left], depth_scale=s*K[0]/1000,
          normal="float32 HWC in CV; PIL RGB truncation; DN reader applies diag(1,-1,-1) once",
          interpolation="OpenCV bilinear RGB, mean pad; torch bilinear align_corners=False, antialias=False",
          train_source="only train PNG normals, sensor Z unchanged", source_transforms_sha256=sha(SCENE / "transforms.json"))
    write(STATE / "prior_identity.json", info)
    ident = sha(STATE / "prior_identity.json")
    for folder in ("normal_cv", "kappa", "depth_canonical", "depth_m", "receipts"):
        (PRIOR / folder).mkdir(parents=True, exist_ok=True)
    norm_dir = SCENE / "normals_from_pretrain"
    norm_dir.mkdir(exist_ok=True)
    def restore(value):
        demand(tuple(value.shape[-2:]) == (616, 1064), "prediction canvas")
        crop = value[..., top:top+hh, left:left+ww]
        return F.interpolate(crop.float(), (h, w), mode="bilinear", align_corners=False, antialias=False)
    rows = []
    with torch.inference_mode():
        for i, frame in enumerate(meta["frames"]):
            path = SCENE / frame["file_path"]
            stem = path.stem
            receipt = PRIOR / "receipts" / (stem + ".json")
            if receipt.is_file():
                row = read(receipt)
                demand(row["identity"] == ident and row["id"] == stem, stem)
                if all((ROOT / r["path"]).is_file() and (ROOT/r["path"]).stat().st_size == r["bytes"] and
                       sha(ROOT/r["path"]) == r["sha256"] for r in row["outputs"]):
                    rows.append(row)
                    continue
            im = cv2.imread(str(path))
            demand(im is not None and im.shape == (h, w, 3), stem)
            rgb = cv2.resize(im[..., ::-1], (ww, hh), interpolation=cv2.INTER_LINEAR)
            padded = np.empty((616, 1064, 3), np.float32)
            padded[:] = mean
            padded[top:top+hh, left:left+ww] = rgb
            inp = torch.from_numpy(((padded-mean)/std).transpose(2, 0, 1).copy()).unsqueeze(0).cuda()
            z, _, outputs = model.inference({"input": inp})
            vector = restore(outputs["prediction_normal"])
            normals = vector[0, :3].permute(1, 2, 0)
            length = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
            invalid = int((~torch.isfinite(length)).sum())
            zero = int((length == 0).sum())
            demand(invalid == 0, "nonfinite normal " + stem)
            normals = (normals / torch.where(length > 0, length, 1.)).cpu().numpy()
            kappa = vector[0, 3].cpu().numpy()
            canonical = restore(z.reshape(1, 1, 616, 1064))[0, 0].cpu().numpy()
            metric = canonical * np.float32(info["depth_scale"])
            outputs_meta = []
            for folder, array in (("normal_cv", normals), ("kappa", kappa), ("depth_canonical", canonical), ("depth_m", metric)):
                target = PRIOR / folder / (stem + ".npy")
                np.save(target, array, allow_pickle=False)
                outputs_meta.append(member(target))
            target = norm_dir / (stem + ".png")
            Image.fromarray((np.clip((normals+1)*.5, 0, 1)*255).astype(np.uint8)).save(target, compress_level=1)
            outputs_meta.append(member(target))
            row = dict(id=stem, identity=ident, outputs=outputs_meta, zero_length=zero, nonfinite_normal=invalid,
                       nonfinite_kappa=int((~np.isfinite(kappa)).sum()), nonpositive_or_invalid_depth=int((~(np.isfinite(metric)&(metric>0))).sum()))
            write(receipt, row)
            rows.append(row)
            if (i+1) % 10 == 0 or i in (0, len(meta["frames"])-1):
                write(STATE / "prior_progress.json", dict(completed=i+1, total=len(meta["frames"]), last=stem))
                print(f"ViT-Large {i+1}/{len(meta['frames'])} {stem}", flush=True)
    demand(len(rows) == len({r["id"] for r in rows}) == 1846, "full priors")
    write(PRIOR / "manifest.json", rows)
    write(REPORT / "normal_prior.json", dict(**info, completed=len(rows), manifest=member(PRIOR / "manifest.json"),
          zero_length=sum(r["zero_length"] for r in rows), nonfinite_normal=sum(r["nonfinite_normal"] for r in rows)))


if __name__ == "__main__":
    main()
