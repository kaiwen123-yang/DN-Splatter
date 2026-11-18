"""Exercise the two historical rasterization paths on a registered VR Room sample."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from gsplat import rasterize_gaussians
from gsplat.rendering import rasterization


def main():
    run = Path(os.environ["DN_RUN_ROOT"])
    scene = Path(os.environ["DN_SCENE_DIR"])
    frame = json.loads((run / "frames.json").read_text())["frames"]["train"][0]
    with Image.open(scene / frame["rgb"]) as image:
        color = np.asarray(image.convert("RGB").resize((1, 1))).reshape(3) / 255
    with Image.open(scene / frame["normal"]) as image:
        prior = np.asarray(image.convert("RGB").resize((1, 1))).reshape(3) / 127.5 - 1
    with Image.open(scene / frame["depth"]) as image:
        depth = np.asarray(image)
        z = float(np.median(depth[depth > 0])) * 0.001
    def tensor(values, grad=False):
        return torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=grad)
    means = tensor([[-0.03*z, 0, z], [0.04*z, 0.015*z, 1.1*z]], True)
    colors = tensor(np.stack([color, np.clip(color * 0.9 + 0.02, 0, 1)]), True)
    normals = tensor(np.stack([prior, prior]), True)
    opacities = tensor([0.7, 0.6], True)
    im, alpha, info = rasterization(means=means, quats=tensor([[1,0,0,0]]*2),
        scales=tensor([[0.08*z, 0.06*z, 0.04*z]]*2), opacities=opacities, colors=colors,
        viewmats=torch.eye(4, device="cuda")[None], Ks=tensor([[[45,0,16],[0,45,16],[0,0,1]]]),
        width=32, height=32, packed=False, tile_size=16, render_mode="RGB+ED")
    nim = rasterize_gaussians(info["means2d"][0].detach(), info["depths"][0], info["radii"][0],
        info["conics"][0], info["tiles_per_gauss"][0], normals, opacities[:,None], 32, 32, 16)
    if not all(torch.isfinite(t).all() for t in (im, alpha, nim)) or not (alpha > 0).any():
        raise RuntimeError("Nonfinite or empty RGB/depth/normal raster")
    norms = {}
    for name, loss, variable in [("rgb", im[..., :3].sum(), colors),
                                 ("depth", im[..., 3].sum(), means), ("legacy_normal", nim.sum(), normals)]:
        grad, = torch.autograd.grad(loss, variable, retain_graph=True)
        if not torch.isfinite(grad).all() or grad.norm() == 0:
            raise RuntimeError(f"Invalid gradient: {name}")
        norms[name] = float(grad.norm().item())
    optimizer = torch.optim.Adam([means, colors, normals, opacities], lr=1e-4)
    before = means.detach().clone()
    (im.mean() + nim.mean()).backward()
    optimizer.step()
    if torch.equal(before, means) or not torch.isfinite(means).all():
        raise RuntimeError("Finite optimizer update failed")
    result = {"sample": frame["rgb"], "gradient_norms": norms, "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "compute_capability": torch.cuda.get_device_capability(),
              "gsplat_backends": ["csrc", "csrc_legacy"], "optimizer_update_verified": True}
    (run / "blackwell.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", action="store_true", help="Run this optional isolated diagnostic")
    if parser.parse_args().standalone:
        main()
    else:
        print("Standalone diagnostic skipped. The formal run proceeds.")
