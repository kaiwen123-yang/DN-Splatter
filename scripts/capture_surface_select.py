"""Validation-only B3 selection. Inputs use the existing B2 evaluator's metric fields."""
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/capture_surface_quality_protocol.json"
PLAN = json.loads(SPEC.read_bytes())
BASE = PLAN["baseline"]
NAMES = ("supported_poisson", "model_tsdf", "new_model_surface")
NODES = tuple(range(29999, 240000, 30000))
COUNT = PLAN["data"]["val_reference_pixels"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def baseline():
    with gzip.open(ROOT / BASE["model_val_source"], "rt", encoding="utf-8") as f:
        result = json.load(f)
    require(result["split"] == "val" and result["step"] == BASE["selected_step"]
            and result["checkpoint_sha256"] == BASE["selected_checkpoint_sha256"]
            and result["depth_pixel_pooled"]["reference_pixels"] == COUNT, "B2 validation identity")
    return result


def choose_checkpoint(validation_files):
    """Read all eight B3 full-val results; rendering promotion additionally checks pooled depth."""
    old = baseline()
    reference_ids = [r["id"] for r in old["frames"]]
    require(len(set(reference_ids)) == len(reference_ids) == 184, "B2 validation frames")
    reference = old["rgb_frame_means"]
    rows = {}
    for file in validation_files:
        path = Path(file)
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        result = json.loads(raw)
        step = result["step"]
        require(result["split"] == "val" and result["frame_count"] == 184 and step in NODES and step not in rows
                and [r["id"] for r in result["frames"]] == reference_ids, "B3 validation identity or ordering")
        rgb, depth = result["rgb_frame_means"], result["depth_pixel_pooled"]
        require(depth["reference_pixels"] == COUNT, "B3 validation denominator")
        mae = depth["MAE_m"]
        complete_depth = depth["invalid_prediction_pixels"] == 0 and mae is not None and math.isfinite(mae)
        rgb_gate = (all(math.isfinite(rgb[k]) for k in reference) and rgb["psnr"] >= reference["psnr"]
                    and rgb["ssim"] >= reference["ssim"] and rgb["lpips"] <= reference["lpips"])
        rows[step] = dict(step=step, rgb=rgb, rgb_gate=rgb_gate, depth_MAE_m=mae,
                          eligible=complete_depth and rgb_gate, checkpoint_sha256=result["checkpoint_sha256"],
                          val_sha256=hashlib.sha256(raw).hexdigest())
    require(set(rows) == set(NODES), "Incomplete B3 validation curve")
    eligible = [r for r in rows.values() if r["eligible"]]
    chosen = min(eligible, key=lambda r: (r["depth_MAE_m"], -r["rgb"]["psnr"], r["step"])) if eligible else rows[NODES[-1]]
    promote = bool(eligible) and chosen["depth_MAE_m"] <= old["depth_pixel_pooled"]["MAE_m"]
    return dict(selected_step=chosen["step"], final_step=NODES[-1], promoted=promote,
                recommended_rendering_model="B3" if promote else "B2", passed_rgb_and_valid_depth=bool(eligible),
                baseline_step=old["step"], baseline_rgb=reference, baseline_depth_MAE_m=old["depth_pixel_pooled"]["MAE_m"],
                candidates=[rows[step] for step in NODES])


def choose_mesh(candidates, trained):
    """Map (B2|B3, model export name) to full-val measures; keep the incumbent on exact ties."""
    expected = {(model, name) for model in (("B2", "B3") if trained else ("B2",)) for name in NAMES}
    require(set(candidates) == expected, "Missing model export or unexpected sensor-only candidate")
    incumbent = ("B2", "supported_poisson")
    old = json.loads((ROOT / BASE["surface_val_source"]).read_bytes())["groups"]["all"]["supported_poisson"]
    for key in ("reference", "hits", "correct_10", "foreground", "MAE_hit_m"):
        require(candidates[incumbent][key] == old[key], "Frozen B2 incumbent metric: " + key)
    qualified = []
    results = {}
    for identity, m in candidates.items():
        n, hit, correct, foreground = (m[key] for key in ("reference", "hits", "correct_10", "foreground"))
        require(n == COUNT and 0 <= correct <= hit <= n and 0 <= foreground <= hit and
                ((hit == 0 and m["MAE_hit_m"] is None) or (hit > 0 and math.isfinite(m["MAE_hit_m"]) and
                 m["MAE_hit_m"] >= 0)), "Invalid mesh validation counts")
        # Integer comparisons keep the pre-registered percentages exact at the boundaries.
        goal = dict(hit=hit*100 >= 90*n, correct10=correct*100 >= 70*n, foreground=foreground*100 <= 15*n)
        qualifies = goal["hit"] and correct >= old["correct_10"] and foreground <= old["foreground"]
        label = "/".join(identity)
        results[label] = dict(reference=n, hits=hit, correct_10=correct, foreground=foreground,
                              MAE_hit_m=m["MAE_hit_m"], eligible=qualifies, goals=goal)
        if qualifies:
            qualified.append((not all(goal.values()), -correct, foreground, m["MAE_hit_m"], identity != incumbent, label))
    winner = min(qualified)[-1] if qualified else "/".join(incumbent)
    return dict(selected=winner, candidates=results, reference_pixels=COUNT)
