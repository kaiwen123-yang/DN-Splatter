"""Use the measured B0 configuration and observe B1 sampling/refinement in the formal trainer."""
from contextlib import ExitStack
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from dn_splatter.dn_model import DNSplatterModel as Model

from capture_metric3d_common import BASE, B0, B0_REPORT, REPORT, RUN, SCENE, STATE, P, demand, read, sha, write


def frozen_config():
    config_path = Path(read(BASE / "training_complete.json")["config"])
    demand(sha(config_path) == P["baseline"]["configuration_sha256"], "trusted config before loading Python YAML")
    before = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    cfg = copy.deepcopy(before)
    demand(cfg.output_dir == BASE/"training" and cfg.pipeline.datamanager.dataparser.data == B0 and
           cfg.load_dir is None and cfg.load_checkpoint is None and cfg.max_num_iterations == 30000, "from scratch")
    cfg.output_dir = RUN / "training"
    cfg.pipeline.datamanager.dataparser.data = SCENE
    cfg.timestamp = "{timestamp}"
    check = copy.deepcopy(cfg)
    check.output_dir = before.output_dir
    check.pipeline.datamanager.dataparser.data = before.pipeline.datamanager.dataparser.data
    check.timestamp = before.timestamp
    demand(yaml.dump(check) == yaml.dump(before), "configuration semantic diff")
    write(REPORT / "config_diff.json", dict(baseline_file=str(config_path), baseline_sha256=sha(config_path),
          fields=dict(output_dir=[str(before.output_dir), str(cfg.output_dir)],
          dataparser_data=[str(B0), str(SCENE)], timestamp=[before.timestamp, cfg.timestamp]),
          other_configuration_differences=[], input_content_change="normal PNG prior only",
          sampler="historical DN next_train, unchanged", seed=cfg.machine.seed))
    return cfg


def main():
    import capture_full_train as full
    import capture_s01_train as trainer

    names = [Path(n).stem for n in read(SCENE / "transforms.json")["train_filenames"]]
    tag = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    with ExitStack() as scope:
        sampled = csv.writer(scope.enter_context((STATE / f"samples-{tag}.csv").open("w", buffering=1, newline="")))
        sampled.writerow(["step", "image_idx", "frame", "gaussians"])
        events = scope.enter_context((STATE / f"refine-{tag}.jsonl").open("w", buffering=1))
        original_loss, original_refine = Model.get_loss_dict, Model.refinement_after
        original_cull, original_split, original_dup = Model.cull_gaussians, Model.split_gaussians, Model.dup_gaussians
        event = {}

        def loss(model, outputs, batch, *args, **kwargs):
            result = original_loss(model, outputs, batch, *args, **kwargs)
            if model.training:
                idx = int(batch["image_idx"])
                sampled.writerow([int(model.step), idx, names[idx], int(model.num_points)])
            return result

        def refine(model, optimizers, step):
            event.clear()
            event.update(step=int(step), before=int(model.num_points), split_parents=0, split_children=0,
                         duplicates=0, opacity=0, world_scale=0, screen_scale=0, parent_replacement=0, cull_union=0)
            result = original_refine(model, optimizers, step)
            event["after"] = int(model.num_points)
            demand(event["after"] == event["before"] + event["split_children"] + event["duplicates"] - event["cull_union"],
                   "refinement accounting")
            print(json.dumps(event), file=events)
            return result

        def split(model, mask, samps):
            result = original_split(model, mask, samps)
            event.update(split_parents=int(mask.sum()), split_children=len(result["means"]))
            return result

        def dup(model, mask):
            result = original_dup(model, mask)
            event["duplicates"] = len(result["means"])
            return result

        def cull(model, extra_cull_mask=None):
            with torch.no_grad():
                low = (torch.sigmoid(model.opacities) < model.config.cull_alpha_thresh).reshape(-1)
                event["opacity"] = int(low.sum())
                union = low.clone()
                if extra_cull_mask is not None:
                    event["parent_replacement"] = int(extra_cull_mask.sum())
                    union |= extra_cull_mask.reshape(-1)
                if model.step > model.config.refine_every * model.config.reset_alpha_every:
                    big = torch.exp(model.scales).max(dim=-1).values > model.config.cull_scale_thresh
                    union |= big
                    event["world_scale"] = int(big.sum())
                    if model.step < model.config.stop_screen_size_at and model.max_2Dsize is not None:
                        screen = model.max_2Dsize.reshape(-1) > model.config.cull_screen_size
                        union |= screen
                        event["screen_scale"] = int(screen.sum())
                deleted = original_cull(model, extra_cull_mask)
                demand(torch.equal(union, deleted.reshape(-1)), "observed vs actual cull union")
                event["cull_union"] = int(deleted.sum())
                return deleted
        for target, method, wrapper in ((full, "config", frozen_config), (trainer, "config", frozen_config),
                (Model, "get_loss_dict", loss), (Model, "refinement_after", refine),
                (Model, "split_gaussians", split), (Model, "dup_gaussians", dup), (Model, "cull_gaussians", cull)):
            scope.enter_context(patch.object(target, method, wrapper))
        full.main()
    from capture_metric3d_prepare import curves
    curves("B1", RUN)
    records = {}
    for p in sorted(STATE.glob("refine-*.jsonl")):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            records[row["step"]] = row
    samples = {}
    for p in sorted(STATE.glob("samples-*.csv")):
        with p.open() as f:
            for row in csv.DictReader(f):
                samples[int(row["step"])] = row
    demand(sorted(samples) == list(range(30000)), "complete formal sample history")
    import gzip
    with gzip.open(REPORT / "B1_samples.csv.gz", "wt", newline="") as f:
        out = csv.DictWriter(f, fieldnames=["step", "image_idx", "frame", "gaussians"])
        out.writeheader()
        out.writerows(samples[s] for s in sorted(samples))
    write(REPORT / "B1_refinement.json", dict(events=[records[s] for s in sorted(records)],
          definition="Reason masks overlap. cull_union is the verified actual number removed."))


if __name__ == "__main__":
    main()
