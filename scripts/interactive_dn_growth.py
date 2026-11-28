"""Historical B3 growth budget and projection-based allocation for R4/R7."""
from contextlib import contextmanager, ExitStack
import json
from unittest.mock import patch

import numpy as np
import torch

from interactive_dn_supervision import digest


def freeze_cap(path, expected_digest):
    if digest(path) != expected_digest:
        raise ValueError("B3 refinement log SHA-256 differs from the published receipt")
    events = {}
    for line in path.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        if row["after"] != row["before"] + row["split_children"] + row["duplicates"] - row["cull_union"]:
            raise ValueError("inconsistent B3 refinement accounting")
        # The last observation is authoritative after restoring a checkpoint.
        events[row["step"]] = row
    if sorted(events) != list(range(0,240000,800)) or sum(r["eligible"] for r in events.values()) != 115:
        raise ValueError("incomplete historical B3 refinement curve")
    return {"sha256": expected_digest, "events": len(events),
            "cap": max(max(r["before"],r["after"]) for r in events.values()),
            "definition": "largest published surviving population, including initialization"}


class Growth:
    """Choose parent IDs from projected, visible Gaussians with independent metric mode evidence."""
    def __init__(self, cap, packets=None):
        self.cap = cap
        self.packets = packets
        self.strength = self.tangent = None
        self.frame = self.camera = None
        self.before = None
        self.event = {}

    def snapshot(self):
        return {"strength":self.strength, "tangent":self.tangent}

    def restore(self, state, device):
        for k in ("strength","tangent"):
            v = state[k]
            setattr(self, k, None if v is None else v.to(device))

    @torch.no_grad()
    def accumulate(self, model):
        if self.packets is None or self.frame is None or model.step >= model.config.stop_split_at:
            return
        rec = self.packets.get(self.frame)
        if not len(rec):
            return
        cam = self.camera
        visible = model.vis_indices
        h, w = model.last_size
        if (h,w) != self.packets.shape or model.xys.shape[1] != model.num_points:
            raise ValueError("projection identity or size differs from the training packet")
        xy = model.xys[0,visible].detach().floor().long()
        inside = (xy[:,0] >= 0) & (xy[:,0] < w) & (xy[:,1] >= 0) & (xy[:,1] < h)
        visible, xy = visible[inside], xy[inside]
        pix = xy[:,1] * w + xy[:,0]
        # Each stored row has verified foreground/background donors. Membership at
        # the projected centre is an overlap approximation, never a rasterizer ID.
        def as_tensor(field):
            return torch.as_tensor(np.array(rec[field],copy=True),device=model.device)
        sparse = as_tensor("index").long()
        where = torch.searchsorted(sparse,pix).clamp_max(len(rec)-1)
        match = sparse[where] == pix
        ids, where = visible[match], where[match]
        z = model.depths.reshape(-1)[ids]
        mode, rival = as_tensor("mode")[where],as_tensor("other")[where]
        near,far=torch.minimum(mode,rival),torch.maximum(mode,rival)
        keep = ((z-mode).abs()<=torch.maximum(mode*.03,mode.new_tensor(.05))) & (
            (z-rival).abs()>torch.maximum(rival*.03,rival.new_tensor(.05)))
        ids, where = ids[keep], where[keep]
        if not ids.numel():
            return
        # World differential corresponding to the observed pixel tangent. The DN
        # camera convention has positive X right and positive Y up.
        txy = as_tensor("tangent")[where]
        local = torch.stack((txy[:,0]/cam.fx.item(),-txy[:,1]/cam.fy.item(),torch.zeros_like(txy[:,0])),dim=-1)
        tangent = torch.nn.functional.normalize(local @ cam.camera_to_worlds[0,:3,:3].T,dim=-1)
        score = model.xys.absgrad[0,ids].norm(dim=-1)
        score = score * ((far-near)[keep] / torch.maximum(near[keep],near.new_tensor(.3))).clamp_max(1)
        if self.strength is None:
            self.strength = torch.zeros(model.num_points,device=model.device)
            self.tangent = torch.zeros((model.num_points,3),device=model.device)
        improves = torch.isfinite(score) & (score>self.strength[ids])
        self.strength[ids[improves]] = score[improves]
        self.tangent[ids[improves]] = tangent[improves]

    def alloc_split(self, model, native, mask, samples):
        aligned = torch.zeros_like(mask)
        if self.strength is not None:
            supported = self.strength > 0
            size = model.scales.exp().amax(dim=-1)
            add_pool = torch.where(supported & ~mask & (size > model.config.densify_size_thresh)
                                   & (model.opacities.sigmoid().flatten() >= model.config.cull_alpha_thresh))[0]
            drop_pool = torch.where(mask & ~supported)[0]
            quota = min(int(mask.sum())//5, add_pool.numel(), drop_pool.numel())
            if quota:
                drops = drop_pool[torch.argsort(self.general[drop_pool],stable=True)[:quota]]
                adds = add_pool[torch.argsort(self.strength[add_pool],descending=True,stable=True)[:quota]]
                mask[drops], mask[adds] = False, True
            aligned = mask & supported
            self.event.update(geometry_supported=int(supported.sum()), reallocated_parents=quota,
                              tangent_split_parents=int(aligned.sum()))
        indices = torch.where(mask)[0]
        self.split_score = self.general[indices].repeat(samples)
        if not aligned.any():
            return native(model, mask, samples)
        if samples != 2:
            raise ValueError("directional split requires the fixed two-child protocol")
        old = model.means[aligned].detach().clone()
        t = self.tangent[aligned]
        from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
        q = torch.nn.functional.normalize(model.quats[aligned],dim=-1)
        axes = torch.bmm(quat_to_rotmat(q).transpose(-1,-2),t[...,None]).squeeze(-1)
        displacement = t * ((axes * model.scales[aligned].exp()).norm(dim=-1,keepdim=True) * .5)
        result = native(model,mask,samples)
        selected = torch.where(aligned[indices])[0]
        result["means"][selected] = old - displacement
        result["means"][selected+len(indices)] = old + displacement
        return result

    def enforce(self, model, native, mask):
        if mask is None:
            return native(model)
        # Native culling sees the actual new scales and parent masks. Additional
        # removals are restricted to children that would otherwise survive.
        ordinary = (model.opacities.sigmoid().flatten() < model.config.cull_alpha_thresh) | mask
        if model.step > model.config.refine_every * model.config.reset_alpha_every:
            ordinary |= model.scales.exp().amax(dim=-1) > model.config.cull_scale_thresh
            if model.step < model.config.stop_screen_size_at and model.max_2Dsize is not None:
                ordinary |= model.max_2Dsize.flatten() > model.config.cull_screen_size
        overflow = max(0,int((~ordinary).sum())-self.cap)
        if overflow:
            survivors = torch.where(~ordinary[self.before:])[0]
            if overflow > len(survivors):
                raise ValueError("common cap cannot be satisfied from new slots")
            rank = torch.cat((self.split_score,self.dup_score))
            losing = survivors[torch.argsort(rank[survivors], stable=True)[:overflow]]
            mask = mask.clone()
            mask[self.before+losing] = True
        self.event["cap_removed_children"] = overflow
        return native(model,mask)

    @contextmanager
    def hooks(self, Model, history, audit, resource_snapshot):
        from capture_full_train import replacement
        original = Model.refinement_after
        diff = []
        refine = replacement(original, "> self.num_train_data + self.config.refine_every",
                             "> 3 * self.num_train_data + self.config.refine_every",diff)
        audit.joinpath("recovery_gate.patch").write_text("".join(diff),encoding="utf8")
        after = Model.after_train
        split, dup, cull = Model.split_gaussians, Model.dup_gaussians, Model.cull_gaussians

        def observe(model, step):
            result = after(model,step)
            self.accumulate(model)
            return result

        def event(model, optimizers, step):
            self.before = int(model.num_points)
            self.event = {"step":step,"before":self.before,"cap":self.cap}
            if model.xys_grad_norm is not None:
                self.general = (model.xys_grad_norm / model.vis_counts).nan_to_num()
            result = refine(model,optimizers,step)
            self.event["after"] = int(model.num_points)
            if self.event["after"] > self.cap:
                raise ValueError("post-refinement Gaussian population exceeds frozen cap")
            if "split_slots" in self.event:
                self.event["resources"] = resource_snapshot()
            print(json.dumps(self.event), file=history, flush=True)
            if step == 5600 and not (audit/"first_refinement.json").exists():
                from interactive_dn_cache import atomic_json
                atomic_json(audit/"first_refinement.json",self.event)
            self.strength = self.tangent = self.split_score = self.dup_score = self.general = None
            return result

        def record_dup(model, mask):
            self.dup_score = self.general[mask]
            self.event["duplicate_slots"] = int(mask.sum())
            return dup(model,mask)

        def record_split(model,mask,n):
            self.event["split_slots"] = int(mask.sum())
            return self.alloc_split(model,split,mask,n)

        with ExitStack() as scope:
            for name,impl in (("after_train",observe),("refinement_after",event),("split_gaussians",record_split),
                              ("dup_gaussians",record_dup),("cull_gaussians",lambda m, mask=None:self.enforce(m,cull,mask))):
                scope.enter_context(patch.object(Model,name,impl))
            yield
