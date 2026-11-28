"""Generate metric candidate surfaces, independent votes, and DN supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

from interactive_depth_geometry import (Footprint, SOURCE_NAMES, atomic_npz, edge_pairs, nearest_map,
                                        other_views, primary, project_native, task_mask, tolerance, valid)
from interactive_depth_inputs import Hall, REPS, RUN, digest_json
from lingbot_common import member, read, require, save, sha


BASE_CHANNELS = ("depth_m", "distance_to_visible_px", "RGB_gradient", "near_visible_depth_gradient_m",
                 "abs_difference_to_NN", "abs_difference_to_footprint", "abs_difference_to_cross_near",
                 "abs_difference_to_cross_far", "abs_difference_to_MDM", "support_count", "free_conflict_count",
                 "occlusion_count", "audit_visible_count", "nearest_audit_difference_m", "pose_flag",
                 "NN", "footprint", "cross_near", "cross_far", "MDM", *REPS)


class CaseGeometry:
    def __init__(self, rgb, visible, footprint, other, pose_flag=False):
        self.nn, self.distance, self.nn_src = nearest_map(visible)
        self.footprint = footprint
        self.other, self.pose_flag = other, pose_flag
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=.125)
        gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=.125)
        self.rgb_grad = np.hypot(gx, gy)
        kernel = np.ones((3,3),np.uint8)
        self.z_grad = cv2.dilate(self.nn,kernel) - cv2.erode(self.nn,kernel)

    def table(self, flat, mdm_depth=None):
        flat = np.asarray(flat)
        mdm = np.zeros(len(flat),np.float32) if mdm_depth is None else np.asarray(mdm_depth,np.float32)
        if mdm.ndim == 2:
            mdm = mdm.ravel()[flat]
        require(mdm.shape == (len(flat),), "MDM scoring grid")
        depths = np.column_stack((self.nn.ravel()[flat], self.footprint.ravel()[flat],
                                  self.other.near.ravel()[flat], self.other.far.ravel()[flat], mdm))
        observations = np.column_stack([z.ravel()[flat] for z in self.other.audit]) if self.other.audit else np.empty((len(flat),0),np.float32)
        local = np.column_stack((self.distance.ravel()[flat], self.rgb_grad.ravel()[flat],
                                 self.z_grad.ravel()[flat], np.full(len(flat), self.pose_flag, np.float32)))
        return depths, observations, local.astype(np.float32,copy=False)


def evidence(z, observations, audit_bits):
    """Counts S (support), C (free space) and O (occlusion); each vote has its own source group."""
    n = len(z)
    support = np.zeros(n,np.uint8)
    conflict = support.copy()
    occluded = support.copy()
    bits = np.zeros(n,np.uint16)
    total = np.zeros(n,np.float32)
    nearest = np.full(n,np.inf,np.float32)
    good = valid(z)
    for j, bit in enumerate(audit_bits):
        ref = observations[:,j]
        measurable = good & valid(ref)
        error = np.zeros(n, np.float32)
        np.subtract(z,ref,out=error,where=measurable)
        allowed = tolerance(ref)
        agrees = measurable & (np.abs(error)<=allowed)
        support += agrees
        conflict += measurable & (error < -allowed)
        occluded += measurable & (error > allowed)
        bits[agrees] |= np.uint16(1 << int(bit))
        total += np.where(agrees, np.abs(error), 0.)
        np.minimum(nearest, np.abs(error), out=nearest, where=measurable)
    return support, conflict, occluded, bits, total, nearest


def feature_table(depths, observations, local, audit_bits, representation):
    """Nx5xF. Invalid candidates are excluded from loss/acceptance, retained in denominators."""
    n, count = depths.shape
    f = np.zeros((n,count,len(BASE_CHANNELS)),np.float32)
    good = valid(depths)
    clean = np.where(good, depths, 0.)
    f[:,:,0] = clean
    f[:,:,1:4] = np.nan_to_num(local[:,:3],posinf=2048.)[:,None,:]
    for c in range(count):
        comparable = good & good[:,c,None]
        f[:,c,4:9] = np.where(comparable, np.abs(clean-clean[:,c,None]), -1.)
        s, k, o, _, _, closest = evidence(depths[:,c],observations,audit_bits)
        f[:,c,9], f[:,c,10], f[:,c,11] = s, k, o
        f[:,c,12] = (observations>0).sum(axis=1)
        f[:,c,13] = np.where(np.isfinite(closest),closest,-1.)
        f[:,c,15+c] = 1.
    f[:,:,14] = local[:,None,3]
    f[:,:,20+REPS.index(representation)] = 1.
    return f


def hard_select(depths, observations, audit_bits):
    n = len(depths)
    best = np.full(n, -np.inf, np.float32)
    choice = np.full(n, -1, np.int8)
    bits = np.zeros(n, np.uint16)
    any_valid = np.zeros(n,bool)
    conflicted = np.zeros(n,bool)
    for c, z in enumerate(depths.T):
        s,k,_,b,err,_ = evidence(z,observations,audit_bits)
        in_range = primary(z)
        any_valid |= in_range
        conflicted |= in_range & (k>0)
        # Maximize independent support, then minimize mean inlier residual. Never average proposals.
        score = s.astype(np.float32) - err / np.maximum(s,1)
        update = in_range & (s>=2) & (k==0) & (score>best)
        best[update], choice[update], bits[update] = score[update],c,b[update]
    reason = np.where(choice>=0,0,np.where(~any_valid,2,np.where(conflicted,4,3))).astype(np.uint8)
    return choice, bits, reason


class HallFrame:
    def __init__(self, hall, fp, row):
        self.row = row
        self.hall, self.fp = hall, fp
        self.raw = hall.native(row)
        self.full = hall.depth(row)
        self.rgb = hall.image(row)
        self.natural = hall.canvas & ~valid(self.full)
        if self.raw is None:
            self.winner = np.where(valid(self.full), np.arange(self.full.size).reshape(self.full.shape), -1).astype(np.int32)
            self.native_target = None
            self.occ = np.zeros(self.full.shape, bool)
        else:
            projected, self.winner, self.native_target, self.occ = project_native(hall,self.raw)
            require(np.array_equal(projected,self.full), "Native winner disagrees with signed full-range Z " + row["id"])
        self.pairs = edge_pairs(self.full)
        self.other = other_views(hall,row)

    def condition(self, kind=None):
        if kind:
            hidden, native_hide, score = task_mask(self.row["id"],kind,self.full,self.raw,self.native_target,self.pairs)
        else:
            hidden = np.zeros(self.full.shape, bool)
            native_hide = None
            score = self.natural
        visible = np.where(hidden,0.,self.full).astype(np.float32)
        if self.raw is not None:
            raw = self.raw if native_hide is None else np.where(native_hide, np.uint16(0), self.raw)
            if kind:
                sparse, winner, _, _ = project_native(self.hall,raw)
                require(np.array_equal(sparse,visible), "Hidden source survived or changed another point")
            else:
                winner = self.winner
            foot, foot_src = self.fp.splat(raw,self.hall.rays)
            if native_hide is not None:
                # A source requires its original four neighbours after hiding. Every emitted
                # interpolant points to a surviving source; no unmasked candidate cache is read.
                parents = foot_src[foot_src>=0]
                require(not native_hide.ravel()[parents].any(), "Footprint has a hidden parent")
        else:
            winner = np.where(hidden,-1,self.winner)
            foot = np.zeros(self.full.shape,np.float32)
            foot_src = np.full(self.full.shape,-1,np.int32)
        ge = CaseGeometry(self.rgb, visible, foot, self.other, self.row["suspect_reference_pose"])
        return dict(hidden=hidden, native_hide=native_hide, score=score, visible=visible, winner=winner,
                    foot=foot, foot_src=foot_src, geometry=ge)


def condition_z(data, representation):
    z = data["visible"]
    if representation == "full_range_sparse":
        return z
    if representation == "primary_range_only":
        return np.where(primary(z),z,0.)
    require(representation == "footprint_aware", "Input representation")
    out = np.where(valid(z),z,data["foot"])
    out[data["hidden"]] = 0.
    return out


def export_selected(folder, frame, data, mdm=None, decision=None, probability=None):
    flat = np.flatnonzero(frame.natural)
    shape = frame.full.shape
    selected = np.zeros(shape,np.float32)
    source = np.zeros(shape,np.uint8)
    reject = np.ones(shape,np.uint8)  # not a natural hole
    support = np.zeros(shape,np.uint16)
    visibility = np.zeros(shape,np.uint8)
    visibility[frame.natural] = 1   # unknown
    risk = np.full(shape,np.nan,np.float32)
    parent = np.full(shape,-1,np.int32)
    parent_frame = np.full(shape,-1,np.int16)
    geo = data["geometry"]
    fid = int(frame.row["id"].split("_")[-1])
    for start in range(0,len(flat),32768):
        sub = flat[start:start+32768]
        depths,obs,local = geo.table(sub, mdm)
        if decision is None:
            choice, bits, reason = hard_select(depths,obs,frame.other.audit_bits)
        else:
            choice, probs = decision(sub,depths,obs,local)
            accepted = choice>=0
            z = depths[np.arange(len(sub)),np.maximum(choice,0)]
            _,_,_,bits,_,_ = evidence(z,obs,frame.other.audit_bits)
            bits[~accepted] = 0
            reason = np.where(accepted,0,np.where(primary(depths).any(axis=1),5,2)).astype(np.uint8)
            risk.flat[sub] = np.where(np.isfinite(probs),1.-probs,np.nan)
        accepted = choice>=0
        reject.flat[sub] = reason
        support.flat[sub] = bits
        visibility.flat[sub[accepted]] = np.where(bits[accepted]>0,3,2)
        src_flat = sub[accepted]
        src_kind = choice[accepted]
        chosen = depths[np.flatnonzero(accepted),src_kind]
        require(np.all(primary(chosen)), "Invalid selected training target")
        selected.flat[src_flat] = chosen
        source.flat[src_flat] = src_kind+1
        for c in range(4):
            p = src_flat[src_kind==c]
            if c==0:
                j = geo.nn_src.ravel()[p]
                parent.flat[p], parent_frame.flat[p] = data["winner"].ravel()[j], fid
            elif c==1:
                parent.flat[p], parent_frame.flat[p] = data["foot_src"].ravel()[p], fid
            elif c==2:
                parent.flat[p], parent_frame.flat[p] = frame.other.near_id.ravel()[p],frame.other.near_frame.ravel()[p]
            else:
                parent.flat[p], parent_frame.flat[p] = frame.other.far_id.ravel()[p],frame.other.far_frame.ravel()[p]
    path = folder / (frame.row["id"]+".npz")
    atomic_npz(path, selected_depth=selected, accepted=source>0, natural_hole=frame.natural,
               selected_source=source, risk=risk, visibility_state=visibility,reject_reason=reject,
               source_parent_index=parent,source_parent_frame=parent_frame,support_group_bits=support,
               donor_groups=frame.other.donor_groups)
    info = member(path)
    return dict(id=frame.row["id"],path=path.name,bytes=info["bytes"],sha256=info["sha256"],
                accepted_pixels=int((source>0).sum()),natural_holes=len(flat),
                accepted_by_source=np.bincount(source[source>0],minlength=6)[1:].tolist(),
                rejected_by_reason=np.bincount(reject.ravel()[flat],minlength=6).tolist())


def boundary(folder, frame):
    near_list, far_list, tangents, fg_bits, bg_bits = [],[],[],[],[]
    pairs_n = 0
    for f,b,t in frame.pairs:
        pairs_n += len(f)
        for i in range(0,len(f),32768):
            x, y = f[i:i+32768], b[i:i+32768]
            def votes(ids):
                obs = np.column_stack([z.ravel()[ids] for z in frame.other.audit]) if frame.other.audit else np.empty((len(ids),0))
                return evidence(frame.full.ravel()[ids],obs,frame.other.audit_bits)
            sf,cf,_,bf,_,_ = votes(x)
            sb,cb,_,bb,_,_ = votes(y)
            keep = (sf>=1)&(sb>=1)&(cf==0)&(cb==0)
            near_list.append(x[keep]); far_list.append(y[keep])
            tangents.append(np.broadcast_to(t,(int(keep.sum()),2)))
            fg_bits.append(bf[keep]); bg_bits.append(bb[keep])
    x = np.concatenate(near_list) if near_list else np.empty(0,np.int32)
    y = np.concatenate(far_list) if far_list else np.empty(0,np.int32)
    path = folder/(frame.row["id"]+".npz")
    atomic_npz(path,flat_index=x, background_flat_index=y,foreground_z_m=frame.full.ravel()[x],
               background_z_m=frame.full.ravel()[y],tangent_xy=np.concatenate(tangents) if tangents else np.empty((0,2),np.float32),
               fg_support_group_bits=np.concatenate(fg_bits) if fg_bits else np.empty(0,np.uint16),
               bg_support_group_bits=np.concatenate(bg_bits) if bg_bits else np.empty(0,np.uint16),
               donor_groups=frame.other.donor_groups)
    return dict(id=frame.row["id"],path=path.name,bytes=path.stat().st_size,sha256=sha(path),
                candidates=pairs_n,accepted_pixels=len(x))


def provenance(folder, frame, data):
    vis = valid(frame.full)
    cause = np.full(frame.full.shape,6,np.uint8)
    cause[frame.fp.outside]=5
    cause[frame.natural & valid(data["foot"])]=3
    cause[vis & ~primary(frame.full)]=2
    cause[primary(frame.full)]=1
    cause[~frame.hall.canvas]=0
    p = folder/(frame.row["id"]+".npz")
    atomic_npz(p,cause=cause, point_source_id=frame.winner, footprint_source_id=data["foot_src"],
               footprint_depth_m=data["foot"], source_collision=frame.occ, range_valid=primary(frame.full))
    return dict(id=frame.row["id"],path=p.name,bytes=p.stat().st_size,sha256=sha(p),
                cause_counts=np.bincount(cause.ravel(),minlength=7).tolist(),competition_pixels=int(frame.occ.sum()))


def frame_result(out, fid, identity, build):
    path = out/"receipts"/(fid+".json")
    if path.exists():
        receipt = read(path)
        require(receipt["identity"]==identity, "Input/code changed; start a separate attempt: "+fid)
        for directory, f in receipt["files"].items():
            p = out/directory/f["path"]
            require(p.stat().st_size==f["bytes"] and sha(p)==f["sha256"], "Damaged result: "+str(p))
        return receipt
    receipt = dict(frame=fid,identity=identity,files=build())
    save(path,receipt)
    return receipt


def worker_init(directory, identity):
    global HALL, FP, IDENTITY
    cv2.setNumThreads(1)
    HALL = Hall(directory)
    FP = Footprint(HALL)
    IDENTITY = identity


def sensor_worker(row):
    def work():
        frame = HallFrame(HALL, FP, row)
        data = frame.condition()
        return {"supervision/R1":export_selected(HALL.directory/"supervision/R1",frame,data),
                "boundary":boundary(HALL.directory/"boundary",frame),
                "provenance":provenance(HALL.directory/"provenance",frame,data)}
    return frame_result(HALL.directory,row["id"],IDENTITY,work)


def finalize(out, rows, dataset):
    extras = dict(source_codes=["none",*SOURCE_NAMES],visibility_states=["outside","unknown","hypothesized","supported"],
                  rejection_codes=["accepted","outside_natural_hole","no_range_valid_candidate","insufficient_independent_support",
                                   "free_space_conflict","quality_rejection"],
                  parent_index="NN/footprint: native flattened measurement (plus four neighbors for footprint), else"
                  " registered pixel when native unavailable. Cross-view: source registered pixel; resolve using its"
                  " provenance/point_source_id. MDM: conditioned RGB/depth input lineage; no single physical parent.",
                  support_group_bits="bit i addresses donor_groups[i]; source-frame donor mapping is frozen in freeze.json",
                  hard_risk="NaN; hard gates do not provide a calibrated probability",
                  boundary_support="Measured source on each side + >=1 independent 10s donor on that side and zero free-space conflicts.",
                  cause_codes=["outside_color_canvas","measured_primary","measured_other_range",
                               "projected_continuous_native_cell","reserved","outside_native_FOV_throughout_primary_range","unknown"])
    for key in rows[0]["files"]:
        frames = [r["files"][key] for r in rows]
        save(out/key/"manifest.json",dict(experiment=key,scope="train",shape=list(reversed(dataset["image_wh"])),
             units="m",frames=frames,freeze_sha256=sha(out/"freeze.json"),implementation=rows[0]["identity"],
             accepted_pixels=sum(r.get("accepted_pixels",0) for r in frames),**extras))


def main(directory, workers):
    dataset=read(directory/"freeze.json")
    identity=digest_json(dict(freeze=sha(directory/"freeze.json"), code=sha(Path(__file__)),
                              geometry=sha(Path(__file__).with_name("interactive_depth_geometry.py"))))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init,initargs=(directory,identity)) as pool:
        for r in pool.map(sensor_worker,dataset["rows"]):
            receipts.append(r)
            if len(receipts)==1 or len(receipts)%20==0:
                print("R1 / lineage / physical edges",len(receipts),"/",len(dataset["rows"]),r["frame"],flush=True)
    finalize(directory,receipts,dataset)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=RUN)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    main(a.out,a.workers)
