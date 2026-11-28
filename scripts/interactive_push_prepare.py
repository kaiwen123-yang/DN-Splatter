"""Freeze near-duplicate episode groups and the independent PushT geometry/camera prior."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from interactive_physics import proxy_scene, value
from interactive_push_assets import file_hash, write_json


def read_episode(root, e):
    cols = ["observation.state", "action", "frame_index", "episode_index", "timestamp"]
    tab = pq.read_table(root/f"data/chunk-000/episode_{e:06d}.parquet", columns=cols).to_pydict()
    t = np.asarray(tab["timestamp"], float)
    if not (np.all(np.asarray(tab["episode_index"])==e) and np.array_equal(tab["frame_index"], np.arange(len(t)))
            and np.allclose(t, np.arange(len(t))/30, atol=3e-6, rtol=1e-7)):
        raise ValueError(f"Episode {e} frame identity/time mismatch")
    cmd, robot = np.array(tab["action"]), np.array(tab["observation.state"])
    if cmd.shape != (len(t), 2) or robot.shape != cmd.shape or not np.isfinite([cmd, robot]).all():
        raise ValueError(f"Episode {e} invalid planar waypoints")
    return t, cmd, robot


def assign_groups(signatures):
    # Compare sampled metric commanded and executed paths. A common start alone does not duplicate a task.
    n = len(signatures)
    parent = list(range(n))
    def leader(i):
        while parent[i] != i:
            i = parent[i]
        return i
    links = []
    for i in range(n):
        for j in range(i):
            distance = np.linalg.norm(signatures[i]-signatures[j], axis=-1)
            if distance.mean()<.006 and np.quantile(distance, .95)<.015:
                parent[leader(i)] = leader(j)
                links.append([j, i])
    groups = {}
    for i in range(n):
        groups.setdefault(leader(i), []).append(i)
    groups = list(groups.values())
    np.random.default_rng(42).shuffle(groups)
    sizes = [36,12,12]
    # Bounded subset sum enforces complete groups and exact role sizes when feasible.
    states = {(0,0): []}
    for g in groups:
        new = {}
        for (a,b), roles in states.items():
            for role, (i,j) in enumerate(((a+len(g),b), (a,b+len(g)), (a,b))):
                if i <= sizes[0] and j <= sizes[1]:
                    new.setdefault((i,j), roles+[role])
        states = new
    key = min(states, key=lambda k: sum(abs(x-y) for x,y in zip([*k,n-sum(k)],sizes)))
    roles = {name: [] for name in ("fit","calibration","test")}
    for g, k in zip(groups, states[key]):
        roles[list(roles)[k]].extend(g)
    return {k:sorted(s) for k,s in roles.items()}, groups, links


def shape_scene(tblock):
    path=tblock/"data/T_0001/final_data.pkl"
    manifest=json.loads((tblock/"source.json").read_text())
    row=next(i for i in manifest["files"] if i["path"]=="data/T_0001/final_data.pkl")
    if file_hash(path)[0]!=row["sha256"]:
        raise ValueError("Static shape identity mismatch")
    with path.open("rb") as f:
        p=np.asarray(pickle.load(f)["object_points"])
    if p.ndim!=3 or p.shape[0]!=1 or not np.isfinite(p).all():
        raise ValueError("Expected a single frame of sampled static geometry")
    p=p[0]
    lo, hi=p.min(0), p.max(0)
    # The official occupied volume has a wide bar at +y and a narrow stem at -y.
    outer = np.abs(p[:,0]-(lo[0]+hi[0])/2) > (hi[0]-lo[0])*.32
    cut = p[outer,1].min()
    stem = p[p[:,1]<cut-.005]
    if not len(stem) or cut<(lo[1]+hi[1])/2 or abs((hi[0]-lo[0])-.15)>.012:
        raise ValueError("Could not identify the measured T convention")
    zhalf=(hi[2]-lo[2])/2
    scene=proxy_scene()
    scene.update(scene_id="public_xarm7_pusht", asset_source_sha256=row["sha256"])
    scene["frames"]["gravity_source"]="official tabletop workspace; gravity magnitude assumed"
    scene["background"].update(support_center_xy=[.46,0.], support_half_xy=[.38,.40],
                               source="bounded workspace covering the recorded xArm tabletop, extent assumed")
    scene["physics"].update(initial_pusher_xy=[.31,-.2],pusher_radius_m=.010,pusher_z_m=.0225)
    parts=[{"center":[float((hi[0]+lo[0])/2),float((hi[1]+cut)/2),0.],
            "half":[float((hi[0]-lo[0])/2),float((hi[1]-cut)/2),float(zhalf)]},
           {"center":[float((stem[:,0].max()+stem[:,0].min())/2),float((cut+lo[1])/2),0.],
            "half":[float(np.ptp(stem[:,0])/2),float((cut-lo[1])/2),float(zhalf)]}]
    obj=scene["objects"][0]
    obj.update(source="known official static xarm T-block", z_extent_m=float(2*zhalf),
               initial_xytheta=[.46,0.,0.], parts=parts, mass_kg=value(.15), geometry_version=1,
               visual_asset="known T-block compound",collision_asset="two metric convex boxes from official point extents",
               unobserved_back="uniform-box approximation to the sampled static volume",support_groups=[row["sha256"]])
    scene["static_shape_audit"]={"sampled_points":len(p),"bounds_m":[lo.tolist(),hi.tolist()],
                                 "bar_lower_y_m":float(cut),"origin":"official XY, centered at thickness midpoint",
                                 "surface_source":str(path),"static_shape_only":True}
    return scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pusht",type=Path);p.add_argument("tblock",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    info=json.loads((a.pusht/"meta/info.json").read_text())
    if (info["total_episodes"],info["total_frames"],info["fps"])!=(60,47582,30):
        raise ValueError("Unexpected public dataset identity")
    signatures=[]; records=[]; offset=[]
    for e in range(60):
        t,cmd,robot=read_episode(a.pusht,e)
        i=np.rint(np.linspace(0,len(t)-1,64)).astype(int)
        signatures.append(np.stack((cmd[i],robot[i]),axis=0))
        records.append({"episode":e,"frames":len(t),"first_pusher_xy":robot[0].tolist(),
                        "parquet_sha256":file_hash(a.pusht/f"data/chunk-000/episode_{e:06d}.parquet")[0]})
        offset.append(float(np.linalg.norm(cmd-robot,axis=-1).mean()))
    roles,groups,links=assign_groups(signatures)
    if sum(i["frames"] for i in records)!=47582:
        raise ValueError("Missing recorded frames")
    a.out.mkdir(parents=True,exist_ok=True)
    write_json(a.out/"split.json",{"dataset":"shashuo0104/xarm7_pusht","revision":"9d9e4e31ff82580ca8841664ba4a891b92dbdc35",
               "seed":42,"units":"meters from official xArm base frame","command":"planar xy waypoint",
               "state":"recorded xy actuator observation; no direct force or object pose","groups":groups,
               "near_duplicate_links":links,"metric":"64 points per complete command/recorded path, mean<6mm and p95<15mm",
               "roles":roles,"episodes":records,"mean_per_episode_command_state_difference_m":float(np.mean(offset))})
    write_json(a.out/"scene_C0.json",shape_scene(a.tblock))
    print(json.dumps({"roles":{k:len(v) for k,v in roles.items()},"groups":len(groups),
                      "fit_first":roles["fit"][:3],"sum_frames":sum(i["frames"] for i in records)}))


if __name__ == "__main__":
    main()
