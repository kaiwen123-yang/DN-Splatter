"""A local metric rigid world with a finite support, one object and an actuated pusher."""

import argparse
import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

from interactive_push_assets import write_json


HZ = 30
SUBSTEPS = 8


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def vec(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def value(number, source="assumed"):
    return {"value": number, "status": source}


def look_at(eye, target, width=240, height=240, focal=235.):
    eye = np.array(eye, dtype=float)
    z = np.asarray(target)-eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    t[:3, 3] = eye
    return {"K": [[focal, 0., (width-1)/2], [0., focal, (height-1)/2], [0., 0., 1.]],
            "T_world_camera_cv": t.tolist(), "size_wh": [width, height], "source": "simulation camera"}


def proxy_scene():
    return {
        "schema": 1, "scene_id": "rigid_local_proxy", "versions": {"scene": 1, "geometry": 1},
        "frames": {"T_physics_reconstruction": np.eye(4).tolist(), "gravity": [0., 0., -9.81],
                   "units": ["m", "s", "kg", "rad"], "quaternion": "wxyz",
                   "gravity_source": "synthetic reference", "camera": "OpenCV integer pixel centers"},
        "background": {"kind": "simulation", "support_half_xy": [.42, .38], "support_center_xy": [0., 0.],
                       "top_z": 0., "thickness": .035, "source": "synthetic bounded table",
                       "unknown_outside": True},
        "objects": [{"object_id": "movable", "source": "synthetic_proxy", "visual_asset": "two box meshes",
                     "collision_asset": "two non-overlapping boxes", "z_extent_m": .024,
                     "mass_kg": value(.10), "inertia_kg_m2": {"status": "uniform-density compound, assumed"},
                     "friction": value(.55), "initial_xytheta": [0., 0., 0.],
                     "origin": "local frame in middle height; xy origin stored explicitly", "geometry_version": 1,
                     "parts": [{"center": [0., .025, 0.], "half": [.053, .012, .012]},
                               {"center": [0., -.012, 0.], "half": [.012, .025, .012]}],
                     "valid_region": "bounded tabletop", "unobserved_back": "synthetic shape",
                     "support_groups": [], "conflicts": []}],
        "physics": {"engine": "MuJoCo CPU", "physics_hz": 240, "control_hz": 30,
                    "pusher_radius_m": .010, "pusher_half_height_m": .020, "pusher_z_m": .0225,
                    "pusher_friction": .45, "pusher_mass_kg": .08, "servo_kp": 420., "servo_damping": 15.,
                    "initial_pusher_xy": [-.115, -.008]},
        "cameras": {"front": look_at([.32, -.35, .52], [0., 0., 0.]),
                    "second": look_at([-.26, .22, .52], [0., 0., 0.])},
        "evidence": {"camera_anchor": "static background only", "fusion": "external versioned state"},
    }


def make_model(scene):
    obj, p, bg = scene["objects"][0], scene["physics"], scene["background"]
    root = ET.Element("mujoco", model=scene["scene_id"])
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true")
    ET.SubElement(root, "option", gravity=vec(scene["frames"]["gravity"]), timestep=str(1/(SUBSTEPS*HZ)),
                  integrator="implicitfast", iterations="50", cone="elliptic")
    world = ET.SubElement(root, "worldbody")
    half_z = bg["thickness"]/2
    ET.SubElement(world, "geom", name="support", type="box",
                  pos=vec([*bg["support_center_xy"], bg["top_z"]-half_z]), size=vec([*bg["support_half_xy"], half_z]))
    body = ET.SubElement(world, "body", name="movable")
    ET.SubElement(body, "freejoint", name="free_object")
    volumes = np.array([np.prod(v["half"]) for v in obj["parts"]])
    contact = ET.SubElement(root, "contact")
    for i, (part, fraction) in enumerate(zip(obj["parts"], volumes/volumes.sum())):
        name = f"part_{i}"
        ET.SubElement(body, "geom", name=name, type="box", size=vec(part["half"]), pos=vec(part["center"]),
                      mass=str(fraction*obj["mass_kg"]["value"]))
        for other, friction in (("support", obj["friction"]["value"]), ("pusher_tip", p["pusher_friction"])):
            ET.SubElement(contact, "pair", geom1=name, geom2=other, condim="4",
                          friction=vec([friction, friction, .001, .0001, .0001]))
    push = ET.SubElement(world, "body", name="pusher", pos=vec([0, 0, p["pusher_z_m"]+bg["top_z"]]))
    actuators = ET.SubElement(root, "actuator")
    for i, axis in enumerate(("1 0 0", "0 1 0")):
        joint = f"slide_{i}"
        ET.SubElement(push, "joint", name=joint, type="slide", axis=axis, damping=str(p["servo_damping"]))
        ET.SubElement(actuators, "position", name=f"waypoint_{i}", joint=joint, kp=str(p["servo_kp"]))
    ET.SubElement(push, "geom", name="pusher_tip", type="cylinder",
                  size=vec([p["pusher_radius_m"], p["pusher_half_height_m"]]), mass=str(p["pusher_mass_kg"]))
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


BOX_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
BOX_FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def mesh(geom_type, half):
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        return BOX_CORNERS*half, BOX_FACES
    a = np.arange(16)*np.pi/8
    ring = np.column_stack((np.cos(a)*half[0], np.sin(a)*half[0]))
    vertices = np.vstack([np.column_stack((ring, np.full(16, z))) for z in (-half[1], half[1])])
    sides = [[i, (i+1)%16, (i+1)%16+16, i+16] for i in range(16)]
    return vertices, [list(range(16)), list(range(16, 32)), *sides]


def rasterize(vertices, faces, color, camera, rgb, zbuf, labels, label):
    t = np.array(camera["T_world_camera_cv"])
    k = np.array(camera["K"])
    xyz = (vertices-t[:3, 3])@t[:3, :3]
    uvw = xyz@k.T
    h, w = zbuf.shape
    for face in faces:
        face_xyz = xyz[face]
        if np.any(face_xyz[:, 2] <= .001):
            continue
        uv = uvw[face, :2]/uvw[face, 2:]
        x0, y0 = np.maximum(np.floor(uv.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(uv.max(0)).astype(int)+1, [w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        poly = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillConvexPoly(poly, np.round(uv-[x0, y0]).astype(np.int32), 1)
        iy, ix = np.nonzero(poly)
        normal = np.cross(face_xyz[1]-face_xyz[0], face_xyz[2]-face_xyz[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        x, y = ix+x0, iy+y0
        den = normal[0]*(x-k[0, 2])/k[0, 0] + normal[1]*(y-k[1, 2])/k[1, 1] + normal[2]
        z = np.divide(normal@face_xyz[0], den, out=np.full_like(den, np.inf), where=np.abs(den)>1e-9)
        valid = (z>.001) & (z < zbuf[y, x])
        x, y, z = x[valid], y[valid], z[valid]
        zbuf[y, x] = z
        rgb[y, x] = np.asarray(color)*(.72 + .28*abs(normal[2]))
        labels[y, x] = label


class RigidWorld:
    def __init__(self, scene):
        self.scene = copy.deepcopy(scene)
        self.model = make_model(scene)
        self.data = mj.MjData(self.model)
        self.obj = self.model.body("movable").id
        self.ps = self.model.body("pusher").id
        self.tip = self.model.geom("pusher_tip").id
        self.geometry = {gid: mesh(self.model.geom_type[gid], self.model.geom_size[gid])
                         for gid in range(self.model.ngeom)}
        self.reset()

    def reset(self, state=None):
        state = state or {}
        mj.mj_resetData(self.model, self.data)
        obj = self.scene["objects"][0]
        xytheta = state.get("object_xytheta", obj["initial_xytheta"])
        pusher = state.get("pusher_xy", self.scene["physics"]["initial_pusher_xy"])
        z = state.get("object_z", self.scene["background"]["top_z"]+obj["z_extent_m"]/2+.00005)
        self.data.qpos[:3] = [*xytheta[:2], z]
        self.data.qpos[3:7] = [math.cos(xytheta[2]/2), 0., 0., math.sin(xytheta[2]/2)]
        self.data.qpos[7:9] = pusher
        self.data.ctrl[:] = pusher
        self.data.qvel[:] = np.asarray(state.get("qvel", np.zeros(self.model.nv)))
        self.version = int(state.get("object_version", obj["geometry_version"]))
        self.counter, self.last_contact, self.penetration = 0, False, 0.
        self.backgrounds, self.ledger, self.calibrator = {}, [], {}
        mj.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        r = self.data.xmat[self.obj].reshape(3, 3)
        pos = self.data.xpos[self.obj]
        return {"step": self.counter, "time_s": self.counter/HZ,
                "object_xytheta": np.array([pos[0], pos[1], math.atan2(r[1, 0], r[0, 0])]),
                "object_xyz": pos.copy(), "quaternion_wxyz": self.data.xquat[self.obj].copy(),
                "pusher_xy": self.data.xpos[self.ps, :2].copy(), "contact": self.last_contact,
                "penetration_m": self.penetration, "version": self.version}

    def correct_from_evidence(self, pose, version, source_group):
        if version <= self.version:
            raise ValueError("Nonmonotonic evidence version")
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Invalid evidence pose")
        # Filtering the latent state at an observation time is independent of the action API.
        self.data.qpos[:2] = pose[:2]
        self.data.qpos[3:7] = [math.cos(pose[2]/2), 0., 0., math.sin(pose[2]/2)]
        self.version = version
        self.ledger.append({"step": self.counter, "version": version, "group": source_group, "pose": pose.tolist()})
        mj.mj_forward(self.model, self.data)

    def step(self, command, semantics="waypoint"):
        cmd = np.asarray(command, dtype=float)
        if semantics == "waypoint" and cmd.shape == (2,) and np.isfinite(cmd).all():
            self.data.ctrl[:] = cmd
        elif semantics in ("force", "impulse") and cmd.shape == (3,) and np.isfinite(cmd).all():
            self.data.xfrc_applied[self.obj, :3] = cmd*(HZ*SUBSTEPS if semantics == "impulse" else 1)
        else:
            raise ValueError("Expected finite world-frame waypoint [m], force [N] or impulse [N s]")
        self.last_contact, self.penetration = False, 0.
        for i in range(SUBSTEPS):
            mj.mj_step(self.model, self.data)
            for c in self.data.contact:
                a, b = int(c.geom1), int(c.geom2)
                if self.tip in (a, b) and self.obj in (self.model.geom_bodyid[a], self.model.geom_bodyid[b]):
                    self.last_contact = True
                self.penetration = max(self.penetration, float(-c.dist))
            if semantics == "impulse" and i == 0:
                self.data.xfrc_applied[self.obj] = 0.
        self.data.xfrc_applied[self.obj] = 0.
        self.counter += 1
        # mj_step has integrated qpos; refresh Cartesian transforms to that same timestamp.
        mj.mj_forward(self.model, self.data)
        return self.state()

    def replay(self, initial, commands):
        self.reset(initial)
        return [self.step(c) for c in commands]

    def _draw(self, gid, camera, rgb, zbuf, labels):
        vertices, faces = self.geometry[gid]
        v = vertices@self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
        label = 0 if gid == self.model.geom("support").id else (2 if gid == self.tip else 1)
        palette = [(154, 146, 127), (199, 93, 53), (64, 126, 169)]
        rasterize(v, faces, palette[label], camera, rgb, zbuf, labels, label)

    def observe(self, name):
        cam = self.scene["cameras"][name]
        if name not in self.backgrounds:
            w, h = cam["size_wh"]
            rgb = np.full((h, w, 3), [57, 61, 63], np.uint8)
            depth = np.full((h, w), np.inf, np.float32)
            ids = np.full((h, w), -1, np.int16)
            source = cam.get("background_rgbd_npz")
            if source:
                with np.load(source, allow_pickle=False) as a:
                    if not np.allclose(a["K"], cam["K"]) or not np.allclose(a["T_world_camera_cv"], cam["T_world_camera_cv"]):
                        raise ValueError("Depth composition requires identical camera and optical Z")
                    rgb = a["rgb"].copy()
                    z = np.asarray(a["depth_m"], dtype=np.float32)
                if rgb.dtype != np.uint8 or rgb.shape != (h, w, 3) or z.shape != (h, w):
                    raise ValueError("Background canvas mismatch")
                # Unknown depth cannot justify drawing an opaque object in front of that ray.
                depth = np.where(np.isfinite(z) & (z>0), z, -np.inf)
            else:
                self._draw(self.model.geom("support").id, cam, rgb, depth, ids)
            self.backgrounds[name] = rgb, depth, ids
        rgb, depth, ids = (v.copy() for v in self.backgrounds[name])
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] != 0:
                self._draw(gid, cam, rgb, depth, ids)
        depth[~np.isfinite(depth)] = 0.
        return {"rgb": rgb, "depth_m": depth, "object_id": ids, "K": np.array(cam["K"]),
                "T_world_camera_cv": np.array(cam["T_world_camera_cv"]),
                "time_s": self.counter/HZ, "object_version": self.version}


def functional(out, scene):
    world = RigidWorld(scene)
    initial = np.asarray(scene["objects"][0]["initial_xytheta"])
    origin = np.asarray(scene["physics"]["initial_pusher_xy"])
    line = np.linspace(origin, initial[:2]+[.09, -.002], 90)
    first = world.replay({}, line)
    obs = world.observe("front")
    second = world.replay({}, line)
    same = all(np.array_equal(a["object_xytheta"], b["object_xytheta"]) and a["contact"] == b["contact"]
               for a, b in zip(first, second))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"scene.json", scene)
    cv2.imwrite(str(out/"last_front.png"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
    stationary = world.replay({}, np.broadcast_to(origin, (45, 2)))
    drop_start = scene["background"]["top_z"] + .2
    world.reset({"object_z": drop_start})
    dropped = world.step(origin)["object_xyz"][2]
    support = world.model.geom("support").id
    below = np.r_[scene["background"]["support_center_xy"], scene["background"]["top_z"]-.12]
    geomid = np.array([-1], dtype=np.int32)
    ray = mj.mj_ray(world.model, world.data, below, np.array([1.,0.,0.]), None, True, -1, geomid)
    report = {"scene_hash": digest(scene), "scope": "simulation_sanity", "mujoco": mj.__version__,
              "source": scene["objects"][0]["source"], "reset_replay_identical": same, "compared_steps": len(first),
              "contact_steps": sum(s["contact"] for s in first), "stationary_control_steps": len(stationary),
              "stationary_contact_steps": sum(s["contact"] for s in stationary),
              "response_displacement_m": float(np.linalg.norm(first[-1]["object_xytheta"][:2]-initial[:2])),
              "peak_penetration_m": max(s["penetration_m"] for s in first), "gravity_drop_m": float(drop_start-dropped),
              "horizontal_ray_below_support_hit": bool(ray >= 0 and geomid[0] == support),
              "foreground_pixels": int((obs["object_id"]==1).sum()), "world_frame": "right-handed z-up"}
    write_json(out/"functional.json", report)
    print(json.dumps(report), flush=True)
    if not (same and report["contact_steps"] and not report["stationary_contact_steps"] and
            report["gravity_drop_m"] > 0 and not report["horizontal_ray_below_support_hit"] and report["foreground_pixels"]):
        raise RuntimeError("One or more functional checks failed; see functional.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--scene", type=Path)
    a = parser.parse_args()
    functional(a.out, json.loads(a.scene.read_text()) if a.scene else proxy_scene())
