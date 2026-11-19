"""Extract selected native Gemini payloads using an existing verified audit and index."""
import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import platform
import sqlite3
import shutil
from importlib.metadata import version

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rosbags.rosbag1 import Reader

from capture_bag_metrics import nearest
from capture_bag_samples import fetch_from_chunk, raw_depth
from capture_bag_scan import RGB, DEPTH, stamp, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    protocol = read(Path("configs/capture_geometry_protocol.json"))
    src, segment = protocol["source"], protocol["segment"]
    if args.full:
        full = read(Path("configs/capture_full_test_protocol.json"))
        capture = full["capture"]
        assert args.run == capture["geometry_run_id"] and src["sha256"] == capture["source_bag_sha256"]
        segment = dict(rgb_start_timestamp_ns=capture["first_rgb_device_timestamp_ns"],
                       rgb_end_timestamp_ns=capture["last_rgb_device_timestamp_ns"],
                       first_depth_device_timestamp_ns=capture["first_depth_device_timestamp_ns"],
                       sampling=dict(target_period_ns=full["selection"]["target_interval_ns"],
                                     max_rgb_depth_offset_ns=full["selection"]["synchronization_tolerance_ms"]*1000000))
    source = Path(src["resolved_windows_file"])
    audit = Path(src["audit_directory"])
    previous = Path(".local/bag-audit") / src["audit_run_id"]
    scan, fp = read(previous/"scan.json"), read(previous.parent/"fingerprint.json")
    before = stamp(source)
    if not (scan["before"] == scan["after"] == before and
            fp["mtime_ns"] == before["mtime_ns"] and fp["bytes"] == src["bytes"] == before["bytes"] and
            Path(fp["path"]) == Path(scan["source"]) == source and
            scan["run_id"] == src["audit_run_id"] and scan["sha256"] == fp["sha256"] == src["sha256"]):
        raise ValueError("Audit source identity mismatch")
    run = Path(full["paths"]["local_extraction"]) if args.full else Path(".local/capture-geometry") / args.run
    packet = run / "transfer"
    packet.mkdir(parents=True, exist_ok=False)
    for folder in ("rgb", "depth"):
        (packet/folder).mkdir()
    cv2.setNumThreads(1)
    ref = read(audit/"samples_metadata.json")["samples"]
    start, end = [int(segment[k]) for k in ("rgb_start_timestamp_ns", "rgb_end_timestamp_ns")]
    if not args.full:
        assert [int(ref[i]["rgb"]["device_ns"]) for i in (0, 3)] == [start, end]
    index = previous/"times.sqlite"
    selected, chunks = [], defaultdict(list)
    with closing(sqlite3.connect(index.resolve().as_uri()+"?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        streams = []
        for topic in (RGB, DEPTH):
            c = scan["catalog"][topic]
            rows = db.execute("select * from times where cid=? order by idx", (c["connection_id"],)).fetchall()
            assert len(rows) == c["expected"] and all(r["idx"] == i for i, r in enumerate(rows))
            streams.append(rows)
        rgb, dep = [np.array([r["device_ns"] for r in rows]) for rows in streams]
        assert np.all(np.diff(rgb)>0) and np.all(np.diff(dep)>0)
        assert dep[0] == int(segment["first_depth_device_timestamp_ns"])
        association = nearest(rgb, dep)
        assert len(np.unique(association)) == len(rgb)
        targets = np.arange(start, end+1, int(segment["sampling"]["target_period_ns"]))
        in_window = np.flatnonzero((rgb>=start) & (rgb<=end))
        picks = in_window[nearest(targets, rgb[in_window])]
        if args.full:
            assert (start,end)==(rgb[0],rgb[-1]) and len(targets)==full["selection"]["expected_regular_grid_targets"]
            if picks[-1] != len(rgb)-1:
                targets=np.append(targets,end)
                picks=np.append(picks,len(rgb)-1)
        used = set()
        for n, (target, i) in enumerate(zip(targets, picks)):
            j = association[i]
            dt = int(dep[j]-rgb[i])
            row = dict(candidate_index_zero_based=n, target_device_timestamp_ns=str(target),
                       rgb_zero_based_index=int(i), actual_rgb_timestamp_ns=str(rgb[i]),
                       rgb_vendor_frame_number=streams[0][i]["number"], depth_minus_rgb_ns=dt,
                       relative_s=float((rgb[i]-dep[0])/1e9), id=f"frame_{n:05d}", drop_reason=None)
            selected.append(row)
            if args.full:
                row.update(depth_zero_based_index=int(j), actual_depth_timestamp_ns=str(dep[j]))
                row["requested_split"] = ("train" if n%10<8 else "val" if n%10==8 else "test")
                row["annotation"] = ("late dynamic interval" if 351<=row["relative_s"]<=355
                                     else "endpoint camera roll" if n==len(targets)-1 else None)
            if i in used or abs(dt)>int(segment["sampling"]["max_rgb_depth_offset_ns"]):
                row["drop_reason"] = "duplicate RGB" if i in used else "outside association tolerance"
                continue
            used.add(i)
            for name, record in zip(("rgb","depth"), (streams[0][i], streams[1][j])):
                chunks[record["chunk"]].append((record, row, name))
    with Reader(source) as reader:
        for pos, entries in sorted(chunks.items()):
            chunk = reader.chunks[pos]
            reader.bio.seek(chunk.datapos)
            block = io.BytesIO(chunk.decompressor(reader.bio.read(chunk.datasize)))
            for r, row, name in entries:
                data, m = fetch_from_chunk(block, r["offset"], r["cid"], r["idx"], r["record_ns"])
                if any(int(m[k]) != r[k] for k in ("record_ns","header_ns","device_ns","system_ns","global_ns","seq")) or m["vendor_frame_number"] != r["number"]:
                    raise ValueError("Indexed identity differs from physical payload")
                fmt = m["format"]
                if name == "rgb":
                    assert fmt["encoding"] == "mjpg"
                    dest = packet / name / (row["id"]+".jpg")
                    dest.write_bytes(data)
                    row["rgb_payload_sha256"] = digest(data)
                    image = cv2.imread(str(dest), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if args.full and (image is None or image.shape != (720,1280,3)):
                        row["drop_reason"]="RGB decode failure"
                        continue
                    assert image is not None and image.shape == (720,1280,3)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    row["gray_mean"] = float(gray.mean())
                    row["laplacian_variance"] = float(cv2.Laplacian(gray,cv2.CV_64F).var())
                else:
                    assert fmt["encoding"] == "mono16"
                    z = raw_depth(data, fmt)
                    assert z.dtype == np.uint16 and z.shape == (480,848)
                    dest = packet / name / (row["id"]+".npz")
                    np.savez_compressed(dest, depth=z)
                    with np.load(dest,allow_pickle=False) as check:
                        assert np.array_equal(z, check["depth"])
                    row["depth_pixels_sha256"] = digest(z.tobytes())
                row[name] = {**m,"topic": RGB if name=="rgb" else DEPTH,
                             "file":dest.relative_to(packet).as_posix(),"sha256":sha256(dest)}
    assert before == stamp(source)
    accepted = [row for row in selected if not row["drop_reason"]]
    assert all(row["rgb_payload_sha256"] == row["rgb"]["sha256"] for row in accepted)
    for row in accepted:
        for sample in ref:
            if row["actual_rgb_timestamp_ns"] == sample["rgb"]["device_ns"] and sample.get("rgb_file"):
                assert row["rgb_payload_sha256"] == sample["rgb_payload_sha256"] and row["depth_pixels_sha256"] == sample["depth_pixels_sha256"]
    write_json(packet/"selection.json", selected)
    shutil.copyfile(audit/"calibration.json",packet/"calibration.json")
    dq = read(audit/"depth_quality.json")
    write_json(packet/"identity.json", dict(run_id=args.run, source=fp, audit_commit=src["audit_commit"],
               source_fingerprint_method="Unchanged path, byte count and nanosecond mtime before/after; audit SHA reused.",
               index_sha256=sha256(index), index_run=scan["run_id"], indexed_chunks_read=len(chunks),
               segment=segment, unit={k:dq[k] for k in ("metres_per_unit","interpretation","unit_evidence")},
               python=platform.python_version(), packages={k:version(k) for k in ("rosbags","numpy","Pillow","opencv-python-headless") if k!="opencv-python-headless"},
               opencv=cv2.__version__,scripts={p:sha256(Path("scripts",p)) for p in ("capture_segment.py","capture_bag_samples.py","capture_bag_scan.py","capture_bag_metrics.py")}))
    for first in range(0, len(accepted), 40):
        subset = accepted[first:first+40]
        canvas = Image.new("RGB",(5*216, ((len(subset)+4)//5)*142), "#17191c")
        pen = ImageDraw.Draw(canvas)
        for k,row in enumerate(subset):
            x, y = k%5*216,k//5*142
            with Image.open(packet/row["rgb"]["file"]) as im:
                canvas.paste(im.resize((212,119),Image.Resampling.LANCZOS),(x,y))
            pen.text((x+2,y+120),f"{row['id']}  {row['relative_s']:.3f}s",fill="white")
        canvas.save(run/f"screen_{first:03d}.jpg",quality=88)
    print(f"{len(targets)} targets; {len(accepted)} decoded pairs; {len(chunks)} indexed chunks")


if __name__ == "__main__":
    main()
