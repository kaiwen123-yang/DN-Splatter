"""Offline Orbbec ROS1 audit. Walk physical chunks once, retain only metadata in SQLite."""
import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import struct

from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import Header, read_uint32
import numpy as np

SDK = 'https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/'
ROLE = {
    '/cam/property': 'device properties', '/cam/deviceInfo': 'device identity',
    '/cam/disparityParam': 'stereo disparity-to-depth calibration',
    '/cam/streamProfileType_2': 'color profile', '/cam/streamProfileType_3': 'depth profile',
    '/cam/streamProfileType_4': 'accel profile', '/cam/streamProfileType_5': 'gyro profile',
    '/cam/sensor_2/frameType_2': 'RGB', '/cam/sensor_3/frameType_3': 'depth',
    '/cam/sensor_4/frameType_4': 'accelerometer', '/cam/sensor_5/frameType_7': 'gyroscope',
}
RGB, DEPTH, ACC, GYRO = list(ROLE)[-4:]
PROPERTY = {14: 'depth_mirror', 15: 'depth_flip', 42: 'hardware_d2c', 81: 'color_mirror',
            82: 'color_flip', 85: 'hardware_disparity_to_depth', 115: 'color_rotation_deg',
            118: 'depth_rotation_deg', 176: 'depth_unit_mm', 3004: 'sdk_disparity_to_depth'}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def stamp(path):
    st = path.stat()
    return {'bytes': st.st_size, 'mtime_ns': str(st.st_mtime_ns),
            'mtime_utc': datetime.fromtimestamp(st.st_mtime, UTC).isoformat()}


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Wire:
    """Length-bounded reads using the SDK serializer order, including vendor-only fields."""
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def fields(self, fmt):
        vals = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += struct.calcsize('<' + fmt)
        return vals

    def array(self):
        n, = self.fields('I')
        stop = self.pos + n
        if stop > len(self.data):
            raise ValueError('array past message boundary')
        a = self.data[self.pos:stop]
        self.pos = stop
        return a

    def string(self):
        return bytes(self.array()).decode()

    def header(self):
        seq, sec, nano = self.fields('III')
        if nano >= 10**9:
            raise ValueError('invalid timestamp')
        return seq, sec * 10**9 + nano, self.string()

    def finish(self):
        if self.pos != len(self.data):
            raise ValueError(f'unconsumed wire data: {len(self.data) - self.pos}')


def image_fields(w):
    h, width = w.fields('II')
    encoding = w.string()
    endian, stride = w.fields('BI')
    data = w.array()
    scale, number, dev, system, glob = w.fields('f4Q')
    meta = w.array()
    meta_size, bits = w.fields('IB')
    w.finish()
    if len(meta) != meta_size:
        raise ValueError('metadata length mismatch')
    if encoding == 'mono16':
        if len(data) != h * stride or stride < width * 2:
            raise ValueError('invalid mono16 stride or length')
    elif encoding == 'mjpg':
        if bytes(data[:2]) != b'\xff\xd8':
            raise ValueError('expected JPEG SOI')
    else:
        raise ValueError(f'unsupported pixel format {encoding}')
    fmt = dict(height=h, width=width, encoding=encoding, step=stride, is_bigendian=endian,
               pixel_bit_size=bits, serialized_depth_units=scale)
    return fmt, data, (number, dev*1000, system*1000, glob*1000)


def meta_fields(w, role):
    if role == 'device properties':
        prop, = w.fields('I')  # uint32 in OBProperty.h, despite uint8 in the embedded text
        data = w.array()
        if w.fields('I')[0] != len(data):
            raise ValueError('property length mismatch')
        if prop not in PROPERTY:
            out = {}
        else:
            value = struct.unpack('<f', data)[0] if prop == 176 else int.from_bytes(data, 'little', signed=True)
            out = {'property_id': prop, 'field': PROPERTY[prop], 'value': value}
    elif role == 'device identity':
        name, full_name, asic = (w.string() for _ in range(3))
        w.fields('HH')
        w.string()  # UID, never retained
        w.string()  # serial number
        fw, hw, supported_sdk, transport = (w.string() for _ in range(4))
        w.fields('HB')
        w.string()
        w.string()
        out = dict(model=full_name, firmware=fw, supported_sdk_field=supported_sdk)
    elif role.endswith('profile'):
        kind, fmt = w.fields('BB')
        if role in ('accel profile', 'gyro profile'):
            ranges = w.fields('4B')
        rot, trans = w.fields('9f'), w.fields('3f')
        out = dict(stream_type=kind, format_enum=fmt, raw_rot=rot, raw_trans_mm=trans)
        if role in ('color profile', 'depth profile'):
            width, height, fps = w.fields('3H')
            out.update(width=width, height=height, fps=fps, intrinsics=w.fields('4f'),
                       distortion=w.fields('8f'), distortion_model=w.fields('B')[0])
        else:
            w.fields('27d')  # noise, random walk, temperature, bias, gravity, misalignment, slope
            out['range_rate_enums'] = ranges
    elif role == 'stereo disparity-to-depth calibration':
        vals = w.fields('ddfdBffBfiiB')
        out = dict(zip('zpd zpps baseline fx bitSize unit minDisparity packMode dispOffset invalidDisp dispIntPlace isDualCamera'.split(), vals))
    else:
        raise ValueError(role)
    w.finish()
    return out


def inspect_imu(w, role):
    orientation, ocov, omega = w.fields('4d'), w.fields('9d'), w.fields('3d')
    wcov, accel, acov = w.fields('9d'), w.fields('3d'), w.fields('9d')
    data = w.array()
    size, number, temp, dev, system, glob = w.fields('IQf3Q')
    w.finish()
    if len(data) != size:
        raise ValueError('IMU size mismatch')
    vec = accel if role == 'accelerometer' else omega
    norm = sum(v*v for v in vec)**0.5
    flags = dict(zero_orientation=not any(orientation), zero_covariances=not any(ocov + wcov + acov),
                 nonfinite=not np.isfinite((*orientation, *ocov, *wcov, *acov, *omega, *accel, temp)).all())
    return (number, dev*1000, system*1000, glob*1000), norm, flags


def scan(source, private):
    catalog, profiles, imu_stats, format_runs = {}, defaultdict(list), defaultdict(Counter), defaultdict(list)
    compression = Counter()
    with Reader(source) as reader, sqlite3.connect(private / 'times.sqlite') as db:
        indexed = defaultdict(dict)
        for c in reader.connections:
            if c.topic not in ROLE or c.topic in catalog:
                raise ValueError(f'unsupported or duplicate connection {c.topic}')
            catalog[c.topic] = dict(connection_id=c.id, type=c.msgtype.replace('/msg/', '/'),
                                    md5=c.digest, definition_sha256=hashlib.sha256(c.msgdef.data.encode()).hexdigest(),
                                    expected=c.msgcount, scanned=0, payload_bytes=0, frame_ids=Counter(), role=ROLE[c.topic])
            for entry in reader.indexes[c.id]:
                indexed[entry.chunk_pos][entry.offset] = (c.id, entry.time)
        lookup = {c.id: c.topic for c in reader.connections}
        db.execute('create table times (cid, idx, chunk, offset, record_ns, header_ns, seq, number, device_ns, system_ns, global_ns, imu_norm)')
        insert = 'insert into times values (?,?,?,?,?,?,?,?,?,?,?,?)'
        for chunk_pos, chunk in sorted(reader.chunks.items()):
            reader.bio.seek(chunk_pos)
            hdr = Header.read(reader.bio)
            compression[hdr['compression'].decode()] += 1
            reader.bio.seek(chunk.datapos)
            block = chunk.decompressor(reader.bio.read(chunk.datasize))
            if len(block) != hdr.get_uint32('size'):
                raise ValueError('decompressed length mismatch')
            f = io.BytesIO(block)
            entries = indexed.pop(chunk_pos)
            while f.tell() < len(block):
                offset = f.tell()
                h = Header.read(f)
                size = read_uint32(f)
                raw = f.read(size)
                if len(raw) != size:
                    raise ValueError('truncated record')
                if h.get_uint8('op') == 7:
                    continue
                cid, t = h.get_uint32('conn'), h.get_time('time')
                if h.get_uint8('op') != 2 or entries.pop(offset) != (cid, t):
                    raise ValueError('physical record differs from index')
                topic = lookup[cid]
                row = catalog[topic]
                idx, role = row['scanned'], row['role']
                w = Wire(raw)
                seq, head, frame = w.header()
                row['frame_ids'][frame] += 1
                timing, norm = (None,)*4, None
                if role in ('RGB', 'depth'):
                    fmt, _, timing = image_fields(w)
                    runs = format_runs[topic]
                    if not runs or runs[-1]['format'] != fmt:
                        runs.append(dict(first_idx=idx, first_header_ns=str(head), format=fmt))
                    runs[-1]['last_idx'] = idx
                elif role in ('accelerometer', 'gyroscope'):
                    timing, norm, flags = inspect_imu(w, role)
                    imu_stats[topic].update({k: int(v) for k, v in flags.items()})
                else:
                    meta = meta_fields(w, role)
                    if meta:
                        profiles[topic].append(dict(header_ns=str(head), idx=idx, values=meta))
                db.execute(insert, (cid, idx, chunk_pos, offset, t, head, seq, *timing, norm))
                row['scanned'] += 1
                row['payload_bytes'] += size
            if entries:
                raise ValueError('missing indexed messages')
        if indexed or any(x['expected'] != x['scanned'] for x in catalog.values()):
            raise ValueError('incomplete scan')
        db.execute('create unique index by_frame on times(cid,idx)')
    return dict(catalog=catalog, metadata=profiles, imu=imu_stats, formats=format_runs, compression=compression)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--fingerprint', type=Path, help='Prior streaming SHA-256 and matching source path, size and mtime_ns')
    args = parser.parse_args()
    start = datetime.now(UTC)
    before = stamp(args.source)
    with args.source.open('rb') as f:
        magic = f.readline(32)
    if magic != b'#ROSBAG V2.0\n':
        raise ValueError('requires ROS1 v2.0')
    if args.fingerprint:
        fp = json.loads(args.fingerprint.read_text(encoding='utf-8-sig'))
        if (Path(fp['path']) != args.source or fp['bytes'] != before['bytes']
                or fp['mtime_ns'] != before['mtime_ns']):
            raise ValueError('fingerprint source identity mismatch')
        digest = fp['sha256']
    else:
        digest = sha256(args.source)
    run = start.strftime('%Y%m%dT%H%M%SZ-') + digest[:8]
    private = args.private_root / run
    private.mkdir(parents=True, exist_ok=False)
    print(run, flush=True)
    data = scan(args.source, private)
    after = stamp(args.source)
    data.update(run_id=run, started_at=start.isoformat(), source=str(args.source), before=before, after=after,
                sha256=digest, magic=magic.decode().rstrip(), logical_cpus=os.cpu_count(), os=platform.platform(),
                output_free_bytes=shutil.disk_usage(private).free)
    write_json(private/'scan.json', data)
    if before != after:
        raise RuntimeError('source changed during scan')
    print('Completed physical scan', flush=True)


if __name__ == '__main__':
    main()
