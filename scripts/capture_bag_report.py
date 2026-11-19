"""Produce Orbbec audit evidence. Finalize only after reviewing contact sheets and the numeric results."""
import argparse
import hashlib
from datetime import UTC, datetime
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from capture_bag_metrics import calibration_reports, pair_report, quantiles, timing_reports
from capture_bag_samples import numeric_samples
from capture_bag_scan import ACC, DEPTH, GYRO, RGB, sha256, stamp, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_review(path):
    if not path.is_dir():
        return read(path)
    summary = read(path/'summary.json')
    keys = ('readiness_checks','recommended_input_path','blocking_missing_items','nonblocking_limitations',
            'unresolved_questions','next_action')
    return {**{k:summary[k] for k in keys}, 'source_sha256':summary['source']['sha256'],
            'overview_md':(path/'overview.md').read_text(encoding='utf-8'),
            'execution_md':(path/'execution.md').read_text(encoding='utf-8'),
            'image_observations':read(path/'image_quality.json')['visual_review'],
            'depth_observations':read(path/'depth_quality.json')['visual_review'],
            'public_rgb_preview_masks':read(path/'samples_metadata.json')['public_rgb_preview_masks']}


def plot_timing(root, times, delta):
    fig, axs = plt.subplots(2,1,figsize=(9,4.9),layout='constrained',sharex=True)
    origin = times[DEPTH][0]
    for topic, label, color in [(DEPTH,'Depth','#31736e'),(RGB,'RGB','#d65d2b')]:
        t = times[topic]
        axs[0].plot((t[1:]-origin)/1e9,np.diff(t)/1e6,'.',ms=1.3,alpha=.6,label=label,color=color)
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('Frame interval (ms)')
    axs[1].plot((times[RGB]-origin)/1e9,delta/1e6,'.',ms=1.3,alpha=.65,color='#385b91')
    axs[1].set_ylabel('Depth minus RGB (ms)')
    axs[1].set_xlabel('Device time since first depth frame (s)')
    for ax in axs:
        ax.grid(alpha=.22)
    fig.savefig(root/'previews/timing.png',dpi=130)
    plt.close(fig)


def finalize(root, base, scan, summary, contract, review):
    """Review supplies observations tied to the hash; refuse to apply them to another recording."""
    if review['source_sha256'] != scan['sha256']:
        raise ValueError('visual review belongs to another source')
    for file, field in [('image_quality.json','image_observations'),('depth_quality.json','depth_observations')]:
        doc = read(root/file)
        doc['visual_review'] = review[field]
        write_json(root/file,doc)
    items = contract['return_package']
    summary.update({k:review[k] for k in ('readiness_checks','recommended_input_path','blocking_missing_items',
                                          'nonblocking_limitations','unresolved_questions','next_action')})
    checks = summary['readiness_checks']
    if set(checks) != set(items['readiness_items']) or any(
        v['status'] not in items['readiness_check_status_values'] for v in checks.values()
    ) or summary['recommended_input_path'] not in items['recommended_input_path_values']:
        raise ValueError('review status outside contract')
    reported = list(items['required_files']) + ['execution.md']
    summary.update(audit_status='complete', finished_at=datetime.now(UTC).isoformat(), report_files=reported,
                   script_hash_basis='UTF-8 source bytes, LF line endings as stored in Git',
                   report_script_sha256={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                         for p in Path(__file__).parent.glob('capture_bag_*.py')})
    write_json(root/'summary.json',summary)
    (root/'overview.md').write_text(review['overview_md'],encoding='utf-8',newline='\n')
    (root/'execution.md').write_text(review['execution_md'],encoding='utf-8',newline='\n')
    (root/'.gitattributes').write_text('.gitattributes text eol=lf\n*.json text eol=lf\n*.md text eol=lf\n'
                                      '*.jpg binary\n*.png binary\n*.npz binary\n',encoding='utf-8',newline='\n')
    missing = [p for p in reported if p != 'artifact_manifest.json' and not (root/p).is_file()]
    if missing or set(items['summary_json_required_fields']) - summary.keys():
        raise ValueError(f'incomplete report: {missing}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    entries = [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in files]
    manifest = dict(files=entries, bytes_excluding_manifest=sum(f['bytes'] for f in entries),
                    total_bytes_including_manifest=0, self_hash_excluded=True, original_bag_included=False,
                    source_unchanged=summary['source_unchanged'], local_only=['source bag','per-message timestamp table'],
                    omissions=['World trajectory and geometric residuals: no recorded dynamic pose.'])
    mp = root/'artifact_manifest.json'
    while True:
        write_json(mp,manifest)
        size = mp.stat().st_size + manifest['bytes_excluding_manifest']
        if size == manifest['total_bytes_including_manifest']:
            break
        manifest['total_bytes_including_manifest'] = size
    if size > items['maximum_publish_bytes']:
        raise ValueError('report exceeds publication limit')
    write_json(base/'latest.json',dict(schema_version=1, status='complete', run_id=root.name,
               source_access_verified=True, source_format=summary['source']['format'],source_sha256=scan['sha256'],
               source_bytes=scan['before']['bytes'], report_directory=root.as_posix(), summary=(root/'summary.json').as_posix(),
               overview=(root/'overview.md').as_posix(), recommended_input_path=summary['recommended_input_path'],
               training_started=False, package_bytes=size, finished_at=summary['finished_at']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--review',type=Path,help='Source-bound visual review JSON or previously published report directory for this bag')
    args = parser.parse_args()
    contract = read(args.config)
    scan = read(args.private/'scan.json')
    root = args.output/scan['run_id']
    source = Path(scan['source'])
    review = load_review(args.review) if args.review else None
    if review and review['source_sha256'] != scan['sha256']:
        raise ValueError('review/source mismatch')
    if stamp(source) != scan['before']:
        raise ValueError('source changed after scan')
    root.mkdir(parents=True,exist_ok=True)
    db_uri = (args.private/'times.sqlite').absolute().as_uri()+'?mode=ro'
    with sqlite3.connect(db_uri,uri=True) as db:
        times, inventory, timing = timing_reports(scan,db)
        sync, ids, delta, good = pair_report(times)
        cal, graph = calibration_reports(scan,times)
        props = {p['field']:p['value'] for p in cal['relevant_properties']}
        # This recording-specific interpretation must not silently transfer to different sensors/modes.
        if (cal['device']['model'] != 'Orbbec Gemini 335' or props.get('depth_unit_mm') != 1.
                or props.get('hardware_disparity_to_depth') != 1 or props.get('hardware_d2c') != 0
                or cal['disparity_calibration']['unit'] != 1
                or any(not c['image_resolution_match'] for c in cal['cameras'].values())
                or not np.all(good)):
            raise ValueError('reassess scale, native grids, or association before reusing this report')
        masks = review.get('public_rgb_preview_masks',{}) if review else {}
        rgb, depth, samples = numeric_samples(source,root,scan,db,times,ids,delta,good,masks)
        imu = {}
        for topic in (ACC,GYRO):
            cid = scan['catalog'][topic]['connection_id']
            norms = [r[0] for r in db.execute('select imu_norm from times where cid=? order by idx',(cid,))]
            imu[topic] = dict(**scan['imu'][topic], vector_norm_raw=quantiles(norms), count=len(norms))
    pose = dict(status='no_usable_camera_poses', pose_topic=None, pose_count=0, rgb_pose_coverage=0.,
                rgb_frames=len(times[RGB]), interpolated_frames=0, imu=imu,
                interpretation='All recorded IMU orientations are zero quaternions and all covariance matrices are zero. SDK stores accelerometer/gyro vectors separately. No position or world reference.',
                direction=None, scale=None, path_length=None, translation_extent=None, rotation_change=None,
                resets_jumps=None, interpolation='none; no dynamic edges', covariance='unspecified',
                geometry_check='Not performed: dynamic camera poses absent.')
    plot_timing(root,times,delta)
    for name,obj in [('topic_inventory',inventory),('timing',timing),('sync',sync),('calibration',cal),
                     ('frame_graph',graph),('trajectory',pose),('image_quality',rgb),('depth_quality',depth),
                     ('samples_metadata',samples)]:
        write_json(root/(name+'.json'),obj)
    after = stamp(source)
    if after != scan['before']:
        raise ValueError('source changed before publication')
    dynamic_start = int(min(a[0] for a in times.values()))
    dynamic_end = int(max(a[-1] for a in times.values()))
    versions = {p:md.version(p) for p in ('rosbags','lz4','numpy','opencv-python-headless','Pillow','matplotlib',
                                          'apsw','backports.zstd','ruamel.yaml')}
    summary = dict(schema_version=1, started_at=scan['started_at'], run_id=root.name, training_started=False,
                   source_unchanged=True, integrity_scope='size and mtime_ns stable across scan/sample export; a streaming SHA-256 was computed once.',
                   source=dict(path=str(source),requested_path=str(source.parent),format='ROS1 bag v2.0 / Orbbec SDK extended messages',
                               magic=scan['magic'],sha256=scan['sha256'],bytes=after['bytes'],before=scan['before'],after=after,
                               sensor_duration_ns=dynamic_end-dynamic_start, first_device_ns=str(dynamic_start),last_device_ns=str(dynamic_end),
                               duration_basis='Dynamic color/depth/IMU device timestamps. The epoch-stamped property topic must not extend sensor duration.'),
                   reader_environment=dict(os=scan['os'],python=platform.python_version(),interpreter='.local/venv/bag-audit/Scripts/python.exe',
                                           logical_cpus=scan['logical_cpus'],output_volume_free_bytes_at_scan=scan['output_free_bytes'],
                                           pixel_workers=1,opencv_threads=1,dependencies=versions),
                   scan_coverage=dict(connection_count=len(inventory['connections']),
                                      indexed_messages=sum(c['expected'] for c in inventory['connections']),
                                      scanned_messages=sum(c['scanned'] for c in inventory['connections']),
                                      wire_decoded=sum(c['decoded_wire_messages'] for c in inventory['connections']),
                                      read_errors=0,scan_order='physical; validated against bag index',
                                      image_pairs_sampled=len(samples['samples']),numeric_pairs=samples['original_value_pairs']),
                   selected_streams=dict(rgb=RGB,depth=DEPTH,imu=[ACC,GYRO],pose=None,association_clock='device/header',
                                         profiles={k:v['profile'] for k,v in cal['cameras'].items()}))
    if review:
        summary['reader_environment'].update(review.get('measured_host',{}))
        finalize(root,args.output,scan,summary,contract,review)
    else:
        write_json(args.private/'summary_draft.json',summary)
    print(root)


if __name__ == '__main__':
    main()
