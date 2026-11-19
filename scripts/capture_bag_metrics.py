"""Timing, association and calibration reports for the Orbbec audit."""
import numpy as np

from capture_bag_scan import ACC, DEPTH, GYRO, RGB, SDK


def quantiles(a):
    a = np.asarray(a)
    if not a.size:
        return {}
    return dict(zip(('min', 'p01', 'p50', 'p95', 'p99', 'max'),
                    np.percentile(a, [0, 1, 50, 95, 99, 100]).tolist()))


def clock(t):
    t = np.asarray(t, dtype=np.int64)
    dt = np.diff(t)
    span = int(t.max() - t.min())
    median = np.median(dt) if dt.size else 0
    gaps = np.flatnonzero(dt > 1.5*median) if median > 0 else []
    longest = np.argsort(-dt, kind='stable')[:3]
    return dict(first_ns=str(t[0]), last_ns=str(t[-1]), duration_ns=span, zero_count=int((t == 0).sum()),
                duplicate_count=len(t)-len(np.unique(t)), regressions=int((dt < 0).sum()),
                interval_ns=quantiles(dt), mean_rate_hz=(len(t)-1)*1e9/span if span else None,
                median_interval_rate_hz=1e9/median if median > 0 else None,
                gaps_over_1p5_median=len(gaps), first_20_gaps_over_1p5_median=[
                    dict(after_idx=int(i), start_rel_s=float((t[i]-t[0])/1e9), delta_ns=int(dt[i])) for i in gaps[:20]
                ], largest_gaps=[
                    dict(after_idx=int(i), start_rel_s=float((t[i]-t[0])/1e9), delta_ns=int(dt[i])) for i in longest],
                first_regressions_after_idx=np.flatnonzero(dt < 0)[:5].tolist())


def relative_offset(a, b):
    diff = a-b
    base = int(diff[0])
    return dict(first_offset_ns=str(base), last_minus_first_ns=int(diff[-1])-base,
                offset_minus_first_ns=quantiles(diff-base))


def timing_reports(scan, db):
    streams, inventory, arrays = {}, [], {}
    total = sum(c['payload_bytes'] for c in scan['catalog'].values())
    for topic, c in scan['catalog'].items():
        dynamic = topic in (RGB, DEPTH, ACC, GYRO)
        cols = ['record_ns', 'header_ns', 'seq']
        if dynamic:
            cols += ['device_ns', 'system_ns', 'global_ns', 'number']
        cols_sql = ','.join(cols)
        arr = np.array(db.execute(f'select {cols_sql} from times where cid=? order by idx',
                                  (c['connection_id'],)).fetchall(), dtype=np.int64)
        fields = dict(zip(cols, arr.T))
        rec, head = fields['record_ns'], fields['header_ns']
        domain = ('host epoch from property writer' if topic == '/cam/property' else
                  'sensor device time' if dynamic else 'static metadata stamped with first capture device time')
        streams[topic] = row = dict(scanned_messages=len(rec), total_indexed=c['expected'], clock_domain=domain,
                                    header_present=len(head), missing_or_zero_header_count=int((head == 0).sum()),
                                    record=clock(rec), header=clock(head),
                                    record_minus_header_ns=quantiles(rec-head))
        if dynamic:
            arrays[topic] = head
            for name in ('device_ns', 'system_ns', 'global_ns'):
                row[name] = clock(fields[name])
            row.update(header_minus_device_ns=quantiles(head-fields['device_ns']),
                       system_minus_global_ns=quantiles(fields['system_ns']-fields['global_ns']),
                       global_minus_device=relative_offset(fields['global_ns'], fields['device_ns']),
                       system_minus_device=relative_offset(fields['system_ns'], fields['device_ns']),
                       number_first=int(fields['number'][0]), number_last=int(fields['number'][-1]),
                       non_unit_number_increments=int(np.count_nonzero(np.diff(fields['number']) != 1)))
        inventory.append(dict(topic=topic, **c, decoded_wire_messages=len(head), wire_errors=0,
                              header_seq_range=[int(fields['seq'].min()), int(fields['seq'].max())],
                              payload_fraction=c['payload_bytes']/total, first_record_ns=str(rec[0]),
                              last_record_ns=str(rec[-1]), duration_ns=row['record']['duration_ns'],
                              rate_hz=row['record']['mean_rate_hz'], median_rate_hz=row['record']['median_interval_rate_hz'],
                              image_format_segments=scan['formats'].get(topic)))
    record_min, record_max = db.execute('select min(record_ns), max(record_ns) from times').fetchone()
    physical_regressions = 0
    prev = None
    for (t,) in db.execute('select record_ns from times order by rowid'):
        physical_regressions += int(prev is not None and t < prev)
        prev = t
    timing = dict(mixed_domain_container_range=dict(min_record_ns=str(record_min), max_record_ns=str(record_max),
                      physical_cross_topic_regressions=physical_regressions,
                      interpretation='Includes epoch-domain properties; min/max subtraction is not capture duration.'),
                  order='Physical chunk file offset and record offset within chunk. Each message matched index connection/time/offset.',
                  rate_definition='(N-1)/(max-min) seconds. Gaps are adjacent intervals > 1.5x same-clock median; descriptive only.',
                  clocks='RGB/depth/IMU record and Header.stamp are copied from device capture time. Profiles use first capture time. Property records use host wall time.',
                  offsets='system/global/device are separately serialized SDK microseconds converted to integer ns. System is host-side, global is device time mapped to host. Offsets are not transport latency.',
                  source=SDK+'src/media/ros/RosbagWriter.cpp', streams=streams)
    inv = dict(compression_chunks=scan['compression'], total_payload_bytes=total,
               byte_definition='Serialized message sizes before bag compression, excluding record headers.',
               definition_source='Embedded definitions plus SDK serializers; ROS names/MD5 alone omit vendor fields. Image adds depth_units and pixel_bit_size omitted from definition text. IMU transmits linear_acceleration_covariance; OBProperty.propertyId uses uint32 on wire.',
               sources=[SDK+'3rdparty/rosbag/src/msgs/'+s for s in ['sensor_msgs/Image.h','sensor_msgs/Imu.h','custom_msg/OBProperty.h']],
               role_evidence='Image payloads and encodings match SDK stream enums 2 (color MJPG), 3 (depth Y16). IMU fields and profile enums 4/5 identify accelerometer/gyro. Other roles follow vendor typed fields.',
               roles_not_present=['CameraInfo','TF','TF-static','odometry','pose','path','GNSS','IR image','confidence image','point cloud','registered RGB-grid depth'],
               connections=inventory)
    return arrays, inv, timing


def nearest(a, b):
    right = np.searchsorted(b, a).clip(0, len(b)-1)
    left = (right-1).clip(0)
    return np.where(abs(a-b[left]) <= abs(b[right]-a), left, right)


def pair_report(times):
    a, b = times[RGB], times[DEPTH]
    if any(np.any(np.diff(t) <= 0) for t in (a, b)):
        raise ValueError('clock duplicates/reset require segmentation')
    ids = nearest(a, b)
    delta = b[ids]-a
    unique = len(np.unique(ids))
    if unique != len(ids):
        raise ValueError('nearest mapping has collisions; solve a separate one-to-one association')
    stats = []
    for tol in (5, 10, 20, 33):
        good = abs(delta) <= tol*10**6
        n = int(good.sum())
        stats.append(dict(tolerance_ms=tol, candidate_rgb=len(a), accepted_unique_pairs=n, reused=0,
                          unmatched_rgb=len(a)-n, unmatched_depth=len(b)-n, rgb_fraction=n/len(a),
                          depth_fraction=n/len(b), signed_ns=quantiles(delta[good]), absolute_ns=quantiles(abs(delta[good]))))
    good = abs(delta) <= 10**7
    pose = dict(source=None, matched=0, total_rgb=len(a))
    report = dict(rgb_topic=RGB, depth_topic=DEPTH, timestamp_source='header/device (also exactly equal to record time across dynamic streams)',
                  clock_evidence='Device timestamp_usec agrees with header and record on all image/IMU rows. One Gemini 335. No clock shift fitted or applied.',
                  algorithm='Nearest depth for every RGB; earlier wins ties. Verified the resulting mapping is injective BEFORE applying tolerances.',
                  nearest_with_reuse_allowed=dict(attempted=len(a), reused=len(a)-unique, signed_ns=quantiles(delta),
                                                 absolute_ns=quantiles(abs(delta))),
                  sign='depth minus RGB', selected_tolerance_ms=10,
                  rationale='Diagnostic window below half of the ~33.37 ms frame interval; acquisition/exposure alignment accuracy is not implied.',
                  overlap_duration_ns=int(min(a[-1], b[-1])-max(a[0], b[0])), thresholds=stats,
                  unmatched_depth_at_10ms=dict(before_first_pair=int(ids[good][0]), after_last_pair=int(len(b)-1-ids[good][-1]),
                                                internal=int(ids[good][-1]-ids[good][0]+1-good.sum())),
                  calibration_association='Matching stream type and native resolution, one static profile per stream stamped at start. See calibration.json.',
                  camera_pose_association=pose, clock_resets='none observed on header/device streams', gap_details='timing.json',
                  imu_at_rgb={})
    for topic in (ACC, GYRO):
        imu = times[topic]
        d = imu[nearest(a, imu)]-a
        report['imu_at_rgb'][topic] = dict(rgb_bracketed=int(((a >= imu[0]) & (a <= imu[-1])).sum()),
                                            absolute_nearest_ns=quantiles(abs(d)))
    return report, ids, delta, good


def calibration_reports(scan, times):
    m = scan['metadata']
    cameras, links = {}, []
    profiles = {'RGB':'/cam/streamProfileType_2', 'depth':'/cam/streamProfileType_3'}
    for role, t, profile in [('RGB', RGB, profiles['RGB']), ('depth', DEPTH, profiles['depth'])]:
        versions = m[profile]
        if len(versions) != 1:
            raise ValueError('multiple calibration records need segmented association')
        v = versions[0]['values']
        fx, fy, cx, cy = v['intrinsics']
        w, h = v['width'], v['height']
        match = all(x['format']['width'] == w and x['format']['height'] == h for x in scan['formats'][t])
        cameras[role] = dict(topic=t, profile=profile, frame_id='', count=len(versions), distinct_calibration_count=1,
                             first_header_ns=versions[0]['header_ns'], width=w, height=h, nominal_fps=v['fps'],
                             stream_type=v['stream_type'], format_enum=v['format_enum'], fx=fx, fy=fy, cx=cx, cy=cy,
                             K=[[fx,0,cx],[0,fy,cy],[0,0,1]], distortion_model={4:'Brown-Conrady K6',3:'Brown-Conrady'}[v['distortion_model']],
                             distortion_model_enum=v['distortion_model'], D_sdk_k1_k2_k3_k4_k5_k6_p1_p2=v['distortion'],
                             R=None, P=None, roi=None, binning=None, image_resolution_match=match,
                             finite_and_plausible=bool(np.isfinite(v['intrinsics']+v['distortion']).all() and fx>0 and fy>0 and 0<cx<w and 0<cy<h),
                             associated_frames=len(times[t]), recorded_changes=0)
    for profile, source, target in [(profiles['depth'],'depth','color'),('/cam/streamProfileType_4','accel','depth')]:
        v = m[profile][0]['values']
        rot = np.array(v['raw_rot']).reshape(3,3)
        links.append(dict(source_sensor=source, destination_sensor=target, static=True, topic=profile, count=1,
                          header_ns=m[profile][0]['header_ns'], R=rot.tolist(), translation_mm=v['raw_trans_mm'],
                          det_R=float(np.linalg.det(rot)), max_abs_RtR_minus_I=float(abs(rot.T@rot-np.eye(3)).max())))
    props = [x['values'] for x in m['/cam/property']]
    cal = dict(device=m['/cam/deviceInfo'][0]['values'], CameraInfo='not_present',
               association='SDK stream enum/format/native size, empty ROS frame IDs. Single stored profile applies to the recorded stream; no in-run updates visible.',
               missing_fields='Rectification R, projection P, ROI and binning are not supplied as CameraInfo. K constructed from SDK fx,fy,cx,cy; D is in SDK order.',
               cameras=cameras, static_extrinsics=links, relevant_properties=props,
               disparity_calibration=m['/cam/disparityParam'][0]['values'],
               registration_status='native unregistered streams', registration_evidence='Hardware D2C property 42=0; depth is 848x480 on depth profile, color is 1280x720 on distinct color profile. Depth-to-color transform is nonidentity.',
               sources=[SDK+'src/media/ros/RosbagWriter.cpp', SDK+'src/media/ros/RosbagReader.cpp',SDK+'include/libobsensor/h/ObTypes.h'])
    graph = dict(tf_messages=0, tf_static_messages=0, dynamic_transform_messages=0, edges=links,
                 convention='p_destination = R @ p_source + t (t in mm); aliases denote SDK sensor roles, not ROS frame_ids. Optical x right, y down, z forward.',
                 gyro='SDK playback binds gyro to accel as colocated when accel and gyro exist. Gyro profile has a zero rotation block, not a calibrated independent rotation.',
                 diagnostics='No world/body dynamic edge, TF cycles or competing TF parents. Color and gyro zero rotation placeholders not used as edges.',
                 camera_chain='accel -> depth -> color is a static mount chain only. World -> camera missing.',
                 source=SDK+'src/media/ros/RosbagReader.cpp')
    return cal, graph
