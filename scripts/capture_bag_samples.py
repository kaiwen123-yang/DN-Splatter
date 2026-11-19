"""Read only selected frames from indexed chunks, write full-value examples and contact sheets."""
import hashlib
import io
import itertools

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rosbags.rosbag1 import Reader
from rosbags.rosbag1.reader import Header, read_uint32

from capture_bag_metrics import nearest, quantiles
from capture_bag_scan import RGB, DEPTH, SDK, Wire, image_fields


def fetch(reader, db, cid, idx):
    chunk_pos, offset, rec = db.execute('select chunk,offset,record_ns from times where cid=? and idx=?',
                                        (cid, int(idx))).fetchone()
    chunk = reader.chunks[chunk_pos]
    reader.bio.seek(chunk.datapos)
    f = io.BytesIO(chunk.decompressor(reader.bio.read(chunk.datasize)))
    return fetch_from_chunk(f, offset, cid, idx, rec)


def fetch_from_chunk(f, offset, cid, idx, rec):
    """Decode one indexed vendor image from an already decompressed chunk."""
    f.seek(offset)
    h = Header.read(f)
    if (h.get_uint32('conn'), h.get_time('time')) != (cid, rec):
        raise ValueError('sample index mismatch')
    w = Wire(f.read(read_uint32(f)))
    seq, head, frame = w.header()
    fmt, payload, (number, device, system, glob) = image_fields(w)
    meta = dict(connection_id=cid, index_zero_based=int(idx), vendor_frame_number=number, seq=seq, frame_id=frame,
                record_ns=str(rec), header_ns=str(head), device_ns=str(device), system_ns=str(system), global_ns=str(glob),
                format=fmt, payload_bytes=len(payload))
    return payload, meta


def raw_depth(payload, fmt):
    dtype = '>u2' if fmt['is_bigendian'] else '<u2'
    return np.ndarray((fmt['height'], fmt['width']), dtype=dtype, buffer=payload, strides=(fmt['step'], 2))


def hist_quantiles(hist, scale=1):
    # Inverse empirical CDF. Keep zeros and documented invalid codes separate.
    cdf = np.cumsum(hist)
    if not cdf[-1]:
        return {}
    return {key: float(scale*np.searchsorted(cdf, max(1, int(np.ceil(cdf[-1]*p)))))
            for key, p in [('min',0),('p01',.01),('p50',.50),('p95',.95),('p99',.99),('max',1)]}


def sheet(out, images, labels, title, depth=False):
    w, h, header, footer = 300, 170, 42, 48 if depth else 8
    ncols = 4
    canvas = Image.new('RGB', (ncols*w, header+((len(images)+ncols-1)//ncols)*(h+23)+footer), '#19212a')
    draw = ImageDraw.Draw(canvas)
    draw.text((10,14), title, fill='white')
    for i, (im, label) in enumerate(zip(images, labels)):
        x = (i % ncols)*w
        y = header + (i//ncols)*(h+23)
        tile = Image.fromarray(im)
        tile.thumbnail((w-4, h), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x+2, y))
        draw.text((x+4, y+h+2), label, fill='white')
    if depth:
        y = canvas.height-42
        grad = np.tile(np.arange(256,dtype=np.uint8), (11,1))
        ramp = cv2.applyColorMap(grad,cv2.COLORMAP_TURBO)[:,:,::-1]
        canvas.paste(Image.fromarray(ramp).resize((450,11)),(10,y))
        for j, metre in enumerate((0, 1, 4, 9, 16, 25)):
            x = 10 + j*90
            draw.line((x,y+10,x,y+14),fill='white')
            draw.text((x,y+15),f'{metre} m',fill='white',anchor='mt' if j else 'lt')
        draw.text((500,y),'Square-root scale  |  Black = invalid  |  Above 25 m clipped for display',fill='white')
    canvas.save(out, quality=88, optimize=True)


def numeric_samples(source, root, scan, db, times, ids, delta, good, preview_masks):
    cv2.setNumThreads(1)
    rgb, dep = times[RGB], times[DEPTH]
    accepted = np.flatnonzero(good)
    # Subtract the first stamp before interpolation to retain integer timestamp precision.
    rel = rgb[accepted]-rgb[accepted[0]]
    uniform = accepted[nearest(np.linspace(0, rel[-1], 24).round().astype(np.int64), rel)]
    worst = int(accepted[np.argmax(abs(delta[accepted]))])
    gap = int(1+np.argmax(np.diff(rgb)))
    picks = list(dict.fromkeys(uniform.tolist()+[worst, gap]))
    if not good[gap]:
        raise ValueError('RGB after largest gap needs an unpaired preview')
    keep = {int(uniform[0]), int(uniform[len(uniform)//2]), int(uniform[-1]), worst}
    quality, depth_frames, samples = [], [], []
    rgb_tiles, depth_tiles, labels, thumbs = [], [], [], []
    total_hist = np.zeros(65536, np.int64)
    (root/'previews').mkdir(exist_ok=True)
    (root/'samples').mkdir(exist_ok=True)
    with Reader(source) as reader:
        for n, i in enumerate(picks, 1):
            p, color = fetch(reader, db, scan['catalog'][RGB]['connection_id'], i)
            d, depth = fetch(reader, db, scan['catalog'][DEPTH]['connection_id'], ids[i])
            fmt = color['format']
            if fmt['encoding'] != 'mjpg' or depth['format']['encoding'] != 'mono16':
                raise ValueError('unexpected sampled image encodings')
            bgr = cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is None or list(bgr.shape) != [fmt['height'], fmt['width'], 3]:
                raise ValueError('color decode / profile mismatch')
            z = raw_depth(d, depth['format'])
            hist = np.bincount(z.ravel(), minlength=65536)
            total_hist += hist
            valid = (z != 0) & (z != 65535)
            valid_hist = hist.copy()
            valid_hist[[0,65535]] = 0
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (96,54), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
            rel_s = float((rgb[i]-dep[0])/1e9)
            color_digest = hashlib.sha256(p).hexdigest()
            quality.append(dict(id=n, time_rel_s=rel_s, decoded_gray_mean=float(gray.mean()), pct_gray_le5=float(100*(gray<=5).mean()),
                                pct_gray_ge250=float(100*(gray>=250).mean()), laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                                jpeg_sha256=color_digest, decoded_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
            depth_frames.append(dict(id=n, time_rel_s=rel_s, total_pixels=int(z.size), zero=int(hist[0]), uint16_max=int(hist[-1]),
                                     valid_fraction=float(valid.mean()), valid_raw=hist_quantiles(valid_hist),
                                     raw_all=hist_quantiles(hist)))
            view = np.rint(255*np.sqrt(np.minimum(z.astype(np.float32),25000)/25000)).astype(np.uint8)
            dvis = cv2.applyColorMap(view, cv2.COLORMAP_TURBO)
            dvis[~valid] = 0
            rgb_tiles.append(cv2.resize(bgr[:,:,::-1],(296,167),interpolation=cv2.INTER_AREA))
            mask = preview_masks.get(color_digest)
            if mask:
                x, y, width, height = mask['xywh_at_296x167']
                rgb_tiles[-1][y:y+height, x:x+width] = (45,48,50)
            depth_tiles.append(cv2.resize(dvis[:,:,::-1],(296,168),interpolation=cv2.INTER_AREA))
            reasons = []
            if i in uniform:
                reasons.append('uniform')
            if i == worst:
                reasons.append('largest absolute RGB-D offset')
            if i == gap:
                reasons.append('after longest RGB header interval')
            labels.append(f'#{n:02}    {rel_s:7.2f} s    RGB {color["vendor_frame_number"]}')
            color.update(topic=RGB, decoded_shape=[*bgr.shape[:2],3], decoded_dtype='uint8', calibration='/cam/streamProfileType_2')
            depth.update(topic=DEPTH, decoded_shape=list(z.shape), decoded_dtype=z.dtype.str, calibration='/cam/streamProfileType_3')
            sample = dict(id=n, preview_cell=n-1, reason=reasons, time_rel_first_depth_s=rel_s, rgb=color, depth=depth,
                          pose=None, pose_status='missing world-camera trajectory', clock='device/header',
                          depth_minus_rgb_ns=int(delta[i]), metres_per_depth_unit=0.001, native_grids=True)
            if mask:
                sample['rgb_preview_mask'] = mask
            if i in keep:
                stem = f'samples/pair_{n:02}'
                jp, zp = root/(stem+'_rgb.jpg'), root/(stem+'_depth.npz')
                # Native camera JPEG; reject EXIF-bearing files pending disclosure review.
                with Image.open(io.BytesIO(p)) as check:
                    if check.getexif():
                        raise ValueError('JPEG contains EXIF; review before publishing')
                jp.write_bytes(p)
                np.savez_compressed(zp, depth=z)
                verify_jpeg = cv2.imdecode(np.frombuffer(jp.read_bytes(),np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                with np.load(zp,allow_pickle=False) as archive:
                    zz = archive['depth']
                    if zz.dtype != z.dtype or not np.array_equal(zz, z) or not np.array_equal(bgr,verify_jpeg):
                        raise ValueError('sample read-back differs')
                sample.update(rgb_file=jp.relative_to(root).as_posix(), depth_file=zp.relative_to(root).as_posix(),
                              rgb_payload_sha256=color_digest, depth_pixels_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
                              returned_transform='none; original JPEG payload; original numeric depth array, NPZ key depth; exact read-back passed')
            samples.append(sample)
    sheet(root/'previews/rgb.jpg',rgb_tiles,labels,'Gemini 335  |  RGB   /   24 uniform device times, then additional timing events')
    sheet(root/'previews/depth.jpg',depth_tiles,labels,'Gemini 335  |  Native depth   /   matched to RGB, on its own pixel grid',True)
    distances = [(float(np.abs(thumbs[i].astype(float)-thumbs[j]).mean()),i+1,j+1)
                 for i,j in itertools.combinations(range(len(thumbs)),2)]
    count = int(total_hist.sum())
    valid_hist = total_hist.copy()
    valid_hist[[0,65535]] = 0
    rgb_report = dict(topic=RGB, sample_count=len(samples), all_frames=len(rgb), frames=quality,
                      decode='JPEG via OpenCV IMREAD_COLOR / ignore EXIF orientation, BGR uint8 -> grayscale; RGB order for display. MJPG step is not a compressed row length.',
                      measures='Native 8-bit grayscale mean and fractions <=5 and >=250. Blur proxy is variance of OpenCV 3x3 footprint Laplacian (ksize=1). Texture and exposure also influence it.',
                      metrics={k:quantiles([r[k] for r in quality]) for k in ('decoded_gray_mean','pct_gray_le5','pct_gray_ge250','laplacian_variance')},
                      exact_decoded_duplicates=len(samples)-len({r['decoded_bgr_sha256'] for r in quality}),
                      near_duplicates=dict(method='All sampled pairs; grayscale area resize to 96x54; MAE in 8-bit levels. Lower implies greater similarity. Adjacent supplementary samples may overlap.',
                                           pairs_below_mae_3=sum(v<3 for v,_,_ in distances), closest_pairs=sorted(distances)[:5]))
    depth_report = dict(topic=DEPTH, sampled_frames=len(samples), total_frames=len(dep), frames=depth_frames,
                        transport='LZ4 bag chunks; uncompressed mono16 Image bytes, SDK Y16, little-endian uint16 with recorded row stride. No compressedDepth transport.',
                        sample_pixels=count, zero_count=int(total_hist[0]), zero_fraction=float(total_hist[0]/count), nan_count=0, inf_count=0, nan_fraction=0, inf_fraction=0,
                        uint16_max_count=int(total_hist[-1]), valid_fraction=float(valid_hist.sum()/count),
                        validity='0 or 65535 excluded (no measurement / invalid in SDK coordinate utilities); validity measures presence, not physical accuracy.',
                        raw_all=hist_quantiles(total_hist), raw_valid=hist_quantiles(valid_hist),
                        range_valid_metres=hist_quantiles(valid_hist,.001), quantile_method='inverse empirical CDF, pooled sampled pixels',
                        metres_per_unit=.001, interpretation='optical-axis Z',
                        unit_evidence='Gemini 335 recording: property 176 float=1.0, SDK G330 uses it for depthUnit; disparityParam.unit=1 agrees; hardware disparity-to-depth property 85=true, frame type DEPTH, format Y16. SDK depth value scale is millimetres per count and its XYZ routine assigns depth to Z. Serialized image.depth_units stays at its zero default. This is a recorded configuration, not accuracy against a reference.',
                        sources=[SDK+p for p in ('include/libobsensor/h/Frame.h','include/libobsensor/h/ObTypes.h','src/device/gemini330/G330Device.cpp','src/media/playback/PlaybackDevice.cpp','src/shared/utils/CoordinateUtil.cpp')],
                        registration='native depth; hardware D2C off, differing intrinsics/extrinsic; no registered depth stream',
                        geometric_check=dict(performed=False, reason='No dynamic camera/world pose; no multi-view projection or registration residual computed.'))
    sample_meta = dict(uniform_count=len(uniform), extra_count=len(samples)-len(uniform), original_value_pairs=len(keep),
                       origin='first depth header', selection='Nearest unique accepted RGB frame to 24 equally spaced device times from first to last accepted RGB, plus worst pair offset and largest RGB interval. Pair offsets retained unmodified.',
                       public_rgb_preview_masks=preview_masks,
                       preview_transforms='Full RGB resized 1280x720 to 296x167, depth colorized on native 848x480 then resized to 296x168 (INTER_AREA); pasted without cropping. Depth: Turbo with sqrt(min(Z,25m)/25m); invalid black. Cells row-major. Designated RGB preview masks applied after metrics, before sheet export.',
                       samples=samples)
    return rgb_report, depth_report, sample_meta
