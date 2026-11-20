#!/usr/bin/env bash
set -eo pipefail
cd /root/autodl-tmp/DN-Splatter
bash scripts/launch_capture_full.sh

# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_rectify.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26/transfer /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26/rect --all --workers 25
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 features --device cuda
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 match --device cuda --exhaustive
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 map --device cuda --exhaustive
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 features --device cuda --bridge
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 match --device cuda --bridge --exhaustive
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 map --device cuda --exhaustive --bridge
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_sfm.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 queries --device cuda
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/capture-geometry/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_localize.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 --map map_full.json
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_metric.py /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_dataset.py export /root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-full-5hz-3982cb26 /root/autodl-tmp/DN-Splatter/.local/data/custom/capture-full/20261119-full-5hz-3982cb26 --full
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_inputs.py copy
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python dn_splatter/scripts/normals_from_pretrain.py --data-dir /root/autodl-tmp/DN-Splatter/.local/runs/capture-full-dn-30k-20261119-v1/inputs --save-path /root/autodl-tmp/DN-Splatter/.local/runs/capture-full-dn-30k-20261119-v1/inputs/normals_from_pretrain --resolution low --model-type omnidata
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_inputs.py manifest
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_train.py reader
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_full_train.py
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_eval.py val
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_eval.py test
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_full_video.py
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_mesh.py export
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_mesh.py validate
