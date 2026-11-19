#!/usr/bin/env bash
set -eo pipefail
cd /root/autodl-tmp/DN-Splatter
# Rechecks this run identity and continues from completed stages.
bash scripts/launch_capture_s01.sh

# Actual commands (working directory listed for each):
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_inputs.py copy
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_diagnostic.py
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python dn_splatter/scripts/normals_from_pretrain.py --data-dir /root/autodl-tmp/DN-Splatter/.local/runs/capture-s01-dn-30k-20261119-v1/inputs --save-path /root/autodl-tmp/DN-Splatter/.local/runs/capture-s01-dn-30k-20261119-v1/inputs/normals_from_pretrain --resolution low --model-type omnidata
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_inputs.py manifest
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_train.py reader
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_train.py train
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_eval.py val
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_eval.py test
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_mesh.py export
# .local/upstream/dn-splatter $ /root/autodl-tmp/DN-Splatter/.local/conda/dn-splatter-bw/bin/python /root/autodl-tmp/DN-Splatter/scripts/capture_s01_mesh.py validate
