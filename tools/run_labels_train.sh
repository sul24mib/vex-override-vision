#!/bin/bash
# Override detector pipeline (2026-10-23), runs on the Jetson in ~/four_cores:
# 1. YOLO speed test -> 2. draft labels (colour + SAM 2) -> 3. dataset split by time -> 4. train + test.
# Progress: ~/four_cores/logs/labels_train.log
# Ultralytics must not install or upgrade packages (it once upgraded numpy and broke VGGT).
export YOLO_AUTOINSTALL=false
cd ~/four_cores
echo "=== 1. YOLO speed test $(date +%T)"
gpuenv/bin/python -u yolo_speed.py data/field_h20 > logs/yolo_speed.log 2>&1
grep -E "^yolo|available|not used|failed" logs/yolo_speed.log
echo "=== 2. draft labels (colour + SAM 2), 4 x 986 pictures $(date +%T)"
for c in 0 1 2 3; do
  gpuenv/bin/python -u colour_labels.py data/field_h20/cam$c --out results/labels_v1/cam$c --field sam 2>&1 \
    | grep -E "pictures:|Error|Traceback"
done
echo "=== 3. dataset (split by time) $(date +%T)"
rm -rf datasets/override_v1
gpuenv/bin/python -u build_dataset.py data/field_h20 results/labels_v1 datasets/override_v1
echo "=== 4. train + test $(date +%T)"
gpuenv/bin/python -u train_yolo.py datasets/override_v1/data.yaml --model auto \
  --speed-log logs/yolo_speed.log --epochs 60 2>&1 | grep -vE "^\s*$" | tail -40
echo "=== done $(date +%T)"
gpuenv/bin/python -c "import numpy; print('numpy still', numpy.__version__)"
