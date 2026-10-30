#!/bin/bash
# Waits until sam3.pt is downloaded and the YOLO speed test is finished, then runs the SAM 3 test.
export YOLO_AUTOINSTALL=false
cd ~/four_cores
until [ -s sam3.pt ] && grep -q "=== 2." logs/labels_train.log; do sleep 30; done
echo "=== SAM 3 test $(date +%T)"
gpuenv/bin/python -u sam3_test.py /tmp/ctest/in2 --out results/sam3_test 2>&1 | grep -vE "warn|Warn|^- |CC|PyTorch"
echo "=== done $(date +%T)"
