#!/bin/bash
export YOLO_AUTOINSTALL=false
cd ~/four_cores
gpuenv/bin/python - << "PY"
import glob, os
from ultralytics import YOLO
m = YOLO("runs/detect/runs_override/v1/weights/best.pt")
r = m.val(data="datasets/override_v1/data.yaml", split="test", device=0, project="runs_override", name="v1_test", exist_ok=True, verbose=False)
lines = [f"TEST (draft labels): mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}  precision {r.box.mp:.3f}  recall {r.box.mr:.3f}"]
for i, c in enumerate(r.box.ap_class_index):
    lines.append(f"  {m.names[int(c)]:12s} mAP50 {r.box.ap50[i]:.3f}")
open("runs/detect/runs_override/v1/test_metrics.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
imgs = sorted(glob.glob("datasets/override_v1/images/test/*.jpg"))
m.predict(imgs[::max(1, len(imgs)//24)][:24], device=0, conf=0.25, save=True, project="runs_override", name="v1_test_pictures", exist_ok=True, verbose=False)
print("pictures saved")
PY
echo "=== SAM 3 test $(date +%T)"
gpuenv/bin/python -u sam3_test.py /tmp/ctest/in2 --out results/sam3_test 2>&1 | grep -vE "warn|Warn|^- |CC|PyTorch" | tail -15
