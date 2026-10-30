#!/usr/bin/env python3
"""
Fine-tunes a pre-trained YOLO on the Override training set (from build_dataset.py) and tests it.

Model size: --model, or "auto" = the largest of yolo11m / yolo11s / yolo11n whose best speed in the
yolo_speed.py log is at least --min-sets-per-s sets of 4 pictures per second at 640 px
(a target of 10 sets/s plus room for distance, memory and markers).

Starts from weights pre-trained on COCO (transfer learning): the network already knows edges, shapes
and colours; fine-tuning teaches it our classes.

Saves (in <project>/<name>/): best.pt, training curves (Ultralytics), test_metrics.txt, and pictures
of predictions on test frames.

Usage (Jetson, gpuenv; YOLO_AUTOINSTALL=false so Ultralytics cannot change installed packages):
    YOLO_AUTOINSTALL=false gpuenv/bin/python train_yolo.py datasets/override_v1/data.yaml --model auto \
        --speed-log logs/yolo_speed.log --epochs 60
"""
import argparse
import glob
import os
import re
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO


def pick_model(speed_log, min_sets):
    """Largest model whose best measured speed at 640 px reaches min_sets sets per second."""
    best = {}
    for line in open(speed_log):
        m = re.match(r"^(yolo\S+\.pt)\s+640\s+\S+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m:
            best[m.group(1)] = max(best.get(m.group(1), 0.0), float(m.group(2)))
    for name in ("yolo11m.pt", "yolo11s.pt", "yolo11n.pt"):
        if best.get(name, 0.0) >= min_sets:
            return name, best
    return "yolo11n.pt", best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="data.yaml from build_dataset.py")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--speed-log", default="logs/yolo_speed.log")
    ap.add_argument("--min-sets-per-s", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs_override")
    ap.add_argument("--name", default="v1")
    args = ap.parse_args()

    model_name = args.model
    if model_name == "auto":
        model_name, speeds = pick_model(args.speed_log, args.min_sets_per_s)
        print(f"speed test (best sets/s at 640 px): {speeds} -> training {model_name}", flush=True)

    model = YOLO(model_name)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0,
                workers=4, patience=15, project=args.project, name=args.name, exist_ok=True,
                plots=True, verbose=False)
    run_dir = os.path.join(args.project, args.name)
    best = YOLO(os.path.join(run_dir, "weights", "best.pt"))

    # Score on the held-out last 20% of the recording. These test labels are unchecked drafts,
    # so this measures agreement with the colour+SAM 2 labels, not true accuracy.
    metrics = best.val(data=args.data, split="test", imgsz=args.imgsz, device=0, project=args.project,
                       name=args.name + "_test", exist_ok=True, plots=True, verbose=False)
    names = best.names
    lines = [f"model {model_name}, {args.epochs} epochs max, imgsz {args.imgsz}",
             f"TEST (draft labels): mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}  "
             f"precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}"]
    for i, c in enumerate(metrics.box.ap_class_index):
        lines.append(f"  {names[int(c)]:12s} mAP50 {metrics.box.ap50[i]:.3f}")
    open(os.path.join(run_dir, "test_metrics.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # Pictures of predictions on a spread of test frames, to look at by eye
    test_imgs = sorted(glob.glob(os.path.join(os.path.dirname(args.data), "images", "test", "*.jpg")))
    sample = test_imgs[:: max(1, len(test_imgs) // 24)][:24]
    best.predict(sample, imgsz=args.imgsz, device=0, conf=0.25, save=True, project=args.project,
                 name=args.name + "_test_pictures", exist_ok=True, verbose=False)
    print(f"saved {run_dir}/weights/best.pt, {run_dir}/test_metrics.txt, "
          f"{args.project}/{args.name}_test_pictures/", flush=True)


if __name__ == "__main__":
    main()
