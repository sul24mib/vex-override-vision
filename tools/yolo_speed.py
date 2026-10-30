#!/usr/bin/env python3
"""
Which YOLO size can the robot afford? Times YOLO models on the Jetson with 4 real camera pictures at
once (one set = what the 4 cameras see at one moment). No training needed: a trained model of the
same size runs at the same speed.

Target (an idea from the 2026-10-21 meeting, to be checked): 4 cameras x 10 times a second
= 10 sets per second, with time left over for distance, memory and markers.

For each model and picture size it reports milliseconds per set and sets per second, for plain
PyTorch and, if TensorRT is installed, for a TensorRT FP16 engine (NVIDIA's Jetson speed-up).

Usage (Jetson, gpuenv):
    gpuenv/bin/python yolo_speed.py data/field_h20
    gpuenv/bin/python yolo_speed.py data/field_h20 --models yolo11n.pt yolo11s.pt --sizes 640 --no-tensorrt
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO

CAMERAS = 4


def load_sets(folder, count):
    """`count` moments spread over the recording, each = the 4 camera pictures at that moment."""
    names = sorted(os.listdir(os.path.join(folder, "cam0")))
    picks = np.linspace(0, len(names) - 1, count).astype(int)
    return [[cv2.imread(os.path.join(folder, f"cam{c}", names[i])) for c in range(CAMERAS)] for i in picks]


def time_model(model, sets, size, repeats):
    for s in sets[:3]:  # warm-up: the first runs include one-time GPU start-up
        model.predict(s, imgsz=size, device=0, verbose=False)
    times = []
    for _ in range(repeats):
        for s in sets:
            t0 = time.perf_counter()
            model.predict(s, imgsz=size, device=0, verbose=False)
            times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    return float(np.median(ms)), float(np.percentile(ms, 95))


def tensorrt_available():
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="recording with cam0..cam3 folders")
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 416])
    ap.add_argument("--sets", type=int, default=20, help="different moments to time on")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-tensorrt", action="store_true")
    args = ap.parse_args()

    sets = load_sets(args.folder, args.sets)
    use_trt = not args.no_tensorrt and tensorrt_available()
    print(f"{len(sets)} sets of {CAMERAS} pictures; TensorRT {'available' if use_trt else 'not used'}", flush=True)
    print(f"{'model':14s} {'size':>5s} {'backend':>9s} {'ms/set (median)':>16s} {'ms/set (95%)':>13s} {'sets/s':>7s}", flush=True)
    for name in args.models:
        for size in args.sizes:
            runs = [("pytorch", YOLO(name))]
            if use_trt:
                try:
                    engine = YOLO(name).export(format="engine", half=True, imgsz=size, batch=CAMERAS,
                                               device=0, verbose=False)
                    runs.append(("tensorrt", YOLO(engine, task="detect")))
                except Exception as e:  # noqa: BLE001  (report and carry on with PyTorch only)
                    print(f"  TensorRT export failed for {name} at {size}: {type(e).__name__}: {e}", flush=True)
            for backend, model in runs:
                med, p95 = time_model(model, sets, size, args.repeats)
                print(f"{name:14s} {size:5d} {backend:>9s} {med:16.1f} {p95:13.1f} {1000 / med:7.1f}", flush=True)


if __name__ == "__main__":
    main()
