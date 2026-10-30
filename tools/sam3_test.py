#!/usr/bin/env python3
"""
SAM 3 (Meta, Nov 2025) on field pictures: find every instance of each named object, from words alone.
Unlike the colour rules, it can be asked for things without a strong colour (grey cups, clear cups, robots).

For each picture: one run per prompt list, boxes + names drawn, counts and time printed.
Needs sam3.pt (gated on Hugging Face; downloaded after the user's access was approved).

Usage (Jetson, gpuenv):
    YOLO_AUTOINSTALL=false gpuenv/bin/python sam3_test.py /tmp/ctest/in2 --out results/sam3_test
"""
import argparse
import glob
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

PROMPTS = ["yellow pin", "blue pin", "red pin", "grey cup", "clear plastic cup", "goal", "robot", "person"]
COLOURS = [(0, 220, 255), (255, 120, 0), (0, 0, 255), (128, 128, 128), (255, 255, 255), (0, 160, 0),
           (255, 0, 255), (0, 128, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/sam3_test")
    ap.add_argument("--model", default="sam3.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    predictor = SAM3SemanticPredictor(overrides={"conf": args.conf, "task": "segment", "mode": "predict",
                                                 "model": args.model, "half": True, "save": False,
                                                 "verbose": False})
    totals = {p: 0 for p in PROMPTS}
    times = []
    for path in sorted(glob.glob(os.path.join(args.folder, "*.jpg"))):
        img = cv2.imread(path)
        t0 = time.perf_counter()
        predictor.set_image(path)
        r = predictor(text=PROMPTS)[0]
        times.append(time.perf_counter() - t0)
        counts = {p: 0 for p in PROMPTS}
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                k = int(cls)
                counts[PROMPTS[k]] += 1
                c = COLOURS[k % len(COLOURS)]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(img, f"{PROMPTS[k]} {conf:.2f}", (int(x1), max(14, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        for p in PROMPTS:
            totals[p] += counts[p]
        cv2.imwrite(os.path.join(args.out, os.path.basename(path)), img)
        print(f"{os.path.basename(path)}: {times[-1]:.1f} s  " +
              ", ".join(f"{p} {n}" for p, n in counts.items() if n), flush=True)
    print(f"\n{len(times)} pictures, {np.median(times):.1f} s each (median; the first includes start-up)")
    print("totals: " + ", ".join(f"{p} {n}" for p, n in totals.items()))
    print(f"saved pictures with boxes in {args.out}")


if __name__ == "__main__":
    main()
