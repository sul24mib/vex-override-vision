#!/usr/bin/env python3
"""
Runs on the Jetson (in ~/four_cores/gpuenv). Runs YOLO-World (CVPR 2024) on a folder of
photos: it finds objects you name in words, with no training. Saves every photo with the
boxes drawn, plus detections.csv, into <folder>/detections/.

Usage:
    gpuenv/bin/python detect_photos.py ~/four_cores/scenes/field1
    gpuenv/bin/python detect_photos.py photos --classes "red cup,blue cup,red pin,blue pin,robot"
    gpuenv/bin/python detect_photos.py photos --camarray     # 5120x800 frames: split into the 4 views first

Phone photos (HEIC) must be converted to JPG first (detect.sh on the Mac does that).
"""
import argparse
import csv
import glob
import os
import time

import cv2
from ultralytics import YOLO

# Override game pieces: pins show red, blue or yellow depending on which half faces up;
# cups are grey on one side and clear on the other; goals are red, blue or neutral.
DEFAULT_CLASSES = ("red pin,blue pin,yellow pin,grey cup,clear plastic cup,"
                   "red goal,blue goal,goal,robot,person")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="comma-separated names to look for")
    ap.add_argument("--model", default="yolov8s-worldv2.pt", help="yolov8s/m/l-worldv2.pt (bigger = slower, better)")
    ap.add_argument("--conf", type=float, default=0.15, help="minimum confidence to keep a box")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--camarray", action="store_true", help="photos are 4-camera Camarray frames: split them")
    ap.add_argument("--labels", action="store_true",
                    help="also save YOLO training labels (detections/labels/<photo>.txt + classes.txt): "
                         "a first draft to check by hand")
    ap.add_argument("--no-pictures", action="store_true", help="don't save the pictures with boxes drawn")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.folder, "*"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"no .jpg/.png photos in {args.folder}")
    out = os.path.join(args.folder, "detections")
    os.makedirs(out, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.labels:
        os.makedirs(os.path.join(out, "labels"), exist_ok=True)
        with open(os.path.join(out, "labels", "classes.txt"), "w") as f:
            f.write("\n".join(classes) + "\n")

    model = YOLO(args.model)
    model.set_classes(classes)
    print(f"{len(files)} photos, looking for: {classes}")

    rows, counts, t_total = [], {}, 0.0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        views = ([img[:, i * img.shape[1] // 4:(i + 1) * img.shape[1] // 4] for i in range(4)]
                 if args.camarray else [img])
        t0 = time.perf_counter()
        results = model.predict(views, device=0, imgsz=args.imgsz, conf=args.conf, verbose=False)
        t_total += time.perf_counter() - t0
        drawn = []
        for v, r in enumerate(results):
            for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = classes[int(cls)]
                counts[name] = counts.get(name, 0) + 1
                rows.append([os.path.basename(path), v, name, round(conf, 3)] + [round(x) for x in box])
            if args.labels:  # YOLO format: class x_centre y_centre width height, all 0..1 of the view
                stem = os.path.splitext(os.path.basename(path))[0] + (f"_view{v}" if args.camarray else "")
                with open(os.path.join(out, "labels", stem + ".txt"), "w") as f:
                    for (x, y, w, h), cls in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
                        f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            if not args.no_pictures:
                drawn.append(r.plot(line_width=2))
        if not args.no_pictures:
            cv2.imwrite(os.path.join(out, os.path.basename(path)), cv2.hconcat(drawn) if args.camarray else drawn[0])

    with open(os.path.join(out, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "view", "class", "confidence", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"{len(rows)} detections in {len(files)} photos, {t_total / len(files) * 1000:.0f} ms per photo on the GPU")
    for name in classes:
        print(f"   {name:12s} {counts.get(name, 0)}")
    print(f"saved pictures with boxes + detections.csv in {out}")


if __name__ == "__main__":
    main()
