#!/usr/bin/env python3
"""
Picks ~200 pictures for people to check by hand, pre-draws the YOLO v1 boxes on them, and splits them into
equal batches (one per person), ready for Roboflow or X-AnyLabeling.

Two kinds of pictures:
  test  evenly spaced in time from the TEST split, per camera (in proportion to how many test pictures each
        camera has). Not chosen by difficulty, so the hand-checked labels give an honest test score for v1.
  hard  from the TRAIN split, where v1 is most unsure (most boxes with confidence 0.25-0.5), at least
        --gap frames apart on the same camera so they are not near-copies. These improve v2's training data.

Each batch mixes both kinds and all cameras, so nobody gets only the easy or only the hard ones.

Output:
    <out>/batch_<k>/images/camN_<frame>.jpg
    <out>/batch_<k>/labels/camN_<frame>.txt   v1's boxes (YOLO format, no confidence), to correct
    <out>/batch_<k>/classes.txt              for X-AnyLabeling
    <out>/batch_<k>/data.yaml                for Roboflow
    <out>/batch_<k>.zip
    <out>/manifest.csv                       file, batch, kind (test/hard), camera, frame, boxes

Usage (Jetson):
    YOLO_AUTOINSTALL=false gpuenv/bin/python make_label_batches.py datasets/override_v1 \
        runs/detect/runs_override/v1/weights/best.pt results/label_batches
"""
import argparse
import collections
import csv
import os
import shutil

import yaml
from ultralytics import YOLO


def camera_and_frame(name):
    cam, frame = os.path.splitext(name)[0].split("_")
    return cam, int(frame)


def even_sample(items, k):
    if k >= len(items):
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--test-n", type=int, default=140)
    ap.add_argument("--hard-n", type=int, default=60)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--gap", type=int, default=300, help="min frame-number gap between two hard pictures of one camera")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--extra-classes", default="robot,person", help="classes v1 does not know, for people to add")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)] + [c for c in args.extra_classes.split(",") if c]
    img_dir = lambda split: os.path.join(args.dataset, "images", split)

    # test: evenly spaced per camera, in proportion to each camera's share of the test split
    by_cam = collections.defaultdict(list)
    for f in sorted(os.listdir(img_dir("test"))):
        by_cam[camera_and_frame(f)[0]].append(f)
    total = sum(len(v) for v in by_cam.values())
    chosen = []
    for cam in sorted(by_cam):
        files = sorted(by_cam[cam], key=lambda f: camera_and_frame(f)[1])
        chosen += [("test", f) for f in even_sample(files, round(args.test_n * len(files) / total))]

    # hard: train pictures where v1 is most unsure
    train = sorted(os.listdir(img_dir("train")))
    unsure = {}
    for i in range(0, len(train), 64):
        part = train[i:i + 64]
        for f, r in zip(part, model.predict([os.path.join(img_dir("train"), f) for f in part],
                                            device=0, conf=args.conf, verbose=False)):
            c = r.boxes.conf.tolist()
            unsure[f] = sum(x < 0.5 for x in c)
    taken = collections.defaultdict(list)
    for f in sorted(unsure, key=lambda f: -unsure[f]):
        if sum(k == "hard" for k, _ in chosen) >= args.hard_n:
            break
        cam, frame = camera_and_frame(f)
        if all(abs(frame - t) >= args.gap for t in taken[cam]):
            taken[cam].append(frame)
            chosen.append(("hard", f))

    # batches: sort by camera and time, deal out round-robin so each batch gets a mix
    chosen.sort(key=lambda kf: (kf[0], camera_and_frame(kf[1])))
    rows = []
    for b in range(args.batches):
        os.makedirs(os.path.join(args.out, f"batch_{b + 1}", "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, f"batch_{b + 1}", "labels"), exist_ok=True)
    for i, (kind, f) in enumerate(chosen):
        b = f"batch_{i % args.batches + 1}"
        src = os.path.realpath(os.path.join(img_dir("test" if kind == "test" else "train"), f))
        shutil.copy(src, os.path.join(args.out, b, "images", f))
        r = model.predict(src, device=0, conf=args.conf, verbose=False)[0]
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
        open(os.path.join(args.out, b, "labels", os.path.splitext(f)[0] + ".txt"), "w").write("\n".join(lines) + "\n" * bool(lines))
        cam, frame = camera_and_frame(f)
        rows.append({"file": f, "batch": b, "kind": kind, "camera": cam, "frame": frame, "boxes": len(lines)})

    for b in range(args.batches):
        d = os.path.join(args.out, f"batch_{b + 1}")
        open(os.path.join(d, "classes.txt"), "w").write("\n".join(names) + "\n")
        yaml.safe_dump({"train": "images", "val": "images", "nc": len(names), "names": names},
                       open(os.path.join(d, "data.yaml"), "w"), sort_keys=False)
        shutil.make_archive(d, "zip", d)
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} pictures: {sum(r['kind'] == 'test' for r in rows)} test, {sum(r['kind'] == 'hard' for r in rows)} hard")
    for b in sorted({r["batch"] for r in rows}):
        rs = [r for r in rows if r["batch"] == b]
        print(f"  {b}: {len(rs)} pictures, {sum(r['boxes'] for r in rs)} pre-drawn boxes, "
              f"cameras {dict(collections.Counter(r['camera'] for r in rs))}")
    print("classes:", ", ".join(f"{i} {n}" for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
