#!/usr/bin/env python3
"""
Turns a recording + draft labels into a YOLO training set, split by TIME (not at random).

Frames next to each other are nearly identical, so a random split would put near-copies of the test
pictures into training and make the score look better than it is. Instead: the first 70% of the
recording trains, the next 10% validates (picks the best epoch), the last 20% tests.

Frames where a camera sees mostly the operator (more than half the picture very dark, e.g. cam2's
jeans) are left out.

Input:
    <recording>/cam0..cam3/<frame>.jpg        pictures (e.g. ~/four_cores/data/field_h20, the 1-in-10 sample)
    <labels>/cam0..cam3/labels/<frame>.txt    YOLO labels from colour_labels.py, + labels/classes.txt
Output:
    <out>/images/{train,val,test}/camN_<frame>.jpg   (links to the original pictures, no copies)
    <out>/labels/{train,val,test}/camN_<frame>.txt
    <out>/data.yaml                                  for Ultralytics training
    <out>/split_report.txt

Usage (Jetson):
    gpuenv/bin/python build_dataset.py data/field_h20 results/labels_v1 datasets/override_v1
"""
import argparse
import os

import cv2

CAMERAS = 4


def mostly_dark(path, share=0.5):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return True
    g = cv2.resize(g, (160, 100))
    return (g < 50).mean() > share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    args = ap.parse_args()

    names = sorted(os.listdir(os.path.join(args.recording, "cam0")))
    n = len(names)
    cut1, cut2 = int(n * args.train), int(n * (args.train + args.val))
    split_of = {name: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, name in enumerate(names)}
    classes = open(os.path.join(args.labels, "cam0", "labels", "classes.txt")).read().split("\n")
    classes = [c for c in classes if c.strip()]

    counts = {s: {"pictures": 0, "boxes": 0, "skipped_dark": 0, "no_label": 0} for s in ("train", "val", "test")}
    for split in counts:
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
    for name in names:
        split = split_of[name]
        stem = os.path.splitext(name)[0]
        for c in range(CAMERAS):
            img = os.path.abspath(os.path.join(args.recording, f"cam{c}", name))
            lab = os.path.join(args.labels, f"cam{c}", "labels", stem + ".txt")
            if not os.path.exists(lab):
                counts[split]["no_label"] += 1
                continue
            if mostly_dark(img):
                counts[split]["skipped_dark"] += 1
                continue
            dst = f"cam{c}_{stem}"
            link = os.path.join(args.out, "images", split, dst + ".jpg")
            if not os.path.lexists(link):
                os.symlink(img, link)
            lines = open(lab).read().strip()
            with open(os.path.join(args.out, "labels", split, dst + ".txt"), "w") as f:
                f.write(lines + ("\n" if lines else ""))
            counts[split]["pictures"] += 1
            counts[split]["boxes"] += len(lines.splitlines()) if lines else 0

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    report = (f"recording {args.recording}: {n} frames x {CAMERAS} cameras; split by time "
              f"(train frames 1-{cut1}, val {cut1 + 1}-{cut2}, test {cut2 + 1}-{n})\nclasses: {classes}\n" +
              "\n".join(f"{s:5s}: {v['pictures']} pictures, {v['boxes']} boxes, {v['skipped_dark']} skipped "
                        f"(mostly dark), {v['no_label']} without labels" for s, v in counts.items()))
    open(os.path.join(args.out, "split_report.txt"), "w").write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
