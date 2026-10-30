#!/usr/bin/env python3
"""
Automatic first-draft labels from colour (option 2). Override pins are bright yellow, blue or red
and the floor is grey, so a pin is a blob of one strong colour. No AI model and no training.

For each picture: convert to HSV (hue = which colour, saturation = how strong, value = how bright),
keep the pixels inside each colour's range, clean the mask, and turn every blob big enough into a box.
Big red/blue blobs low in the picture are goal bases; blobs whose top is above the field wall line are
ignored (the wall toggles and things outside the field).

Saves YOLO labels (<out>/labels/<photo>.txt + classes.txt) and pictures with the boxes drawn.

Usage (Jetson or Mac, needs OpenCV):
    python3 colour_labels.py ~/four_cores/data/field_h20/cam2 --out results/colour_cam2
    python3 colour_labels.py photos --show-masks      # also save the colour masks, to tune the ranges

Known limits: grey and clear cups are not found (grey = floor colour, clear = no colour); pieces of
the same colour that touch become one box; the colour ranges were set on one recording's lighting.
"""
import argparse
import glob
import os

import cv2
import numpy as np

# Option A (decided 2026-10-23): a pin box marks one coloured PART of a pin, so a two-coloured pin gets two
# boxes; the line between them gives the pin's orientation. Goal bases are one box each.
CLASSES = ["yellow part", "blue part", "red part", "red goal", "blue goal"]
# OpenCV hue runs 0..180. (low, high) HSV ranges; red wraps around 0, so it has two.
# Saturation/brightness limits are low enough to keep pins in shadow (they turn pale).
RANGES = {
    "yellow": [((20, 75, 70), (34, 255, 255))],
    "blue":   [((100, 110, 50), (125, 255, 255))],
    "red":    [((0, 120, 60), (6, 255, 255)), ((172, 120, 60), (180, 255, 255))],
}
STRIP = 128  # width of the vertical strips used to find the field wall
COLOURS = {"yellow part": (0, 220, 255), "blue part": (255, 120, 0), "red part": (0, 0, 255),
           "red goal": (0, 0, 160), "blue goal": (160, 60, 0)}


def field_wall(hsv):
    """Bottom edge of the black-and-white checkered band on the field wall, per column (or None).

    In each vertical strip, a row belongs to the band if it switches between black and white many
    times. Everything above the band's bottom edge is outside the field."""
    H, W = hsv.shape[:2]
    s, v = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    tone = np.zeros((H, W), np.int8)            # -1 black, +1 white, 0 neither
    tone[(v < 80) & (s < 90)] = -1
    tone[(v > 150) & (s < 60)] = 1
    flips = (tone[:, 1:] * tone[:, :-1] < 0)    # black next to white along the row
    wall = np.full(W, -1.0)
    for x0 in range(0, W, STRIP):
        per_row = flips[:, x0:x0 + STRIP].sum(axis=1)
        per_row = np.convolve(per_row, np.ones(9) / 9, mode="same")  # smooth over 9 rows
        rows = np.where(per_row >= 3)[0]          # at least 3 black/white switches per strip
        if len(rows):
            # the band is the largest run of such rows; its bottom is the wall line
            runs = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
            band = max(runs, key=len)
            if len(band) >= 6:
                wall[x0:x0 + STRIP] = band[-1]
    if (wall < 0).all():
        return None
    xs = np.arange(W)
    known = wall >= 0
    return np.interp(xs, xs[known], wall[known])  # fill strips where the band is hidden


def field_mask(hsv):
    """Where the field floor is: the RoboCup method (NimbRo 2016, arXiv 1809.11127; NUbots, arXiv
    1310.5781) with grey foam tiles instead of green carpet.

    Floor-coloured pixels (grey: little colour, medium brightness) that connect to the bottom of the
    picture form the floor; its convex hull fills the holes left by pieces lying on it. Anything outside
    the hull is outside the field. Returns a 0/255 mask at full size, or None if no floor is seen."""
    H, W = hsv.shape[:2]
    small = cv2.resize(hsv, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    s, v = small[:, :, 1], small[:, :, 2]
    floor = ((s < 45) & (v > 60) & (v < 190)).astype(np.uint8)
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(floor)
    bottom = set(labels[-2:, :].ravel()) - {0}   # pieces of floor touching the bottom edge
    if not bottom:
        return None
    biggest = max(bottom, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    if stats[biggest, cv2.CC_STAT_AREA] < 0.05 * floor.size:  # too little floor to trust
        return None
    # Grey lab walls look like the grey floor, so a convex hull leaks into the room. Instead: the floor
    # region itself, holes from pieces closed, and nothing above the checkered field-wall band.
    region = (labels == biggest).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.resize(region * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    wall = field_wall(hsv)
    if wall is not None:
        rows = np.arange(H)[:, None]
        mask[rows < wall[None, :]] = 0
    return mask


_SAM = None


def sam_field_mask(img, hsv):
    """Where the field floor is, found by SAM 2 (Meta, 2024; freely downloadable): we give it a few
    points near the bottom of the picture that look like floor ("this is floor") and one point at the
    top ("this is not"), and it outlines the whole region those points belong to.
    Returns a 0/255 mask, or None if no floor-like point is found near the bottom."""
    global _SAM
    if _SAM is None:
        try:
            from ultralytics import SAM
        except ImportError:
            print("ultralytics not installed: no field finder (run on the Jetson in gpuenv for --field sam)")
            _SAM = False
        else:
            _SAM = SAM("sam2.1_b.pt")
    if _SAM is False:
        return None
    H, W = hsv.shape[:2]
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    floorish = (s < 45) & (v > 60) & (v < 190)
    points, labels = [], []
    for fx in (0.15, 0.35, 0.5, 0.65, 0.85):        # candidate floor points along the bottom
        x = int(fx * W)
        for fy in (0.95, 0.88, 0.8):
            y = int(fy * H)
            if floorish[max(0, y - 10):y + 10, max(0, x - 10):x + 10].mean() > 0.8:
                points.append([x, y]); labels.append(1)
                break
    if not points:
        return None
    points.append([W // 2, int(0.03 * H)]); labels.append(0)  # top of the picture: not floor
    r = _SAM(img, points=[points], labels=[labels], verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8) * 255
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    # Pieces lying on the floor punch holes in the floor outline: fill every hole inside it.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c for c in contours if cv2.contourArea(c) > 0.01 * H * W], -1, 255, cv2.FILLED)
    return filled


def split_touching(component):
    """Split one blob into pieces when it looks like several pins touching (watershed).
    Each pin has its own thick middle, so the distance-to-edge map has one peak per pin."""
    dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    cores = (dist > 0.55 * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(cores)
    if n <= 2:  # background + one core = one pin
        return [component]
    markers = markers + 1
    markers[(component > 0) & (cores == 0)] = 0  # unknown: let watershed decide
    cv2.watershed(cv2.cvtColor(component * 255, cv2.COLOR_GRAY2BGR), markers)
    return [((markers == k) & (component > 0)).astype(np.uint8) for k in range(2, n + 1)]


def boxes_for(hsv, colour, min_area, goal_area):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RANGES[colour]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # fill small holes
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_area:
            continue
        if colour in ("red", "blue") and area >= goal_area:
            found.append((f"{colour} goal", x, y, x + w, y + h))
            continue
        pieces = [(labels[y:y + h, x:x + w] == k).astype(np.uint8)]
        if area >= 3 * min_area:
            pieces = split_touching(pieces[0])
        for p in pieces:
            if cv2.countNonZero(p) < min_area / 2:
                continue
            px, py, pw, ph = cv2.boundingRect(p)
            found.append((f"{colour} part", x + px, y + py, x + px + pw, y + py + ph))
    return found, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", help="output folder (default <folder>/colour_labels)")
    ap.add_argument("--min-area", type=int, default=250, help="smallest blob in pixels (full-size picture)")
    ap.add_argument("--goal-area", type=int, default=8000, help="red/blue blobs at least this big are goal bases")
    ap.add_argument("--horizon", type=float, default=0.25,
                    help="ignore blobs whose bottom is in the top part of the picture (fraction of height)")
    ap.add_argument("--field", choices=["sam", "floor", "wall", "none"], default="sam",
                    help="how to find the field: sam = SAM 2 outlines the floor from a few floor points "
                         "(default; tested 2026-10-21: removed every off-field box on 8 pictures and kept most "
                         "pins; needs ultralytics, run on the Jetson), floor = grey floor region below the wall band (RoboCup-style; "
                         "tested 2026-10-21: removes some off-field boxes but also drops real pins, because the "
                         "grey floor looks like the grey lab walls), wall = below the checkered wall band, "
                         "none = only ignore the top of the picture (default until a better field finder exists)")
    ap.add_argument("--show-masks", action="store_true")
    ap.add_argument("--max", type=int, default=0, help="only the first N pictures (0 = all)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.jpg")))
    if args.max:
        files = files[:args.max]
    out = args.out or os.path.join(args.folder, "colour_labels")
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    with open(os.path.join(out, "labels", "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    counts = {c: 0 for c in CLASSES}
    for path in files:
        img = cv2.imread(path)
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        field = field_mask(hsv) if args.field == "floor" else sam_field_mask(img, hsv) if args.field == "sam" else None
        if field is not None:  # "near the floor": grow the floor by ~20 px so pins lying on cups still count
            near_floor = cv2.dilate(field, np.ones((41, 41), np.uint8))
        wall = field_wall(hsv) if args.field == "wall" else None
        boxes, masks = [], []
        for colour in RANGES:
            found, mask = boxes_for(hsv, colour, args.min_area, args.goal_area)
            for b in found:
                bottom, centre_x = min(b[4], H - 1), (b[1] + b[3]) // 2
                if field is not None:      # keep a piece whose bottom part is on or near the floor
                    strip = near_floor[max(0, bottom - 10):min(H, bottom + 12), max(0, b[1]):b[3]]
                    keep = bottom >= H - 5 or (strip.size > 0 and strip.mean() > 0.1 * 255)
                elif wall is not None:     # older rule: bottom below the checkered wall band
                    keep = bottom > wall[centre_x]
                else:                      # no field found: ignore the top part of the picture
                    keep = bottom > args.horizon * H
                if keep:
                    boxes.append(b)
            masks.append(mask)
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out, "labels", stem + ".txt"), "w") as f:
            for name, x1, y1, x2, y2 in boxes:
                counts[name] += 1
                f.write(f"{CLASSES.index(name)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                        f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        drawn = img.copy()
        if field is not None:  # the field outline used for the filter
            contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(drawn, contours, -1, (255, 0, 255), 2)
        if wall is not None:  # the field-wall line used for the filter
            cv2.polylines(drawn, [np.column_stack([np.arange(W), wall]).astype(np.int32)], False, (255, 0, 255), 1)
        for name, x1, y1, x2, y2 in boxes:
            cv2.rectangle(drawn, (x1, y1), (x2, y2), COLOURS[name], 2)
            cv2.putText(drawn, name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[name], 2)
        cv2.imwrite(os.path.join(out, stem + ".jpg"), drawn)
        if args.show_masks:
            cv2.imwrite(os.path.join(out, stem + "_masks.jpg"), cv2.merge(masks))  # B=yellow G=blue R=red
    print(f"{len(files)} pictures: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    print(f"saved labels + pictures in {out}")


if __name__ == "__main__":
    main()
