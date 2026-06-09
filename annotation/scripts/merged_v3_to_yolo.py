"""
merged_v3_to_yolo.py — Convert Abhijeet's merged_dataset_v3 (COCO with the
tip/base point shapes already merged onto each cotton_boll annotation as a
real `keypoints` array) into two sibling Ultralytics YOLO datasets:

    <out>/seg/                    # YOLOv11-seg
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt  # `cls x1 y1 x2 y2 ...` (normalized)
        data.yaml

    <out>/pose/                   # YOLOv11-pose (2 keypoints: tip, base)
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt  # `cls cx cy w h kp1x kp1y v1 kp2x kp2y v2`
        data.yaml

Input layout (as delivered by Abhijeet):

    new_dataset/merged_dataset_v3/
        backdown/
            images/<frame>.jpg          # actual files, NO src1_ prefix
            train.json                  # COCO; image entries have file_name="src1_<frame>.jpg"
            val.json                    # but original_name="<frame>.jpg" — use original_name.
        frontdown/
            images/<frame>.jpg
            train.json
            val.json

We preserve the train/val split exactly as Abhijeet shipped it (15+25=40 train,
2+4=6 val). To prevent filename collisions across the two recordings, output
images are renamed `<recording>__<frame>.jpg` (e.g. `back__frame_000838.jpg`,
`front__frame_001259.jpg`).

Pose dataset specifics (matches the v3 schema):
  - Keypoint order: [tip, base]   (matches COCO `categories[0].keypoints`)
  - kpt_shape: [2, 3]
  - flip_idx: [0, 1]              (identity — tip/base do NOT swap on hflip)
  - Visibility flags 0/1/2 are passed through unchanged.

If a polygon has fewer than 3 vertices (degenerate), the instance is skipped
and a warning is printed. If an image has zero valid annotations after that
filter, it is still copied to images/<split>/ with an empty label file
(Ultralytics treats it as a negative).

Usage:
    python annotation/scripts/merged_v3_to_yolo.py \
        --root new_dataset/merged_dataset_v3 \
        --out  annotation/datasets/v3
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def largest_polygon(segmentation):
    """COCO `segmentation` is a list of polygons (each a flat [x1, y1, x2, y2, ...]).
    Return the single polygon with the largest contour area, or None if none usable.
    """
    polys = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue  # need at least 3 vertices to form a polygon
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        area = cv2.contourArea(pts)
        polys.append((area, poly))
    if not polys:
        return None
    polys.sort(key=lambda t: -t[0])
    return polys[0][1]


def normalize_polygon(poly, W, H):
    """Flatten poly [x1, y1, x2, y2, ...] -> normalized [x1/W, y1/H, ...] clipped to [0, 1]."""
    out = []
    it = iter(poly)
    for x in it:
        y = next(it)
        out.append(max(0.0, min(1.0, float(x) / W)))
        out.append(max(0.0, min(1.0, float(y) / H)))
    return out


def bbox_coco_to_yolo(bbox, W, H):
    """COCO bbox [x, y, w, h] (top-left + size) -> YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def kpts_coco_to_yolo(keypoints, W, H, num_kpts=2):
    """COCO keypoints flat [x1, y1, v1, x2, y2, v2, ...] -> YOLO [x1/W, y1/H, v1, ...]
    with coords clipped to [0, 1] and visibility kept as-is.
    """
    out = []
    if keypoints is None:
        keypoints = [0.0, 0.0, 0] * num_kpts
    if len(keypoints) < num_kpts * 3:
        keypoints = list(keypoints) + [0.0, 0.0, 0] * (num_kpts - len(keypoints) // 3)
    for i in range(num_kpts):
        x, y, v = keypoints[3 * i], keypoints[3 * i + 1], keypoints[3 * i + 2]
        nx = max(0.0, min(1.0, float(x) / W))
        ny = max(0.0, min(1.0, float(y) / H))
        out.extend([nx, ny, int(v)])
    return out


def fmt_floats(xs, prec=6):
    """Format a list of floats compactly (no trailing zeros, fixed precision)."""
    fmt = "{:." + str(prec) + "f}"
    parts = []
    for x in xs:
        if isinstance(x, float):
            s = fmt.format(x).rstrip("0").rstrip(".")
            parts.append(s if s else "0")
        else:
            parts.append(str(x))
    return " ".join(parts)


# --------------------------------------------------------------------------
# Core conversion
# --------------------------------------------------------------------------

def convert_one_split(coco_path, src_images_dir, recording_tag,
                      seg_dir, pose_dir, split, stats):
    """Convert one COCO file (e.g. backdown/train.json) into label files inside
    seg_dir and pose_dir, copying images to <dir>/images/<split>/<recording>__<orig>.
    """
    data = json.loads(coco_path.read_text())
    images = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c for c in data["categories"]}

    if 1 not in cats or cats[1]["name"] != "cotton_boll":
        print(f"WARN: unexpected category schema in {coco_path}: {cats}", file=sys.stderr)

    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    seg_img_dir = seg_dir / "images" / split
    seg_lbl_dir = seg_dir / "labels" / split
    pose_img_dir = pose_dir / "images" / split
    pose_lbl_dir = pose_dir / "labels" / split
    for d in (seg_img_dir, seg_lbl_dir, pose_img_dir, pose_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    for img_id, im in images.items():
        # Use original_name (true filename on disk), NOT file_name (which has src1_ prefix)
        orig_name = im.get("original_name") or im["file_name"]
        if orig_name.startswith("src1_"):
            orig_name = orig_name[len("src1_"):]
        elif orig_name.startswith("src2_"):
            orig_name = orig_name[len("src2_"):]

        src = src_images_dir / orig_name
        if not src.exists():
            print(f"WARN: image not found on disk, skipping: {src}", file=sys.stderr)
            stats["images_missing"] += 1
            continue

        W, H = im["width"], im["height"]
        out_basename = f"{recording_tag}__{orig_name}"
        out_stem = Path(out_basename).stem

        seg_lines = []
        pose_lines = []

        for ann in anns_by_img.get(img_id, []):
            if ann.get("category_id") != 1:
                stats["ann_skipped_non_boll"] += 1
                continue

            poly = largest_polygon(ann.get("segmentation") or [])
            if poly is None:
                stats["ann_skipped_bad_polygon"] += 1
                continue
            norm_poly = normalize_polygon(poly, W, H)
            if len(norm_poly) < 6:
                stats["ann_skipped_bad_polygon"] += 1
                continue
            seg_lines.append("0 " + fmt_floats(norm_poly))

            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                stats["ann_skipped_no_bbox"] += 1
                continue
            cx, cy, nw, nh = bbox_coco_to_yolo(bbox, W, H)
            kpts = kpts_coco_to_yolo(ann.get("keypoints"), W, H, num_kpts=2)
            pose_lines.append(
                f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} "
                + " ".join(
                    f"{kpts[3 * i]:.6f} {kpts[3 * i + 1]:.6f} {kpts[3 * i + 2]}"
                    for i in range(2)
                )
            )
            stats[f"vis_tip_{kpts[2]}"] += 1
            stats[f"vis_base_{kpts[5]}"] += 1

        dst_seg = seg_img_dir / out_basename
        dst_pose = pose_img_dir / out_basename
        if not dst_seg.exists():
            shutil.copy2(src, dst_seg)
        if not dst_pose.exists():
            shutil.copy2(src, dst_pose)

        (seg_lbl_dir / f"{out_stem}.txt").write_text(
            "\n".join(seg_lines) + ("\n" if seg_lines else "")
        )
        (pose_lbl_dir / f"{out_stem}.txt").write_text(
            "\n".join(pose_lines) + ("\n" if pose_lines else "")
        )

        stats[f"images_{split}"] += 1
        stats[f"labels_{split}_seg"] += len(seg_lines)
        stats[f"labels_{split}_pose"] += len(pose_lines)


def write_data_yaml(out_dir, kind):
    """Write Ultralytics-format data.yaml. `kind` is 'seg' or 'pose'.

    `path: .` makes the file portable: Ultralytics resolves a non-absolute
    `path` relative to the data.yaml's own location, so the dataset folder
    can be moved between machines (Linux sandbox -> Windows workstation)
    without regenerating.
    """
    if kind == "seg":
        body = (
            "# YOLOv11 segmentation dataset - generated by merged_v3_to_yolo.py\n"
            "# `path: .` is resolved relative to this data.yaml's location.\n"
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names: [cotton_boll]\n"
        )
    elif kind == "pose":
        body = (
            "# YOLOv11 pose dataset - generated by merged_v3_to_yolo.py\n"
            "# Keypoints: [tip, base] - flip_idx is identity because tip/base do not\n"
            "# swap on horizontal flip.\n"
            "# `path: .` is resolved relative to this data.yaml's location.\n"
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names: [cotton_boll]\n"
            "kpt_shape: [2, 3]\n"
            "flip_idx: [0, 1]\n"
        )
    else:
        raise ValueError(kind)
    (out_dir / "data.yaml").write_text(body)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert merged_dataset_v3 -> YOLOv11 seg + pose datasets."
    )
    ap.add_argument("--root", required=True, type=Path,
                    help="Path to merged_dataset_v3 (containing backdown/ and frontdown/).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output root. Will create <out>/seg/ and <out>/pose/.")
    ap.add_argument("--clean", action="store_true",
                    help="If set, wipe existing <out>/seg and <out>/pose first.")
    args = ap.parse_args()

    root = args.root
    out = args.out

    if not root.is_dir():
        print(f"ERROR: --root not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    seg_dir = out / "seg"
    pose_dir = out / "pose"

    if args.clean:
        for d in (seg_dir, pose_dir):
            if d.exists():
                shutil.rmtree(d)

    recordings = [
        ("backdown", "back"),
        ("frontdown", "front"),
    ]
    splits = ["train", "val"]

    stats = Counter()

    for sub, tag in recordings:
        rec_dir = root / sub
        if not rec_dir.is_dir():
            print(f"WARN: missing recording dir {rec_dir}, skipping", file=sys.stderr)
            continue
        for split in splits:
            coco_path = rec_dir / f"{split}.json"
            images_dir = rec_dir / "images"
            if not coco_path.exists():
                print(f"WARN: missing {coco_path}, skipping", file=sys.stderr)
                continue
            if not images_dir.is_dir():
                print(f"WARN: missing {images_dir}, skipping", file=sys.stderr)
                continue
            print(f"-> Converting {sub}/{split} (tag={tag}) ...")
            convert_one_split(coco_path, images_dir, tag,
                              seg_dir, pose_dir, split, stats)

    write_data_yaml(seg_dir, "seg")
    write_data_yaml(pose_dir, "pose")

    print("\n=== Done ===")
    print("Output:")
    print(f"  {seg_dir}")
    print(f"  {pose_dir}")
    print("\nStats:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
