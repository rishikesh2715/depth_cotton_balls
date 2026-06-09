"""
merged_v4_to_yolo.py — Convert Abhijeet's combined_all_sequential dataset
(COCO with tip_polygon / base_rectangle + per-keypoint visibility strings)
into two sibling Ultralytics YOLO datasets:

    <out>/seg/                    # YOLOv11-seg
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt  # `cls x1 y1 x2 y2 ...` (normalized)
        data.yaml

    <out>/pose/                   # YOLOv11-pose (2 keypoints: tip, base)
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt  # `cls cx cy w h kp1x kp1y v1 kp2x kp2y v2`
        data.yaml

Input layout (as delivered by Abhijeet):

    combined_all_sequential/                # <-- --images-root
        back/<frame>.jpg
        front/<frame>.jpg                   # (not annotated yet in seq3)
        track/<frame>.jpg
    combined_all_sequential 3/
        combined_all_sequential/
            output_coco_dir/                # <-- --coco-dir
                train.json                  # 64 images, 470 anns
                val.json                    # 63 images, 264 anns

Schema differences from v3:
  - file_name already carries the camera subfolder (e.g. "track/frame_xxx.jpg")
  - Keypoints are NOT in a flat `keypoints` array; instead each ann has:
      tip_polygon       : 4-point polygon for the calyx (when visible)
      tip_point         : single point (when "guessed")
      tip_visibility    : "visible" | "guessed" | "unknown"
      base_rectangle    : 4-corner box for the base (when visible)
      base_point        : single point (when "guessed")
      base_visibility   : "visible" | "guessed" | "unknown"
  - occluded_by has 5 values; we collapse `other_boll + frame_edge -> other`
    to stay compatible with the 4-way aux head ({none, leaf, stem, other}).

Keypoint derivation (output goes to YOLO with discrete v in {0,1,2}):
  - "visible" -> centroid of polygon (tip) / rectangle (base), v=2
  - "guessed" -> the placed point coords, v=1
  - "unknown" -> (0, 0, 0)

Filename collisions across cameras would be possible but rare; we prefix
the output filename with the task tag (e.g. `track__frame_001009_t16p83s.jpg`)
to be safe.

Usage:
    python annotation/scripts/merged_v4_to_yolo.py \
        --images-root "combined_all_sequential" \
        --coco-dir   "combined_all_sequential 3/combined_all_sequential/output_coco_dir" \
        --out        annotation/datasets/v4
"""
from __future__ import annotations

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
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        area = cv2.contourArea(pts)
        polys.append((area, poly))
    if not polys:
        return None
    polys.sort(key=lambda t: -t[0])
    return polys[0][1]


def normalize_polygon(poly, W, H):
    out = []
    it = iter(poly)
    for x in it:
        y = next(it)
        out.append(max(0.0, min(1.0, float(x) / W)))
        out.append(max(0.0, min(1.0, float(y) / H)))
    return out


def bbox_coco_to_yolo(bbox, W, H):
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


def fmt_floats(xs, prec=6):
    fmt = "{:." + str(prec) + "f}"
    parts = []
    for x in xs:
        if isinstance(x, float):
            s = fmt.format(x).rstrip("0").rstrip(".")
            parts.append(s if s else "0")
        else:
            parts.append(str(x))
    return " ".join(parts)


def polygon_centroid(poly_xy):
    """poly_xy is either a list of [x,y] pairs OR a flat list [x1,y1,x2,y2,...].
    Returns (cx, cy) via cv2 moments; falls back to mean of vertices."""
    if not poly_xy:
        return None
    if isinstance(poly_xy[0], (list, tuple)):
        pts = np.array(poly_xy, dtype=np.float32)
    else:
        pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    M = cv2.moments(pts.astype(np.float32))
    if abs(M["m00"]) < 1e-6:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


def derive_keypoint(visibility, polygon_or_rect, point):
    """Return (x, y, v) for a single keypoint given the v4 schema fields.

    visibility: "visible" | "guessed" | "unknown" | None
    polygon_or_rect: the visible-case shape (tip_polygon OR base_rectangle)
    point: the guessed-case fallback (tip_point OR base_point)
    """
    if visibility == "visible" and polygon_or_rect:
        c = polygon_centroid(polygon_or_rect)
        if c is not None:
            return c[0], c[1], 2
    if visibility == "guessed" and point:
        # point is [x, y] (or sometimes [[x, y]])
        if isinstance(point[0], (list, tuple)):
            return float(point[0][0]), float(point[0][1]), 1
        return float(point[0]), float(point[1]), 1
    # Fall back: if visibility is "visible" but polygon missing, try point;
    # if "guessed" but no point, try polygon centroid.
    if polygon_or_rect:
        c = polygon_centroid(polygon_or_rect)
        if c is not None:
            return c[0], c[1], 2 if visibility == "visible" else 1
    if point:
        if isinstance(point[0], (list, tuple)):
            return float(point[0][0]), float(point[0][1]), 1
        return float(point[0]), float(point[1]), 1
    return 0.0, 0.0, 0


def normalize_kpt(kpt, W, H):
    x, y, v = kpt
    nx = max(0.0, min(1.0, float(x) / W))
    ny = max(0.0, min(1.0, float(y) / H))
    return nx, ny, int(v)


# --------------------------------------------------------------------------
# Core conversion
# --------------------------------------------------------------------------

OCC_COLLAPSE = {
    "none":       "none",
    "leaf":       "leaf",
    "stem":       "stem",
    "other_boll": "other",
    "frame_edge": "other",
    None:         "none",
}


def convert_one_split(coco_path, images_root, seg_dir, pose_dir, split, stats,
                      keep_unannotated_images=False):
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
        file_name = im["file_name"]              # e.g. "track/frame_001009_t16p83s.jpg"
        task = im.get("task") or file_name.split("/", 1)[0]

        src = images_root / file_name
        if not src.exists():
            print(f"WARN: image not found on disk, skipping: {src}", file=sys.stderr)
            stats["images_missing"] += 1
            continue

        W, H = im["width"], im["height"]

        # Tag the output filename with the camera task so we never collide
        # if frame numbers ever repeat across cameras.
        bare = Path(file_name).name
        out_basename = f"{task}__{bare}"
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

            tip = derive_keypoint(
                ann.get("tip_visibility"),
                ann.get("tip_polygon"),
                ann.get("tip_point"),
            )
            base = derive_keypoint(
                ann.get("base_visibility"),
                ann.get("base_rectangle"),
                ann.get("base_point"),
            )
            ntx, nty, tv = normalize_kpt(tip, W, H)
            nbx, nby, bv = normalize_kpt(base, W, H)

            pose_lines.append(
                f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} "
                f"{ntx:.6f} {nty:.6f} {tv} "
                f"{nbx:.6f} {nby:.6f} {bv}"
            )
            stats[f"vis_tip_{tv}"] += 1
            stats[f"vis_base_{bv}"] += 1
            stats[f"occ_{OCC_COLLAPSE.get(ann.get('occluded_by'), 'none')}"] += 1

        if not seg_lines and not keep_unannotated_images:
            stats["images_no_anns"] += 1
            continue

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
        stats[f"images_{split}_{task}"] += 1


def write_data_yaml(out_dir, kind):
    # Absolute, forward-slash path. Ultralytics resolves a relative `path:`
    # against its global `datasets_dir` setting, NOT the data.yaml's own
    # location, so `path: .` is fragile across machines. An absolute path
    # always works.
    abs_path = str(Path(out_dir).resolve()).replace("\\", "/")
    if kind == "seg":
        body = (
            "# YOLOv11 segmentation dataset - generated by merged_v4_to_yolo.py\n"
            "# Absolute path so Ultralytics resolves images/ correctly\n"
            "# regardless of cwd or the global `datasets_dir` setting.\n"
            f"path: {abs_path}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names: [cotton_boll]\n"
        )
    elif kind == "pose":
        body = (
            "# YOLOv11 pose dataset - generated by merged_v4_to_yolo.py\n"
            "# Keypoints: [tip, base]\n"
            "# Visibility mapping in this dataset (from v4 schema):\n"
            "#   COCO 'visible' -> v=2 (centroid of polygon/rectangle)\n"
            "#   COCO 'guessed' -> v=1 (the placed point)\n"
            "#   COCO 'unknown' -> v=0\n"
            "# Absolute path so Ultralytics resolves images/ correctly\n"
            "# regardless of cwd or the global `datasets_dir` setting.\n"
            f"path: {abs_path}\n"
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
        description="Convert combined_all_sequential -> YOLOv11 seg + pose datasets."
    )
    ap.add_argument("--images-root", required=True, type=Path,
                    help="Directory containing back/, front/, track/ subfolders with .jpg files.")
    ap.add_argument("--coco-dir", required=True, type=Path,
                    help="Directory containing train.json and val.json (the CORRECTED sequential split).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output root. Will create <out>/seg/ and <out>/pose/.")
    ap.add_argument("--clean", action="store_true",
                    help="If set, wipe existing <out>/seg and <out>/pose first.")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep images that have no annotations (default: skip).")
    args = ap.parse_args()

    images_root = args.images_root
    coco_dir = args.coco_dir
    out = args.out

    if not images_root.is_dir():
        print(f"ERROR: --images-root not a directory: {images_root}", file=sys.stderr)
        sys.exit(2)
    if not coco_dir.is_dir():
        print(f"ERROR: --coco-dir not a directory: {coco_dir}", file=sys.stderr)
        sys.exit(2)

    seg_dir = out / "seg"
    pose_dir = out / "pose"

    if args.clean:
        for d in (seg_dir, pose_dir):
            if d.exists():
                shutil.rmtree(d)

    stats = Counter()
    for split in ("train", "val"):
        coco_path = coco_dir / f"{split}.json"
        if not coco_path.exists():
            print(f"WARN: missing {coco_path}", file=sys.stderr)
            continue
        print(f"[convert] {split}: {coco_path}")
        convert_one_split(
            coco_path, images_root, seg_dir, pose_dir, split, stats,
            keep_unannotated_images=args.keep_empty,
        )

    write_data_yaml(seg_dir, "seg")
    write_data_yaml(pose_dir, "pose")

    # Pretty-print stats
    print("\n=== conversion summary ===")
    for k in sorted(stats):
        print(f"  {k:<35s} {stats[k]}")

    print(f"\nWrote:")
    print(f"  {seg_dir}/")
    print(f"  {pose_dir}/")
    print(f"  {seg_dir}/data.yaml")
    print(f"  {pose_dir}/data.yaml")


if __name__ == "__main__":
    main()
