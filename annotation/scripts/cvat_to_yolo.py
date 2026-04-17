"""
cvat_to_yolo.py — Convert a CVAT COCO 1.0 export into Ultralytics-style
YOLOv8 *seg* and *pose* dataset layouts.

Expected CVAT export (COCO 1.0) shape:
  annotations.json
  images/<file>.jpg

We produce two sibling datasets that share the same image splits:

  <out>/seg/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt      # YOLOv8-seg: cls x1 y1 x2 y2 ... (norm)
      data.yaml

  <out>/pose/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt      # YOLOv8-pose: cls cx cy w h  kp1x kp1y v1 ...
      data.yaml

Split strategy (pick one):
  --split-by filename   : use a --split-pattern regex to assign images to splits
  --split-by ratio      : random 80/20 (NOT recommended for frames from same video)

Keypoints included (in this fixed order):
  0: base
  1: tip
  2: midpoint
  3: width_left
  4: width_right
A keypoint that is absent or flagged 0 is written as "0 0 0".

Run:
  python cvat_to_yolo.py --coco annotations.json --images images/ \
      --out datasets/ --split-by filename \
      --train-pattern "handheld_1" --val-pattern "handheld_2"
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

KP_ORDER = ["kp_base", "kp_tip", "kp_midpoint", "kp_width_left", "kp_width_right"]
KP_INDEX = {name: i for i, name in enumerate(KP_ORDER)}


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    images_by_id = {im["id"]: im for im in data["images"]}
    cats_by_id = {c["id"]: c for c in data["categories"]}
    # Group annotations by image
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return images_by_id, cats_by_id, anns_by_img


def poly_to_yolo_seg(polys, W, H):
    """Take COCO segmentation polygons and normalize coordinates to [0,1].
    YOLOv8-seg supports multiple polygons per instance — we concatenate and
    use the largest polygon (YOLO seg expects a single polygon per line)."""
    largest = max(polys, key=lambda p: cv2.contourArea(np.array(p).reshape(-1, 2).astype(np.float32)))
    xs = largest[0::2]
    ys = largest[1::2]
    norm = []
    for x, y in zip(xs, ys):
        norm.append(max(0.0, min(1.0, x / W)))
        norm.append(max(0.0, min(1.0, y / H)))
    return norm


def bbox_to_yolo(bbox, W, H):
    """COCO bbox [x, y, w, h] → YOLO (cx, cy, w, h) normalized."""
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / W
    cy = (y + bh / 2) / H
    return cx, cy, bw / W, bh / H


def collect_keypoints(anns_per_img, cats_by_id, W, H):
    """Given all boll + keypoint anns on one image, return per-boll dict:
       fruit_id -> {'boll': boll_ann, 'kps': {kp_name: (x, y, v)}}"""
    by_fruit = defaultdict(lambda: {"boll": None, "kps": {}})
    for a in anns_per_img:
        cat_name = cats_by_id[a["category_id"]]["name"]
        attrs = a.get("attributes", {})
        fid = attrs.get("fruit_id")
        if fid is None:
            continue
        try:
            fid = int(fid)
        except (ValueError, TypeError):
            continue
        if cat_name == "boll":
            by_fruit[fid]["boll"] = a
        elif cat_name in KP_INDEX:
            # Point annotation: segmentation = [[x, y]] or bbox = [x, y, 0, 0]
            if "bbox" in a and a["bbox"]:
                x, y = a["bbox"][0], a["bbox"][1]
            elif a.get("segmentation"):
                seg = a["segmentation"][0] if isinstance(a["segmentation"], list) else a["segmentation"]
                if isinstance(seg, list) and len(seg) >= 2:
                    x, y = seg[0], seg[1]
                else:
                    continue
            else:
                continue
            v = int(attrs.get("visible", 2))
            by_fruit[fid]["kps"][cat_name] = (x, y, v)
    return by_fruit


def write_seg_label(boll_anns, W, H, out_path: Path):
    lines = []
    for a in boll_anns:
        polys = a.get("segmentation") or []
        if not polys:
            continue
        norm = poly_to_yolo_seg(polys, W, H)
        if len(norm) < 6:
            continue
        lines.append(" ".join(["0"] + [f"{v:.6f}" for v in norm]))
    out_path.write_text("\n".join(lines))


def write_pose_label(per_fruit, W, H, out_path: Path):
    lines = []
    for fid, bundle in per_fruit.items():
        boll = bundle["boll"]
        if not boll or not boll.get("bbox"):
            continue
        cx, cy, bw, bh = bbox_to_yolo(boll["bbox"], W, H)
        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for name in KP_ORDER:
            if name in bundle["kps"]:
                x, y, v = bundle["kps"][name]
                parts.extend([f"{x/W:.6f}", f"{y/H:.6f}", str(int(v))])
            else:
                parts.extend(["0", "0", "0"])
        lines.append(" ".join(parts))
    out_path.write_text("\n".join(lines))


def assign_split(filename, args):
    if args.split_by == "filename":
        if re.search(args.val_pattern, filename):
            return "val"
        if re.search(args.train_pattern, filename):
            return "train"
        return None  # ignore
    else:
        # ratio
        import random
        return "val" if random.random() < args.val_ratio else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-by", choices=["filename", "ratio"], default="filename")
    ap.add_argument("--train-pattern", default="handheld_1", help="regex")
    ap.add_argument("--val-pattern", default="handheld_2", help="regex")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    coco_path = Path(args.coco)
    images_dir = Path(args.images)
    out_root = Path(args.out)

    images_by_id, cats_by_id, anns_by_img = load_coco(coco_path)

    seg_root = out_root / "seg"
    pose_root = out_root / "pose"
    for root in (seg_root, pose_root):
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_seg = {"train": 0, "val": 0}
    n_pose = {"train": 0, "val": 0}

    for img_id, im in images_by_id.items():
        fn = im["file_name"]
        split = assign_split(fn, args)
        if split is None:
            continue
        W, H = im["width"], im["height"]
        src = images_dir / fn
        if not src.exists():
            print(f"[warn] missing image: {src}")
            continue

        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        # --- Seg dataset ---
        boll_anns = [a for a in anns if cats_by_id[a["category_id"]]["name"] == "boll" and a.get("segmentation")]
        if boll_anns:
            dst_img = seg_root / "images" / split / fn
            if not dst_img.exists():
                shutil.copy2(src, dst_img)
            write_seg_label(boll_anns, W, H, seg_root / "labels" / split / (Path(fn).stem + ".txt"))
            n_seg[split] += 1

        # --- Pose dataset ---
        per_fruit = collect_keypoints(anns, cats_by_id, W, H)
        has_any_kp = any(bundle["kps"] for bundle in per_fruit.values())
        if has_any_kp:
            dst_img = pose_root / "images" / split / fn
            if not dst_img.exists():
                shutil.copy2(src, dst_img)
            write_pose_label(per_fruit, W, H, pose_root / "labels" / split / (Path(fn).stem + ".txt"))
            n_pose[split] += 1

    # data.yaml files
    seg_yaml = seg_root / "data.yaml"
    seg_yaml.write_text(
        "path: {}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: boll\n".format(seg_root.resolve())
    )
    pose_yaml = pose_root / "data.yaml"
    # flip_idx: base/tip/midpoint are symmetric; width_left(3) <-> width_right(4)
    pose_yaml.write_text(
        "path: {}\n"
        "train: images/train\nval: images/val\n"
        "kpt_shape: [{n}, 3]\n"
        "flip_idx: [0, 1, 2, 4, 3]\n"
        "names:\n  0: boll\n".format(pose_root.resolve(), n=len(KP_ORDER))
    )

    print(f"SEG  : train={n_seg['train']}  val={n_seg['val']}  → {seg_root}")
    print(f"POSE : train={n_pose['train']}  val={n_pose['val']}  → {pose_root}")


if __name__ == "__main__":
    main()
