"""
coco_to_yolov11_seg.py — Convert the coworker's COCO-export folders
(dataset/0Down and dataset/180down) into an Ultralytics YOLO segmentation
dataset.

Input layout (as received):
  dataset/0Down/
      annotations/instances_default.json
      images/default/*.jpg
  dataset/180down/
      annotations/instances_default.json
      images/default/*.jpg

Output layout (Ultralytics seg):
  <out>/images/train/*.jpg
  <out>/images/val/*.jpg
  <out>/labels/train/*.txt       # cls x1 y1 x2 y2 ... (normalized)
  <out>/labels/val/*.txt
  <out>/data.yaml

Split modes:
  --split by_recording   : 0Down -> train, 180down -> val (default)
  --split combined       : merge both, random 80/20 with seed

Each COCO segmentation is written as ONE YOLO line using the largest
polygon if multiple disconnected polygons exist.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def largest_polygon(segmentation):
    """COCO segmentation → largest polygon as [x1,y1,x2,y2,...]."""
    if not segmentation:
        return None
    polys = []
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        area = cv2.contourArea(pts)
        polys.append((area, poly))
    if not polys:
        return None
    polys.sort(key=lambda t: -t[0])
    return polys[0][1]


def normalize_polygon(poly, W, H):
    xs = poly[0::2]
    ys = poly[1::2]
    out = []
    for x, y in zip(xs, ys):
        out.append(max(0.0, min(1.0, float(x) / W)))
        out.append(max(0.0, min(1.0, float(y) / H)))
    return out


def read_coco(coco_path: Path, images_dir: Path, split_label: str):
    """Return list of records: {image_file, W, H, lines: [...], split_label}"""
    d = json.loads(coco_path.read_text())
    imgs = {im["id"]: im for im in d["images"]}
    cat_map = {c["id"]: c["name"] for c in d["categories"]}

    # Group annotations by image
    anns_by_img = {}
    for a in d["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    records = []
    for img_id, im in imgs.items():
        fn = im["file_name"]
        W, H = im["width"], im["height"]
        src = images_dir / fn
        if not src.exists():
            print(f"[warn] missing image: {src}")
            continue
        lines = []
        for a in anns_by_img.get(img_id, []):
            if cat_map.get(a["category_id"]) != "cotton_boll":
                continue
            poly = largest_polygon(a.get("segmentation"))
            if poly is None:
                continue
            norm = normalize_polygon(poly, W, H)
            if len(norm) < 6:
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
        if not lines:
            continue
        records.append({
            "src": src,
            "file_name": fn,
            "W": W,
            "H": H,
            "lines": lines,
            "source_split": split_label,
        })
    return records


def write_split(records, out_root: Path, split_name: str):
    img_dir = out_root / "images" / split_name
    lbl_dir = out_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        # Prefix the file name with the source split so 0Down and 180down
        # images don't collide (they can share filenames in principle).
        safe_name = f"{r['source_split']}__{r['file_name']}"
        shutil.copy2(r["src"], img_dir / safe_name)
        (lbl_dir / (Path(safe_name).stem + ".txt")).write_text("\n".join(r["lines"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset",
                    help="Folder containing 0Down/ and 180down/")
    ap.add_argument("--out", default="annotation/datasets/coworker_seg",
                    help="Output Ultralytics YOLO seg dataset root")
    ap.add_argument("--split", choices=["by_recording", "combined"],
                    default="by_recording")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Used only when --split combined")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaml-path", default=None,
                    help="Override the `path:` field written into data.yaml. "
                         "Useful when converting on one machine (e.g. a Linux "
                         "sandbox) but training on another (Windows). "
                         "Example: --yaml-path C:/Users/rrishike/depth_cotton_balls/annotation/datasets/coworker_seg")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    records_a = read_coco(
        root / "0Down" / "annotations" / "instances_default.json",
        root / "0Down" / "images" / "default",
        split_label="0Down",
    )
    records_b = read_coco(
        root / "180down" / "annotations" / "instances_default.json",
        root / "180down" / "images" / "default",
        split_label="180down",
    )

    if args.split == "by_recording":
        write_split(records_a, out_root, "train")
        write_split(records_b, out_root, "val")
        print(f"Train (0Down)     : {len(records_a)} images, {sum(len(r['lines']) for r in records_a)} instances")
        print(f"Val   (180down)   : {len(records_b)} images, {sum(len(r['lines']) for r in records_b)} instances")
    else:
        all_records = records_a + records_b
        rng = random.Random(args.seed)
        rng.shuffle(all_records)
        n_val = max(1, int(round(len(all_records) * args.val_ratio)))
        val = all_records[:n_val]
        train = all_records[n_val:]
        write_split(train, out_root, "train")
        write_split(val, out_root, "val")
        print(f"Train (combined)  : {len(train)} images, {sum(len(r['lines']) for r in train)} instances")
        print(f"Val   (combined)  : {len(val)} images, {sum(len(r['lines']) for r in val)} instances")

    # data.yaml — use --yaml-path override if provided, else resolved absolute path.
    # Forward slashes work on both Linux and Windows and avoid YAML escape issues.
    yaml_path = out_root / "data.yaml"
    dataset_root_str = args.yaml_path if args.yaml_path else str(out_root.resolve()).replace("\\", "/")
    yaml_path.write_text(
        f"path: {dataset_root_str}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: cotton_boll\n"
    )
    print(f"\nWrote dataset root to: {out_root.resolve()}")
    print(f"Point YOLO at: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
