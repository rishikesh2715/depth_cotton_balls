"""
bootstrap_from_sam2.py — Convert existing SAM 2 per-boll masks into a
COCO-style JSON that CVAT can import as pre-annotations, matching the
new 3-label schema in annotation/cvat/cvat_labels.json (cotton_boll
polygon + tip points + base points).

Input:
  - A sample_manifest.csv produced by sample_frames.py
  - Per-recording masks under <work>/masks/<frame_idx>/<boll_id>.png

Output:
  - A single sam2_bootstrap_coco.json that CVAT can import via
    "Upload annotations" → "COCO 1.0".

What the bootstrap does and does NOT do:
  - DOES: write one `cotton_boll` polygon per SAM 2 mask, with boll_id
    pre-filled from the SAM 2 tag_id. Defaults the other attributes to
    their most common value (annotation_type=whole_visible,
    occluded_by=none, tip/base_visibility=visible) — you override per
    boll in CVAT where wrong.
  - DOES NOT: place tip / base point annotations. SAM 2 masks don't
    encode anatomy, so you click those two per boll manually. The
    cotton_boll `boll_id` value is what you'll type into the tip/base
    point `boll_id` fields to link them.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

MIN_AREA = 50  # px — drop specks


def mask_to_polygons(mask: np.ndarray):
    """Return list of [x1,y1,x2,y2,...] polygons (COCO segmentation format)."""
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        c = c.squeeze(1)  # (N, 2)
        if c.ndim != 2 or c.shape[0] < 3:
            continue
        polys.append(c.flatten().astype(float).tolist())
    return polys


def bbox_from_polygons(polys):
    xs, ys = [], []
    for poly in polys:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs:
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def area_from_polygons(polys):
    total = 0.0
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        total += cv2.contourArea(pts)
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="sample_manifest.csv from sample_frames.py")
    ap.add_argument("--sampled-dir", required=True,
                    help="Folder with the copied sampled frames.")
    ap.add_argument("--out", required=True,
                    help="Output COCO JSON path (e.g., sam2_bootstrap_coco.json)")
    args = ap.parse_args()

    sampled_dir = Path(args.sampled_dir)
    out_path = Path(args.out)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for r in rows:
        sampled_fn = r["sampled_filename"]
        work_dir = Path(r["source_work_dir"])
        frame_idx = int(r["source_frame_idx"])
        img_path = sampled_dir / sampled_fn
        if not img_path.exists():
            print(f"[warn] missing sampled image: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] cannot read: {img_path}")
            continue
        H, W = img.shape[:2]
        images.append({
            "id": img_id,
            "file_name": sampled_fn,
            "width": W,
            "height": H,
        })

        masks_dir = work_dir / "masks" / f"{frame_idx:05d}"
        if not masks_dir.is_dir():
            img_id += 1
            continue

        for m_path in sorted(masks_dir.glob("*.png")):
            try:
                sam2_tag_id = int(m_path.stem)
            except ValueError:
                continue
            mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != (H, W):
                print(f"[warn] mask size mismatch for {m_path}")
                continue
            polys = mask_to_polygons(mask)
            if not polys:
                continue
            bbox = bbox_from_polygons(polys)
            if bbox is None:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # cotton_boll
                "segmentation": polys,
                "bbox": bbox,
                "area": area_from_polygons(polys),
                "iscrowd": 0,
                "attributes": {
                    "boll_id": sam2_tag_id,
                    "tip_visibility": "visible",
                    "base_visibility": "visible",
                    "occluded_by": "none",
                    "annotation_type": "whole_visible",
                },
            })
            ann_id += 1
        img_id += 1

    coco = {
        "info": {
            "description": "SAM 2 bootstrap pre-annotations for CVAT "
                           "(cotton_boll polygons only; tip/base points "
                           "to be added manually during annotation)",
            "version": "0.2",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "cotton_boll", "supercategory": ""},
            {"id": 2, "name": "tip",         "supercategory": ""},
            {"id": 3, "name": "base",        "supercategory": ""},
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(coco, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations (pre-polys): {len(annotations)}")


if __name__ == "__main__":
    main()
