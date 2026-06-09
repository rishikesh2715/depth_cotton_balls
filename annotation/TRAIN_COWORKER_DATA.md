# Train YOLOv11-seg on the coworker's pilot dataset

**Dataset state as received:**
- 47 images total (0Down: 29, 180down: 18)
- 281 `cotton_boll` segmentation polygons
- Categories `tip` and `base` are defined in the COCO but **no annotations use them** — tip/base are stored only as *visibility string attributes*, not as point coordinates
- Conclusion: we can train segmentation tonight, but keypoint/pose training has to wait until the annotation schema is fixed (see "Roboflow + keypoints" at the bottom of this file)

---

## One-shot: convert + train

From the workspace folder:

```bash
# 1) Convert COCO → YOLOv11 seg dataset (by-recording split: 0Down train, 180down val)
python annotation/scripts/coco_to_yolov11_seg.py \
    --dataset-root dataset \
    --out annotation/datasets/coworker_seg \
    --split by_recording

# 2) Train YOLOv11n-seg
pip install ultralytics      # only needed once

python annotation/scripts/train_yolov11_seg.py \
    --data annotation/datasets/coworker_seg/data.yaml \
    --model yolo11n-seg.pt \
    --epochs 150 --imgsz 1280 --batch 4 \
    --project runs/coworker_seg --name v0
```

Expected wall-clock on a single consumer GPU (RTX 3060/4060/4070 class):
- 30–60 seconds per epoch at imgsz=1280, batch=4
- Full 150 epochs: ~1.5–2.5 hours (early-stop usually fires around epoch 60–100)

Watch `runs/coworker_seg/v0/results.png` — it live-updates.

---

## Settings rationale (so you can tune)

**Why `imgsz=1280`, not 640?**  
Your images are portrait phone shots, ~2988×5312. At `imgsz=640` YOLO downscales the long side to 640 — each cotton boll ends up <40 pixels, below the minimum feature size a nano-seg head can resolve reliably. 1280 keeps bolls in the ~80–120 px range. Bump to 1536 if VRAM allows (it helps mAP noticeably on small bolls).

**Why `batch=4`?**  
At 1280 × 1280 with a nano model, batch=4 fits in ~6 GB VRAM. If you're on an 8 GB card, try 8. If OOM, drop to 2 with `--workers 2`.

**Why `yolo11n-seg.pt` to start?**  
47 images is small. A nano model regularizes well on small data. If val mAP plateaus below 0.7 (box) / 0.55 (mask), upgrade:

```bash
python annotation/scripts/train_yolov11_seg.py \
    --data annotation/datasets/coworker_seg/data.yaml \
    --model yolo11s-seg.pt --epochs 200 --imgsz 1280 --batch 4 \
    --project runs/coworker_seg --name v1_small
```

**Why `flipud=0.0`, `fliplr=0.5`?**  
Vertical flip would put the base above the tip, which matters once we add keypoints later. Horizontal flip is fine for a boll.

**Why `close_mosaic=15`?**  
Mosaic helps with diverse small-object training but hurts final-epoch mask quality. Turning it off for the last 15 epochs lets the decoder sharpen edges.

---

## What "good" looks like

For a 47-image pilot with clean polygons, expect:

| Metric           | Floor | Decent | Good   |
|------------------|-------|--------|--------|
| box mAP50        | 0.70  | 0.85   | 0.92+  |
| mask mAP50       | 0.55  | 0.72   | 0.85+  |
| box mAP50-95     | 0.40  | 0.55   | 0.65+  |
| mask mAP50-95    | 0.30  | 0.45   | 0.58+  |

If your val numbers are much worse than train, it's probably cross-recording domain shift between 0Down and 180down (different angles, different lighting). Run an alternative split to diagnose:

```bash
python annotation/scripts/coco_to_yolov11_seg.py \
    --dataset-root dataset \
    --out annotation/datasets/coworker_seg_combined \
    --split combined --val-ratio 0.2 --seed 42
```

If combined-split val mAP is much higher than by_recording val mAP, the model isn't overfit — it just hasn't seen enough angle diversity. That's useful information for the meeting.

---

## After training — predict + measure

```bash
# Inference on held-out val images
yolo predict model=runs/coworker_seg/v0/weights/best.pt \
    source=annotation/datasets/coworker_seg/images/val \
    imgsz=1280 save=True conf=0.25

# Results land in runs/segment/predict/ — flip through them
```

To plug into the existing `pipeline/04_measure_bolls.py`, load the model and write out per-frame per-boll binary masks matching the SAM 2 naming scheme:

```python
from ultralytics import YOLO
import cv2, numpy as np
from pathlib import Path

model = YOLO("runs/coworker_seg/v0/weights/best.pt")
out_root = Path("work/handheld_1/masks_yolo")  # parallel to masks/

for img_path in sorted(Path("work/handheld_1/frames").glob("*.jpg")):
    res = model(str(img_path), imgsz=1280, conf=0.25)[0]
    if res.masks is None:
        continue
    frame_idx = img_path.stem
    frame_dir = out_root / frame_idx
    frame_dir.mkdir(parents=True, exist_ok=True)
    # YOLO gives no fruit_id — use a running counter or match by IoU to the SAM 2 masks
    for i, mask in enumerate(res.masks.data.cpu().numpy()):
        cv2.imwrite(str(frame_dir / f"{i}.png"), (mask * 255).astype(np.uint8))
```

Note YOLO segmentation doesn't carry tag IDs — you'll need an IoU-based matching pass against the SAM 2 masks to recover `fruit_id`, or re-annotate with actual keypoints and train a pose head that picks up the physical tag via a nearby OCR/symbol step.

---

## Roboflow + keypoints — what happened

Your question: *"I uploaded this dataset to Roboflow and saw the masks but no keypoints. Is it because I made it an instance segmentation project?"*

Two separate issues, both apply here:

**(1) Roboflow project type.** Yes — Instance Segmentation projects in Roboflow render polygons only. To display keypoints you need a **Keypoint Detection** project (different schema, different SKU). Re-uploading the same file to an instance-seg project will never show keypoints even if they exist in the JSON.

**(2) Your COCO file does not contain keypoint coordinates at all.** I checked the export directly. Every annotation looks like this:

```json
{
  "category_id": 1,               // "cotton_boll"
  "segmentation": [[...polygon...]],
  "bbox": [x, y, w, h],
  "attributes": {
    "tip_visibility": "visible",  // just a string
    "base_visibility": "guessed",
    "occluded": false
  }
}
```

There are no `"keypoints": [x, y, v, x, y, v]` arrays and zero annotations with `category_id=2` (tip) or `category_id=3` (base). The coworker set up tip/base as dropdown **attributes**, not as point shapes on the image. The CVAT UI probably shows nothing to click for placing tip/base locations.

**Fix for next annotation round (share with the coworker):** in the CVAT project, add two separate labels of type **points**: `tip` and `base`, each with a `visible` attribute (select: 2/1/0). When annotators label a boll, they also drop two points — one at the tip, one at the base — and set the visibility flag on each. Then the COCO export will contain real `keypoints` arrays on the boll annotation (or separate point-category annotations depending on export format), and a Keypoint Detection Roboflow project (or a YOLO pose training run) will have something to work with.

The existing `Cotton_Boll_Annotation_Guidelines.docx` already specifies this schema in detail — Section 5 (keypoints) and Section 9.3 (point labels). Point the coworker at those sections before round 2.
