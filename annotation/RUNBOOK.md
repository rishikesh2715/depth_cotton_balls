# Tonight's Runbook — Annotation + YOLOv8 Pilot

End-to-end: sample frames → pre-bootstrap from SAM 2 → annotate in CVAT → convert → train YOLOv8-seg + YOLOv8-pose → evaluate.

Everything in this folder is designed so you can re-run any single stage without redoing the ones before it.

---

## 0. Prerequisites (15 min, one-time)

```bash
# Python deps
pip install ultralytics opencv-python numpy

# Docker (for CVAT)
# Make sure Docker + docker-compose are installed on your workstation.

# Spin up CVAT locally
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
# CVAT is now at http://localhost:8080 — create an admin user.
```

If you already have CVAT (self-hosted or cvat.ai cloud), skip Docker setup.

---

## 1. Sample frames (~1 min)

We have ~730 extracted frames across `work/handheld_1/frames/` and `work/handheld_2/frames/`. Annotating all of them is a waste — pick the ~150 most informative ones.

```bash
cd /path/to/depth_cotton_balls

python annotation/scripts/sample_frames.py \
    --work work/handheld_1 work/handheld_2 \
    --out annotation/sampled_frames \
    --target 150 \
    --min-stride 5
```

Output: `annotation/sampled_frames/` with ~150 JPEGs and a `sample_manifest.csv` that tracks where each frame came from.

**Why 150?** Small enough to annotate tonight, large enough to train a nano-sized YOLO model that reads cotton-boll geometry rather than memorizes individual frames. If you have less time, 50–80 is still enough to produce a working proof-of-concept.

---

## 2. Pre-bootstrap from existing SAM 2 masks (~1 min)

Rather than drawing every mask from scratch, convert the SAM 2 masks you already have into a CVAT-importable COCO JSON.

```bash
python annotation/scripts/bootstrap_from_sam2.py \
    --manifest annotation/sampled_frames/sample_manifest.csv \
    --sampled-dir annotation/sampled_frames \
    --out annotation/sampled_frames/sam2_bootstrap_coco.json
```

This gives annotators a starting point with `fruit_id` already filled in (from the SAM 2 object_id = physical tag number).

---

## 3. Set up CVAT project (~5 min)

1. Log into CVAT. **Projects → New**.
2. Name: `cotton_bolls_pilot_v0`.
3. **Constructor → Raw → Paste** the contents of `annotation/cvat/cotton_boll_cvat_labels.json`.
4. **Create task** inside the project:
   - Name: `handheld_pilot`
   - Select Files → upload the `annotation/sampled_frames/*.jpg` files (drag the whole folder).
   - Image quality: 95.
   - Start upload.
5. Once the task is created, open it → **Actions → Upload annotations → COCO 1.0** → select `annotation/sampled_frames/sam2_bootstrap_coco.json`.
6. You now have ~150 images with pre-drawn boll polygons and fruit_ids. Annotators refine edges + add keypoints + fill attributes.

---

## 4. Annotate (~2–3 hours for 150 frames, two annotators)

Follow `Cotton_Boll_Annotation_Guidelines.docx` — that's the source of truth. Key reminders:

- Clean up mask edges around leaves and tags.
- Add `kp_base` and `kp_tip` per boll, with the correct `visible` flag (2 = visible, 1 = occluded, 0 = not labeled).
- Fill `visibility_fraction`, `occlusion_type`, `usable_for_size`, `boll_stage`.
- Optional but recommended on at least 50% of frames: `kp_midpoint`, `kp_width_left`, `kp_width_right`.

Use CVAT's **AI Tools → Segment Anything** for any bolls the SAM 2 bootstrap missed.

---

## 5. Export from CVAT (~30 sec)

Project → Actions → **Export dataset → COCO 1.0** → download the zip.

Unzip into `annotation/cvat_export/`. You should have:

```
annotation/cvat_export/
    annotations/instances_default.json
    images/<file>.jpg
```

---

## 6. Convert to YOLOv8 format (~30 sec)

```bash
python annotation/scripts/cvat_to_yolo.py \
    --coco annotation/cvat_export/annotations/instances_default.json \
    --images annotation/cvat_export/images \
    --out annotation/datasets \
    --split-by filename \
    --train-pattern "handheld_1" \
    --val-pattern "handheld_2"
```

The split uses `handheld_1` for train and `handheld_2` for val. If you only annotated one recording, change to `--split-by ratio --val-ratio 0.2`, but prefer the recording-level split — it tells you whether the model generalizes to a new handheld pass rather than just to held-out frames from the same video.

You end up with two sibling datasets:

```
annotation/datasets/seg/
    images/{train,val}/...
    labels/{train,val}/...    # YOLO seg format
    data.yaml
annotation/datasets/pose/
    images/{train,val}/...
    labels/{train,val}/...    # YOLO pose format (5 kps × 3)
    data.yaml
```

---

## 7. Train YOLOv8-seg (~30–60 min on a single GPU)

```bash
python annotation/scripts/train_yolo_seg.py \
    --data annotation/datasets/seg/data.yaml \
    --model yolov8n-seg.pt \
    --epochs 100 --imgsz 1024 --batch 8 \
    --project runs/boll_seg --name v0
```

Watch `runs/boll_seg/v0/results.png` live. Expect:
- **box mAP50** > 0.85 within ~30 epochs for clearly-visible bolls.
- **mask mAP50** > 0.75 by epoch 60.
- If loss plateaus early, bump to `yolov8s-seg.pt`.

---

## 8. Train YOLOv8-pose (~30–60 min)

```bash
python annotation/scripts/train_yolo_pose.py \
    --data annotation/datasets/pose/data.yaml \
    --model yolov8n-pose.pt \
    --epochs 150 --imgsz 1024 --batch 8 \
    --project runs/boll_pose --name v0
```

The pose model trains on detection box + 5 keypoints per box. Visibility flags {0,1,2} are respected natively by YOLO's OKS loss — keypoints flagged 0 contribute zero loss, flagged 1 and 2 both contribute but the model learns the distinction from the data.

---

## 9. Sanity-check visualizations (~5 min)

```bash
# seg predictions
yolo predict model=runs/boll_seg/v0/weights/best.pt \
    source=annotation/datasets/seg/images/val \
    imgsz=1024 save=True

# pose predictions
yolo predict model=runs/boll_pose/v0/weights/best.pt \
    source=annotation/datasets/pose/images/val \
    imgsz=1024 save=True
```

Output PNGs in `runs/detect/predict/`. Flip through a few — you are looking for:

- Masks that hug the boll edges cleanly (not spilling onto leaves).
- Keypoints on the base/tip endpoints, not on the middle of the boll.
- Occluded bolls still detected, with keypoints at sensible (inferred) positions.

---

## 10. Where this plugs into the existing pipeline

Your existing `pipeline/04_measure_bolls.py` reads per-boll binary masks. The trained seg model writes compatible masks:

```python
from ultralytics import YOLO
m = YOLO("runs/boll_seg/v0/weights/best.pt")
res = m(img, imgsz=1024)[0]
# res.masks.data is a tensor (N, H, W) of binary masks
# res.boxes.conf gives per-detection confidence
```

For the pose model, lifting (base, tip) keypoints into 3D via the RealSense depth map gives you the pose-aware measurement-validity check the professor's doc asks for:

```python
# depth is the aligned RealSense depth frame, HxW float32 in meters
base_3d = depth[int(base_y), int(base_x)] * pixel_ray(base_x, base_y, K)
tip_3d  = depth[int(tip_y),  int(tip_x) ] * pixel_ray(tip_x,  tip_y,  K)
length_m = np.linalg.norm(tip_3d - base_3d)

# Measurement validity rule:
camera_axis = np.array([0, 0, 1])
boll_axis   = (tip_3d - base_3d) / (np.linalg.norm(tip_3d - base_3d) + 1e-9)
cos_angle   = abs(np.dot(boll_axis, camera_axis))
# If cos_angle ≈ 1  → boll pointing at camera → width reliable, length unreliable
# If cos_angle ≈ 0  → boll perpendicular to camera → length reliable, width reliable
```

Use this to populate the per-frame confidence value (CV) your professor mentioned, along with the predicted `visibility_fraction` (a small regression head you can train after tonight's pilot).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ultralytics` can't find dataset | Wrong `path:` in `data.yaml` | Edit the absolute path in `datasets/seg/data.yaml` and `datasets/pose/data.yaml` |
| mAP stays at zero | Val split contains zero labels | Check `labels/val/` — not empty? Each `.txt` well-formed? |
| Pose keypoints flip during training | `flip_idx` in `data.yaml` wrong | Should be `[0, 1, 2, 4, 3]` (width_left ↔ width_right) |
| GPU OOM | `imgsz=1024` too big | Drop to `imgsz=768 --batch 4` first; if still OOM, `imgsz=640` |
| Model misses small bolls | `imgsz` too small | Go back up to 1024 and cut batch to fit |

---

## File map

| Path | Purpose |
|---|---|
| `Cotton_Boll_Annotation_Guidelines.docx` | What annotators read |
| `cvat/cotton_boll_cvat_labels.json` | Paste into CVAT "Raw" to set up labels |
| `scripts/sample_frames.py` | Pick ~150 informative frames from work/ |
| `scripts/bootstrap_from_sam2.py` | Turn SAM 2 masks into CVAT pre-annotations |
| `scripts/cvat_to_yolo.py` | CVAT COCO export → YOLOv8 seg + pose format |
| `scripts/train_yolo_seg.py` | Train seg model |
| `scripts/train_yolo_pose.py` | Train pose model |
| `sampled_frames/` | Output of step 1 (gitignored) |
| `cvat_export/` | Place your CVAT export here (gitignored) |
| `datasets/` | YOLO-format dataset (gitignored) |
