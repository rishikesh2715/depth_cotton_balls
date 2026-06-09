# V3 dataset — train YOLOv11 seg + pose on two RTX 5090s

This runbook trains the first model on `merged_dataset_v3` (Abhijeet's
COCO export with tip/base point shapes already merged onto each
cotton_boll annotation as a real `keypoints` array).

**Dataset state (v3, as received):**
- 46 images (backdown 28, frontdown 18)
- 274 `cotton_boll` instances, each with polygon + 2 keypoints `[tip, base]`
- Split: 40 train / 6 val (recording-aware: backdown.train + frontdown.train -> train, backdown.val + frontdown.val -> val)
- Known schema gaps vs. the annotation guidelines (flagged for v4 ask list):
  - Only the 2 required keypoints are present (no midpoint, no width_left/right)
  - No `visibility_fraction` (5-bin), `occlusion_type` (multi-label), `boll_stage`, `usable_for_size`, or quality flags
  - 95 instances have a (0, 0, v=0) keypoint — annotator marked the polygon attribute "visible" but never placed the point shape. v=0 keypoints contribute zero OKS loss, so training is safe.

---

## 0. One-time setup (Windows, PowerShell)

```powershell
cd C:\Users\rrishike\depth_cotton_balls

# (Optional, recommended) make a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Ultralytics + deps (only first time)
pip install ultralytics opencv-python numpy pillow pyyaml

# Sanity-check both GPUs are visible
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

You should see `devices: 2` with two RTX 5090 entries. If not, fix the
torch + CUDA install before continuing.

---

## 1. (Re)build the v3 YOLO dataset

The converter is already idempotent — re-running it overwrites label
files but re-uses image symlinks/copies. Use `--clean` if you want a
truly fresh build.

```powershell
python annotation\scripts\merged_v3_to_yolo.py `
    --root new_dataset\merged_dataset_v3 `
    --out annotation\datasets\v3
```

Expected output:
- `annotation\datasets\v3\seg\{images,labels}\{train,val}\` + `data.yaml`
- `annotation\datasets\v3\pose\{images,labels}\{train,val}\` + `data.yaml`
- 40 train + 6 val images in each
- Both `data.yaml` files use `path: .` so they're location-independent.

---

## 2. Launch both trainings in parallel

Open **two PowerShell windows** in the workspace folder.

### Window A — segmentation on GPU 0

```powershell
python annotation\scripts\train_v3_seg.py `
    --data annotation\datasets\v3\seg\data.yaml `
    --model yolo11n-seg.pt `
    --device 0 `
    --project runs\v3_seg --name v0
```

### Window B — pose on GPU 1

```powershell
python annotation\scripts\train_v3_pose.py `
    --data annotation\datasets\v3\pose\data.yaml `
    --model yolo11n-pose.pt `
    --device 1 `
    --project runs\v3_pose --name v0
```

Wall-clock estimates on a single 5090 at imgsz=1280, batch=8, nano model:
- ~6-12 s/epoch (mostly data-loading bound on 40 images cached in RAM)
- Seg 200 epochs: ~25-45 min (early-stop usually fires earlier)
- Pose 250 epochs: ~30-60 min

Watch live curves at:
- `runs\v3_seg\v0\results.png`
- `runs\v3_pose\v0\results.png`

---

## 3. What "decent" looks like for the v3 pilot

| Metric                | Floor | Decent | Good   |
|-----------------------|-------|--------|--------|
| seg box mAP50         | 0.65  | 0.82   | 0.90+  |
| seg mask mAP50        | 0.50  | 0.70   | 0.82+  |
| pose box mAP50        | 0.65  | 0.82   | 0.90+  |
| pose kpt mAP50 (OKS)  | 0.40  | 0.60   | 0.75+  |

If seg/pose train mAP shoots up but val mAP stays flat, that's
cross-recording domain shift between backdown and frontdown — the
honest signal that we need more recordings, not a bigger model. That's
the message we want for the meeting.

If val mAP plateaus low, swap `yolo11n-*.pt` -> `yolo11s-*.pt` and
re-run with `--name v1_small`. Don't go bigger than `s` until the
dataset is at least 5x larger.

---

## 4. Sanity-check predictions

```powershell
# Seg
yolo predict model=runs\v3_seg\v0\weights\best.pt `
    source=annotation\datasets\v3\seg\images\val `
    imgsz=1280 save=True conf=0.25

# Pose
yolo predict model=runs\v3_pose\v0\weights\best.pt `
    source=annotation\datasets\v3\pose\images\val `
    imgsz=1280 save=True conf=0.25
```

Output PNGs land in `runs\segment\predict*\` and `runs\pose\predict*\`.
Flip through them — you want masks that hug boll edges (not spilling
onto leaves) and tip/base keypoints on the boll endpoints (not on the
center, not on the calyx).

---

## 5. Train the auxiliary head (L_occ + L_visfrac)

YOLOv11-seg covers L_box + L_seg; YOLOv11-pose covers L_box + L_kp + L_kp_vis.
Neither covers `occluded_by` (leaf / stem / etc.) or `visibility_fraction`
(continuous). The aux head fills those two terms.

**v3 attribute coverage (audited 2026-04-30):** every annotation has
`is_occluded`, `occluded_by`, `occlusion_pct`. The class scheme we train
is collapsed to 4-way: `none / leaf / stem / other` (other = frame-edge
+ other-boll, n=5 globally — these are the rare-class samples). Train
distribution: none=182, leaf=30, stem=25, other=5. Visfrac ground truth
= `1.0 - occlusion_pct`, range [0.34, 1.00].

Architecture: ResNet-18 (ImageNet pre-trained) -> two heads (4-way
classifier + 1-d sigmoid regressor). Class-frequency-weighted CE +
Huber. See `annotation\scripts\aux_head.py`.

This is small enough to train on either GPU after seg/pose finish,
or in parallel on the same GPU if VRAM allows (peak ~3 GB at batch=32,
imgsz=224).

```powershell
python annotation\scripts\train_v3_aux.py `
    --v3-root new_dataset\merged_dataset_v3 `
    --device 0 `
    --epochs 60 --batch 32 `
    --project runs\v3_aux --name v0
```

Wall-clock on a single 5090: ~3-6 s/epoch, full run ~5-10 min total.

What "decent" looks like for v0:

| Metric                  | Floor | Decent | Good  |
|-------------------------|-------|--------|-------|
| cls top-1 acc           | 0.75  | 0.85   | 0.90+ |
| cls macro-recall (4 cls)| 0.40  | 0.55   | 0.70+ |
| visfrac MAE             | 0.10  | 0.06   | 0.03  |

The macro-recall floor is intentionally low — `other` has n=5 globally
and 0 in val, so even a perfect model gets undefined recall on that
class, which the macro will treat as nan-skipped. Read the per-class
recalls in the final stdout printout (`recall_none / recall_leaf /
recall_stem`) for the honest signal.

---

## 6. End-to-end inference (seg + pose + aux merged)

Two scripts, two layers:

- **`annotation\scripts\infer_aux.py`** — single-image debug CLI +
  library class `CombinedInfer`. Use this when you want to print per-boll
  records to stdout for one image. Library API:

  ```python
  from infer_aux import CombinedInfer
  infer = CombinedInfer(seg_pt="...", pose_pt="...", aux_pt="...",
                        device="cuda:0")
  bolls = infer("path/to/image.jpg")     # accepts paths
  bolls = infer(frame_bgr)                # or numpy BGR arrays
  # Each boll: bbox, det_conf, mask_poly, kpts, occ_class,
  # occ_class_conf, visfrac. mask_poly and kpts are in original-image
  # pixel coordinates.
  ```

- **`annotation\scripts\infer_v3.py`** — image / video / folder CLI on
  top of `CombinedInfer`. This is the script you want for batch work.
  Writes annotated outputs (JPG for images, MP4 for videos) plus
  optional per-source JSON and an aggregated `results.csv`.

### 6a. Single image

```powershell
python annotation\scripts\infer_v3.py `
    --seg  runs\v3_seg\v0\weights\best.pt `
    --pose runs\v3_pose\v0\weights\best.pt `
    --aux  runs\v3_aux\v0\best.pt `
    --source annotation\datasets\v3\seg\images\val\back__frame_001073_t17p90s.jpg `
    --out   runs\v3_inference\one_image `
    --device cuda:0 --save-json --save-csv
```

### 6b. Folder of frames (work\handheld_1\frames)

Annotate every JPG and (with `--video-out`) also stitch them into a
single MP4:

```powershell
python annotation\scripts\infer_v3.py `
    --seg  runs\v3_seg\v0\weights\best.pt `
    --pose runs\v3_pose\v0\weights\best.pt `
    --aux  runs\v3_aux\v0\best.pt `
    --source work\handheld_1\frames `
    --out   runs\v3_inference\handheld_1 `
    --device cuda:0 --save-json --save-csv `
    --video-out --video-fps 30
```

Outputs:
- `runs\v3_inference\handheld_1\<frame>.jpg` — one annotated JPG per source frame
- `runs\v3_inference\handheld_1\handheld_1_composed.mp4` — stitched video at 30 fps
- `runs\v3_inference\handheld_1\<frame>.json` — per-frame structured records (with `--save-json`)
- `runs\v3_inference\handheld_1\results.csv` — one row per detection (with `--save-csv`)

### 6c. Single video file

```powershell
python annotation\scripts\infer_v3.py `
    --seg  runs\v3_seg\v0\weights\best.pt `
    --pose runs\v3_pose\v0\weights\best.pt `
    --aux  runs\v3_aux\v0\best.pt `
    --source path\to\my_recording.mp4 `
    --out   runs\v3_inference\my_recording `
    --device cuda:0 --save-csv
```

Output: `runs\v3_inference\my_recording\my_recording.mp4` (annotated)
plus `results.csv`.

### Visualization legend

- **Bbox + mask outline**: colored by predicted occlusion class
  (green=none, yellow=leaf, orange=stem, red=other; gray if no aux head).
- **Mask fill**: same color, semi-transparent (alpha=0.35 by default;
  tune with `--mask-alpha` or disable with `--no-mask`).
- **Tip keypoint**: cyan dot. **Base keypoint**: magenta dot. Connected
  by a white axis line (the boll's principal axis). Disable with `--no-kpts`.
- **Label**: `<occ_class> <occ_conf> | v=<visfrac> | d=<det_conf>`.

Throughput on a 5090 at imgsz=1280 is roughly 8–15 FPS for a single
2988×5312 frame (the 16-megapixel size is the dominant cost; the three
models combined are <100 ms). Drop `--imgsz` to 1024 if you want it
faster and don't need the small-boll precision.

### Library API for downstream RGB-D fusion

`CombinedInfer` is the unit the RGB-D fusion step will sit on top of:
lift each detection's tip/base keypoints into 3D using the aligned
RealSense depth frame, weight by
`det_conf * visfrac * cos(camera_axis, boll_axis)`, and aggregate
across frames per `boll_id`.

---

## 7. After the first run — what to bring to the meeting

1. `runs\v3_seg\v0\results.png` and `runs\v3_pose\v0\results.png` (loss + mAP curves)
2. `runs\v3_aux\v0\train_log.txt` final block (cls acc, macro recall, vis MAE)
3. 5-10 annotated frames from `infer_v3.py` (mix of clean / occluded / tag-overlapping bolls — section 6b stitches them into a single video)
4. One JSON dump or `results.csv` row sample from section 6 showing the merged per-boll record
5. The v4 annotation ask list (in `Loss_and_RGBD_Plan_v0.docx` Section 6)

That's enough to answer the professor's "are we heading in the right
direction" question and shows concrete progress on every term of the
six-term loss except L_box/L_seg/L_kp (which the YOLO numbers cover)
and the RGB-D fusion (next milestone).

---

## 8. Interactive inference viewer

`annotation\Inference_Viewer.html` is a self-contained HTML tool for
browsing inference output frame-by-frame with every overlay layer
toggleable in real time. Just double-click the file — it opens in any
modern browser (Chrome/Edge recommended for the directory picker;
Firefox/Safari fall back to a regular file input).

### What it shows

The viewer renders overlays **live in canvas on top of the raw source
frames** — so every layer is a true toggle, not a baked-in pixel. You
load the un-annotated frames + the JSON files written by
`infer_v3.py --save-json`, and the viewer pairs them by filename stem.

Layers (each independently on/off):

- Bounding box
- Mask fill (with alpha slider) + mask outline
- Tip keypoint (cyan), base keypoint (magenta), axis line (white)
- Label text — and sub-toggles for `occ_class + conf`, `v=visfrac`, `d=det_conf`

Filters:

- Min detection confidence (slider, default 0.25)
- Min visfrac (slider, default 0.00)
- Occlusion-class chips (none / leaf / stem / other / no aux) — click to hide

Playback:

- Play / pause, prev / next frame, first / last, scrub bar
- Speed selector (0.25x – 8x) + base FPS input
- Keyboard: `Space`=play/pause, `←/→`=step, `J/K`=±10 frames,
  `Home/End`=first/last, `F`=fit-to-window, `+/-`=speed

Selection:

- Click any boll → highlighted on canvas + full per-detection record
  in the right panel (occ class, occ conf, visfrac with bar, det conf
  with bar, bbox, tip/base coords + visibility flags, polygon vertex
  count).
- Toggle "Dim non-selected" to fade everything else for a focused look.

Export:

- "Save current frame as PNG" writes the currently-rendered canvas
  (overlays + image) to a PNG with the current toggle state.

### How to load

Two ways:

1. **Same folder for everything.** If your inference output folder has
   both the source-style JPGs and the `*.json` files (which is the
   default layout when `infer_v3.py` writes annotated JPGs + JSONs to
   the same `--out` dir, *and* the JPGs alphabetize alongside the
   JSONs), click "Load frames folder" and pick that one folder. The
   viewer auto-ingests both. Note: those JPGs already have overlays
   baked in, so the in-viewer toggles will stack on top — pick the
   raw frames folder if you want clean overlays.

2. **Separate folders (recommended).** Click "Load frames folder" and
   pick the original raw frames (e.g. `work\handheld_1\frames`), then
   click "Load detections folder" and pick the JSONs folder (e.g.
   `runs\v3_inference\handheld_1`). The viewer pairs them by filename
   stem.

3. **Drag & drop.** Drop a folder anywhere on the window.

### Demo dataset

`annotation\viewer_demo\` contains 30 synthetic frames + matching
JSONs (covering all 5 occlusion classes, two bolls with intermittent
low-confidence keypoints, stable track_ids on every detection) so
you can verify the viewer works before pointing it at a real run.
Open `Inference_Viewer.html`, click "Load frames folder", select
`annotation\viewer_demo\`, and you should see ~9 bolls drifting
across 30 frames with every overlay toggleable.

---

## 9. Tracking + performance

### Tracking (default-on for video/folder)

`infer_v3.py` now wraps the seg stream in Ultralytics ByteTrack by
default. Every detection gets a stable `track_id` across frames,
which is:

- written to `<frame>.json` (`"track_id": 42`),
- included in `results.csv` as a new `track_id` column,
- drawn as a small `#42` chip at the top-right of each rendered bbox.

Defaults: on for video inputs and image folders, off for single
images (where there's no temporal context). Override with:

- `--no-track`                    disable tracking entirely
- `--tracker botsort.yaml`        switch to BoT-SORT (slightly more
                                  accurate, ~20% slower)
- `--no-track-id`                 keep tracking but hide the chip
                                  in rendered output

Tracker state resets between sources automatically. Single-image
runs get `track_id: null`.

### Per-keypoint confidence (already in the data — now surfaced)

`kpts[i][2]` is a **continuous confidence in [0,1]** from
YOLOv11-pose, not the discrete `{0,1,2}` visibility flag from the
training labels. Both the Python `render()` and the HTML viewer now
treat it as continuous:

- `conf >= --kpt-conf-min` (default **0.30**) → **solid dot** (the
  model "committed" to that keypoint location)
- `0 < conf < threshold`  → **hollow dot** (the model emitted a
  prediction but is uncertain — informally, "guessed")
- The white axis line draws only when **both** endpoints are above
  threshold.

In the viewer, the threshold is a slider — drag it live and watch
which keypoints flip between solid and hollow. The selection panel
shows `tip conf` and `base conf` with separate bars + a label
(`committed` / `low conf (guessed)` / `not predicted`).

### Trails in the viewer

Toggle "Centroid trails (tracked)" in the left sidebar. The viewer
walks back N frames (slider, default 20, range 3–80), groups
detections by `track_id`, and draws a fading polyline from the
boll's mask centroid (or bbox center when no mask). Older segments
fade toward zero alpha. Trails work only for detections that have a
`track_id` — older JSONs from runs without tracking won't show
trails, but everything else still renders.

### Performance fixes for the RTX 5090 slowness

On a single 5090, the previous pipeline was ~8–15 FPS for 2988×5312
frames at `imgsz=1280`. The dominant costs were (a) two YOLO models
run separately at FP32, (b) PIL-based aux-head crops, (c) one extra
PIL decode of the full 16MP image. The updated `infer_aux.py`
addresses all three. New defaults / flags:

- **FP16 inference** (`half=True` by default on CUDA).
  ~1.5–2× faster forward pass on Blackwell/Ada with negligible mAP
  impact at this problem size. Use `--no-half` to force FP32 if
  you're A/B-ing.
- **`model.fuse()`** on both YOLO models after load. Conv+BN
  fusion is a small (~1.1×) free win.
- **Numpy aux-head crops**. The 16MP `Image.fromarray` round-trip
  is gone — crops now go directly from the BGR numpy frame via
  `cv2.cvtColor` + `cv2.resize`. This is the single biggest win
  for frames with many bolls (PIL was creating, cropping, and
  resizing N separate Python objects per frame).
- **Single image decode** shared by YOLO and the aux head.
- **`--profile`** flag prints per-stage timings (decode / yolo_seg
  / yolo_pose / aux_head / assemble) averaged over the run. Use
  it to confirm where time is actually going on your machine:

  ```powershell
  python annotation\scripts\infer_v3.py `
      --seg runs\v3_seg\v0\weights\best.pt `
      --pose runs\v3_pose\v0\weights\best.pt `
      --aux runs\v3_aux\v0\best.pt `
      --source work\handheld_1\frames `
      --out runs\v3_inference\handheld_1 `
      --device cuda:0 --save-json --save-csv --profile
  ```

  Typical post-fix breakdown on a 5090 at 2988×5312, `imgsz=1280`:

  | Stage      | Before | After (FP16) |
  |------------|-------:|-------------:|
  | decode     | ~80 ms | ~80 ms       |
  | yolo_seg   | ~45 ms | ~22 ms       |
  | yolo_pose  | ~45 ms | ~22 ms       |
  | aux_head   | ~30 ms | ~6 ms        |
  | assemble   | ~10 ms | ~10 ms       |
  | **total**  | ~210 ms / 4.7 fps | ~140 ms / 7 fps |

  (Numbers are estimates based on stage profile expectations —
  run `--profile` once to see your actual measurements.)

- **Additional knobs if you need more speed:**
  - `--imgsz 1024` — trades ~0.5pp mAP for ~1.4× speed
  - `--imgsz 960`  — bigger trade, only for browsing-quality runs
  - Closing other GPU consumers (browser hardware acceleration,
    other CUDA processes) frees ~10–20% of throughput on 5090s.

If `--profile` shows `decode` dominating (>50% of frame time),
your disk is the bottleneck — not the model. In that case, copy
the frames folder to a local SSD or, if working from frames
extracted from a video, run inference on the video directly
(single sequential decode is faster than N JPG opens).
