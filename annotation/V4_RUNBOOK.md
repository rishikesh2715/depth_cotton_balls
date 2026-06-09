# V4 dataset — train seg + pose + aux on combined_all_sequential

This runbook trains the second iteration of all three models on the
combined multi-camera dataset Abhijeet shipped (`combined_all_sequential`,
with the corrected JSONs from `combined_all_sequential 3`).

**Why this is a meaningful upgrade over v3:**

- ~3× the annotations: **734** (470 train / 264 val) vs v3's 274 (242/32).
- Honest val signal: **sequential split** holds out a contiguous block at
  the end of the timeline, so val frames are genuinely unseen. Random
  splits on continuous video leak temporal context — neighboring frames
  are nearly identical and inflate mAP.
- Two camera angles annotated: **back + track**. Front frames exist on
  disk but aren't annotated yet (we'll either ask Abhijeet to label them
  or skip them).
- Richer keypoint signal: instead of single `tip_point` / `base_point`
  shapes, annotators drew `tip_polygon` (4-corner calyx) and
  `base_rectangle` (4-corner base box) when visible, falling back to a
  single point when only "guessed". Visibility is now a 3-level string
  (`visible` / `guessed` / `unknown`) that maps cleanly to YOLO's
  {2, 1, 0}.
- Bigger rare-class signal: stems in val went from **0** (v3) to **60**;
  total stem instances went from 25 to 67; "other" went from 5 to 32.
  The aux head finally has enough rare-class examples to be evaluated
  honestly.

**Dataset state (v4, as received):**

| | back | track | total |
|---|---:|---:|---:|
| train imgs | 14 | 50 | **64** |
| val imgs   |  3 | 60 | **63** |
| train anns |    |    | **470** |
| val anns   |    |    | **264** |

Class distribution (`occluded_by`, collapsed to 4-way):

| | none | leaf | stem | other |
|---|---:|---:|---:|---:|
| train | 355 | 76 | 7   | 32 |
| val   | 138 | 66 | **60** | 0 |

⚠️ Two distribution notes worth flagging in the meeting:

1. **Sequential split pushed most stem occlusions into val** (60 vs 7 in
   train). That's the honest signal of the discipline — the held-out
   tail of the timeline happens to contain more stem-occluded bolls.
   Read `recall_stem` in the aux-head printout as a hard generalization
   test, not a weakness of the dataset.
2. **Val has zero "other" instances.** `recall_other` will be NaN in val
   and gets skipped from the macro mean. Read the per-class recalls in
   the final stdout block, not just the macro.

---

## 0. (Re)build the v4 YOLO dataset

Already done (path `annotation/datasets/v4/`). If you ever need to
re-generate from scratch (e.g. Abhijeet re-issues the COCO JSONs):

```powershell
cd C:\Users\rrishike\depth_cotton_balls

python annotation\scripts\merged_v4_to_yolo.py `
    --images-root "combined_all_sequential" `
    --coco-dir   "combined_all_sequential 3/combined_all_sequential/output_coco_dir" `
    --out        annotation\datasets\v4 `
    --clean
```

(`--clean` wipes existing `<out>/seg` and `<out>/pose` so you start
fresh. Drop it for incremental re-runs — the converter is idempotent
and re-uses image copies.)

Expected output:

- `annotation\datasets\v4\seg\{images,labels}\{train,val}\` + `data.yaml`
- `annotation\datasets\v4\pose\{images,labels}\{train,val}\` + `data.yaml`
- 64 train + 63 val images in each
- Both `data.yaml` files use `path: .` so the dataset folder is portable.

---

## 1. Launch seg + pose training in parallel

Open **two PowerShell windows** in the workspace folder.

### Window A — segmentation on GPU 0

```powershell
python annotation\scripts\train_v4_seg.py `
    --data annotation\datasets\v4\seg\data.yaml `
    --device 0 `
    --project runs\v4_seg --name v0
```

### Window B — pose on GPU 1

```powershell
python annotation\scripts\train_v4_pose.py `
    --data annotation\datasets\v4\pose\data.yaml `
    --device 1 `
    --project runs\v4_pose --name v0
```

Defaults vs v3:

| Knob | v3 | v4 | Why |
|---|---|---|---|
| seg epochs | 200 | 250 | ~3× the data |
| pose epochs | 250 | 300 | same reason |
| patience | 50 | 80 | bigger val => more reliable signal, less twitchy stop |
| everything else | same | same | model size, imgsz=1280, batch=8, augmentation balance |

Wall-clock estimates on a single RTX 5090 at `imgsz=1280, batch=8, nano`:

- ~12–22 s/epoch (data-loading bound — 64 train images cached in RAM)
- Seg 250 epochs: ~50–95 min (early-stop usually fires earlier)
- Pose 300 epochs: ~60–110 min

Live curves:

- `runs\v4_seg\v0\results.png`
- `runs\v4_pose\v0\results.png`

---

## 2. What "decent" looks like for v4

The honest comparison is **v4 val** vs **v3 val**, but remember the v3
val was 6 images / 32 anns and the v4 val is 63 images / 264 anns — the
v4 numbers are harder-earned even if they're similar.

| Metric                | Floor | Decent | Good   |
|-----------------------|-------|--------|--------|
| seg box mAP50         | 0.65  | 0.82   | 0.90+  |
| seg mask mAP50        | 0.55  | 0.75   | 0.85+  |
| seg mask mAP50–95     | 0.40  | 0.55   | 0.65+  |
| pose box mAP50        | 0.70  | 0.85   | 0.92+  |
| pose kpt mAP50 (OKS)  | 0.50  | 0.68   | 0.80+  |
| pose kpt mAP50–95     | 0.35  | 0.55   | 0.70+  |

If val mAP stays flat while train mAP climbs, that's cross-recording or
cross-camera domain shift between back and track — exactly the signal
you want to show the professor. The fix is more annotated frames, not a
bigger model. Don't jump to `yolo11s-*` until the dataset is at least
5× larger.

---

## 3. Sanity-check predictions

```powershell
yolo predict model=runs\v4_seg\v0\weights\best.pt `
    source=annotation\datasets\v4\seg\images\val `
    imgsz=1280 save=True conf=0.25

yolo predict model=runs\v4_pose\v0\weights\best.pt `
    source=annotation\datasets\v4\pose\images\val `
    imgsz=1280 save=True conf=0.25
```

Output PNGs land under `runs\segment\predict*\` and `runs\pose\predict*\`.
Flip through — you want masks that hug boll edges and tip/base
keypoints on the boll endpoints (not the center / not the calyx face).

---

## 4. Train the aux head on v4

```powershell
python annotation\scripts\train_v4_aux.py `
    --images-root "combined_all_sequential" `
    --coco-dir   "combined_all_sequential 3/combined_all_sequential/output_coco_dir" `
    --device 0 `
    --epochs 80 --batch 32 `
    --project runs\v4_aux --name v0
```

Wall-clock on a 5090: ~5–10 s/epoch, full run ~8–15 min total.

What "decent" looks like for v4:

| Metric                  | Floor | Decent | Good  |
|-------------------------|-------|--------|-------|
| cls top-1 acc           | 0.70  | 0.82   | 0.90+ |
| cls macro-recall (4 cls)| 0.45  | 0.60   | 0.75+ |
| recall_none             | 0.85  | 0.92   | 0.97+ |
| recall_leaf             | 0.55  | 0.72   | 0.85+ |
| recall_stem             | 0.30  | 0.55   | 0.75+ |
| visfrac MAE             | 0.04  | 0.025  | 0.015 |

Read `recall_stem` carefully — with only 7 stem instances in train and
60 in val, this is the hardest generalization test. A low stem recall
here is a signal that we need more stem-occluded annotations from
Abhijeet, not a model failure.

`recall_other` will print as `nan` (val has 0 "other" instances). Don't
worry about it. The macro mean skips NaN-class recalls.

---

## 5. End-to-end inference with the new weights

The inference scripts (`infer_v3.py`, `infer_aux.py`) and the HTML
viewer (`Inference_Viewer.html`) don't need any changes — they already
work with arbitrary weight paths. Just point at the v4 best.pt files:

```powershell
python annotation\scripts\infer_v3.py `
    --seg  runs\v4_seg\v0\weights\best.pt `
    --pose runs\v4_pose\v0\weights\best.pt `
    --aux  runs\v4_aux\v0\best.pt `
    --source work\my_recording_frames `
    --out   runs\v4_inference\my_recording `
    --device cuda:0 `
    --save-json --save-csv --profile
```

All the v3 inference features still apply: tracking on by default,
FP16, per-stage `--profile` output, the same JSON schema, the same
viewer.

Open the viewer (`annotation\Inference_Viewer.html`), load
`work\my_recording_frames` and `runs\v4_inference\my_recording`, and
you'll see the v4 model's behavior with all the same toggles.

---

## 6. After the first run — what to bring to the meeting

1. `runs\v4_seg\v0\results.png` and `runs\v4_pose\v0\results.png` (loss + mAP curves)
2. `runs\v4_aux\v0\train_log.txt` final block (cls acc, macro recall,
   per-class recalls, vis MAE)
3. Side-by-side: v3 val numbers (the existing "Inference_Visualization_Legend.md"
   has them) vs v4 val numbers — frame the comparison around "v4 is the
   first val we can actually trust" rather than apples-to-apples mAP.
4. 5–10 annotated frames from `infer_v3.py` running with v4 weights —
   ideally a mix from both back and track cameras.
5. A note about `recall_stem` and the sequential split — show the
   professor we understand why this metric is hard and what would
   improve it.

---

## 7. What's still on the v5 ask list

- **Front camera annotations.** Front frames exist on disk
  (`combined_all_sequential\front\`, 29 images) but aren't annotated.
  Even a small front pass would let us measure cross-camera
  generalization more cleanly.
- **More stem-occluded instances in train.** Currently 7 train / 60 val
  is the wrong way around for the rarer class.
- **Front-frame `other_boll` instances in val.** Val has zero "other"
  right now — even a handful would let us put a real number on that
  recall.
- **RGB-D fusion** — next milestone. Now that the seg + pose + aux
  pipeline is producing structured per-boll records with track_ids,
  lifting the principal axis into 3D and aggregating across frames per
  track_id is the next thing to ship.
