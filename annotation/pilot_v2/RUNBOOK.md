# Pilot v2 Annotation — 3-Hour Runbook

Generated 2026-04-24. 50 sampled frames, 653 SAM 2 pre-polygons queued.

## Files you need (all inside this folder)

- `frames/` — 50 JPGs, upload these as the CVAT task
- `frames/sample_manifest.csv` — maps each sampled filename back to its source recording and original frame index; keep for post-processing
- `sam2_bootstrap_coco.json` — pre-filled polygons + pre-filled `boll_id` (=SAM 2 tag). Import this AFTER uploading frames.
- `../cvat/cvat_labels_v2.json` — paste this into CVAT project's Raw labels tab

## 0:00 – 0:15   Setup in CVAT

1. Create a new CVAT project.
2. Project settings → pencil → **Raw** tab → paste `cvat_labels_v2.json` → Save. You should now see 3 labels: `cotton_boll` (polygon), `tip` (points), `base` (points).
3. Add a task under this project. Name it `pilot_v2`. Upload all 50 JPGs from `frames/` (drop the `sample_manifest.csv` out of the upload; CVAT will complain about non-image files).
4. Once the task is created, open the job → top menu → **Actions → Upload annotations** → format **COCO 1.0** → pick `sam2_bootstrap_coco.json`. You should see polygons appear on every frame.

Sanity check: open frame 1, you should see multiple orange polygons each with `boll_id` already set to an integer.

## 0:15 – 2:45   Annotate (≈3 min / frame)

Per frame, work boll by boll. Average 13 bolls/frame × 50 frames × ~14 sec/boll = 150 min. That's your budget.

### Keyboard shortcuts worth memorizing (saves ~30 min over 3 hrs)

- `N` — new object (for tip/base points)
- `F` — next frame
- `D` — previous frame
- `Q` / `W` — cycle through labels
- `Esc` — deselect / finish current shape
- `Space` — play through frames (skip past ones that look fine)

### Per-boll workflow (fast path)

For each of the pre-filled `cotton_boll` polygons:
1. Click the polygon to select it. Check that the outline matches the boll. Edit nodes only if it's clearly wrong — SAM 2 is usually 80–90% correct; chasing perfection costs you the deadline.
2. Note the `boll_id` value in the right sidebar (pre-filled from SAM 2). You'll reuse it for the two points.
3. Press `N`, label = `tip`, click once on the tip of the boll, set `boll_id` = same integer.
4. Press `N`, label = `base`, click once on the base, set `boll_id` = same integer.

Default attributes (`whole_visible`, `none`, `visible`, `visible`) are what you want for ~70% of bolls. Only change them when:

- Boll is partly behind leaf/stem/flower/other boll → change `annotation_type` → `visible_parts`, then either redraw the polygon to just the visible piece OR add a second `whole_estimate` polygon covering the full guessed outline. Set `occluded_by` on the `whole_estimate` polygon.
- You can't see the tip or base clearly → switch `tip_visibility` / `base_visibility` to `guessed`, place the point where you think it is.
- You genuinely can't place a point at all → set visibility to `unknown` AND don't create the point annotation. This is the only case where the boll has a polygon but no point.

### When to skip a SAM 2 polygon entirely

- It's actually a cluster of leaves, not a boll. Delete the polygon.
- The boll is so occluded you can't usefully estimate `whole_estimate` either. Delete polygon, move on.
- Motion blur makes any annotation unreliable. Delete.

Skipping is fine — 40 good frames beats 50 sloppy ones.

## 2:45 – 3:00   Export

Actions → **Export annotations** → format **COCO 1.0** with images = no (just the JSON). Save as `pilot_v2_annotations.json` back into this folder.

Post-deadline, the next session can run a new `cvat_to_yolo_v2.py` against it to produce paired seg + pose datasets. The old `cvat_to_yolo.py` won't match this schema — don't try to use it.

## What to flag if you fall behind

If at 2:00 you've only done 15 frames, stop and export whatever you have. 15 frames × 13 bolls × 2 keypoints = 390 labeled keypoints is still enough to fine-tune a pose head, and you can keep annotating tomorrow against the same CVAT task.
