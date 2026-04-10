# Cotton Boll RGBD Measurement Pipeline

A 5-stage pipeline that takes RealSense `.bag` recordings of tagged
cotton bolls and produces per-boll size measurements validated against
caliper ground truth.

## What this gives you

For each numbered boll, the pipeline produces:
- Median height and width in cm (aggregated across many frames)
- Standard deviation across frames (your measurement noise)
- Per-boll error vs caliper ground truth
- Overall metrics: MAE, RMSE, R², Pearson r, Bland-Altman bias
- Scatter plots and Bland-Altman plots ready to drop into a slide deck

## Why 5 stages instead of one big script

Each stage writes its output to disk and the next stage reads it.
This means you can re-run just the part you need. If your masks are
fine but your ground-truth file has a typo, you only re-run stage 5,
not the slow .bag extraction. **This will save you tonight.**

```
.bag  ──[01]──>  frames/ + depth/ + metadata.json
                                  │
                            [02]  │  pick anchor frame
                                  ▼
                              anchor.json
                                  │
                            [03]  │  click bolls in SAM 2 → propagate
                                  ▼
                              masks/<frame>/<boll_id>.png
                                  │
                            [04]  │  measure each (frame, boll) pair
                                  ▼
                              measurements_per_frame.csv
                                  │
                            [05]  │  aggregate, join GT, plot
                                  ▼
                              report/  ← deliverable
```

## Setup (one-time, ~15 min)

```bash
# Existing deps you already have:
pip install pyrealsense2 opencv-python numpy matplotlib

# SAM 2:
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# Download the large checkpoint (~900 MB, best accuracy):
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ../..
```

You'll reference these two paths when running stage 3:
- `--sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt`
- `--sam2-config configs/sam2.1/sam2.1_hiera_l.yaml`  (path inside the sam2 repo)

## Before you start: fill in the coverage table

You have many .bag files (trolley at different heights/angles + handheld
passes), and each physical boll is visible in some subset of them. The
purpose of all those recordings is coverage, not redundancy — a boll
hidden behind a leaf at 0° may become visible at 45°.

**Before touching the pipeline, fill in `coverage_template.csv`.**
Scrub through each recording (RealSense Viewer or the stage 2 anchor
picker works for this) and write down which boll tag numbers are visible
in each. Takes ~20 minutes. This is:

1. What your professor meant by "spend some time manually annotating"
2. Your guide for stage 3 — tells you which tag numbers to click in
   each recording
3. A deliverable in its own right — "here's the coverage matrix for
   our experimental design"

Save it as `coverage.csv` in your working directory when you're done.
**Tag numbers must be unique across all plants** — don't let two bolls
share the number "5" even if they're on different plants.

## Running the pipeline (per .bag)

Run stages 1–4 once per .bag, into its own working directory. Stage 5
combines them all at the end.

### Stage 1 — Extract frames (slow, ~1 min per minute of recording)

```bash
python 01_extract_frames.py --bag trolley_0deg_low.bag \
    --out work/rec_0deg_low --recording-id 0deg_low
```

The `--recording-id` is a short label baked into the metadata and
carried through to the final CSV. Use names like `0deg_low`,
`0deg_high`, `45deg_low`, `handheld_1` so the per-recording bias plot
is readable later.

If your recording is long or high-FPS, thin it out with `--stride`.
Stride 2 keeps every 2nd frame (cuts work in half), stride 3 every 3rd.
Aim for ~150–400 frames per recording — that's plenty of samples per
boll without grinding SAM 2.

```bash
python 01_extract_frames.py --bag trolley_0deg_low.bag \
    --out work/rec_0deg_low --recording-id 0deg_low --stride 2
```

### Stage 2 — Pick the anchor frame

The anchor is the frame where the most numbered tags are visible
clearly. SAM 2 propagates BOTH directions from the anchor, so the
middle of the recording is usually the right choice.

```bash
python 02_pick_anchor.py --work work/rec_0deg_low
```

Scrub with `a/d`, jump with `j/l` (10) or `h/;` (50), `g` to type a
specific frame number, SPACE to lock it in.

### Stage 3 — Annotate with SAM 2 (the manual part, ~1–2 min per boll)

```bash
python 03_annotate_sam2.py --work work/rec_0deg_low \
    --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
```

**Pull out your coverage.csv before you start.** The list of boll tag
numbers you wrote down for this specific recording is your checklist.
Click each one, in any order.

Workflow for each boll:
1. Press `n`, type the boll's tag number (e.g. `35`), hit Enter
2. Left-click on the boll. The point appears in yellow.
3. Add 1–3 positive clicks per boll (center + maybe edges). One click
   is often enough for SAM 2.
4. Press `n` again to commit and move to the next boll.
5. Before pressing `p` to propagate, cross-reference against your
   coverage.csv — did you get every boll on the list?

After all bolls are committed, press `p` to propagate. SAM 2 runs
forward from the anchor through the end, then backward from the anchor
through the start.

**Do this for every recording.** The same boll #35 annotated in
`rec_0deg_low` and again in `rec_45deg_high` will be merged
automatically in stage 5 because the obj_id is the same tag number.
You don't need to do anything special to link them.

### Stage 4 — Measure (fast, < 1 min per recording)

Run once per recording:

```bash
python 04_measure_bolls.py --work work/rec_0deg_low
python 04_measure_bolls.py --work work/rec_0deg_high
python 04_measure_bolls.py --work work/rec_45deg_low
# ... etc
```

Reads every (frame, boll) mask and computes distance, height, width,
and area using your camera intrinsics. Two height/width metrics are
written for each measurement:
- `H_aa, W_aa` — axis-aligned bounding box (your original method)
- `H_rot, W_rot` — rotated minAreaRect (matches caliper measurement)

Stage 5 uses the rotated version because that's what calipers measure.

### Stage 5 — Report

```bash
python 05_make_report.py --work work/plant_a --ground-truth gt.csv
```

If your ground truth CSV uses different column names, override them:

```bash
python 05_make_report.py --work work/plant_a \
    --ground-truth gt.csv \
    --gt-cols id=Boll_Tag height=Height_mm width=Width_mm \
    --units mm
```

Outputs land in `work/plant_a/report/`:
- `scatter_height.png`, `scatter_width.png` — **for the slide deck**
- `bland_altman_height.png`, `bland_altman_width.png` — **also for slides**
- `per_boll_summary.csv` — table of every boll with measured + GT + error
- `overall_metrics.json` — MAE, RMSE, R², Pearson r
- `summary.txt` — human-readable one-pager

## Combining results across multiple .bag files

Run stages 1–4 separately for each .bag, into separate working dirs.
For the report, the simplest path tonight is to merge the per-frame CSVs:

```bash
# Merge all per-frame measurements into one file
head -1 work/plant_a/measurements_per_frame.csv > work/all_measurements.csv
for d in work/plant_*/; do
    tail -n +2 "$d/measurements_per_frame.csv" >> work/all_measurements.csv
done

# Tell stage 5 to read it
python 05_make_report.py --work work/all --ground-truth gt.csv \
    --measurements work/all_measurements.csv
```
(Make sure boll tag numbers are unique across plants — if two
different plants both have a boll #5, you'll need to renumber.)

## Quality gates

Stage 4 flags each measurement with three quality bits:
- `ok_mask_size`  — mask has at least 200 pixels
- `ok_distance`   — boll is between 15 cm and 250 cm from camera
- `ok_depth_coverage` — at least 30% of mask pixels have valid depth

Stage 5 only aggregates measurements where all three are 1. Tune the
thresholds at the top of `04_measure_bolls.py` if you need to.

## Talking points for the meeting

1. **Validation methodology.** Caliper-measured ground truth as
   independent reference, per-boll comparison via SAM 2 tag IDs that
   match the physical tag numbers exactly. R²/MAE/Bland-Altman.
2. **Why SAM 2 instead of training a YOLO model.** Zero training data
   needed, one click per boll, handles partial occlusion via temporal
   memory. Tag IDs flow directly from physical tags into the
   measurement records — no separate annotation step.
3. **Why rotated rect, not axis-aligned.** Calipers measure across the
   real major/minor axis of the boll regardless of how it's oriented in
   the frame. Both metrics are in the CSV; rotated is the apples-to-
   apples comparison.
4. **Per-frame median = noise reduction.** Single-shot RGBD distance is
   noisy at ±0.5–1 cm. Aggregating ~50–200 frames per boll gives a
   median + std, and the std *is* a useful number — it tells you the
   measurement uncertainty per boll.
5. **Where this beats Depth Anything.** Depth Anything is monocular,
   relative depth, no metric scale without a reference object in frame.
   RealSense gives you metric depth out of the box, so "boll size in
   cm" needs no scale calibration step.

## Known limitations to mention before he asks

- Boll height measured here is the projected silhouette long axis, not
  the through-axis depth. For a roughly spherical boll, projection
  overestimates by zero; for an elongated boll viewed end-on, it
  underestimates. A second measurement angle would resolve this.
- SAM 2 can lose a boll under heavy occlusion or motion blur. The
  per-boll `n_frames` column shows how many frames each boll was
  successfully tracked through.
- Depth holes in shiny/wet regions of bolls can bias the median
  distance. The hole-filling filter in stage 1 mitigates but doesn't
  eliminate this.

## File index

| File | What it does |
|---|---|
| `01_extract_frames.py` | .bag → frames/ + depth/ + metadata.json |
| `02_pick_anchor.py` | Pick best frame to start annotating from |
| `03_annotate_sam2.py` | Click bolls + SAM 2 propagation → masks/ |
| `04_measure_bolls.py` | Per-frame measurements CSV |
| `05_make_report.py` | Aggregate, join GT, scatter + Bland-Altman plots |
| `test_synthetic.py` | End-to-end test of stages 4+5 with fake data |
