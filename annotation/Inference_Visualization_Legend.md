# Cotton-boll inference frames — what you're looking at

The annotated frames and videos are output from one forward pass of three
models trained on `merged_dataset_v3` (40 training + 6 validation images,
274 cotton-boll instances). Each annotated boll shows the combined
output of:

- **YOLOv11n-seg** — bounding box + segmentation polygon
- **YOLOv11n-pose** — tip and base keypoints (2 keypoints per boll, with
  visibility flag in {0, 1, 2})
- **ResNet-18 auxiliary head** — predicts `occluded_by` class (4-way:
  none / leaf / stem / other) and a continuous `visibility_fraction` in
  [0, 1] from the bbox crop

Together these cover four of the six terms in the planned loss function
(L_box, L_seg, L_kp + L_kp_vis, L_occ, L_visfrac). The remaining pieces
— RGB-D measurement-validity weighting and multi-frame aggregation —
are the next milestone and are not visualized here.

## Reading the overlay

**Colored rectangle and polygon outline (with semi-transparent fill):**
the seg model's bounding box and segmentation polygon. The color encodes
the auxiliary head's predicted occlusion class:

- green = `none` (unobstructed)
- yellow = `leaf` (leaf occluding part of the boll)
- orange = `stem` (stem or branch occluding part of the boll)
- red = `other` (frame-edge cutoff or another boll)
- gray = the auxiliary head was not run on this frame; only the seg
  detection is shown

**Cyan dot, magenta dot, and the white line connecting them:** the pose
model's `tip` (cyan) and `base` (magenta) keypoint predictions, with the
white line being the boll's predicted principal axis. A missing dot
means the pose model assigned visibility flag = 0 to that endpoint —
i.e. it judged the endpoint as not labelable / not visible. The 3D lift
of this axis using the aligned RealSense depth frame is what will
produce the cos(θ) measurement-confidence weight in the upcoming RGB-D
fusion step.

**Label text above each box,** read as
`<occ_class> <occ_conf> | v=<visfrac> | d=<det_conf>`:

- `<occ_class> <occ_conf>` — the auxiliary head's classifier output and
  its softmax confidence. Example: `leaf 0.74` means "74% confident this
  is leaf-occluded."
- `v=<visfrac>` — the auxiliary head's continuous visibility estimate
  in [0, 1], trained on `1 − occlusion_pct`. `v=1.00` means fully visible
  to the camera; `v=0.40` means roughly 60% of the boll is occluded.
- `d=<det_conf>` — the seg model's per-detection confidence (objectness
  × class probability).

The diagnostically interesting combination is **high `d` with low `v`**:
the seg model is confident the boll exists, but the auxiliary head says
most of it is hidden. That signal will tell the downstream measurement
step to down-weight this frame's contribution to that boll's size
estimate, instead of treating every visible polygon as equally
trustworthy.

## v0 validation metrics

For calibration on whether the overlays should be trusted (recording-aware
split: backdown + frontdown each contribute to train and val):

| Model | Metric                | Val (n=6 images, 32 instances) |
|-------|-----------------------|--------------------------------|
| Seg   | box mAP50             | 0.96 |
| Seg   | mask mAP50            | 0.96 |
| Seg   | mask mAP50–95         | 0.69 |
| Pose  | box mAP50             | 0.98 |
| Pose  | keypoint mAP50 (OKS)  | 0.92 |
| Pose  | keypoint mAP50–95     | 0.85 |

These are above the "good" floor on every metric, but the validation set
is small (6 images, no `other`-class instances), so these numbers are
encouraging rather than conclusive. The v4 annotation round listed in
the planning document is meant to expand validation coverage —
particularly for occluded bolls, hard angles, and lighting variation
between recording passes.

## Caveats worth flagging

1. **Auxiliary head label noise.** v3 has 95 instances where the
   annotator marked tip or base as "visible" in the polygon attributes
   but never placed the keypoint. Those land as `(0, 0, v=0)` in the
   pose labels, which YOLO's OKS loss skips natively, so training is
   unaffected — but the resulting model has slightly less keypoint
   supervision than the n=274 instance count suggests.

2. **Class imbalance for `occluded_by`.** Train distribution is
   none=182, leaf=30, stem=25, other=5. The "other" class will be
   under-trained until v4. Frame-edge cutoffs and other-boll occlusions
   may be misclassified as `leaf` or `stem`.

3. **Detection confidence threshold = 0.25.** Lowering this surfaces
   more occluded / partially-visible bolls but adds false positives.
   The threshold is exposed in the inference script as `--conf` if you
   want to sweep it on a sample frame.
