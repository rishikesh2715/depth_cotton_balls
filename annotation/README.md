# annotation/ — supervised YOLO track

Everything for turning SAM 2 outputs and annotator COCO exports into
trained YOLOv11 models.

## Read in this order

1. `Cotton_Boll_Annotation_Guidelines.docx` — annotation schema; source
   of truth for annotators.
2. `RUNBOOK.md` — v1 pilot: sample frames → bootstrap from SAM 2 →
   annotate in CVAT → convert → train. CVAT setup lives here.
3. `TRAIN_COWORKER_DATA.md` — training on the first coworker pilot
   dataset (seg only; keypoint schema wasn't fixed yet).
4. `V3_RUNBOOK.md` — v3 dataset (46 images / 274 instances), first
   seg + pose + aux training.
5. `V4_RUNBOOK.md` — **current.** v4 dataset (734 instances, two camera
   angles, sequential split), second iteration of all three models.

## Contents

- `scripts/` — dataset conversion (`merged_v4_to_yolo.py`, `cvat_to_yolo.py`,
  `coco_to_yolov11_seg.py`, …), training (`train_v4_{seg,pose,aux}.py` and
  earlier versions), inference (`infer_v3.py`, `predict_and_save.py`,
  `predict_video.py`), frame sampling and SAM 2 bootstrapping.
- `cvat/` — label constructor JSONs to paste into CVAT (v1 and v2 schema).
- `pilot_v2/` — v1 pilot working dir: runbook, annotation tracker CSV
  (sampled frames + bootstrap COCO are gitignored, regenerable via
  `scripts/sample_frames.py` / `scripts/bootstrap_from_sam2.py`).
- `Inference_Viewer.html` — standalone browser viewer for frame JPEG+JSON
  prediction pairs; `viewer_demo/` is a small committed example;
  `Inference_Visualization_Legend.md` explains the overlay.
- `datasets/` (gitignored) — YOLO-format datasets produced by the
  conversion scripts.
- `*_demo.mp4` (gitignored) — rendered overlay videos for sharing.
