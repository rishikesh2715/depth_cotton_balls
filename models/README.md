# models/ — trained weights (not in git)

Weights are gitignored; only this README is tracked. Expected layout:

- `segment/best.pt`, `segment/best_v4.pt` — YOLOv11 seg (v3 / v4 datasets)
- `pose/best.pt` — YOLOv11 pose (tip/base keypoints)
- `auxliary/best.pt` — ResNet-18 aux head (occlusion class + visibility fraction)
- `pretrained/` — stock Ultralytics checkpoints (`yolo11n-seg.pt`, …);
  Ultralytics re-downloads these automatically if missing.

To regenerate: run the training scripts in `annotation/scripts/`
(`train_v4_seg.py`, `train_v4_pose.py`, `train_v4_aux.py`) per
`annotation/V4_RUNBOOK.md`; copy `runs/.../weights/best.pt` here.
