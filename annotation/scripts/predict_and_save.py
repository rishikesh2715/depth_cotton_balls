"""
predict_and_save.py — Run a trained YOLOv11-seg model on a recording's
frames and save per-boll binary masks in the *same layout* the existing
SAM 2 pipeline produces, so `pipeline/04_measure_bolls.py` can ingest the
output unchanged.

SAM 2 pipeline layout (reproduced here):
    work/<recording>/masks/<frame_idx>/<boll_id>.png      # grayscale, >0 inside

This script writes, by default:
    work/<recording>/masks_yolo/<frame_idx>/<int_id>.png

so a run of `04_measure_bolls.py` just needs `--work work/<recording>` with
`masks_yolo` symlinked/renamed to `masks`, OR pass `--out` pointing at a
fresh `masks` folder in a new work dir.

Two ID modes:

  default:
      YOLO detections have no tag knowledge. Each frame gets integer IDs
      0, 1, 2, … in detection order. Good for "what are the sizes of all
      detected bolls" but loses cross-frame identity.

  --match-sam2 <sam2_masks_dir>:
      For each frame, compare every YOLO prediction to the existing SAM 2
      masks in <sam2_masks_dir>/<frame_idx>/*.png via mask-IoU. If the
      best IoU ≥ --iou-thresh, inherit that SAM 2 filename stem as the
      integer ID (= physical paper tag). Unmatched YOLO detections get
      new IDs starting above the max SAM 2 ID. Matched SAM 2 masks that
      weren't "claimed" by any YOLO prediction are *dropped*, i.e. the
      output reflects what YOLO saw.

Run:
    python annotation/scripts/predict_and_save.py \
        --model runs/coworker_seg/v1_small/weights/best.pt \
        --source-frames work/handheld_1/frames \
        --out work/handheld_1/masks_yolo \
        --imgsz 1280 --conf 0.25

With SAM 2 tag inheritance:
    python annotation/scripts/predict_and_save.py \
        --model runs/coworker_seg/v1_small/weights/best.pt \
        --source-frames work/handheld_1/frames \
        --out work/handheld_1/masks_yolo \
        --match-sam2 work/handheld_1/masks \
        --iou-thresh 0.3
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def load_sam2_frame_masks(sam2_frame_dir: Path, target_hw):
    """
    Load every <id>.png under sam2_frame_dir as a binary bool array at
    target_hw = (H, W). Returns {int_id: bool_array}.
    """
    out = {}
    if not sam2_frame_dir.is_dir():
        return out
    H, W = target_hw
    for p in sorted(sam2_frame_dir.iterdir()):
        if p.suffix.lower() != ".png":
            continue
        try:
            tag_id = int(p.stem)
        except ValueError:
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out[tag_id] = m > 0
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two bool masks of identical shape."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def predict_one_frame(model, img_path: Path, imgsz: int, conf: float,
                      device: str):
    """Return list of (H, W) bool masks in the frame's original resolution."""
    res = model(str(img_path), imgsz=imgsz, conf=conf, device=device,
                verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return []

    # res.masks.xy is polygons in ORIGINAL image coordinates (handles
    # letterbox/scale internally), which is what we want so the output
    # lines up with the untouched frame file on disk.
    H, W = res.orig_shape  # (h, w)
    polys = res.masks.xy   # list of (K, 2) arrays

    out = []
    for poly in polys:
        m = np.zeros((H, W), dtype=np.uint8)
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        out.append(m > 0)
    return out


def assign_ids_by_iou(pred_masks, sam2_masks_by_id, iou_thresh: float,
                       fallback_start: int):
    """
    Greedy max-IoU matching predictions -> SAM 2 IDs.

    Returns list of int IDs, one per pred_masks entry (parallel order).
    Unmatched predictions are numbered starting at fallback_start and
    incrementing, guaranteed not to collide with existing SAM 2 IDs.
    """
    n = len(pred_masks)
    ids = [None] * n
    used_sam_ids = set()

    # Score all (pred, sam2) pairs and greedily pick best IoU first
    pairs = []
    sam_items = list(sam2_masks_by_id.items())
    for pi, pm in enumerate(pred_masks):
        for tag_id, sm in sam_items:
            iou = mask_iou(pm, sm)
            if iou >= iou_thresh:
                pairs.append((iou, pi, tag_id))
    pairs.sort(reverse=True)  # highest IoU first

    for iou, pi, tag_id in pairs:
        if ids[pi] is None and tag_id not in used_sam_ids:
            ids[pi] = tag_id
            used_sam_ids.add(tag_id)

    next_fallback = fallback_start
    for pi in range(n):
        if ids[pi] is None:
            # avoid colliding with any existing SAM 2 id OR a newly assigned one
            while next_fallback in sam2_masks_by_id or next_fallback in ids:
                next_fallback += 1
            ids[pi] = next_fallback
            next_fallback += 1
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to trained YOLO weights, e.g. "
                         "runs/coworker_seg/v1_small/weights/best.pt")
    ap.add_argument("--source-frames", required=True,
                    help="Folder of frame JPGs, e.g. work/handheld_1/frames")
    ap.add_argument("--out", required=True,
                    help="Output masks root, e.g. work/handheld_1/masks_yolo")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Inference image size (match training imgsz)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold")
    ap.add_argument("--device", default="0",
                    help="CUDA id, 'cpu', or comma list")
    ap.add_argument("--match-sam2", default=None,
                    help="Optional: SAM 2 masks dir "
                         "(e.g. work/handheld_1/masks) to inherit tag IDs "
                         "by mask IoU")
    ap.add_argument("--iou-thresh", type=float, default=0.3,
                    help="Min IoU for SAM 2 tag inheritance "
                         "(only used with --match-sam2)")
    ap.add_argument("--glob", default="*.jpg",
                    help="Frame filename glob inside --source-frames")
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only process the first N frames (debug)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(args.source_frames)
    if not frames_dir.is_dir():
        sys.exit(f"ERROR: --source-frames not found: {frames_dir}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sam2_root = Path(args.match_sam2) if args.match_sam2 else None
    if sam2_root is not None and not sam2_root.is_dir():
        sys.exit(f"ERROR: --match-sam2 dir not found: {sam2_root}")

    model = YOLO(args.model)

    frame_paths = sorted(frames_dir.glob(args.glob))
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]

    if not frame_paths:
        sys.exit(f"ERROR: no frames matched {frames_dir}/{args.glob}")

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Frames: {len(frame_paths)} from {frames_dir}")
    print(f"[INFO] Output: {out_root}")
    print(f"[INFO] Tag mode: "
          f"{'inherit from ' + str(sam2_root) if sam2_root else 'running counter'}")

    n_frames_with_preds = 0
    n_total_preds = 0
    n_matched = 0

    for idx, fp in enumerate(frame_paths):
        frame_stem = fp.stem  # e.g. "00000"
        pred_masks = predict_one_frame(
            model, fp, imgsz=args.imgsz, conf=args.conf, device=args.device)

        if not pred_masks:
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(frame_paths)}] {frame_stem}: no detections")
            continue

        n_frames_with_preds += 1
        n_total_preds += len(pred_masks)

        H, W = pred_masks[0].shape

        if sam2_root is not None:
            sam2_masks_by_id = load_sam2_frame_masks(
                sam2_root / frame_stem, target_hw=(H, W))
            fallback_start = (max(sam2_masks_by_id.keys()) + 1
                              if sam2_masks_by_id else 0)
            ids = assign_ids_by_iou(
                pred_masks, sam2_masks_by_id,
                iou_thresh=args.iou_thresh,
                fallback_start=fallback_start,
            )
            n_matched += sum(1 for i in ids if i in sam2_masks_by_id)
        else:
            ids = list(range(len(pred_masks)))

        frame_out = out_root / frame_stem
        frame_out.mkdir(parents=True, exist_ok=True)
        for boll_id, pm in zip(ids, pred_masks):
            mask_png = (pm.astype(np.uint8)) * 255
            cv2.imwrite(str(frame_out / f"{boll_id}.png"), mask_png)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(frame_paths):
            print(f"  [{idx+1}/{len(frame_paths)}] "
                  f"{frame_stem}: {len(pred_masks)} preds, "
                  f"ids={ids}")

    print("\n=== Done ===")
    print(f"Frames processed     : {len(frame_paths)}")
    print(f"Frames with preds    : {n_frames_with_preds}")
    print(f"Total predictions    : {n_total_preds}")
    if sam2_root is not None:
        print(f"Predictions matched  : {n_matched} "
              f"({(100.0*n_matched/n_total_preds if n_total_preds else 0):.1f}%)")
    print(f"\nNext step:")
    print(f"  Point pipeline/04_measure_bolls.py at this output. Either:")
    print(f"    (a) rename {out_root.name} -> masks and rerun 04, or")
    print(f"    (b) copy metadata.json + depth/ next to {out_root.parent}/masks_yolo_work/")
    print(f"        and rerun: python pipeline/04_measure_bolls.py --work <that dir>")


if __name__ == "__main__":
    main()
