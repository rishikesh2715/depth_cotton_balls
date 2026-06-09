"""
predict_video.py — Run the trained YOLOv11-seg model on a video file,
overlay per-boll masks + persistent track IDs + mask-derived axial
endpoints, and write the result to an annotated output video.

HONEST DISCLOSURE (also burned into the on-screen legend):
    The current checkpoint was trained on 47 COCO images containing only
    `cotton_boll` POLYGON annotations — there are no annotated tip/base
    keypoint coordinates in that dataset. The two endpoint dots drawn on
    every boll here are a GEOMETRIC HEURISTIC (principal-axis endpoints
    via PCA on mask pixels), not a learned pose output.
    Semantic tip-vs-base labeling, along with occlusion-aware usability
    gating, requires the keypoint-pose model planned for annotation
    round 2 (see annotation/Cotton_Boll_Annotation_Guidelines.docx §5).

Run:
    python annotation/scripts/predict_video.py \
        --model runs/coworker_seg/v1_small/weights/best.pt \
        --source path/to/video.mp4 \
        --out path/to/video_annotated.mp4 \
        --imgsz 1280 --conf 0.25
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Drawing helpers ──────────────────────────────────────────────────────


def color_for_id(tid: int):
    """Deterministic, well-spread BGR color per track ID."""
    hue = (tid * 23 + 7) % 180  # OpenCV HSV hue is 0..179
    c = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def mask_axial_endpoints(mask_bool: np.ndarray):
    """
    Return (pt_a, pt_b, length_px) of the mask's principal axis via PCA.

    pt_a and pt_b are the two extreme pixel positions along the first
    singular vector of the zero-meaned mask-pixel cloud.
    """
    ys, xs = np.where(mask_bool)
    if xs.size < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    # SVD — vt[0] is the principal direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    pt_a = pts[int(np.argmin(proj))]
    pt_b = pts[int(np.argmax(proj))]
    return pt_a, pt_b, float(proj.max() - proj.min())


def put_text_outlined(img, text, org, scale=0.5, color=(255, 255, 255),
                      thickness=1):
    """White text with black outline — readable over any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_legend(img):
    lines = [
        "YOLOv11-seg pilot (47-img coworker COCO, trained 2026-04-17)",
        "Colored overlay = cotton_boll instance mask",
        "White/colored dots = principal-axis endpoints (GEOMETRIC proxy)",
        "TIP vs BASE semantics require round-2 keypoint model",
    ]
    y = 22
    for ln in lines:
        put_text_outlined(img, ln, (10, y), scale=0.5)
        y += 20


def draw_overlay(frame, masks, tids, confs, alpha=0.45):
    """Composite colored mask overlay + axis endpoints + ID/conf labels."""
    out = frame.copy()
    blended = out.copy()

    for m, tid, c in zip(masks, tids, confs):
        color = color_for_id(tid)

        # colored fill
        color_layer = np.zeros_like(out)
        color_layer[m] = color
        blended = np.where(m[..., None],
                           (blended * (1 - alpha) + color_layer * alpha).astype(np.uint8),
                           blended)

        # contour
        cnts, _ = cv2.findContours(m.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, color, 2, lineType=cv2.LINE_AA)

        # principal-axis endpoints (tip/base PROXY)
        ep = mask_axial_endpoints(m)
        if ep is not None:
            a, b, _ = ep
            a_i = tuple(a.astype(int))
            b_i = tuple(b.astype(int))
            cv2.line(blended, a_i, b_i, color, 1, cv2.LINE_AA)
            for pt in (a_i, b_i):
                cv2.circle(blended, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(blended, pt, 6, color, 2, cv2.LINE_AA)

        # label at centroid
        ys, xs = np.where(m)
        if ys.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            put_text_outlined(blended, f"#{tid}  {c:.2f}",
                              (max(0, cx - 30), cy), scale=0.55)

    return blended


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to trained YOLO weights, e.g. "
                         "runs/coworker_seg/v1_small/weights/best.pt")
    ap.add_argument("--source", required=True,
                    help="Input video file (.mp4/.mov/.avi) OR a folder of "
                         "frames (e.g. work/handheld_1/frames)")
    ap.add_argument("--out", required=True,
                    help="Output annotated video path (.mp4)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Inference image size (match training imgsz)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="ultralytics tracker yaml (bytetrack.yaml / botsort.yaml)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="If > 0, only process the first N frames (debug)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Output video FPS. Only used when --source is a "
                         "frames folder; ignored for video inputs (input fps "
                         "is reused).")
    ap.add_argument("--frame-glob", default="*.jpg",
                    help="Glob pattern for frames when --source is a folder")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"ERROR: --source not found: {src}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe once to get fps + frame size for the writer. Works for both a
    # video file (via VideoCapture) and a folder of frames (peek first img).
    if src.is_file():
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            sys.exit(f"ERROR: could not open video: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    elif src.is_dir():
        frame_paths = sorted(src.glob(args.frame_glob))
        if not frame_paths:
            sys.exit(f"ERROR: no frames in {src} matching {args.frame_glob}")
        probe = cv2.imread(str(frame_paths[0]))
        if probe is None:
            sys.exit(f"ERROR: could not read {frame_paths[0]}")
        H, W = probe.shape[:2]
        fps = float(args.fps)
        n_frames_total = len(frame_paths)
    else:
        sys.exit(f"ERROR: --source is neither a file nor a directory: {src}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        sys.exit(f"ERROR: cv2.VideoWriter failed to open {out_path}")

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Source: {src}  ({W}x{H} @ {fps:.2f} fps, {n_frames_total} frames)")
    print(f"[INFO] Out   : {out_path}")

    model = YOLO(args.model)

    # stream=True: iterate frame-by-frame without loading everything.
    # persist=True: keep tracker state across the iterator.
    results = model.track(
        source=str(src),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    n_done = 0
    n_with_det = 0
    n_total_det = 0

    for res in results:
        frame = res.orig_img  # BGR, original resolution
        if frame is None:
            continue

        if res.masks is None or len(res.masks) == 0:
            overlay = frame.copy()
        else:
            fh, fw = res.orig_shape
            polys = res.masks.xy  # original-resolution polygons

            # Track IDs may be None for the first frame until tracker warms up
            if res.boxes is not None and res.boxes.id is not None:
                tids = res.boxes.id.int().cpu().tolist()
            else:
                tids = list(range(len(polys)))

            if res.boxes is not None and res.boxes.conf is not None:
                confs = res.boxes.conf.cpu().tolist()
            else:
                confs = [0.0] * len(polys)

            masks = []
            for poly in polys:
                m = np.zeros((fh, fw), dtype=np.uint8)
                if poly is not None and len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
                masks.append(m.astype(bool))

            overlay = draw_overlay(frame, masks, tids, confs)
            n_with_det += 1
            n_total_det += len(masks)

        draw_legend(overlay)

        # HUD — top-right: frame index / total
        hud = f"frame {n_done+1}/{n_frames_total}"
        put_text_outlined(overlay, hud, (W - 180, 22), scale=0.55)

        writer.write(overlay)
        n_done += 1

        if n_done % 30 == 0:
            print(f"  [{n_done}/{n_frames_total}]  det_frames={n_with_det} "
                  f"total_det={n_total_det}")

        if args.max_frames and n_done >= args.max_frames:
            break

    writer.release()
    print("\n=== Done ===")
    print(f"Frames written      : {n_done}")
    print(f"Frames w/ detections: {n_with_det}")
    print(f"Total detections    : {n_total_det}")
    print(f"Output              : {out_path}")


if __name__ == "__main__":
    main()
