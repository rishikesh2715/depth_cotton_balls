"""
infer_v3.py — End-to-end inference (seg + pose + aux) on images, videos,
or directories.

Loads YOLOv11-seg, YOLOv11-pose, and AuxHead once and runs them on:
  - a single image file (.jpg .jpeg .png .bmp .tif .tiff)
  - a single video file (.mp4 .mov .avi .mkv .webm)
  - a directory containing any mix of the above (recurses)

For each input, writes:
  - <out>/<stem>.<ext>      annotated image, OR
  - <out>/<stem>.mp4        annotated video
  - <out>/<stem>.json       per-frame structured records (with --save-json)
And aggregates everything into:
  - <out>/results.csv       one row per detection (with --save-csv)

Visualization overlays:
  - mask polygon outline + semi-transparent fill, color by occluded_by class
  - bbox rectangle + label "<occ_class> <occ_conf> | v=<visfrac> | d=<det_conf>"
  - tip (cyan) and base (magenta) keypoints connected by a white axis line

Run (from depth_cotton_balls/):
    python annotation\\scripts\\infer_v3.py ^
        --seg  runs\\v3_seg\\v0\\weights\\best.pt ^
        --pose runs\\v3_pose\\v0\\weights\\best.pt ^
        --aux  runs\\v3_aux\\v0\\best.pt ^
        --source work\\handheld_1\\frames ^
        --out   runs\\v3_inference\\handheld_1 ^
        --device cuda:0 ^
        --save-json --save-csv --video-out --video-fps 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_aux import CombinedInfer  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# BGR (cv2 convention).
COLOR_BY_CLASS = {
    "none":  (60, 220, 60),    # green
    "leaf":  (60, 220, 220),   # yellow
    "stem":  (60, 165, 255),   # orange
    "other": (60, 60, 220),    # red
    None:    (200, 200, 200),  # gray (no aux)
}
COLOR_TIP = (255, 220, 60)     # cyan-ish
COLOR_BASE = (220, 60, 220)    # magenta-ish
COLOR_AXIS = (255, 255, 255)   # white


def render(image_bgr: np.ndarray, per_boll: list, *,
           draw_mask: bool = True, draw_kpts: bool = True,
           alpha: float = 0.35,
           kpt_conf_min: float = 0.3,
           draw_track_id: bool = True) -> np.ndarray:
    """Return an annotated copy of image_bgr.

    kpts[i][2] is treated as a continuous confidence in [0,1]. Keypoints
    with conf < kpt_conf_min are drawn as hollow circles (model "guessed"),
    >= threshold as filled (model "committed").
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]
    lw = max(2, int(min(W, H) / 600))
    fs = max(0.5, min(W, H) / 1500.0)
    text_thick = max(1, lw // 2)

    # 1) Mask fills (alpha blend on a separate overlay so outlines stay sharp)
    if draw_mask:
        overlay = out.copy()
        any_filled = False
        for r in per_boll:
            poly = r.get("mask_poly")
            if not poly:
                continue
            color = COLOR_BY_CLASS.get(r.get("occ_class"), COLOR_BY_CLASS[None])
            cv2.fillPoly(overlay, [np.asarray(poly, dtype=np.int32)], color)
            any_filled = True
        if any_filled:
            cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    # 2) Outlines, bboxes, kpts, labels
    for r in per_boll:
        color = COLOR_BY_CLASS.get(r.get("occ_class"), COLOR_BY_CLASS[None])
        x, y, w, h = r["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        if draw_mask and r.get("mask_poly"):
            cv2.polylines(out, [np.asarray(r["mask_poly"], dtype=np.int32)],
                          True, color, lw)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)

        # Keypoints + axis line. Treat kpt[2] as continuous confidence:
        #   conf >= kpt_conf_min -> filled  ("model committed")
        #   0 < conf <  kpt_conf_min -> hollow ("model guessed")
        if draw_kpts and r.get("kpts"):
            kpts = r["kpts"]
            if len(kpts) >= 2:
                tx, ty, tc = kpts[0]
                bx, by, bc = kpts[1]
                kr = max(5, lw * 2)
                # Axis line only if BOTH endpoints crossed the threshold.
                if tc >= kpt_conf_min and bc >= kpt_conf_min:
                    cv2.line(out, (int(tx), int(ty)), (int(bx), int(by)),
                             COLOR_AXIS, lw)
                # Tip
                if tc > 0:
                    if tc >= kpt_conf_min:
                        cv2.circle(out, (int(tx), int(ty)), kr, COLOR_TIP, -1)
                    else:
                        cv2.circle(out, (int(tx), int(ty)), kr, COLOR_TIP, max(1, lw // 2))
                # Base
                if bc > 0:
                    if bc >= kpt_conf_min:
                        cv2.circle(out, (int(bx), int(by)), kr, COLOR_BASE, -1)
                    else:
                        cv2.circle(out, (int(bx), int(by)), kr, COLOR_BASE, max(1, lw // 2))

        # Track id chip (top-right of bbox)
        if draw_track_id and r.get("track_id") is not None:
            tag = f"#{r['track_id']}"
            ((tw_, th_), _) = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                              fs * 0.85, text_thick)
            cx0 = x2 - tw_ - 6
            cy0 = y1
            cv2.rectangle(out, (cx0 - 3, cy0), (cx0 + tw_ + 4, cy0 + th_ + 6),
                          (0, 0, 0), -1)
            cv2.rectangle(out, (cx0 - 3, cy0), (cx0 + tw_ + 4, cy0 + th_ + 6),
                          color, max(1, lw // 2))
            cv2.putText(out, tag, (cx0, cy0 + th_ + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, color, text_thick,
                        cv2.LINE_AA)

        # Label
        parts = []
        if r.get("occ_class") is not None:
            parts.append(f"{r['occ_class']} {r['occ_class_conf']:.2f}")
        if r.get("visfrac") is not None:
            parts.append(f"v={r['visfrac']:.2f}")
        if r.get("det_conf") is not None:
            parts.append(f"d={r['det_conf']:.2f}")
        text = " | ".join(parts)
        if text:
            ((tw, th), bl) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                              fs, text_thick)
            ty_text = max(th + 4, y1 - 4)
            cv2.rectangle(out, (x1, ty_text - th - 6),
                          (x1 + tw + 6, ty_text + 4), color, -1)
            cv2.putText(out, text, (x1 + 3, ty_text),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), text_thick,
                        cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# CSV row assembly
# ---------------------------------------------------------------------------
CSV_HEADER = [
    "source", "frame_idx", "boll_idx", "track_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "det_conf",
    "tip_x", "tip_y", "tip_conf",
    "base_x", "base_y", "base_conf",
    "occ_class", "occ_class_conf", "visfrac",
]


def _csv_rows_for(source: str, frame_idx: int, per_boll: list) -> list:
    rows = []
    for i, r in enumerate(per_boll):
        bx, by, bw, bh = r["bbox"]
        kpts = r.get("kpts") or []
        tip = kpts[0] if len(kpts) > 0 else (None, None, None)
        base = kpts[1] if len(kpts) > 1 else (None, None, None)
        rows.append({
            "source": source,
            "frame_idx": frame_idx,
            "boll_idx": i,
            "track_id": r.get("track_id"),
            "bbox_x": round(bx, 2), "bbox_y": round(by, 2),
            "bbox_w": round(bw, 2), "bbox_h": round(bh, 2),
            "det_conf": round(float(r.get("det_conf", 0.0)), 4),
            "tip_x": tip[0], "tip_y": tip[1], "tip_conf": tip[2],
            "base_x": base[0], "base_y": base[1], "base_conf": base[2],
            "occ_class": r.get("occ_class"),
            "occ_class_conf": r.get("occ_class_conf"),
            "visfrac": r.get("visfrac"),
        })
    return rows


# ---------------------------------------------------------------------------
# Source enumeration
# ---------------------------------------------------------------------------
def _classify_path(p: Path) -> Optional[str]:
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def _iter_inputs(source: Path) -> Iterable[tuple]:
    """Yield (relative_path_from_source, kind) tuples."""
    if source.is_file():
        kind = _classify_path(source)
        if kind:
            yield Path(source.name), kind
        return
    if source.is_dir():
        # Walk recursively, preserving relative paths
        for p in sorted(source.rglob("*")):
            if not p.is_file():
                continue
            kind = _classify_path(p)
            if kind:
                yield p.relative_to(source), kind
        return
    raise FileNotFoundError(source)


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------
def process_image(infer: CombinedInfer, src_path: Path, out_path: Path,
                  args, csv_writer=None, source_label: Optional[str] = None) -> int:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[skip] could not read {src_path}", file=sys.stderr)
        return 0
    per_boll = infer(img)
    annotated = render(img, per_boll,
                       draw_mask=not args.no_mask,
                       draw_kpts=not args.no_kpts,
                       alpha=args.mask_alpha,
                       kpt_conf_min=args.kpt_conf_min,
                       draw_track_id=not args.no_track_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    if args.save_json:
        Path(out_path).with_suffix(".json").write_text(
            json.dumps([_strip_for_json(r) for r in per_boll], indent=2))

    if csv_writer is not None:
        for row in _csv_rows_for(source_label or str(src_path), 0, per_boll):
            csv_writer.writerow(row)

    return len(per_boll)


def process_video(infer: CombinedInfer, src_path: Path, out_path: Path,
                  args, csv_writer=None, source_label: Optional[str] = None) -> int:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[skip] could not open {src_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path.with_suffix(".mp4")), fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"[skip] could not open writer for {out_path}", file=sys.stderr)
        cap.release()
        return 0

    label = source_label or str(src_path)
    total_dets = 0
    frame_idx = 0
    json_records = [] if args.save_json else None
    t0 = time.time()
    last_print = t0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        per_boll = infer(frame)
        annotated = render(frame, per_boll,
                           draw_mask=not args.no_mask,
                           draw_kpts=not args.no_kpts,
                           alpha=args.mask_alpha)
        writer.write(annotated)
        if json_records is not None:
            json_records.append({
                "frame": frame_idx,
                "bolls": [_strip_for_json(r) for r in per_boll],
            })
        if csv_writer is not None:
            for row in _csv_rows_for(label, frame_idx, per_boll):
                csv_writer.writerow(row)
        total_dets += len(per_boll)
        frame_idx += 1

        # Progress every 1s
        now = time.time()
        if now - last_print > 1.0:
            speed = frame_idx / (now - t0 + 1e-9)
            done = f"{frame_idx}/{n_frames}" if n_frames > 0 else f"{frame_idx}"
            print(f"  [{label}] frame {done}  {speed:.2f} fps  {total_dets} dets so far",
                  end="\r", flush=True)
            last_print = now

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"  [{label}] done: {frame_idx} frames, {total_dets} total dets, "
          f"{frame_idx/elapsed:.2f} fps, {elapsed:.1f}s")

    if json_records is not None:
        Path(out_path).with_suffix(".json").write_text(
            json.dumps(json_records, indent=2))

    return total_dets


def process_image_folder_as_video(infer: CombinedInfer,
                                  frames: list,
                                  out_path: Path, fps: float,
                                  args, csv_writer=None,
                                  source_label: Optional[str] = None) -> int:
    """Compose a folder's worth of (already enumerated) image frames into one
    annotated video. `frames` is a list of (rel_path, abs_path) tuples in the
    desired play order."""
    if not frames:
        return 0
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        return 0
    H, W = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    label = source_label or "frames"
    total_dets = 0
    json_records = [] if args.save_json else None
    t0 = time.time()
    last_print = t0
    for frame_idx, (rel, abs_) in enumerate(frames):
        img = cv2.imread(str(abs_), cv2.IMREAD_COLOR)
        if img is None:
            continue
        per_boll = infer(img)
        annotated = render(img, per_boll,
                           draw_mask=not args.no_mask,
                           draw_kpts=not args.no_kpts,
                           alpha=args.mask_alpha)
        writer.write(annotated)
        if json_records is not None:
            json_records.append({
                "frame": frame_idx, "src": str(rel),
                "bolls": [_strip_for_json(r) for r in per_boll],
            })
        if csv_writer is not None:
            for row in _csv_rows_for(label, frame_idx, per_boll):
                csv_writer.writerow(row)
        total_dets += len(per_boll)
        now = time.time()
        if now - last_print > 1.0:
            speed = (frame_idx + 1) / (now - t0 + 1e-9)
            print(f"  [{label}] frame {frame_idx+1}/{len(frames)}  "
                  f"{speed:.2f} fps  {total_dets} dets so far",
                  end="\r", flush=True)
            last_print = now
    writer.release()
    elapsed = time.time() - t0
    print(f"  [{label}] composed video: {len(frames)} frames, "
          f"{total_dets} dets, {len(frames)/elapsed:.2f} fps, {elapsed:.1f}s")
    if json_records is not None:
        out_path.with_suffix(".json").write_text(
            json.dumps(json_records, indent=2))
    return total_dets


def _strip_for_json(r: dict) -> dict:
    """Make a per-boll record JSON-safe (mask_poly is already plain list)."""
    return dict(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # Models
    ap.add_argument("--seg", required=True, help="YOLOv11-seg best.pt")
    ap.add_argument("--pose", default=None, help="YOLOv11-pose best.pt (optional)")
    ap.add_argument("--aux", default=None, help="AuxHead best.pt (optional)")
    # Source / output
    ap.add_argument("--source", required=True,
                    help="Image, video, or directory.")
    ap.add_argument("--out", required=True,
                    help="Output directory.")
    # Inference settings
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5,
                    help="IoU threshold for matching seg boxes to pose boxes.")
    # Visualization toggles
    ap.add_argument("--no-mask", action="store_true",
                    help="Skip mask polygon overlay.")
    ap.add_argument("--no-kpts", action="store_true",
                    help="Skip tip/base keypoints + axis line.")
    ap.add_argument("--mask-alpha", type=float, default=0.35,
                    help="Mask fill alpha (0=transparent, 1=opaque).")
    ap.add_argument("--kpt-conf-min", type=float, default=0.30,
                    help="Per-keypoint conf threshold for solid vs hollow dot.")
    ap.add_argument("--no-track-id", action="store_true",
                    help="Hide the #<track_id> chip on each rendered bbox.")
    # Speed / runtime
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="Force FP32 inference (default is FP16 on CUDA, ~1.5-2x faster).")
    ap.set_defaults(half=True)
    ap.add_argument("--profile", action="store_true",
                    help="Print per-stage timing (decode/yolo_seg/yolo_pose/aux_head/assemble).")
    # Tracking
    ap.add_argument("--no-track", dest="track", action="store_false",
                    help="Disable ByteTrack/BoT-SORT tracking on the seg stream.")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config (bytetrack.yaml or botsort.yaml).")
    ap.set_defaults(track=None)  # resolved per-source below (off for single image)
    # Output toggles
    ap.add_argument("--save-json", action="store_true",
                    help="Also write per-source <stem>.json with structured records.")
    ap.add_argument("--save-csv", action="store_true",
                    help="Also write <out>/results.csv with one row per detection.")
    # Image-folder -> composed video
    ap.add_argument("--video-out", action="store_true",
                    help="If --source is a directory of images, also compose a "
                         "single annotated video at <out>/<dirname>.mp4.")
    ap.add_argument("--video-fps", type=float, default=30.0)
    args = ap.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide tracking default: on for video/folder, off for single-image source.
    if args.track is None:
        args.track = source.is_dir() or _classify_path(source) == "video"

    # Build the inference pipeline once.
    print(f"[load] seg={args.seg}  half={args.half}  track={args.track}"
          + (f"  tracker={args.tracker}" if args.track else ""))
    if args.pose:  print(f"[load] pose={args.pose}")
    if args.aux:   print(f"[load] aux={args.aux}")
    infer = CombinedInfer(
        seg_pt=args.seg, pose_pt=args.pose, aux_pt=args.aux,
        device=args.device, imgsz=args.imgsz,
        conf=args.conf, iou_match=args.iou_match,
        half=args.half, track=args.track, tracker=args.tracker,
        profile=args.profile,
    )

    # CSV setup
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = out_dir / "results.csv"
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        csv_writer.writeheader()
        print(f"[csv ] {csv_path}")

    # Enumerate inputs
    inputs = list(_iter_inputs(source))
    if not inputs:
        print(f"[err ] no images or videos found at {source}", file=sys.stderr)
        if csv_file: csv_file.close()
        sys.exit(2)

    print(f"[scan] {len(inputs)} input(s) under {source}")

    # Stash image-folder frames separately so we can optionally compose them
    # into a single video at the end.
    image_frames_for_video = []
    total_dets = 0
    n_processed = 0
    t0 = time.time()
    last_print = t0

    for rel, kind in inputs:
        src_path = source / rel if source.is_dir() else source
        if kind == "image":
            out_path = out_dir / rel if source.is_dir() else out_dir / rel.name
            # Ensure annotated image keeps original suffix
            out_path = out_path.with_suffix(rel.suffix)
            n = process_image(infer, src_path, out_path, args,
                              csv_writer=csv_writer,
                              source_label=str(rel))
            total_dets += n
            n_processed += 1
            if args.video_out and source.is_dir():
                image_frames_for_video.append((rel, src_path))
            # Progress (image-folder mode is otherwise silent).
            now = time.time()
            if now - last_print > 1.0 or n_processed == len(inputs):
                speed = n_processed / (now - t0 + 1e-9)
                eta_s = (len(inputs) - n_processed) / max(speed, 1e-9)
                print(f"  [{n_processed}/{len(inputs)}]  {speed:5.2f} fps  "
                      f"{total_dets} dets  eta {eta_s/60:5.1f} min",
                      end="\r", flush=True)
                last_print = now
        elif kind == "video":
            # Reset tracker state at the start of each video so IDs don't
            # carry across distinct scenes.
            if args.track:
                infer.reset_tracker()
            out_path = (out_dir / rel).with_suffix(".mp4") if source.is_dir() \
                else out_dir / (rel.stem + ".mp4")
            n = process_video(infer, src_path, out_path, args,
                              csv_writer=csv_writer,
                              source_label=str(rel))
            total_dets += n
            n_processed += 1

    # Compose folder-of-images into one video if requested.
    if args.video_out and image_frames_for_video and source.is_dir():
        # Order by source-relative path for deterministic playback.
        image_frames_for_video.sort(key=lambda t: str(t[0]))
        composed_out = out_dir / f"{source.name}_composed.mp4"
        # Reset tracker so the composed pass starts fresh (the per-image
        # pass above already ran tracking through these frames in order,
        # so its IDs are already committed in the JSONs).
        if args.track:
            infer.reset_tracker()
        print(f"[video] composing {len(image_frames_for_video)} frames -> {composed_out}")
        # Re-runs inference on each frame so the composed video matches
        # the per-frame annotated images. Slightly redundant compute,
        # but keeps annotation deterministic and avoids loading all
        # frames into memory.
        process_image_folder_as_video(
            infer, image_frames_for_video, composed_out,
            fps=args.video_fps, args=args,
            csv_writer=None,                # already wrote CSV in image pass
            source_label=f"{source.name}_composed",
        )

    if csv_file:
        csv_file.close()

    # Finish the progress carriage-return line with a newline.
    print("")
    print(f"[done] processed {n_processed} input(s), {total_dets} total detections, "
          f"out -> {out_dir}")

    if args.profile:
        print("\n[profile]")
        print(infer.profile_dump())


if __name__ == "__main__":
    main()
