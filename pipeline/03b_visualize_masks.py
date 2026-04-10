"""
Stage 3.5 — Overlay SAM 2 masks + boll IDs on the extracted frames,
             write to an MP4 for the demo / meeting
====================================================================
Walks every frame in <work>/frames/, looks for matching masks in
<work>/masks/<frame_idx>/<boll_id>.png, composites them with
transparency + a colored outline + the boll ID label, and writes
the result to an MP4 video.

Each boll gets a distinct color (cycled from the tab10 palette).
The same boll keeps the same color across all frames.

Usage:
    python 03b_visualize_masks.py --work work/handheld_1
    python 03b_visualize_masks.py --work work/handheld_1 --fps 10
    python 03b_visualize_masks.py --work work/handheld_1 --output demo.mp4
    python 03b_visualize_masks.py --work work/handheld_1 --only-masked
        (skip frames that have no masks)
"""

import argparse
import os
import sys

import cv2
import numpy as np


# Distinct colors for up to 20 bolls, will cycle after that.
# BGR format for OpenCV.
PALETTE = [
    (255,  64,  64),  # blue
    ( 64, 255,  64),  # green
    ( 64,  64, 255),  # red
    ( 64, 255, 255),  # yellow
    (255, 255,  64),  # cyan
    (255,  64, 255),  # magenta
    (128, 200, 255),  # orange-ish
    (200, 128, 255),  # pink-ish
    (255, 200, 128),  # light blue
    (128, 255, 200),  # mint
    ( 64, 128, 255),  # orange
    (255, 128,  64),  # sky
    (200,  64, 128),  # purple
    ( 64, 200, 128),  # lime
    (128,  64, 200),  # plum
    (200, 255,  64),  # aqua
    ( 64, 200, 255),  # amber
    (255,  64, 128),  # rose
    (128, 255,  64),  # spring green
    ( 64, 255, 200),  # turquoise
]


def color_for_boll(boll_id: int) -> tuple[int, int, int]:
    return PALETTE[boll_id % len(PALETTE)]


def overlay_masks_on_frame(
    frame: np.ndarray,
    mask_files: list[tuple[int, str]],  # (boll_id, mask_path)
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Return a new image with translucent colored fills + solid outlines
    + boll-ID labels for each mask in mask_files.
    """
    disp = frame.copy()
    overlay = frame.copy()

    # Paint translucent fills first so the outlines can draw on top
    for boll_id, mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        binary = mask > 0
        if not binary.any():
            continue
        color = color_for_boll(boll_id)
        overlay[binary] = color

    cv2.addWeighted(overlay, alpha, disp, 1.0 - alpha, 0, disp)

    # Now outlines + labels on the blended image
    for boll_id, mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if not (mask > 0).any():
            continue

        color = color_for_boll(boll_id)

        # Find contours for clean outlines
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(disp, contours, -1, color, 2)

        # Label at the centroid of the biggest contour
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback: top-left of contour bbox
                x, y, _, _ = cv2.boundingRect(largest)
                cx, cy = x, y

            label = f"#{boll_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2

            # Measure the text so we can draw a filled background box
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            pad = 4
            bx1 = cx - tw // 2 - pad
            by1 = cy - th // 2 - pad
            bx2 = cx + tw // 2 + pad
            by2 = cy + th // 2 + pad + baseline

            # Clip label box to image bounds
            h_img, w_img = disp.shape[:2]
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(w_img - 1, bx2); by2 = min(h_img - 1, by2)

            cv2.rectangle(disp, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, 1)
            cv2.putText(disp, label,
                        (cx - tw // 2, cy + th // 2),
                        font, scale, (255, 255, 255), thickness,
                        cv2.LINE_AA)

    return disp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Working dir from stage 1")
    p.add_argument("--output", default=None,
                   help="Output video path (default: <work>/overlay_demo.mp4)")
    p.add_argument("--fps", type=int, default=10,
                   help="Playback frame rate (default: 10)")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="Mask fill transparency 0..1 (default: 0.45)")
    p.add_argument("--only-masked", action="store_true",
                   help="Skip frames that have no masks at all")
    p.add_argument("--hud", action="store_true", default=True,
                   help="Show frame counter + boll count overlay (default: on)")
    args = p.parse_args()

    frames_dir = os.path.join(args.work, "frames")
    masks_root = os.path.join(args.work, "masks")

    if not os.path.isdir(frames_dir):
        sys.exit(f"ERROR: {frames_dir} not found")
    if not os.path.isdir(masks_root):
        sys.exit(f"ERROR: {masks_root} not found. Run stage 3 first.")

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not files:
        sys.exit("ERROR: no frames")

    out_path = args.output or os.path.join(args.work, "overlay_demo.mp4")

    # Probe first frame to get size
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]

    # mp4v is the most portable fourcc for MP4; works on Windows out of the box
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"ERROR: could not open video writer for {out_path}")

    print(f"[INFO] {len(files)} frames, output: {out_path}")
    print(f"[INFO] {w}x{h} @ {args.fps} fps")

    total_masked_frames = 0
    total_mask_instances = 0
    all_bolls_seen = set()

    for i, fname in enumerate(files):
        frame_idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))

        # Find masks for this frame (if any)
        frame_mask_dir = os.path.join(masks_root, f"{frame_idx:05d}")
        mask_files = []
        if os.path.isdir(frame_mask_dir):
            for mf in sorted(os.listdir(frame_mask_dir)):
                if mf.endswith(".png"):
                    try:
                        boll_id = int(os.path.splitext(mf)[0])
                    except ValueError:
                        continue
                    mask_files.append((boll_id, os.path.join(frame_mask_dir, mf)))

        if args.only_masked and not mask_files:
            continue

        if mask_files:
            total_masked_frames += 1
            total_mask_instances += len(mask_files)
            for bid, _ in mask_files:
                all_bolls_seen.add(bid)

        disp = overlay_masks_on_frame(frame, mask_files, alpha=args.alpha)

        # HUD: frame counter + boll count in top-left
        if args.hud:
            hud1 = f"Frame {frame_idx}"
            hud2 = f"{len(mask_files)} bolls visible"
            cv2.rectangle(disp, (0, 0), (260, 70), (0, 0, 0), -1)
            cv2.putText(disp, hud1, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, hud2, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(disp)

        if (i + 1) % 50 == 0:
            print(f"  ...wrote {i + 1}/{len(files)} frames")

    writer.release()

    print()
    print(f"[DONE] Wrote video: {out_path}")
    print(f"[STATS] {total_masked_frames} frames had masks, "
          f"{total_mask_instances} total mask instances")
    print(f"[STATS] Unique bolls seen: {sorted(all_bolls_seen)}")


if __name__ == "__main__":
    main()