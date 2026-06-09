"""
RGBD Recording Analyzer — Cotton Boll Size Measurement
=======================================================
Replays a .bag file recorded with record_rgbd.py (or any RealSense
.bag recording) and runs a segmentation model on each frame to
measure cotton boll dimensions using the preserved depth data.

This is a TEMPLATE — plug in your segmentation model when ready.

Requirements:
    pip install pyrealsense2 opencv-python numpy
    pip install ultralytics  (when you have the model)

Usage:
    python analyze_recording.py --bag greenhouse_session.bag --model cotton_boll_seg.pt
    python analyze_recording.py --bag recording.bag --model best.pt --output results.csv
    python analyze_recording.py --bag recording.bag --model best.pt --visualize

What's preserved in the .bag file:
    - Raw 16-bit depth (Z16) with full metric precision
    - Color frames (BGR8) synchronized with depth
    - Camera intrinsics (fx, fy, ppx, ppy) and depth scale
    - Original timestamps
"""

import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2")


# ── Depth & Measurement Helpers (from your original script) ──────────────


def get_distance_at_mask(
    depth_frame,
    mask: np.ndarray,
    depth_scale: float,
    percentile: float = 50,
) -> float:
    """Compute distance (meters) using the median depth under a binary mask."""
    depth_image = np.asanyarray(depth_frame.get_data())
    masked_depth = depth_image[mask > 0]
    valid = masked_depth[masked_depth > 0]

    if len(valid) == 0:
        return -1.0

    return round(float(np.percentile(valid, percentile)) * depth_scale, 3)


def get_distance_at_point(
    depth_frame,
    cx: int,
    cy: int,
    depth_scale: float,
    kernel: int = 5,
) -> float:
    """Fallback: depth at a point averaged over a small kernel."""
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    half = kernel // 2
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    region = depth_image[y1:y2, x1:x2]
    valid = region[region > 0]
    if len(valid) == 0:
        return -1.0
    return round(float(np.median(valid)) * depth_scale, 3)


def compute_real_area(
    mask: np.ndarray,
    distance_m: float,
    depth_intrinsics,
) -> float:
    """Compute real-world area (cm²) of a segmented region at a given depth."""
    if distance_m <= 0:
        return -1.0
    pixel_count = int(np.sum(mask > 0))
    if pixel_count == 0:
        return -1.0
    pixel_width_m = distance_m / depth_intrinsics.fx
    pixel_height_m = distance_m / depth_intrinsics.fy
    total_area_m2 = pixel_count * pixel_width_m * pixel_height_m
    return round(total_area_m2 * 1e4, 2)


def compute_real_dimensions(
    mask: np.ndarray,
    distance_m: float,
    depth_intrinsics,
) -> dict:
    """
    Compute real-world bounding dimensions of a segmented object.

    Returns dict with:
        - height_cm: vertical extent of the mask in cm
        - width_cm:  horizontal extent of the mask in cm
        - area_cm2:  total projected area in cm²

    For cotton bolls, 'height_cm' is the measurement you likely want
    (total height of the boll).
    """
    if distance_m <= 0:
        return {"height_cm": -1.0, "width_cm": -1.0, "area_cm2": -1.0}

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {"height_cm": -1.0, "width_cm": -1.0, "area_cm2": -1.0}

    # Pixel extents
    pixel_height = int(ys.max() - ys.min()) + 1
    pixel_width = int(xs.max() - xs.min()) + 1

    # Convert to real-world units
    m_per_pixel_x = distance_m / depth_intrinsics.fx
    m_per_pixel_y = distance_m / depth_intrinsics.fy

    height_cm = round(pixel_height * m_per_pixel_y * 100, 2)
    width_cm = round(pixel_width * m_per_pixel_x * 100, 2)
    area_cm2 = compute_real_area(mask, distance_m, depth_intrinsics)

    return {
        "height_cm": height_cm,
        "width_cm": width_cm,
        "area_cm2": area_cm2,
    }


# ── Replay & Analysis ────────────────────────────────────────────────────


def replay_bag(args):
    """Replay a .bag file, run segmentation, and measure cotton bolls."""

    if not os.path.isfile(args.bag):
        sys.exit(f"ERROR: Bag file not found: {args.bag}")

    # ── Load segmentation model ──────────────────────────────────────────
    model = None
    if args.model:
        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading model: {args.model}")
            model = YOLO(args.model)
            print("[INFO] Model loaded.")
        except ImportError:
            sys.exit("ERROR: ultralytics not found. Install with: pip install ultralytics")
    else:
        print("[INFO] No model specified — running in preview-only mode.")
        print("       Use --model path/to/model.pt to enable analysis.")

    # ── Configure pipeline for .bag playback ─────────────────────────────
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(args.bag, repeat_playback=False)

    profile = pipeline.start(config)

    # Get the playback device and disable real-time to process every frame
    playback = profile.get_device().as_playback()
    playback.set_real_time(args.realtime)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Retrieve depth scale and intrinsics from the recorded stream
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrinsics = depth_stream.get_intrinsics()

    print(f"[INFO] Replaying: {args.bag}")
    print(f"[INFO] Depth scale: {depth_scale:.6f} m/unit")
    print(f"[INFO] Intrinsics: fx={depth_intrinsics.fx:.1f}, fy={depth_intrinsics.fy:.1f}")

    # ── Optional depth filters ───────────────────────────────────────────
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # ── CSV output setup ─────────────────────────────────────────────────
    csv_file = None
    csv_writer = None
    if args.output:
        csv_file = open(args.output, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "detection_id", "class", "confidence",
            "distance_m", "height_cm", "width_cm", "area_cm2",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ])
        print(f"[INFO] Results will be saved to: {args.output}")

    # ── Main replay loop ─────────────────────────────────────────────────
    frame_idx = 0
    total_detections = 0

    print("[INFO] Press 'q' to stop  |  Space to pause/resume")
    print()

    paused = False

    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # End of bag file
                print(f"\n[INFO] End of recording reached at frame {frame_idx}.")
                break

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Apply depth filters
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            frame_idx += 1

            # ── Run segmentation (if model loaded) ───────────────────────
            if model is not None:
                results = model(color_image, conf=args.conf, verbose=False)
                result = results[0]

                if result.masks is not None and len(result.boxes) > 0:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        conf = float(confs[i])
                        cls_id = int(classes[i])
                        cls_name = model.names.get(cls_id, str(cls_id))

                        # Resize mask to frame dimensions
                        mask = cv2.resize(
                            masks[i],
                            (color_image.shape[1], color_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        binary_mask = (mask > 0.5).astype(np.uint8)

                        # Distance
                        dist = get_distance_at_mask(
                            depth_frame, binary_mask, depth_scale
                        )
                        if dist < 0:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            dist = get_distance_at_point(
                                depth_frame, cx, cy, depth_scale
                            )

                        # Dimensions
                        dims = compute_real_dimensions(
                            binary_mask, dist, depth_intrinsics
                        )

                        total_detections += 1

                        # Log to console
                        print(
                            f"  Frame {frame_idx:>5d} | {cls_name}: "
                            f"dist={dist:.3f}m  "
                            f"H={dims['height_cm']:.1f}cm  "
                            f"W={dims['width_cm']:.1f}cm  "
                            f"area={dims['area_cm2']:.1f}cm²  "
                            f"conf={conf:.0%}"
                        )

                        # Write to CSV
                        if csv_writer:
                            csv_writer.writerow([
                                frame_idx, total_detections, cls_name, f"{conf:.3f}",
                                dist, dims["height_cm"], dims["width_cm"],
                                dims["area_cm2"],
                                x1, y1, x2, y2,
                            ])

                        # ── Visualization ────────────────────────────────
                        if args.visualize:
                            color = (0, 255, 0)
                            # Mask overlay
                            overlay = display.copy()
                            overlay[binary_mask > 0] = color
                            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

                            # Bounding box
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                            # Labels
                            label = f"{cls_name} {dist:.2f}m ({conf:.0%})"
                            cv2.putText(
                                display, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                            )
                            dim_label = f"H:{dims['height_cm']:.1f}cm W:{dims['width_cm']:.1f}cm"
                            cv2.putText(
                                display, dim_label, (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                            )

            # ── Frame counter on display ─────────────────────────────────
            cv2.putText(
                display, f"Frame: {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            if args.visualize or model is None:
                cv2.imshow("RGBD Replay", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = True

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

    print(f"\n[SUMMARY] Processed {frame_idx} frames, {total_detections} total detections.")
    if args.output:
        print(f"[SUMMARY] Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay RGBD .bag recording and analyze with segmentation model"
    )
    parser.add_argument(
        "--bag", type=str, required=True,
        help="Path to the .bag recording file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to segmentation model (.pt). Omit for preview-only mode.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold for detections (default: 0.4)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output CSV file for measurements",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show detections overlaid on video during replay",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Replay at original speed (default: process as fast as possible)",
    )
    args = parser.parse_args()

    replay_bag(args)


if __name__ == "__main__":
    main()