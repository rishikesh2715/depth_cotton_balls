"""
Tennis Ball Distance Detection
==============================
Uses YOLOv11 instance segmentation + Intel RealSense depth camera
to detect tennis balls and measure their real-world distance,
surface area, and estimated volume.

Requirements:
    pip install ultralytics pyrealsense2 opencv-python numpy

Usage:
    python tennis_ball_distance.py --model path/to/your/model.pt
    python tennis_ball_distance.py --model best.pt --conf 0.5
    python tennis_ball_distance.py --model best.pt --show-mask --depth-colormap
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit(
        "ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2"
    )

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "ERROR: ultralytics not found. Install with: pip install ultralytics"
    )


# ── RealSense Pipeline Setup ────────────────────────────────────────────────

def create_realsense_pipeline(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> tuple:
    """
    Initialize the RealSense pipeline for aligned color + depth streams.

    Returns:
        (pipeline, align, depth_scale, depth_intrinsics)
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Check that a device is connected
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit("ERROR: No Intel RealSense device detected.")

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # Align depth frames to the color frame
    align = rs.align(rs.stream.color)

    # Depth scale: converts raw depth units → meters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Optional: set high-accuracy preset if available
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy

    # Intrinsics (useful if you want 3-D coordinates later)
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrinsics = depth_stream.get_intrinsics()

    print(f"[INFO] RealSense started  |  depth scale = {depth_scale:.6f} m/unit")
    print(f"[INFO] Resolution: {width}x{height} @ {fps} FPS")
    print(f"[INFO] Intrinsics: fx={depth_intrinsics.fx:.1f}, fy={depth_intrinsics.fy:.1f}")

    return pipeline, align, depth_scale, depth_intrinsics


# ── Depth Helpers ────────────────────────────────────────────────────────────

def get_distance_at_mask(
    depth_frame: rs.depth_frame,
    mask: np.ndarray,
    depth_scale: float,
    percentile: float = 50,
) -> float:
    """
    Compute the distance (in meters) of an object defined by a binary mask,
    using the median (or chosen percentile) of valid depth pixels under the mask.

    This is more robust than a single-pixel lookup because it averages over
    the whole segmented region and ignores zero (invalid) readings.
    """
    depth_image = np.asanyarray(depth_frame.get_data())

    # Extract depth values inside the mask
    masked_depth = depth_image[mask > 0]

    # Filter out zero / invalid readings
    valid = masked_depth[masked_depth > 0]

    if len(valid) == 0:
        return -1.0  # no valid depth

    distance_m = np.percentile(valid, percentile) * depth_scale
    return round(distance_m, 3)


def get_distance_at_point(
    depth_frame: rs.depth_frame,
    cx: int,
    cy: int,
    depth_scale: float,
    kernel: int = 5,
) -> float:
    """
    Fallback: get depth at a single point (center of bounding box),
    averaged over a small kernel to reduce noise.
    """
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


def pixel_to_3d(
    depth_frame: rs.depth_frame,
    cx: int,
    cy: int,
    depth_intrinsics,
    depth_scale: float,
) -> tuple:
    """
    Convert a 2-D pixel + depth into a 3-D point (X, Y, Z) in meters,
    using the camera intrinsics.  Useful for downstream robotics / tracking.
    """
    depth_value = depth_frame.get_distance(cx, cy)
    if depth_value <= 0:
        return (0.0, 0.0, 0.0)
    point_3d = rs.rs2_deproject_pixel_to_point(
        depth_intrinsics, [cx, cy], depth_value
    )
    return tuple(round(v, 3) for v in point_3d)


# ── Area & Volume Helpers ────────────────────────────────────────────────────

def compute_real_area(
    mask: np.ndarray,
    distance_m: float,
    depth_intrinsics,
) -> float:
    """
    Compute the real-world area (in cm²) of a segmented region.

    Each pixel at distance Z covers a real-world area of:
        pixel_area_m² = (Z / fx) * (Z / fy)

    where fx, fy are the focal lengths in pixels. Summing over all mask
    pixels gives the total projected area.

    For a sphere like a tennis ball, this gives the visible cross-sectional
    area (i.e. the area of the circular disc you see from the camera).
    """
    if distance_m <= 0:
        return -1.0

    pixel_count = int(np.sum(mask > 0))
    if pixel_count == 0:
        return -1.0

    # Real-world size of one pixel at this depth
    pixel_width_m = distance_m / depth_intrinsics.fx
    pixel_height_m = distance_m / depth_intrinsics.fy
    pixel_area_m2 = pixel_width_m * pixel_height_m

    total_area_m2 = pixel_count * pixel_area_m2
    total_area_cm2 = total_area_m2 * 1e4  # m² → cm²

    return round(total_area_cm2, 2)


def compute_volume_from_area(area_cm2: float) -> float:
    """
    Estimate the volume (in cm³) of a tennis ball assuming it's a sphere.

    The segmentation mask captures the visible cross-section, which for a
    sphere viewed head-on is a circle of area π·r². We solve for r, then
    compute V = (4/3)·π·r³.

    This is a good approximation when the ball faces the camera roughly
    head-on, which is typical for a sphere from any angle.
    """
    if area_cm2 <= 0:
        return -1.0

    # Cross-sectional area of a sphere = π·r²  →  r = sqrt(A / π)
    radius_cm = math.sqrt(area_cm2 / math.pi)
    volume_cm3 = (4.0 / 3.0) * math.pi * (radius_cm ** 3)

    return round(volume_cm3, 2)


def compute_diameter_from_area(area_cm2: float) -> float:
    """Derive the estimated ball diameter (cm) from cross-sectional area."""
    if area_cm2 <= 0:
        return -1.0
    radius_cm = math.sqrt(area_cm2 / math.pi)
    return round(radius_cm * 2, 2)


# ── Visualization ────────────────────────────────────────────────────────────

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)


def distance_color(dist_m: float) -> tuple:
    """Color-code by distance: green < 1 m, yellow 1-3 m, red > 3 m."""
    if dist_m < 1.0:
        return COLOR_GREEN
    elif dist_m < 3.0:
        return COLOR_YELLOW
    return COLOR_RED


def draw_detection(
    frame: np.ndarray,
    bbox: list[int],
    distance_m: float,
    conf: float,
    area_cm2: float = -1.0,
    volume_cm3: float = -1.0,
    diameter_cm: float = -1.0,
    mask: np.ndarray | None = None,
    show_mask: bool = True,
    point_3d: tuple | None = None,
) -> None:
    """Draw bounding box, optional mask overlay, distance, area, and volume."""
    x1, y1, x2, y2 = bbox
    color = distance_color(distance_m) if distance_m > 0 else COLOR_WHITE

    # Mask overlay
    if show_mask and mask is not None:
        overlay = frame.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── Primary label: distance + confidence ─────────────────────────────
    if distance_m > 0:
        label = f"Tennis Ball  {distance_m:.2f} m  ({conf:.0%})"
    else:
        label = f"Tennis Ball  N/A  ({conf:.0%})"

    # Background rectangle for text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
    )

    # ── Secondary labels below the bounding box ─────────────────────────
    line_y = y2 + 18
    line_spacing = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1

    # Area
    if area_cm2 > 0:
        area_text = f"Area: {area_cm2:.1f} cm2"
        cv2.putText(frame, area_text, (x1, line_y), font, font_scale, COLOR_CYAN, thickness)
        line_y += line_spacing

    # Volume
    if volume_cm3 > 0:
        vol_text = f"Vol: {volume_cm3:.1f} cm3"
        cv2.putText(frame, vol_text, (x1, line_y), font, font_scale, COLOR_CYAN, thickness)
        line_y += line_spacing

    # Diameter
    if diameter_cm > 0:
        diam_text = f"Diam: {diameter_cm:.1f} cm"
        cv2.putText(frame, diam_text, (x1, line_y), font, font_scale, COLOR_CYAN, thickness)
        line_y += line_spacing

    # Optional 3-D coordinates
    if point_3d and any(v != 0 for v in point_3d):
        coord_text = f"XYZ: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
        cv2.putText(frame, coord_text, (x1, line_y), font, font_scale, color, thickness)


def build_depth_colormap(depth_frame) -> np.ndarray:
    """Generate a JET colormap from a depth frame."""
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET,
    )
    return depth_colormap


# ── Main Loop ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tennis ball distance detection with YOLOv11 + RealSense"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to your YOLOv11 instance-segmentation model (.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Stream width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Stream height (default: 480)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Stream FPS (default: 30)",
    )
    parser.add_argument(
        "--show-mask", action="store_true",
        help="Overlay the segmentation mask on the color image",
    )
    parser.add_argument(
        "--show-3d", action="store_true",
        help="Show 3-D (X, Y, Z) coordinates for each detection",
    )
    parser.add_argument(
        "--depth-colormap", action="store_true",
        help="Show the depth colormap side-by-side",
    )
    parser.add_argument(
        "--max-dist", type=float, default=10.0,
        help="Ignore detections beyond this distance in meters (default: 10)",
    )
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print("[INFO] Model loaded successfully.")

    # ── Start RealSense ──────────────────────────────────────────────────
    pipeline, align, depth_scale, depth_intrinsics = create_realsense_pipeline(
        args.width, args.height, args.fps
    )

    # Temporal filter smooths depth across frames (reduces flicker)
    temporal_filter = rs.temporal_filter()
    spatial_filter = rs.spatial_filter()
    hole_filling = rs.hole_filling_filter()

    print("[INFO] Press 'q' to quit  |  's' to save a screenshot (color + depth)")

    fps_time = time.time()
    frame_count = 0

    # Keep a reference to the latest depth frame for screenshots
    latest_depth_colormap = None

    try:
        while True:
            # ── Grab frames ──────────────────────────────────────────────
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Apply depth post-processing filters
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            # Always build depth colormap (needed for screenshots)
            latest_depth_colormap = build_depth_colormap(depth_frame)

            # ── Run YOLO inference ───────────────────────────────────────
            results = model.track(
                color_image,
                conf=args.conf,
                verbose=False,
            )

            result = results[0]

            # ── Process detections ───────────────────────────────────────
            if result.masks is not None and len(result.boxes) > 0:
                masks = result.masks.data.cpu().numpy()        # (N, H, W)
                boxes = result.boxes.xyxy.cpu().numpy()        # (N, 4)
                confs = result.boxes.conf.cpu().numpy()        # (N,)
                classes = result.boxes.cls.cpu().numpy()        # (N,)

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    conf = float(confs[i])

                    # Resize mask to match the color frame
                    mask = cv2.resize(
                        masks[i],
                        (color_image.shape[1], color_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # ── Distance from segmentation mask ──────────────────
                    dist = get_distance_at_mask(
                        depth_frame, binary_mask, depth_scale
                    )

                    # Fallback to center-point if mask depth failed
                    if dist < 0:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        dist = get_distance_at_point(
                            depth_frame, cx, cy, depth_scale
                        )

                    # Skip if beyond max distance
                    if 0 < dist > args.max_dist:
                        continue

                    # ── Area & Volume ────────────────────────────────────
                    area_cm2 = compute_real_area(
                        binary_mask, dist, depth_intrinsics
                    )
                    volume_cm3 = compute_volume_from_area(area_cm2)
                    diameter_cm = compute_diameter_from_area(area_cm2)

                    # Print to console for logging
                    if dist > 0 and area_cm2 > 0:
                        print(
                            f"  Ball: dist={dist:.3f}m  "
                            f"area={area_cm2:.1f}cm²  "
                            f"vol={volume_cm3:.1f}cm³  "
                            f"diam={diameter_cm:.1f}cm  "
                            f"conf={conf:.0%}"
                        )

                    # Optional 3-D point
                    pt3d = None
                    if args.show_3d:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        pt3d = pixel_to_3d(
                            depth_frame, cx, cy,
                            depth_intrinsics, depth_scale,
                        )

                    draw_detection(
                        display,
                        [x1, y1, x2, y2],
                        dist, conf,
                        area_cm2=area_cm2,
                        volume_cm3=volume_cm3,
                        diameter_cm=diameter_cm,
                        mask=binary_mask,
                        show_mask=args.show_mask,
                        point_3d=pt3d,
                    )

            # ── FPS counter ──────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
            else:
                fps = frame_count / max(elapsed, 1e-6)

            cv2.putText(
                display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2,
            )

            # ── Display ─────────────────────────────────────────────────
            if args.depth_colormap:
                combined = np.hstack((display, latest_depth_colormap))
                cv2.imshow("Tennis Ball Detection + Depth", combined)
            else:
                cv2.imshow("Tennis Ball Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting...")
                break
            elif key == ord("s"):
                timestamp = int(time.time())

                # Always save both color+detections AND depth colormap
                # as a combined side-by-side image
                combined_save = np.hstack((display, latest_depth_colormap))
                combined_fname = f"screenshot_{timestamp}_combined.png"
                cv2.imwrite(combined_fname, combined_save)

                # Also save them individually for convenience
                color_fname = f"screenshot_{timestamp}_color.png"
                depth_fname = f"screenshot_{timestamp}_depth.png"
                cv2.imwrite(color_fname, display)
                cv2.imwrite(depth_fname, latest_depth_colormap)

                print(f"[INFO] Screenshots saved:")
                print(f"       Combined : {combined_fname}")
                print(f"       Color    : {color_fname}")
                print(f"       Depth    : {depth_fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped. Goodbye!")


if __name__ == "__main__":
    main()