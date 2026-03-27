"""
RGBD Video Recorder for Intel RealSense
========================================
Records synchronized color + depth streams to a .bag file,
preserving full 16-bit depth with metric precision and camera
intrinsics. The .bag can be replayed later through the same
RealSense pipeline API as if the camera were live.

Requirements:
    pip install pyrealsense2 opencv-python numpy

Usage:
    python record_rgbd.py
    python record_rgbd.py --width 1280 --height 720 --fps 30
    python record_rgbd.py --output greenhouse_session_1.bag
    python record_rgbd.py --high-accuracy --preview-depth

Controls (in preview window):
    r  — Start / stop recording  (toggles)
    s  — Save a snapshot (color + depth PNG)
    q  — Quit

The depth stream is saved as raw 16-bit Z16 format, so every pixel
retains its exact distance value.  To convert any pixel to meters:
    distance_m = pixel_value * depth_scale
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2")


def make_output_filename(prefix: str, extension: str) -> str:
    """Generate a timestamped filename."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension}"


def build_depth_colormap(depth_frame) -> np.ndarray:
    """Generate a JET colormap from a depth frame for preview only."""
    depth_image = np.asanyarray(depth_frame.get_data())
    return cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record RGBD video from Intel RealSense to .bag file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .bag filename (default: auto-timestamped)",
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
        "--high-accuracy", action="store_true",
        help="Enable high-accuracy depth preset",
    )
    parser.add_argument(
        "--preview-depth", action="store_true",
        help="Show depth colormap alongside color in the preview window",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory to save recordings and snapshots (default: current dir)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Start pipeline WITHOUT recording (live preview) ────────
    pipeline = rs.pipeline()
    config = rs.config()

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit("ERROR: No Intel RealSense device detected.")

    device_name = devices[0].get_info(rs.camera_info.name)
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"[INFO] Found device: {device_name} (SN: {serial})")

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    # Depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    if args.high_accuracy and depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)
        print("[INFO] High-accuracy depth preset enabled")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Intrinsics (print for reference)
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = depth_stream.get_intrinsics()
    print(f"[INFO] Depth scale: {depth_scale:.6f} m/unit")
    print(f"[INFO] Resolution: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"[INFO] Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
          f"ppx={intrinsics.ppx:.1f}, ppy={intrinsics.ppy:.1f}")
    print()
    print("[CONTROLS]  r = start/stop recording  |  s = snapshot  |  q = quit")
    print()

    # ── State ────────────────────────────────────────────────────────────
    recording = False
    recorder = None  # rs.recorder
    rec_pipeline = None
    rec_align = None
    frame_count = 0
    rec_frame_count = 0
    fps_time = time.time()
    display_fps = 0.0
    bag_path = None
    snapshot_count = 0

    try:
        while True:
            # ── Grab frames ──────────────────────────────────────────────
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            # ── FPS counter ──────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()

            # ── Recording indicator ──────────────────────────────────────
            if recording:
                rec_frame_count += 1
                # Red recording dot + frame count
                cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(
                    display, f"REC  {rec_frame_count} frames",
                    (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            else:
                cv2.putText(
                    display, "STANDBY - press 'r' to record",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
                )

            # FPS
            cv2.putText(
                display, f"FPS: {display_fps:.1f}",
                (args.width - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # ── Preview ──────────────────────────────────────────────────
            if args.preview_depth:
                depth_colormap = build_depth_colormap(depth_frame)
                combined = np.hstack((display, depth_colormap))
                cv2.imshow("RGBD Recorder", combined)
            else:
                cv2.imshow("RGBD Recorder", display)

            # ── Key handling ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Quitting...")
                break

            elif key == ord("r"):
                if not recording:
                    # ── Start recording ──────────────────────────────────
                    # Stop the current preview pipeline
                    pipeline.stop()

                    # Create a new pipeline with recording enabled
                    bag_path = args.output or os.path.join(
                        args.output_dir,
                        make_output_filename("rgbd_recording", "bag"),
                    )

                    rec_config = rs.config()
                    rec_config.enable_stream(
                        rs.stream.color, args.width, args.height,
                        rs.format.bgr8, args.fps,
                    )
                    rec_config.enable_stream(
                        rs.stream.depth, args.width, args.height,
                        rs.format.z16, args.fps,
                    )
                    rec_config.enable_record_to_file(bag_path)

                    pipeline = rs.pipeline()
                    profile = pipeline.start(rec_config)
                    align = rs.align(rs.stream.color)

                    # Re-apply high accuracy if needed
                    if args.high_accuracy:
                        ds = profile.get_device().first_depth_sensor()
                        if ds.supports(rs.option.visual_preset):
                            ds.set_option(rs.option.visual_preset, 3)

                    recording = True
                    rec_frame_count = 0
                    print(f"[REC] Recording started → {bag_path}")

                else:
                    # ── Stop recording ───────────────────────────────────
                    pipeline.stop()
                    recording = False
                    print(f"[REC] Recording stopped. {rec_frame_count} frames saved.")
                    print(f"      File: {bag_path}")

                    # Restart preview-only pipeline
                    preview_config = rs.config()
                    preview_config.enable_stream(
                        rs.stream.color, args.width, args.height,
                        rs.format.bgr8, args.fps,
                    )
                    preview_config.enable_stream(
                        rs.stream.depth, args.width, args.height,
                        rs.format.z16, args.fps,
                    )

                    pipeline = rs.pipeline()
                    profile = pipeline.start(preview_config)
                    align = rs.align(rs.stream.color)

                    if args.high_accuracy:
                        ds = profile.get_device().first_depth_sensor()
                        if ds.supports(rs.option.visual_preset):
                            ds.set_option(rs.option.visual_preset, 3)

                    # Reset for next recording
                    bag_path = None

            elif key == ord("s"):
                # ── Snapshot ─────────────────────────────────────────────
                snapshot_count += 1
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base = os.path.join(args.output_dir, f"snapshot_{ts}")

                # Color frame
                color_path = f"{base}_color.png"
                cv2.imwrite(color_path, color_image)

                # Raw 16-bit depth as PNG (lossless, preserves metric data)
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_path = f"{base}_depth_raw.png"
                cv2.imwrite(depth_path, depth_image)

                # Depth colormap for visual reference
                depth_vis_path = f"{base}_depth_vis.png"
                cv2.imwrite(depth_vis_path, build_depth_colormap(depth_frame))

                print(f"[SNAP] Snapshot #{snapshot_count} saved:")
                print(f"       Color     : {color_path}")
                print(f"       Depth raw : {depth_path}")
                print(f"       Depth vis : {depth_vis_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        if recording and bag_path:
            print(f"[INFO] Recording was in progress. File saved: {bag_path}")

        print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()