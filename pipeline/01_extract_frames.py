"""
Stage 1 — Extract frames from a RealSense .bag recording
=========================================================
Reads a .bag file, aligns depth to color, and writes:
    <out_dir>/frames/00000.jpg, 00001.jpg, ...      (color frames for SAM 2)
    <out_dir>/depth/00000.npy,  00001.npy,  ...     (raw aligned depth, uint16)
    <out_dir>/metadata.json                          (intrinsics + depth scale)

Run once per .bag.  Subsequent stages only need <out_dir>, not the .bag.

Usage:
    python 01_extract_frames.py --bag plant_a.bag --out work/plant_a
    python 01_extract_frames.py --bag plant_a.bag --out work/plant_a --stride 2
        (--stride 2 keeps every 2nd frame; use to thin dense recordings)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2")


def extract(bag_path: str, out_dir: str, stride: int, jpeg_quality: int,
            recording_id: str):
    if not os.path.isfile(bag_path):
        sys.exit(f"ERROR: bag file not found: {bag_path}")

    frames_dir = os.path.join(out_dir, "frames")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    profile = pipeline.start(config)

    # Process every frame, not real-time
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    # After aligning depth -> color, depth uses color intrinsics
    intrinsics = {
        "width": color_intr.width,
        "height": color_intr.height,
        "fx": color_intr.fx,
        "fy": color_intr.fy,
        "ppx": color_intr.ppx,
        "ppy": color_intr.ppy,
        "model": str(color_intr.model),
        "coeffs": list(color_intr.coeffs),
    }

    # Optional depth post-processing — same filters as analyze script
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    print(f"[INFO] Extracting from: {bag_path}")
    print(f"[INFO] Recording ID:    {recording_id}")
    print(f"[INFO] Output dir:      {out_dir}")
    print(f"[INFO] Resolution:      {color_intr.width}x{color_intr.height}")
    print(f"[INFO] Depth scale:     {depth_scale:.6f} m/unit")
    print(f"[INFO] fx={color_intr.fx:.2f} fy={color_intr.fy:.2f} "
          f"ppx={color_intr.ppx:.2f} ppy={color_intr.ppy:.2f}")
    print(f"[INFO] Stride:          {stride}")
    print()

    src_idx = 0       # frames read from bag
    out_idx = 0       # frames written to disk
    timestamps = []   # bag timestamps for each output frame, ms

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                break  # end of bag

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if src_idx % stride == 0:
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = hole_filling.process(depth_frame)

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

                fname = f"{out_idx:05d}"
                cv2.imwrite(
                    os.path.join(frames_dir, f"{fname}.jpg"),
                    color,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                np.save(os.path.join(depth_dir, f"{fname}.npy"), depth)
                timestamps.append(color_frame.get_timestamp())
                out_idx += 1

                if out_idx % 50 == 0:
                    print(f"  ...wrote {out_idx} frames")

            src_idx += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()

    metadata = {
        "recording_id": recording_id,
        "source_bag": os.path.abspath(bag_path),
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
        "stride": stride,
        "num_frames": out_idx,
        "frame_timestamps_ms": timestamps,
        "depth_units": "uint16, multiply by depth_scale to get meters",
        "depth_alignment": "depth aligned to color (use color intrinsics)",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[DONE] Extracted {out_idx} frames from {src_idx} source frames.")
    print(f"[DONE] Metadata: {os.path.join(out_dir, 'metadata.json')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True, help="Path to .bag recording")
    p.add_argument("--out", required=True, help="Output working directory")
    p.add_argument("--recording-id", default=None,
                   help="Short label for this recording (e.g. '0deg_low', "
                        "'45deg_high', 'handheld_1'). Defaults to basename of --out.")
    p.add_argument("--stride", type=int, default=1,
                   help="Keep every Nth frame (default: 1 = all frames)")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for color frames (default: 95)")
    args = p.parse_args()
    rec_id = args.recording_id or os.path.basename(os.path.normpath(args.out))
    extract(args.bag, args.out, args.stride, args.jpeg_quality, rec_id)


if __name__ == "__main__":
    main()
