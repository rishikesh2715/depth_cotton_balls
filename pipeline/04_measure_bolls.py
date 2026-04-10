"""
Stage 4 — Measure each boll in each frame
==========================================
For every (frame, boll) mask saved in stage 3, compute:
    - distance to camera (median depth under mask, in meters)
    - axis-aligned bounding box  H_aa, W_aa  (cm)
    - rotated minAreaRect        H_rot, W_rot (cm)  <-- match caliper measurement
    - projected area in cm^2
    - mask pixel count
    - quality flags

Writes:
    <work_dir>/measurements_per_frame.csv
        frame_idx, boll_id, distance_m, depth_pix_count, mask_pix,
        H_aa_cm, W_aa_cm, H_rot_cm, W_rot_cm, area_cm2,
        ok_distance, ok_mask_size, ok_depth_coverage

Usage:
    python 04_measure_bolls.py --work work/plant_a
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np


# ── Geometry helpers (extends your original analyze_rgbd.py) ─────────────


def median_distance_under_mask(depth: np.ndarray, mask: np.ndarray,
                                depth_scale: float) -> tuple[float, int]:
    """Median depth in meters under the mask. Returns (dist_m, n_valid_px)."""
    valid = depth[(mask > 0) & (depth > 0)]
    if valid.size == 0:
        return -1.0, 0
    return float(np.median(valid)) * depth_scale, int(valid.size)


def real_dimensions_aabb(mask: np.ndarray, distance_m: float,
                          fx: float, fy: float) -> tuple[float, float, float]:
    """Axis-aligned bounding box in cm + total area in cm^2."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or distance_m <= 0:
        return -1.0, -1.0, -1.0
    pix_h = int(ys.max() - ys.min()) + 1
    pix_w = int(xs.max() - xs.min()) + 1
    m_per_px_x = distance_m / fx
    m_per_px_y = distance_m / fy
    h_cm = pix_h * m_per_px_y * 100.0
    w_cm = pix_w * m_per_px_x * 100.0
    area_cm2 = int(np.sum(mask > 0)) * m_per_px_x * m_per_px_y * 1e4
    return round(h_cm, 3), round(w_cm, 3), round(area_cm2, 3)


def real_dimensions_rotated(mask: np.ndarray, distance_m: float,
                             fx: float, fy: float) -> tuple[float, float]:
    """
    Rotated minimum-area-rectangle dimensions in cm.

    This matches what calipers measure: longest dimension across the boll
    (= height) and the perpendicular dimension (= width), regardless of
    boll orientation in the image.

    Returns (long_cm, short_cm) where long >= short.
    """
    if distance_m <= 0:
        return -1.0, -1.0
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0, -1.0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return -1.0, -1.0
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect

    # The rotated rect axes are not aligned with image x/y, so the
    # m/pixel scale should use the average of fx and fy. RealSense
    # depth-aligned-to-color usually has fx ≈ fy so the error is small.
    m_per_px = distance_m / ((fx + fy) / 2.0)
    long_pix = max(rw, rh)
    short_pix = min(rw, rh)
    long_cm = long_pix * m_per_px * 100.0
    short_cm = short_pix * m_per_px * 100.0
    return round(long_cm, 3), round(short_cm, 3)


# ── Quality gates ────────────────────────────────────────────────────────

MIN_MASK_PIXELS = 200       # discard tiny masks (likely tracker drift)
MIN_DEPTH_COVERAGE = 0.30   # at least 30% of mask must have valid depth
MIN_DIST_M = 0.15           # RealSense min range
MAX_DIST_M = 2.50           # too far -> too noisy for boll measurement


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Working dir")
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <work>/measurements_per_frame.csv)")
    args = p.parse_args()

    meta_path = os.path.join(args.work, "metadata.json")
    if not os.path.isfile(meta_path):
        sys.exit(f"ERROR: {meta_path} not found")
    with open(meta_path) as f:
        meta = json.load(f)
    fx = meta["intrinsics"]["fx"]
    fy = meta["intrinsics"]["fy"]
    depth_scale = meta["depth_scale"]
    recording_id = meta.get("recording_id", os.path.basename(os.path.normpath(args.work)))

    masks_root = os.path.join(args.work, "masks")
    depth_dir = os.path.join(args.work, "depth")
    if not os.path.isdir(masks_root):
        sys.exit(f"ERROR: {masks_root} not found. Run 03_annotate_sam2.py first.")

    out_csv = args.out or os.path.join(args.work, "measurements_per_frame.csv")

    frame_dirs = sorted(d for d in os.listdir(masks_root)
                        if os.path.isdir(os.path.join(masks_root, d)))
    print(f"[INFO] {len(frame_dirs)} frame dirs with masks")
    print(f"[INFO] fx={fx:.2f} fy={fy:.2f} depth_scale={depth_scale:.6f}")

    n_rows = 0
    n_skipped = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "recording_id",
            "frame_idx", "boll_id",
            "distance_m", "depth_valid_px", "mask_pix",
            "H_aa_cm", "W_aa_cm",
            "H_rot_cm", "W_rot_cm",
            "area_cm2",
            "ok_mask_size", "ok_distance", "ok_depth_coverage",
        ])

        for frame_dir in frame_dirs:
            frame_idx = int(frame_dir)
            depth_path = os.path.join(depth_dir, f"{frame_idx:05d}.npy")
            if not os.path.isfile(depth_path):
                n_skipped += 1
                continue
            depth = np.load(depth_path)

            mask_files = sorted(os.listdir(os.path.join(masks_root, frame_dir)))
            for mf in mask_files:
                if not mf.endswith(".png"):
                    continue
                boll_id = int(mf.split(".")[0])
                mask = cv2.imread(os.path.join(masks_root, frame_dir, mf),
                                  cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                if mask.shape != depth.shape:
                    mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                mask_pix = int(np.sum(mask > 0))
                if mask_pix == 0:
                    continue

                dist_m, valid_px = median_distance_under_mask(
                    depth, mask, depth_scale)
                h_aa, w_aa, area = real_dimensions_aabb(mask, dist_m, fx, fy)
                h_rot, w_rot = real_dimensions_rotated(mask, dist_m, fx, fy)

                ok_mask = mask_pix >= MIN_MASK_PIXELS
                ok_dist = MIN_DIST_M <= dist_m <= MAX_DIST_M
                ok_depth = (valid_px / max(mask_pix, 1)) >= MIN_DEPTH_COVERAGE

                w.writerow([
                    recording_id,
                    frame_idx, boll_id,
                    round(dist_m, 4), valid_px, mask_pix,
                    h_aa, w_aa, h_rot, w_rot, area,
                    int(ok_mask), int(ok_dist), int(ok_depth),
                ])
                n_rows += 1

    print(f"[DONE] Wrote {n_rows} rows to {out_csv}")
    if n_skipped:
        print(f"[INFO] Skipped {n_skipped} frame dirs with no depth file")


if __name__ == "__main__":
    main()
