"""
Synthetic end-to-end test of stages 4 + 5 with MULTIPLE recordings.

Simulates 4 recordings of the same 5 bolls, where each boll appears in a
DIFFERENT subset of recordings (matching the real-world situation: some
bolls visible only from certain angles).
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

# Camera intrinsics — RealSense D435 at 640x480
W, H = 640, 480
fx, fy = 615.0, 615.0
ppx, ppy = 320.0, 240.0
DEPTH_SCALE = 0.001

# Ground truth for 5 bolls
TRUE_BOLLS = [
    # (boll_id, true_H_cm, true_W_cm)
    (10, 4.0, 3.0),
    (20, 5.5, 4.2),
    (30, 3.2, 2.8),
    (40, 6.1, 4.8),
    (50, 4.7, 3.5),
]

# Recordings: name, distance_m, and which bolls are visible
RECORDINGS = [
    ("rec_0deg_low",  0.50, [10, 20, 30, 40]),      # 4 bolls
    ("rec_0deg_high", 0.55, [10, 20, 40, 50]),      # 4 bolls, #30 hidden
    ("rec_45deg",     0.60, [20, 30, 40, 50]),      # 4 bolls, #10 hidden
    ("handheld_1",    0.30, [10, 20, 30, 40, 50]),  # all 5 close up
]

ROOT = "/tmp/synthetic_multi"
if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

def make_ellipse_mask(cx, cy, h_cm, w_cm, dist_m):
    pix_h = int(round(h_cm / 100 / dist_m * fy))
    pix_w = int(round(w_cm / 100 / dist_m * fx))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (pix_w // 2, pix_h // 2),
                0, 0, 360, 255, -1)
    return mask

np.random.seed(7)
N_FRAMES_PER_REC = 15
work_dirs = []

for rec_name, base_dist, visible_ids in RECORDINGS:
    work = os.path.join(ROOT, rec_name)
    os.makedirs(f"{work}/frames")
    os.makedirs(f"{work}/depth")
    os.makedirs(f"{work}/masks")

    with open(f"{work}/metadata.json", "w") as f:
        json.dump({
            "recording_id": rec_name,
            "depth_scale": DEPTH_SCALE,
            "intrinsics": {"width": W, "height": H, "fx": fx, "fy": fy,
                           "ppx": ppx, "ppy": ppy, "model": "Brown_Conrady",
                           "coeffs": [0, 0, 0, 0, 0]},
        }, f)

    # Lay out visible bolls horizontally
    positions = {bid: (100 + i * 100, 240) for i, bid in enumerate(visible_ids)}

    for frame_idx in range(N_FRAMES_PER_REC):
        depth = np.zeros((H, W), dtype=np.uint16)
        frame_mask_dir = f"{work}/masks/{frame_idx:05d}"
        os.makedirs(frame_mask_dir)
        for boll_id, h_cm, w_cm in TRUE_BOLLS:
            if boll_id not in visible_ids:
                continue
            cx, cy = positions[boll_id]
            jitter = np.random.uniform(-0.005, 0.005)
            d_eff = base_dist + jitter
            mask = make_ellipse_mask(cx, cy, h_cm, w_cm, d_eff)
            depth[mask > 0] = int(round(d_eff / DEPTH_SCALE))
            cv2.imwrite(f"{frame_mask_dir}/{boll_id}.png", mask)
        np.save(f"{work}/depth/{frame_idx:05d}.npy", depth)

    work_dirs.append(work)

# Ground truth CSV
gt_path = f"{ROOT}/ground_truth.csv"
with open(gt_path, "w") as f:
    f.write("boll_id,height_cm,width_cm\n")
    for bid, h, w in TRUE_BOLLS:
        f.write(f"{bid},{h},{w}\n")

print(f"Created {len(work_dirs)} synthetic recordings:")
for w in work_dirs:
    print(f"  {w}")
print()

# Stage 4 on each work dir
print("=== Stage 4 (measure) on each recording ===")
for wd in work_dirs:
    r = subprocess.run([sys.executable,
                        "/home/claude/cotton_pipeline/04_measure_bolls.py",
                        "--work", wd], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STAGE 4 FAILED for {wd}:")
        print(r.stdout); print(r.stderr)
        sys.exit(1)
    print(f"  {wd}: OK")

# Stage 5 combining all work dirs
print()
print("=== Stage 5 (report) combining all recordings ===")
cmd = [sys.executable, "/home/claude/cotton_pipeline/05_make_report.py",
       "--work"] + work_dirs + [
       "--ground-truth", gt_path,
       "--report-dir", f"{ROOT}/combined_report"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

print()
print("=== Summary ===")
with open(f"{ROOT}/combined_report/summary.txt") as f:
    print(f.read())

print()
print("=== per_boll_per_recording.csv (first 20 rows) ===")
with open(f"{ROOT}/combined_report/per_boll_per_recording.csv") as f:
    for i, line in enumerate(f):
        if i > 20:
            break
        print(line.rstrip())
