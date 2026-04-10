"""
Synthetic end-to-end test of stages 4 + 5.

We can't test stages 1-3 here (need a real .bag and a GPU + SAM 2),
but we CAN test the measurement and reporting pipeline by faking
masks + depth and verifying the math comes out right.
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

WORK = "/tmp/synthetic_test"
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(f"{WORK}/frames")
os.makedirs(f"{WORK}/depth")
os.makedirs(f"{WORK}/masks")

# Camera intrinsics — close to a RealSense D435 at 640x480
W, H = 640, 480
fx, fy = 615.0, 615.0
ppx, ppy = 320.0, 240.0
DEPTH_SCALE = 0.001  # 1 unit = 1 mm

with open(f"{WORK}/metadata.json", "w") as f:
    json.dump({
        "depth_scale": DEPTH_SCALE,
        "intrinsics": {"width": W, "height": H, "fx": fx, "fy": fy,
                       "ppx": ppx, "ppy": ppy, "model": "Brown_Conrady",
                       "coeffs": [0, 0, 0, 0, 0]},
    }, f)

# Create 5 synthetic bolls at known sizes & distances.
# Each boll: ellipse on the image plane representing its projection.
# The math: pixels-to-cm = distance_m / fx * 100
#   so for a true height H_cm at distance d_m:
#   pixel_height = H_cm/100 / d_m * fy
TRUE_BOLLS = [
    # (boll_id, true_H_cm, true_W_cm, distance_m)
    (10, 4.0, 3.0, 0.50),
    (20, 5.5, 4.2, 0.60),
    (30, 3.2, 2.8, 0.45),
    (40, 6.1, 4.8, 0.70),
    (50, 4.7, 3.5, 0.55),
]

def make_ellipse_mask(cx, cy, true_h_cm, true_w_cm, dist_m):
    pix_h = int(round(true_h_cm / 100 / dist_m * fy))
    pix_w = int(round(true_w_cm / 100 / dist_m * fx))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (pix_w // 2, pix_h // 2),
                0, 0, 360, 255, -1)
    return mask, pix_h, pix_w

# Generate 20 synthetic frames. Each frame has all 5 bolls. Add small
# random noise to simulate measurement variance.
np.random.seed(42)
N_FRAMES = 20
positions = {bid: (100 + i * 100, 240) for i, (bid, *_) in enumerate(TRUE_BOLLS)}

for frame_idx in range(N_FRAMES):
    depth = np.zeros((H, W), dtype=np.uint16)
    frame_mask_dir = f"{WORK}/masks/{frame_idx:05d}"
    os.makedirs(frame_mask_dir)
    for boll_id, h_cm, w_cm, d_m in TRUE_BOLLS:
        cx, cy = positions[boll_id]
        # Add tiny per-frame jitter to distance (simulating depth noise)
        jitter = np.random.uniform(-0.005, 0.005)
        d_eff = d_m + jitter
        mask, _, _ = make_ellipse_mask(cx, cy, h_cm, w_cm, d_eff)
        # Stamp depth values under the mask (uint16 mm)
        depth[mask > 0] = int(round(d_eff / DEPTH_SCALE))
        cv2.imwrite(f"{frame_mask_dir}/{boll_id}.png", mask)
    np.save(f"{WORK}/depth/{frame_idx:05d}.npy", depth)

# Ground truth CSV
gt_path = f"{WORK}/ground_truth.csv"
with open(gt_path, "w") as f:
    f.write("boll_id,height_cm,width_cm\n")
    for bid, h, w, _ in TRUE_BOLLS:
        f.write(f"{bid},{h},{w}\n")

print("Synthetic data created.")
print(f"  frames: {N_FRAMES},  bolls: {len(TRUE_BOLLS)}")
print()

# Run stage 4
print("=== Running stage 4 (measure) ===")
r = subprocess.run([sys.executable, "/home/claude/cotton_pipeline/04_measure_bolls.py",
                    "--work", WORK], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

# Run stage 5
print("=== Running stage 5 (report) ===")
r = subprocess.run([sys.executable, "/home/claude/cotton_pipeline/05_make_report.py",
                    "--work", WORK, "--ground-truth", gt_path],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

# Print summary
print("=== Summary ===")
with open(f"{WORK}/report/summary.txt") as f:
    print(f.read())
