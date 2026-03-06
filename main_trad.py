import os, glob, re

os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache/hub"

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from sklearn.linear_model import LinearRegression

# -----------------
# Config
# -----------------
CAP_DIR = "captures"
OUT_DIR = "model_comparison_output"
YOLO_WEIGHTS = "exp.pt" 

# --- Model Selector ---
# ZoeDepth (The one you were using)
ZOE_ID = "Intel/zoedepth-nyu-kitti"

# Depth Anything V3 (From your screenshot)
# Try "depth-anything/Depth-Anything-V3-Small" if Large crashes your GPU
DA3_ID = "depth-anything/Depth-Anything-V2-Small-hf" 

# Settings
DEVICE = 0 if torch.cuda.is_available() else -1
EPS = 1e-6
alpha = 0.5 

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------
# 1. Load Models
# -----------------
print(f"Loading YOLO: {YOLO_WEIGHTS}...")
yolo = YOLO(YOLO_WEIGHTS)

print(f"Loading Model A (ZoeDepth): {ZOE_ID}...")
pipe_zoe = pipeline("depth-estimation", model=ZOE_ID, device=DEVICE)

print(f"Loading Model B (DA3): {DA3_ID}...")
# Note: Ensure you have latest transformers: pip install --upgrade transformers
pipe_da3 = pipeline("depth-estimation", model=DA3_ID, device=DEVICE)

# -----------------
# 2. Helpers
# -----------------
def parse_distance_inches(path):
    m = re.search(r"d(\d+)_", path.replace("\\", "/"))
    if not m: return None
    return float(m.group(1))

def get_data_bundle(img_path):
    """
    Runs YOLO + Zoe + DA3. 
    Returns dict with all raw values.
    """
    # 1. Load Image
    img = cv2.imread(str(img_path))
    if img is None: return None
    h, w, _ = img.shape
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 2. Run YOLO
    yolo_res = yolo.predict(img, verbose=False)[0]
    if len(yolo_res.boxes) == 0: return None

    # Get Best Box
    best_idx = int(np.argmax(yolo_res.boxes.conf.cpu().numpy()))
    
    # Width (for Pinhole)
    w_px = yolo_res.boxes.xywh[best_idx].cpu().numpy()[2]
    
    # Box & Mask
    box_xy = yolo_res.boxes.xyxy[best_idx].cpu().numpy()
    mask = None
    if yolo_res.masks is not None:
        raw_mask = yolo_res.masks.data[best_idx].cpu().numpy()
        if raw_mask.shape[:2] != (h, w):
            mask = cv2.resize(raw_mask, (w, h))
        else:
            mask = raw_mask
        mask = mask.astype(bool)

    # 3. Run Depth Models
    def get_depth_val(pipe, img_pil, mask, box_xy):
        out = pipe(img_pil)
        d_tensor = out["predicted_depth"] if "predicted_depth" in out else out["depth"]
        if hasattr(d_tensor, "detach"):
            d_map = d_tensor.detach().cpu().numpy()
        else:
            d_map = np.array(d_tensor)
        
        if d_map.ndim == 3: d_map = d_map[0]
        d_map = cv2.resize(d_map, (w, h))
        
        # Get Mean Value of Ball
        if mask is not None:
            val = np.mean(d_map[mask])
        else:
            x1, y1, x2, y2 = map(int, box_xy)
            val = np.mean(d_map[y1:y2, x1:x2])
        return d_map, val

    map_zoe, val_zoe = get_depth_val(pipe_zoe, img_pil, mask, box_xy)
    map_da3, val_da3 = get_depth_val(pipe_da3, img_pil, mask, box_xy)

    return {
        "img": img,
        "width_px": w_px,
        "box": box_xy,
        "mask": mask,
        "map_zoe": map_zoe,
        "val_zoe": val_zoe,
        "map_da3": map_da3,
        "val_da3": val_da3
    }

# -----------------
# 3. Calibration Pass
# -----------------
print("\n--- Step 1: Gathering Data & Calibrating All Models ---")
img_paths = sorted(glob.glob(os.path.join(CAP_DIR, "*.jpg")))
calib_data = []

for p in img_paths:
    d_true = parse_distance_inches(p)
    if d_true is None: continue
    
    data = get_data_bundle(p)
    if data is None: continue
    
    calib_data.append({
        "d_true": d_true,
        "inv_width": 1.0 / (data["width_px"] + EPS),
        "val_zoe": data["val_zoe"],
        "val_da3": data["val_da3"]
    })

if not calib_data: raise RuntimeError("No data found.")
df = pd.DataFrame(calib_data)

# Calibrate Pinhole
reg_pin = LinearRegression().fit(df[["inv_width"]], df["d_true"])
m_pin, c_pin = reg_pin.coef_[0], reg_pin.intercept_

# Calibrate Zoe
reg_zoe = LinearRegression().fit(df[["val_zoe"]], df["d_true"])
m_zoe, c_zoe = reg_zoe.coef_[0], reg_zoe.intercept_

# Calibrate DA3
reg_da3 = LinearRegression().fit(df[["val_da3"]], df["d_true"])
m_da3, c_da3 = reg_da3.coef_[0], reg_da3.intercept_

print(f"Pinhole: Dist = {m_pin:.1f} * (1/w) + {c_pin:.1f}")
print(f"ZoeDepth: Dist = {m_zoe:.2f} * z + {c_zoe:.2f}")
print(f"DA3:      Dist = {m_da3:.2f} * z + {c_da3:.2f}")

# -----------------
# 4. Visualization
# -----------------
print("\n--- Step 2: Generating 3-Way Visuals ---")
metrics = []

for p in img_paths:
    d_true = parse_distance_inches(p)
    if d_true is None: continue

    data = get_data_bundle(p)
    if data is None: continue
    
    # Estimates
    est_pin = m_pin * (1.0/(data["width_px"]+EPS)) + c_pin
    est_zoe = m_zoe * data["val_zoe"] + c_zoe
    est_da3 = m_da3 * data["val_da3"] + c_da3
    
    metrics.append({
        "real": d_true,
        "err_pin": abs(est_pin - d_true),
        "err_zoe": abs(est_zoe - d_true),
        "err_da3": abs(est_da3 - d_true)
    })

    # Prepare Visuals
    def colorize(d_map):
        d_norm = (d_map - d_map.min()) / (d_map.max() - d_map.min() + EPS)
        return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    img = data["img"]
    vis_zoe = colorize(data["map_zoe"])
    vis_da3 = colorize(data["map_da3"]) # Magma/Inferno usually looks best

    # Draw Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Panel 1: RGB + Pinhole

    if data["mask"] is not None:
        overlay = img.copy()
        overlay[data["mask"]] = (255, 0, 0)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (0,0), (420, 130), (0,0,0), -1)
    cv2.putText(img, "Geometric + Neural", (10,30), font, 0.6, (200,200,200), 1)
    cv2.putText(img, f"Real:  {d_true}\"", (10,55), font, 0.8, (0,255,0), 2)
    cv2.putText(img, f"Est:   {est_pin:.1f}\"", (10,85), font, 0.8, (0,255,255), 2)
    cv2.putText(img, f"Error: {abs(est_pin - d_true):.1f}\"", (10,115), font, 0.8, (0,165,255), 2)

    # Panel 2: Zoe
    cv2.rectangle(vis_zoe, (0,0), (420, 130), (0,0,0), -1)
    cv2.putText(vis_zoe, "ZOEDEPTH", (10,30), font, 0.6, (200,200,200), 1)
    cv2.putText(vis_zoe, f"Real:  {d_true}\"", (10,55), font, 0.8, (0,255,0), 2)
    cv2.putText(vis_zoe, f"Est:   {est_zoe:.1f}\"", (10,85), font, 0.8, (0,255,255), 2)
    cv2.putText(vis_zoe, f"Error: {abs(est_zoe - d_true):.1f}\"", (10,115), font, 0.8, (0,165,255), 2)

    # Panel 3: DA3
    cv2.rectangle(vis_da3, (0,0), (420, 130), (0,0,0), -1)
    cv2.putText(vis_da3, "DEPTH ANYTHING V3", (10,30), font, 0.6, (200,200,200), 1)
    cv2.putText(vis_da3, f"Real:  {d_true}\"", (10,55), font, 0.8, (0,255,0), 2)
    cv2.putText(vis_da3, f"Est:   {est_da3:.1f}\"", (10,85), font, 0.8, (0,255,255), 2)
    cv2.putText(vis_da3, f"Error: {abs(est_da3 - d_true):.1f}\"", (10,115), font, 0.8, (0,165,255), 2)

    # Combine
    combined = cv2.hconcat([img, vis_zoe, vis_da3])
    cv2.imwrite(os.path.join(OUT_DIR, f"comp_{os.path.basename(p)}"), combined)
    print(f"Saved {os.path.basename(p)}")

# -----------------
# 5. Plot Comparison
# -----------------
print("\n--- Step 3: Saving Comparison Plot ---")
df_m = pd.DataFrame(metrics)
plt.figure(figsize=(12, 6))

plt.plot([0, df_m["real"].max()], [0, df_m["real"].max()], "k--", alpha=0.3, label="Ideal")

# Group by distance to get mean error bars
grp = df_m.groupby("real").mean().reset_index()

plt.plot(grp["real"], grp["err_pin"], "g-o", label=f"Pinhole MAE: {df_m['err_pin'].mean():.2f}")
plt.plot(grp["real"], grp["err_zoe"], "r-x", label=f"Zoe MAE: {df_m['err_zoe'].mean():.2f}")
plt.plot(grp["real"], grp["err_da3"], "b-s", label=f"DA3 MAE: {df_m['err_da3'].mean():.2f}")

plt.title("Error Analysis: Which Model is Best?")
plt.xlabel("Distance (inches)")
plt.ylabel("Mean Absolute Error (inches)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("model_showdown.png")
print("Done. Check 'model_showdown.png'.")