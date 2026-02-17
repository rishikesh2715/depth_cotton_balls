import os, glob, re

# Set Hugging Face cache directory to D: drive to avoid disk space issues on C:
# This must be set BEFORE importing transformers or huggingface_hub
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache/hub"

import numpy as np
import pandas as pd

from ultralytics import YOLO
from transformers import pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # we'll compute MAE manually

# -----------------
# Config
# -----------------
CAP_DIR = "captures"
YOLO_WEIGHTS = "best.pt"

DEPTH_MODEL = "Intel/zoedepth-nyu-kitti"  # if this fails, switch to "Intel/dpt-hybrid-midas"
DEVICE = 0
EPS = 1e-6

# ROI / depth-stat stability knobs
ERODE_ITERS = 2          # increase to 3 if edges are noisy
TRIM_LOW = 0.20          # keep middle 60% of depth values
TRIM_HIGH = 0.80

# -----------------
# Load models
# -----------------
yolo = YOLO(YOLO_WEIGHTS)
depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)

# -----------------
# Helpers
# -----------------
def parse_distance_inches(path):
    m = re.search(r"/d(\d+)_", path.replace("\\", "/"))
    if not m:
        return None
    return float(m.group(1))

def erode_mask(mask, iters=2):
    try:
        import cv2
        k = np.ones((5, 5), np.uint8)
        m = (mask.astype(np.uint8) * 255)
        m = cv2.erode(m, k, iterations=iters)
        return m > 0
    except Exception:
        return mask

def get_best_mask(img_path):
    res = yolo.predict(img_path, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return None

    conf = res.boxes.conf.cpu().numpy()
    i = int(np.argmax(conf))

    m = res.masks.data[i].cpu().numpy().astype(bool)
    m = erode_mask(m, iters=ERODE_ITERS)  # reduce edge mixing
    return m

def _to_numpy_depth(pred_depth):
    """Convert predicted_depth (torch/np) to float32 numpy HxW."""
    D = pred_depth
    if hasattr(D, "detach"):  # torch tensor
        D = D.detach().cpu().numpy()
    D = np.asarray(D).astype(np.float32)
    if D.ndim == 3:
        # sometimes shape is (1, H, W) or (B,H,W)
        D = D[0]
    return D

def depth_scalar_from_mask(img_path, mask):
    out = depth_pipe(img_path)

    # Prefer numeric depth if available
    if isinstance(out, dict) and "predicted_depth" in out:
        D = _to_numpy_depth(out["predicted_depth"])
    else:
        # fallback: visualization depth image (often normalized)
        D = np.array(out["depth"]).astype(np.float32)

    # Align sizes if needed
    if D.shape[:2] != mask.shape[:2]:
        import cv2
        D = cv2.resize(D, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    vals = D[mask]
    if vals.size < 50:
        return None

    # Trim extremes (robust against boundary bleed + outliers)
    vals = np.sort(vals)
    lo = int(TRIM_LOW * len(vals))
    hi = int(TRIM_HIGH * len(vals))
    if hi <= lo + 10:
        # not enough values after trim; just use median
        return float(np.median(vals))

    vals_mid = vals[lo:hi]
    return float(np.median(vals_mid))

def mae_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # no squared kwarg
    return mae, rmse

# -----------------
# Build dataset
# -----------------
rows = []
for p in sorted(glob.glob(os.path.join(CAP_DIR, "*.jpg"))):
    d_in = parse_distance_inches(p)
    if d_in is None:
        continue

    mask = get_best_mask(p)
    if mask is None:
        continue

    z = depth_scalar_from_mask(p, mask)
    if z is None or not np.isfinite(z):
        continue

    rows.append((p, d_in, z, str(d_in)))
    print(d_in, z)

df = pd.DataFrame(rows, columns=["image_path", "distance_in", "z", "dist_group"])
print("Usable samples:", len(df))
if len(df) == 0:
    raise RuntimeError("No usable samples. Check captures/, YOLO masks, and depth model loading.")
print(df.groupby("distance_in").size())

# -----------------
# Train/test split by distance
# -----------------
if df["dist_group"].nunique() < 2:
    raise RuntimeError("Need at least 2 different distances to do a train/test split.")

gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
tr_i, te_i = next(gss.split(df, groups=df["dist_group"]))
train, test = df.iloc[tr_i].copy(), df.iloc[te_i].copy()

# -----------------
# Fit calibration: linear vs inverse
# -----------------
def fit(form):
    yte = test["distance_in"].values

    if form == "linear":
        Xtr = train[["z"]].values
        Xte = test[["z"]].values
        ytr = train["distance_in"].values
    else:  # inverse
        Xtr = 1.0 / (train[["z"]].values + EPS)
        Xte = 1.0 / (test[["z"]].values + EPS)
        ytr = train["distance_in"].values

    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mae, rmse = mae_rmse(yte, pred)
    return reg, mae, rmse

reg_lin, mae_lin, rmse_lin = fit("linear")
reg_inv, mae_inv, rmse_inv = fit("inverse")

print(f"Linear  MAE={mae_lin:.2f} in, RMSE={rmse_lin:.2f} in")
print(f"Inverse MAE={mae_inv:.2f} in, RMSE={rmse_inv:.2f} in")

best_form = "inverse" if mae_inv < mae_lin else "linear"
best_reg = reg_inv if best_form == "inverse" else reg_lin
print("Best:", best_form)

# -----------------
# Final per-sample errors
# -----------------
if best_form == "linear":
    X = test[["z"]].values
else:
    X = 1.0 / (test[["z"]].values + EPS)

test["d_est_in"] = best_reg.predict(X)
test["abs_err_in"] = np.abs(test["d_est_in"] - test["distance_in"])

overall_mae = float(test["abs_err_in"].mean())
overall_rmse = float(np.sqrt(np.mean((test["d_est_in"] - test["distance_in"]) ** 2)))

print("\nOverall Test MAE (in):", overall_mae)
print("Overall Test RMSE (in):", overall_rmse)

print("\nError by distance:")
print(
    test.groupby("distance_in")["abs_err_in"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("distance_in")
)