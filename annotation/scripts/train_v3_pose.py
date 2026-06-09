"""
train_v3_pose.py — Train YOLOv11-pose on merged_dataset_v3.

Dataset: 40 train + 6 val images, 274 cotton_boll instances each with 2
keypoints in the order [tip, base]. Visibility flags v in {0, 1, 2} are
respected by Ultralytics' OKS loss natively.

  - kpt_shape: [2, 3]
  - flip_idx:  [0, 1]   (identity — tip/base do not swap on horizontal flip)

Note on the v3 schema: ~95 instances have a (0, 0, v=0) keypoint where the
annotator marked tip/base as "visible" in the polygon attributes but never
placed a point shape. These contribute zero loss because v=0 — they will
not corrupt training, but they also do not supervise. Flagged for the v4
ask list to the coworker.

Hardware target: RTX 5090 on GPU 1, running in parallel with train_v3_seg.py
on GPU 0.

Run (from depth_cotton_balls/):
    python annotation/scripts/train_v3_pose.py \\
        --data annotation/datasets/v3/pose/data.yaml \\
        --model yolo11n-pose.pt \\
        --device 1 \\
        --project runs/v3_pose --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to v3 pose data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="yolo11n-pose.pt | yolo11s-pose.pt | yolo11m-pose.pt")
    ap.add_argument("--epochs", type=int, default=250,
                    help="Pose tends to need more epochs than seg on small data.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs/v3_pose")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="1", help="CUDA id (default 1 — pair with seg on 0)")
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.model)
    cache_arg = False if args.cache == "false" else args.cache

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        cache=cache_arg,
        # ---- Augmentation: gentler than seg because keypoints are more
        # sensitive to geometric distortion than masks ----
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=8,
        translate=0.05,
        scale=0.3,
        shear=0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,        # safe: flip_idx=[0,1] keeps tip/base assignments
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        # ---- Pose-specific loss weights ----
        # pose: weight on kpt regression (OKS-based)
        # kobj: weight on kpt objectness (per-kpt visibility classifier)
        pose=12.0,
        kobj=2.0,
    )

    print("\n=== Training done ===")
    if hasattr(results, "results_dict"):
        print(results.results_dict)

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    if hasattr(metrics, "results_dict"):
        print("\n=== Final val ===")
        for k, v in metrics.results_dict.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
