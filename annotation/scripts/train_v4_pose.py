"""
train_v4_pose.py — Train YOLOv11-pose on the combined_all_sequential dataset.

Dataset: 64 train + 63 val images, 734 instances. Keypoints: [tip, base].
Visibility distribution in train labels:
  tip:  v=0 (unknown) =  16,  v=1 (guessed) =  71,  v=2 (visible) = 647
  base: v=0           = 140,  v=1           =  57,  v=2          = 537

v=0 keypoints contribute zero OKS loss (Ultralytics skips them
natively), so the (0, 0, 0) entries from "unknown" tips/bases are
harmless during training.

Source: annotation/datasets/v4/pose/data.yaml
       (produced by merged_v4_to_yolo.py).

Run (from depth_cotton_balls/), in parallel with seg on the other GPU:
    python annotation/scripts/train_v4_pose.py `
        --data annotation/datasets/v4/pose/data.yaml `
        --device 1 `
        --project runs/v4_pose --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="annotation/datasets/v4/pose/data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="yolo11n-pose.pt | yolo11s-pose.pt | yolo11m-pose.pt")
    ap.add_argument("--epochs", type=int, default=300,
                    help="Pose tends to need a bit longer than seg; ~3x v3 data.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs/v4_pose")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="1")
    ap.add_argument("--patience", type=int, default=80)
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
        # Augmentation
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.4,
        shear=2,
        perspective=0.0,
        flipud=0.0,    # NEVER vertical-flip: tip/base would swap meaning
        fliplr=0.5,    # OK because flip_idx=[0,1] (identity)
        mosaic=0.6,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        # Pose-specific loss weights
        pose=12.0,     # Higher than default (12); penalize kpt error more
        kobj=2.0,      # Keypoint objectness
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
