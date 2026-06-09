"""
train_v3_seg.py — Train YOLOv11-seg on merged_dataset_v3.

Dataset: 40 train + 6 val images, 274 cotton_boll polygon instances.
Source: annotation/datasets/v3/seg/data.yaml (produced by merged_v3_to_yolo.py).

Hardware target: RTX 5090 (32 GB VRAM). Defaults below assume the seg model
runs alone on GPU 0 while train_v3_pose.py runs on GPU 1.

Run (from depth_cotton_balls/):
    python annotation/scripts/train_v3_seg.py \\
        --data annotation/datasets/v3/seg/data.yaml \\
        --model yolo11n-seg.pt \\
        --device 0 \\
        --project runs/v3_seg --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to v3 seg data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="yolo11n-seg.pt | yolo11s-seg.pt | yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8,
                    help="RTX 5090 (32 GB) handles batch=8 at imgsz=1280 with "
                         "yolo11n-seg comfortably. Drop if OOM.")
    ap.add_argument("--project", default="runs/v3_seg")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="0", help="CUDA id, 'cpu', or comma list")
    ap.add_argument("--patience", type=int, default=50)
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
        # ---- Augmentation tuned for a small in-greenhouse dataset ----
        # Color: keep gentle — boll color is diagnostic of stage
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        # Geometry: bolls appear at varied scales/orientations
        degrees=15,
        translate=0.1,
        scale=0.4,
        shear=2,
        perspective=0.0,
        flipud=0.0,     # Vertical flip would invert tip/base; we share images
                        # with the pose model so keep them consistent.
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        # ---- Mask-specific ----
        overlap_mask=True,
        mask_ratio=4,
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
