"""
train_yolov11_seg.py — Fine-tune YOLOv11-seg on the coworker's 47-image
pilot dataset.

Notes tailored to this dataset:
  - Images are portrait ~2988 x 5312 (phone camera, landscape-rotated).
    Longest edge ~5312 px. Training at imgsz=640 would destroy small boll
    detail, so we default to 1280. Bump to 1536 if your GPU has the VRAM.
  - 47 images / 281 instances is small → expect easy overfitting.
    We use moderate augmentation, patience-based early stop, and close
    mosaic in the last 15 epochs.
  - Set --model to yolo11n-seg.pt first. Upgrade to yolo11s-seg.pt only
    if val mAP plateaus.

Run:
    python train_yolov11_seg.py \
        --data annotation/datasets/coworker_seg/data.yaml \
        --model yolo11n-seg.pt \
        --epochs 150 --imgsz 1280 --batch 4 \
        --project runs/coworker_seg --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="yolo11n-seg.pt | yolo11s-seg.pt | yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--project", default="runs/coworker_seg")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="0", help="CUDA id, 'cpu', or comma list")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                    help="Dataset cache. 'ram' is fastest if you have RAM for it.")
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
        # ---- Augmentation tuned for a small, in-greenhouse dataset ----
        # Color: keep gentle — boll color (white/green) is diagnostic
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        # Geometry: bolls appear at varied scales/orientations
        degrees=15,
        translate=0.1,
        scale=0.4,
        shear=2,
        perspective=0.0,
        flipud=0.0,     # Don't flip vertically — boll orientation matters
        fliplr=0.5,
        # Compositing
        mosaic=0.6,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,
        # ---- Mask-specific ----
        overlap_mask=True,
        mask_ratio=4,
    )

    print("\n=== Training done ===")
    if hasattr(results, "results_dict"):
        print(results.results_dict)

    # Final val
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    if hasattr(metrics, "results_dict"):
        print("\n=== Final val ===")
        for k, v in metrics.results_dict.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
