"""
train_v4_seg.py — Train YOLOv11-seg on the combined_all_sequential dataset.

Dataset: 64 train + 63 val images, 734 cotton_boll polygon instances
across two camera angles (back + track). Front images aren't annotated
in this drop; ignore them for now.

Source: annotation/datasets/v4/seg/data.yaml
       (produced by merged_v4_to_yolo.py).

Validation discipline: SEQUENTIAL split — val frames are a contiguous
block at the end of the timeline. mAP here is the honest "unseen
scene" number, not the leakage-inflated random-split version.

Run (from depth_cotton_balls/):
    python annotation/scripts/train_v4_seg.py `
        --data annotation/datasets/v4/seg/data.yaml `
        --device 0 `
        --project runs/v4_seg --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="annotation/datasets/v4/seg/data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="yolo11n-seg.pt | yolo11s-seg.pt | yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=250,
                    help="More than v3 (was 200) because v4 has ~3x the data.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8,
                    help="RTX 5090 32 GB at imgsz=1280 with yolo11n-seg.")
    ap.add_argument("--project", default="runs/v4_seg")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=80,
                    help="Bigger v4 val (63 imgs) gives a more reliable signal -> "
                         "less twitchy early-stop. Was 50 in v3.")
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
        # Augmentation — same as v3, gentle hue since color is diagnostic.
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.4,
        shear=2,
        perspective=0.0,
        flipud=0.0,   # Avoid vertical flip: would swap tip/base across pose head
        fliplr=0.5,
        mosaic=0.6,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
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
