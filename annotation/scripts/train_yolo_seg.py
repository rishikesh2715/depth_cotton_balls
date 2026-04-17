"""
train_yolo_seg.py — Fine-tune YOLOv8/11-seg on the cotton boll segmentation
dataset produced by cvat_to_yolo.py.

Requirements:
    pip install ultralytics

Run:
    python train_yolo_seg.py \
        --data datasets/seg/data.yaml \
        --model yolov8n-seg.pt \
        --epochs 100 --imgsz 1024 --batch 8 \
        --project runs/boll_seg --name v0

For a small pilot (50–200 labeled frames), start with:
  * model: yolov8n-seg.pt  (nano, fastest, reasonable baseline)
  * epochs: 100–200 (watch val/seg loss plateau)
  * imgsz: 1024 (bolls are small — keep resolution high)
  * batch: 4–8 depending on VRAM
  * close_mosaic: 10 (turn off mosaic aug in final epochs)
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--model", default="yolov8n-seg.pt",
                    help="Base checkpoint. Options: yolov8n-seg.pt, yolov8s-seg.pt, yolov11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs/boll_seg")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    ap.add_argument("--patience", type=int, default=30, help="Early-stopping patience")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        # Augmentation — keep mild for a small dataset.
        # Cotton boll color is distinctive, so avoid heavy HSV shifts.
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=10,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        # Mask-specific
        overlap_mask=True,
        mask_ratio=4,
    )
    print("Training complete. Metrics:", results.results_dict if hasattr(results, "results_dict") else results)

    # Quick val on same split for sanity
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("Val metrics:", metrics.results_dict if hasattr(metrics, "results_dict") else metrics)


if __name__ == "__main__":
    main()
