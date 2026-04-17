"""
train_yolo_pose.py — Fine-tune YOLOv8/11-pose on the cotton boll keypoint
dataset produced by cvat_to_yolo.py.

Keypoints (5): base, tip, midpoint, width_left, width_right
Each keypoint has a visibility flag {0, 1, 2} — YOLOv8-pose handles this
natively via the last channel of kpt_shape: [5, 3].

Requirements:
    pip install ultralytics

Run:
    python train_yolo_pose.py \
        --data datasets/pose/data.yaml \
        --model yolov8n-pose.pt \
        --epochs 100 --imgsz 1024 --batch 8 \
        --project runs/boll_pose --name v0
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n-pose.pt",
                    help="Options: yolov8n-pose.pt, yolov8s-pose.pt, yolov11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="runs/boll_pose")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=40)
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
        # Pose is more sensitive to keypoint aug: avoid mirroring that
        # swaps left/right width keypoints unless flip_idx is set correctly.
        # The flip_idx in data.yaml points 3 <-> 4 so fliplr is OK.
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=8,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.3,
        close_mosaic=15,
        # Pose-specific loss weight
        pose=12.0,
        kobj=2.0,
    )
    print("Training complete.")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("Val metrics:", metrics.results_dict if hasattr(metrics, "results_dict") else metrics)


if __name__ == "__main__":
    main()
