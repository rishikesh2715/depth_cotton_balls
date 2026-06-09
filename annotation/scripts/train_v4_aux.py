"""
train_v4_aux.py — Train the auxiliary head (L_occ + L_visfrac) on the
combined_all_sequential dataset (v4).

Differences from train_v3_aux.py:
  - Uses BollCropDatasetV4 (reads single train.json/val.json with v4 schema:
    file_name includes camera subfolder, occlusion fields top-level on ann)
  - Two CLI roots instead of one: --images-root and --coco-dir
  - Slightly larger epochs / patience because v4 has ~3x the annotations
    (734 vs 274) and a much bigger val set (264 vs 32) — more reliable signal,
    less twitchy early stop.

v4 class distribution (collapsed to 4-way):
    train:  none=355, leaf= 76, stem=  7, other=32        (470 anns)
    val:    none=138, leaf= 66, stem= 60, other= 0        (264 anns)
  -> Train still skewed toward `none`; class-frequency-weighted CE helps.
  -> Val has 0 "other" samples (the held-out continuous block didn't
     contain any frame_edge / other_boll cases). Recall for "other" will
     be undefined in val and skipped from the macro mean — read the per-
     class numbers in the final printout.

Run (from depth_cotton_balls/):
    python annotation/scripts/train_v4_aux.py `
        --images-root combined_all_sequential `
        --coco-dir   "combined_all_sequential 3/combined_all_sequential/output_coco_dir" `
        --device 0 `
        --epochs 80 --batch 32 `
        --project runs/v4_aux --name v0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aux_head import (  # noqa: E402
    AuxHead, AuxLoss, BollCropDatasetV4, OCC_CLASSES,
)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_cls_pred, all_cls_true, all_vis_pred, all_vis_true = [], [], [], []
    for batch in loader:
        x = batch["crop"].to(device, non_blocking=True)
        out = model(x)
        all_cls_pred.append(out["cls_logits"].argmax(-1).cpu().numpy())
        all_cls_true.append(batch["cls_id"].numpy())
        all_vis_pred.append(out["visfrac"].cpu().numpy())
        all_vis_true.append(batch["visfrac"].numpy())
    cls_pred = np.concatenate(all_cls_pred) if all_cls_pred else np.array([])
    cls_true = np.concatenate(all_cls_true) if all_cls_true else np.array([])
    vis_pred = np.concatenate(all_vis_pred) if all_vis_pred else np.array([])
    vis_true = np.concatenate(all_vis_true) if all_vis_true else np.array([])
    metrics: dict = {}
    if len(cls_true):
        metrics["cls_acc"] = float((cls_pred == cls_true).mean())
        recalls = []
        for c in range(len(OCC_CLASSES)):
            mask = cls_true == c
            if mask.sum() == 0:
                metrics[f"recall_{OCC_CLASSES[c]}"] = float("nan")
                continue
            r = float((cls_pred[mask] == c).mean())
            metrics[f"recall_{OCC_CLASSES[c]}"] = r
            recalls.append(r)
        metrics["cls_macro_recall"] = float(np.mean(recalls)) if recalls else 0.0
    if len(vis_true):
        metrics["vis_mae"] = float(np.mean(np.abs(vis_pred - vis_true)))
        metrics["vis_rmse"] = float(np.sqrt(np.mean((vis_pred - vis_true) ** 2)))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=Path,
                    help="Directory with back/, front/, track/ subfolders.")
    ap.add_argument("--coco-dir", required=True, type=Path,
                    help="Directory with train.json + val.json (sequential split).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--pad-frac", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--w-occ", type=float, default=1.0)
    ap.add_argument("--w-vis", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0",
                    help="CUDA id, 'cpu', or 'cuda:N'.")
    ap.add_argument("--project", default="runs/v4_aux")
    ap.add_argument("--name", default="v0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25,
                    help="Bumped from 20 (v3) — v4 val is more reliable.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {run_dir}")

    ds_train = BollCropDatasetV4(args.images_root, args.coco_dir,
                                 split="train", imgsz=args.imgsz,
                                 pad_frac=args.pad_frac, augment=True)
    ds_val = BollCropDatasetV4(args.images_root, args.coco_dir,
                               split="val", imgsz=args.imgsz,
                               pad_frac=args.pad_frac, augment=False)
    print(f"[data] train={len(ds_train)}  val={len(ds_val)}")
    train_counts = ds_train.class_counts()
    val_counts = ds_val.class_counts()
    print(f"[data] train class counts: {train_counts}")
    print(f"[data] val   class counts: {val_counts}")

    if len(ds_train) == 0:
        print("ERROR: no training records found. Check --images-root and --coco-dir.",
              file=sys.stderr)
        sys.exit(1)

    pin = (device.type == "cuda")
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          drop_last=False,
                          persistent_workers=(args.workers > 0))
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=max(1, args.workers // 2), pin_memory=pin,
                        drop_last=False,
                        persistent_workers=(args.workers > 0))

    model = AuxHead(num_classes=len(OCC_CLASSES), pretrained=True).to(device)
    loss_fn = AuxLoss(w_occ=args.w_occ, w_vis=args.w_vis,
                      class_counts=train_counts).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_recall = -1.0
    epochs_since_improve = 0
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        sum_l = sum_lo = sum_lv = 0.0
        n_seen = 0
        for batch in dl_train:
            x = batch["crop"].to(device, non_blocking=True)
            out = model(x)
            losses = loss_fn(out, batch)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            bs = x.size(0)
            sum_l += losses["loss"].item() * bs
            sum_lo += losses["l_occ"].item() * bs
            sum_lv += losses["l_vis"].item() * bs
            n_seen += bs
        sched.step()
        train_l = sum_l / max(n_seen, 1)
        train_lo = sum_lo / max(n_seen, 1)
        train_lv = sum_lv / max(n_seen, 1)

        val_metrics = evaluate(model, dl_val, device) if len(ds_val) else {}
        dt = time.time() - t0

        msg = (f"epoch {epoch:3d}/{args.epochs}  "
               f"loss={train_l:.4f} (occ={train_lo:.4f} vis={train_lv:.4f})  "
               f"val_acc={val_metrics.get('cls_acc', float('nan')):.3f}  "
               f"val_macro_recall={val_metrics.get('cls_macro_recall', float('nan')):.3f}  "
               f"val_vis_mae={val_metrics.get('vis_mae', float('nan')):.4f}  "
               f"({dt:.1f}s)")
        print(msg)
        log_lines.append(msg)

        cur = val_metrics.get("cls_macro_recall", -1.0)
        if cur > best_recall:
            best_recall = cur
            epochs_since_improve = 0
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "occ_classes": list(OCC_CLASSES),
                "epoch": epoch,
                "best_macro_recall": best_recall,
            }
            torch.save(ckpt, run_dir / "best.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"[early-stop] no val_macro_recall improvement for "
                      f"{args.patience} epochs (best={best_recall:.3f})")
                break

    (run_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\n[done] best val_macro_recall={best_recall:.3f}")
    print(f"[done] checkpoint: {run_dir / 'best.pt'}")

    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "occ_classes": list(OCC_CLASSES),
                "epoch": "final"}, run_dir / "last.pt")

    if len(ds_val):
        final = evaluate(model, dl_val, device)
        print("\n=== Final val ===")
        for k, v in final.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
