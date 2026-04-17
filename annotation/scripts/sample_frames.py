"""
sample_frames.py — Pick a representative subset of frames from one or more
working directories (e.g., work/handheld_1, work/handheld_2) for pilot
annotation in CVAT.

Strategy:
  1. For each working dir, list frames that have SAM 2 masks available
     under masks/<frame_idx>/*.png.
  2. Rank frames by number of distinct boll_ids present — frames with more
     bolls are more informative per minute of annotator time.
  3. Apply a minimum-stride constraint to avoid near-duplicate frames.
  4. Output: a copy of the sampled JPEGs into an output folder, plus a
     manifest CSV linking each sampled frame back to its source recording
     and original frame index.

Usage:
  python sample_frames.py \
      --work work/handheld_1 work/handheld_2 \
      --out annotation/sampled_frames \
      --target 150 \
      --min-stride 5
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def list_mask_frames(work_dir: Path):
    """Return list of (frame_idx, set_of_boll_ids) for frames that have masks."""
    masks_root = work_dir / "masks"
    if not masks_root.is_dir():
        return []
    out = []
    for frame_dir in sorted(masks_root.iterdir(), key=lambda p: p.name):
        if not frame_dir.is_dir():
            continue
        try:
            idx = int(frame_dir.name)
        except ValueError:
            continue
        boll_ids = set()
        for p in frame_dir.glob("*.png"):
            try:
                boll_ids.add(int(p.stem))
            except ValueError:
                pass
        if boll_ids:
            out.append((idx, boll_ids))
    return out


def pick_frames(frame_infos, target: int, min_stride: int):
    """Greedy: prefer frames with more bolls, enforce a minimum-stride gap."""
    # Sort descending by number of bolls, tie-break by frame idx
    ranked = sorted(frame_infos, key=lambda t: (-len(t[1]), t[0]))
    chosen = []
    chosen_idx = set()
    for idx, bolls in ranked:
        if len(chosen) >= target:
            break
        if any(abs(idx - c) < min_stride for c in chosen_idx):
            continue
        chosen.append((idx, bolls))
        chosen_idx.add(idx)
    chosen.sort(key=lambda t: t[0])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", nargs="+", required=True,
                    help="One or more working directories (each containing frames/ and masks/).")
    ap.add_argument("--out", required=True, help="Output folder for sampled frames.")
    ap.add_argument("--target", type=int, default=150,
                    help="Approx target number of frames TOTAL across all --work dirs.")
    ap.add_argument("--min-stride", type=int, default=5,
                    help="Minimum frame-index gap between chosen frames in a single recording.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dirs = [Path(w) for w in args.work]
    # Gather per-work info, and distribute the target across them.
    per_work_infos = [(w, list_mask_frames(w)) for w in work_dirs]
    total_avail = sum(len(infos) for _, infos in per_work_infos)
    if total_avail == 0:
        print("ERROR: no mask frames found under any --work dir.", file=sys.stderr)
        sys.exit(1)

    manifest_rows = []
    for work_dir, infos in per_work_infos:
        if not infos:
            print(f"[skip] {work_dir}: no mask frames")
            continue
        # Per-dir share, proportional to availability
        share = max(1, round(args.target * len(infos) / total_avail))
        picked = pick_frames(infos, target=share, min_stride=args.min_stride)
        print(f"[{work_dir.name}] picked {len(picked)} / target ~{share}  (available: {len(infos)})")
        for idx, bolls in picked:
            src = work_dir / "frames" / f"{idx:05d}.jpg"
            if not src.exists():
                # fall back to png
                src_png = work_dir / "frames" / f"{idx:05d}.png"
                if src_png.exists():
                    src = src_png
                else:
                    print(f"  [warn] no frame file for idx {idx}", file=sys.stderr)
                    continue
            dst_name = f"{work_dir.name}__{idx:05d}{src.suffix}"
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            manifest_rows.append({
                "sampled_filename": dst_name,
                "source_work_dir": str(work_dir),
                "source_frame_idx": idx,
                "n_bolls_sam2": len(bolls),
                "boll_ids_sam2": ",".join(sorted(str(b) for b in bolls)),
            })

    manifest_path = out_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sampled_filename", "source_work_dir", "source_frame_idx",
            "n_bolls_sam2", "boll_ids_sam2",
        ])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} frames copied to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
