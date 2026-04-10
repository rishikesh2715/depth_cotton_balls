"""
Stage 5 — Aggregate per-frame measurements across MULTIPLE recordings,
          compare to ground truth, and produce plots + report
=======================================================================
A single physical boll may appear in several recordings (different camera
heights, plant rotations, handheld passes).  This stage:

  1. Reads per-frame measurements from ONE OR MORE working dirs
  2. Aggregates per boll across ALL recordings (noise reduction)
  3. Also aggregates per (boll, recording) for view-consistency plots
  4. Joins against greenhouse caliper ground truth by boll tag number
  5. Produces scatter + Bland-Altman + per-boll spread + per-recording bias

Ground truth CSV columns (case-insensitive, override with --gt-cols):
    boll_id, height_cm, width_cm

Usage:
    # Single recording
    python 05_make_report.py --work work/plant_a --ground-truth gt.csv

    # Multiple recordings, merged automatically
    python 05_make_report.py \\
        --work work/rec_0deg_low work/rec_0deg_high work/rec_45deg_low \\
               work/rec_45deg_high work/handheld_1 work/handheld_2 \\
        --ground-truth gt.csv \\
        --report-dir work/combined_report

    # Override ground-truth column names + units
    python 05_make_report.py --work work/rec_a work/rec_b --ground-truth gt.csv \\
        --gt-cols id=Tag height=Height_mm width=Width_mm --units mm
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("ERROR: matplotlib not found. Install with: pip install matplotlib")


# ── CSV helpers (no pandas dependency) ───────────────────────────────────


def read_csv_dict(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_col(headers: list[str], aliases: list[str]) -> str | None:
    lookup = {h.lower().strip(): h for h in headers}
    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]
    return None


def parse_gt_cols(arg: list[str]) -> dict:
    out = {}
    for tok in arg or []:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


# ── Aggregation ──────────────────────────────────────────────────────────


def load_measurements(work_dirs: list[str]) -> list[dict]:
    """Load and merge per-frame measurements from all working dirs."""
    all_rows = []
    for wd in work_dirs:
        path = os.path.join(wd, "measurements_per_frame.csv")
        if not os.path.isfile(path):
            print(f"[WARN] No measurements in {wd}, skipping")
            continue
        rows = read_csv_dict(path)
        default_rec = os.path.basename(os.path.normpath(wd))
        for r in rows:
            if not r.get("recording_id"):
                r["recording_id"] = default_rec
        all_rows.extend(rows)
        print(f"[INFO] Loaded {len(rows):>5d} rows from {wd}")
    return all_rows


def passes_quality(r: dict) -> bool:
    try:
        if not (int(r["ok_mask_size"]) and int(r["ok_distance"])
                and int(r["ok_depth_coverage"])):
            return False
        return float(r["H_rot_cm"]) > 0 and float(r["W_rot_cm"]) > 0
    except (KeyError, ValueError):
        return False


def aggregate_per_boll(rows: list[dict]) -> dict:
    """Group quality-passing rows by boll_id across ALL recordings."""
    by_boll = defaultdict(list)
    for r in rows:
        if not passes_quality(r):
            continue
        by_boll[int(r["boll_id"])].append({
            "H_rot": float(r["H_rot_cm"]),
            "W_rot": float(r["W_rot_cm"]),
            "dist": float(r["distance_m"]),
            "recording": r["recording_id"],
        })

    summary = {}
    for boll_id, samples in by_boll.items():
        arr_h = np.array([s["H_rot"] for s in samples])
        arr_w = np.array([s["W_rot"] for s in samples])
        arr_d = np.array([s["dist"] for s in samples])
        recordings_seen = sorted(set(s["recording"] for s in samples))
        summary[boll_id] = {
            "n_frames": len(samples),
            "n_recordings": len(recordings_seen),
            "recordings": recordings_seen,
            "H_median_cm": float(np.median(arr_h)),
            "H_std_cm": float(np.std(arr_h)),
            "W_median_cm": float(np.median(arr_w)),
            "W_std_cm": float(np.std(arr_w)),
            "dist_median_m": float(np.median(arr_d)),
        }
    return summary


def aggregate_per_boll_per_recording(rows: list[dict]) -> dict:
    """Group by (boll_id, recording_id) for view-consistency plots."""
    by_key = defaultdict(list)
    for r in rows:
        if not passes_quality(r):
            continue
        key = (int(r["boll_id"]), r["recording_id"])
        by_key[key].append({
            "H_rot": float(r["H_rot_cm"]),
            "W_rot": float(r["W_rot_cm"]),
        })

    out = {}
    for (boll_id, rec), samples in by_key.items():
        arr_h = np.array([s["H_rot"] for s in samples])
        arr_w = np.array([s["W_rot"] for s in samples])
        out[(boll_id, rec)] = {
            "n_frames": len(samples),
            "H_median_cm": float(np.median(arr_h)),
            "W_median_cm": float(np.median(arr_w)),
        }
    return out


# ── Metrics ──────────────────────────────────────────────────────────────


def metrics(measured: np.ndarray, truth: np.ndarray) -> dict:
    err = measured - truth
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    nonzero = truth > 1e-6
    mape = float(np.mean(abs_err[nonzero] / truth[nonzero]) * 100) if nonzero.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if len(measured) >= 2 and measured.std() > 0 and truth.std() > 0:
        pearson = float(np.corrcoef(measured, truth)[0, 1])
    else:
        pearson = float("nan")
    return {
        "n": int(len(measured)),
        "MAE_cm": round(mae, 3),
        "RMSE_cm": round(rmse, 3),
        "bias_cm": round(bias, 3),
        "MAPE_pct": round(mape, 2),
        "R2": round(r2, 4),
        "pearson_r": round(pearson, 4),
    }


# ── Plots ────────────────────────────────────────────────────────────────


def scatter_plot(out_path, measured, truth, labels, dim_name, units, m):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(truth, measured, s=70, alpha=0.75, edgecolor="k", linewidth=0.5)
    for x, y, lab in zip(truth, measured, labels):
        ax.annotate(f"#{lab}", (x, y), fontsize=7, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")
    lo = float(min(truth.min(), measured.min())) * 0.9
    hi = float(max(truth.max(), measured.max())) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x (perfect)")
    if len(truth) >= 2:
        slope, intercept = np.polyfit(truth, measured, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, slope * xs + intercept, "r-", alpha=0.7,
                label=f"fit: y = {slope:.3f}x + {intercept:+.3f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Ground truth {dim_name} ({units})")
    ax.set_ylabel(f"RGBD-measured {dim_name} ({units})")
    ax.set_title(
        f"Cotton boll {dim_name}: RGBD vs caliper\n"
        f"n={m['n']}  MAE={m['MAE_cm']} {units}  "
        f"RMSE={m['RMSE_cm']} {units}  R²={m['R2']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def bland_altman_plot(out_path, measured, truth, dim_name, units):
    mean = (measured + truth) / 2
    diff = measured - truth
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, s=60, alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.axhline(md, color="r", linestyle="-",
               label=f"mean diff = {md:+.3f}")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--",
               label=f"+1.96 SD = {md + 1.96*sd:+.3f}")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--",
               label=f"-1.96 SD = {md - 1.96*sd:+.3f}")
    ax.set_xlabel(f"Mean of methods ({units})")
    ax.set_ylabel(f"RGBD - caliper ({units})")
    ax.set_title(f"Bland-Altman: {dim_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_boll_spread_plot(out_path, per_br: dict, matched: list[dict],
                          dim: str, units: str):
    """
    For each matched boll, plot one dot per recording (measurement in that
    recording) and a horizontal line at the ground truth value.  Shows how
    consistent the RGBD measurement is across different views of the same boll.
    """
    if not matched:
        return
    matched_sorted = sorted(matched, key=lambda r: r[f"gt_{dim}_cm"])
    boll_ids = [r["boll_id"] for r in matched_sorted]
    gt_vals = [r[f"gt_{dim}_cm"] for r in matched_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(boll_ids) * 0.35), 6))

    rec_names = sorted({rec for (_, rec) in per_br.keys()})
    cmap = plt.get_cmap("tab10")
    rec_colors = {r: cmap(i % 10) for i, r in enumerate(rec_names)}

    for x, boll_id in enumerate(boll_ids):
        for rec in rec_names:
            key = (boll_id, rec)
            if key in per_br:
                val = per_br[key][f"{dim.upper()}_median_cm"]
                ax.scatter([x], [val], s=55,
                           color=rec_colors[rec], alpha=0.75,
                           edgecolor="k", linewidth=0.4)
        ax.plot([x - 0.3, x + 0.3], [gt_vals[x], gt_vals[x]],
                color="red", linewidth=2.0, zorder=3)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          color=rec_colors[r], markersize=7,
                          markeredgecolor="k", markeredgewidth=0.4, label=r)
               for r in rec_names]
    handles.append(plt.Line2D([0], [0], color="red", linewidth=2.0,
                              label="ground truth"))
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              ncol=max(1, (len(rec_names) + 1) // 6))

    ax.set_xticks(range(len(boll_ids)))
    ax.set_xticklabels([f"#{b}" for b in boll_ids], rotation=60, fontsize=8)
    ax.set_xlabel("Boll (sorted by ground-truth value)")
    ax.set_ylabel(f"{dim} ({units})")
    ax.set_title(f"Per-boll measurement spread across recordings ({dim})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_recording_bias_plot(out_path, per_br: dict, gt_lookup: dict,
                              dim: str, units: str):
    """
    For each recording, compute mean signed error (bias) vs ground truth.
    Tells you whether any one camera setup is systematically over/under.
    """
    by_rec = defaultdict(list)
    for (boll_id, rec), data in per_br.items():
        if boll_id not in gt_lookup:
            continue
        gt_val = gt_lookup[boll_id][0 if dim == "H" else 1]
        meas = data[f"{dim}_median_cm"]
        by_rec[rec].append(meas - gt_val)

    if not by_rec:
        return

    recs = sorted(by_rec.keys())
    biases = [float(np.mean(by_rec[r])) for r in recs]
    stds = [float(np.std(by_rec[r])) for r in recs]
    ns = [len(by_rec[r]) for r in recs]

    fig, ax = plt.subplots(figsize=(max(6, len(recs) * 0.9), 5))
    xs = np.arange(len(recs))
    ax.bar(xs, biases, yerr=stds, capsize=5, alpha=0.8,
           color=["#4C72B0"] * len(recs), edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.8)
    for x, bias, n in zip(xs, biases, ns):
        ax.text(x, bias + (0.02 if bias >= 0 else -0.02),
                f"n={n}", ha="center",
                va="bottom" if bias >= 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(recs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Mean error ({units})   [RGBD − caliper]")
    ax.set_title(f"Per-recording bias ({dim})\nError bars = 1 SD across bolls")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", nargs="+", required=True,
                   help="One or more working directories (stage-4 output)")
    p.add_argument("--ground-truth", required=True,
                   help="Ground truth CSV with boll_id, height_cm, width_cm")
    p.add_argument("--gt-cols", nargs="*", default=[],
                   help="Override column names: id=Foo height=Bar width=Baz")
    p.add_argument("--units", choices=["cm", "mm", "in"], default="cm",
                   help="Units of the ground truth file (converted to cm)")
    p.add_argument("--report-dir", default=None,
                   help="Output directory for the report "
                        "(default: <first work dir>/report)")
    args = p.parse_args()

    rows = load_measurements(args.work)
    if not rows:
        sys.exit("ERROR: No measurements found in any --work directory.")
    print(f"[INFO] Total: {len(rows)} per-frame rows from {len(args.work)} recording(s)")

    summary = aggregate_per_boll(rows)
    per_br = aggregate_per_boll_per_recording(rows)
    print(f"[INFO] {len(summary)} unique bolls passed quality gates")
    if not summary:
        sys.exit("ERROR: No bolls passed quality gates. Check thresholds in stage 4.")

    gt_rows = read_csv_dict(args.ground_truth)
    if not gt_rows:
        sys.exit("ERROR: ground truth CSV is empty")
    headers = list(gt_rows[0].keys())
    overrides = parse_gt_cols(args.gt_cols)
    id_col = overrides.get("id") or find_col(
        headers, ["boll_id", "boll", "tag", "id", "number", "tag_number"])
    h_col = overrides.get("height") or find_col(
        headers, ["height_cm", "height", "h_cm", "h", "length", "length_cm"])
    w_col = overrides.get("width") or find_col(
        headers, ["width_cm", "width", "w_cm", "w", "diameter_cm", "diameter"])
    if not (id_col and h_col and w_col):
        sys.exit(
            f"ERROR: Could not find id/height/width columns in ground truth.\n"
            f"  Headers: {headers}\n"
            f"  Use --gt-cols id=NAME height=NAME width=NAME to specify."
        )
    print(f"[INFO] GT columns: id='{id_col}'  height='{h_col}'  width='{w_col}'")

    unit_scale = {"cm": 1.0, "mm": 0.1, "in": 2.54}[args.units]

    gt_lookup = {}
    for r in gt_rows:
        try:
            bid = int(float(r[id_col]))
            gt_lookup[bid] = (float(r[h_col]) * unit_scale,
                              float(r[w_col]) * unit_scale)
        except (ValueError, KeyError):
            continue
    print(f"[INFO] Loaded {len(gt_lookup)} ground truth bolls")

    matched = []
    only_measured = []
    only_gt = []
    for bid, s in summary.items():
        if bid in gt_lookup:
            gt_h, gt_w = gt_lookup[bid]
            matched.append({
                "boll_id": bid,
                "n_frames": s["n_frames"],
                "n_recordings": s["n_recordings"],
                "recordings_seen": "|".join(s["recordings"]),
                "dist_median_m": round(s["dist_median_m"], 3),
                "gt_H_cm": round(gt_h, 3),
                "gt_W_cm": round(gt_w, 3),
                "meas_H_cm": round(s["H_median_cm"], 3),
                "meas_W_cm": round(s["W_median_cm"], 3),
                "H_std_cm": round(s["H_std_cm"], 3),
                "W_std_cm": round(s["W_std_cm"], 3),
                "err_H_cm": round(s["H_median_cm"] - gt_h, 3),
                "err_W_cm": round(s["W_median_cm"] - gt_w, 3),
                "err_H_pct": round((s["H_median_cm"] - gt_h) / gt_h * 100, 2) if gt_h > 0 else None,
                "err_W_pct": round((s["W_median_cm"] - gt_w) / gt_w * 100, 2) if gt_w > 0 else None,
            })
        else:
            only_measured.append(bid)
    for bid in gt_lookup:
        if bid not in summary:
            only_gt.append(bid)

    print(f"[INFO] Matched {len(matched)} bolls")
    if only_measured:
        print(f"[WARN] In measurements but not in GT: {sorted(only_measured)}")
    if only_gt:
        print(f"[WARN] In GT but not in measurements: {sorted(only_gt)}")

    if not matched:
        sys.exit("ERROR: No bolls matched between measurements and ground truth.")

    report_dir = args.report_dir or os.path.join(args.work[0], "report")
    os.makedirs(report_dir, exist_ok=True)

    per_boll_path = os.path.join(report_dir, "per_boll_summary.csv")
    fieldnames = list(matched[0].keys())
    with open(per_boll_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(matched, key=lambda r: r["boll_id"]):
            w.writerow(row)
    print(f"[OUT] {per_boll_path}")

    per_br_path = os.path.join(report_dir, "per_boll_per_recording.csv")
    with open(per_br_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boll_id", "recording_id", "n_frames",
                    "H_median_cm", "W_median_cm",
                    "gt_H_cm", "gt_W_cm",
                    "err_H_cm", "err_W_cm"])
        for (bid, rec), data in sorted(per_br.items()):
            gt_h, gt_w = gt_lookup.get(bid, (None, None))
            w.writerow([
                bid, rec, data["n_frames"],
                round(data["H_median_cm"], 3),
                round(data["W_median_cm"], 3),
                round(gt_h, 3) if gt_h is not None else "",
                round(gt_w, 3) if gt_w is not None else "",
                round(data["H_median_cm"] - gt_h, 3) if gt_h is not None else "",
                round(data["W_median_cm"] - gt_w, 3) if gt_w is not None else "",
            ])
    print(f"[OUT] {per_br_path}")

    arr_meas_h = np.array([m["meas_H_cm"] for m in matched])
    arr_meas_w = np.array([m["meas_W_cm"] for m in matched])
    arr_gt_h = np.array([m["gt_H_cm"] for m in matched])
    arr_gt_w = np.array([m["gt_W_cm"] for m in matched])
    metrics_h = metrics(arr_meas_h, arr_gt_h)
    metrics_w = metrics(arr_meas_w, arr_gt_w)

    metrics_path = os.path.join(report_dir, "overall_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "height": metrics_h,
            "width": metrics_w,
            "matched_bolls": len(matched),
            "total_per_frame_rows": len(rows),
            "recordings": sorted({r["recording_id"] for r in rows}),
        }, f, indent=2)
    print(f"[OUT] {metrics_path}")

    labels = [m["boll_id"] for m in matched]
    scatter_plot(os.path.join(report_dir, "scatter_height.png"),
                 arr_meas_h, arr_gt_h, labels, "height", "cm", metrics_h)
    scatter_plot(os.path.join(report_dir, "scatter_width.png"),
                 arr_meas_w, arr_gt_w, labels, "width", "cm", metrics_w)
    bland_altman_plot(os.path.join(report_dir, "bland_altman_height.png"),
                      arr_meas_h, arr_gt_h, "height", "cm")
    bland_altman_plot(os.path.join(report_dir, "bland_altman_width.png"),
                      arr_meas_w, arr_gt_w, "width", "cm")
    per_boll_spread_plot(os.path.join(report_dir, "spread_height.png"),
                         per_br, matched, "H", "cm")
    per_boll_spread_plot(os.path.join(report_dir, "spread_width.png"),
                         per_br, matched, "W", "cm")
    per_recording_bias_plot(os.path.join(report_dir, "bias_per_recording_height.png"),
                             per_br, gt_lookup, "H", "cm")
    per_recording_bias_plot(os.path.join(report_dir, "bias_per_recording_width.png"),
                             per_br, gt_lookup, "W", "cm")
    print(f"[OUT] 8 plots in {report_dir}/")

    recordings_present = sorted({r["recording_id"] for r in rows})
    summary_path = os.path.join(report_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Cotton Boll RGBD Measurement Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recordings combined:  {len(recordings_present)}\n")
        for rec in recordings_present:
            rec_bolls = sorted({bid for (bid, r) in per_br.keys() if r == rec})
            f.write(f"  {rec:<20s}  {len(rec_bolls)} bolls\n")
        f.write(f"\nMatched bolls (meas & GT): {len(matched)}\n")
        f.write(f"GT-only bolls:             {len(only_gt)}  "
                f"{sorted(only_gt) if only_gt else ''}\n")
        f.write(f"Meas-only bolls:           {len(only_measured)}  "
                f"{sorted(only_measured) if only_measured else ''}\n\n")
        f.write("HEIGHT (cm)\n")
        for k, v in metrics_h.items():
            f.write(f"  {k:12s} = {v}\n")
        f.write("\nWIDTH (cm)\n")
        for k, v in metrics_w.items():
            f.write(f"  {k:12s} = {v}\n")
        f.write("\nPer-boll detail:\n")
        f.write(f"  {'boll':>6} {'nRec':>5} {'nFr':>5} "
                f"{'gt_H':>7} {'meas_H':>8} {'errH':>7} "
                f"{'gt_W':>7} {'meas_W':>8} {'errW':>7}\n")
        for r in sorted(matched, key=lambda x: x["boll_id"]):
            f.write(f"  {r['boll_id']:>6d} {r['n_recordings']:>5d} "
                    f"{r['n_frames']:>5d} "
                    f"{r['gt_H_cm']:>7.2f} {r['meas_H_cm']:>8.2f} "
                    f"{r['err_H_cm']:>+7.2f} "
                    f"{r['gt_W_cm']:>7.2f} {r['meas_W_cm']:>8.2f} "
                    f"{r['err_W_cm']:>+7.2f}\n")
    print(f"[OUT] {summary_path}")

    print()
    print("=" * 60)
    print(f"  HEIGHT:  MAE={metrics_h['MAE_cm']} cm  "
          f"RMSE={metrics_h['RMSE_cm']} cm  R²={metrics_h['R2']}")
    print(f"  WIDTH:   MAE={metrics_w['MAE_cm']} cm  "
          f"RMSE={metrics_w['RMSE_cm']} cm  R²={metrics_w['R2']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
