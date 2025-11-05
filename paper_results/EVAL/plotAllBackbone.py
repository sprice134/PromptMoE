#!/usr/bin/env python3
import os
import glob
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# IO helpers
# ----------------------------
def _list_csvs(path: str):
    return sorted([p for p in glob.glob(os.path.join(path, "*.csv")) if os.path.isfile(p)])

def _load_and_pair_deltas(csv_paths):
    """
    Load CSVs with columns:
      unrefined_iou, refined_iou, unrefined_boundary_iou, refined_boundary_iou
    Return concatenated per-image deltas: (d_iou%, d_biou%) in percentage points.
    """
    d_iou_all, d_bio_all = [], []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = ["unrefined_iou","refined_iou","unrefined_boundary_iou","refined_boundary_iou"]
        if not set(cols).issubset(df.columns):
            continue
        df = df.dropna(subset=cols)
        if df.empty:
            continue
        # CHANGE: compute per-CSV (per-model) means so models are equally weighted,
        # rather than concatenating all per-image deltas
        d_iou_all.append(float((df["refined_iou"] - df["unrefined_iou"]).mean()))
        d_bio_all.append(float((df["refined_boundary_iou"] - df["unrefined_boundary_iou"]).mean()))
    if d_iou_all:
        d_iou = np.array(d_iou_all, dtype=float) * 100.0
        d_bio = np.array(d_bio_all, dtype=float) * 100.0
    else:
        d_iou = np.array([], dtype=float)
        d_bio = np.array([], dtype=float)
    return d_iou, d_bio


# ----------------------------
# Data harvesting (explicit dataset directories)
# ----------------------------
def gather_backbone_deltas_from_dirs(ours_dirs, baseline_dirs, backbones=("vit_b","vit_l","vit_h"), ours_method="SamRefiner-MoE"):
    """
    ours_dirs:      list of absolute dataset directories for OURS (same order as baseline_dirs)
    baseline_dirs:  list of absolute dataset directories for SR baseline
    backbones:      iterable of backbone subdir names present inside each dataset dir
    ours_method:    optional subfolder under each backbone dir for our CSVs

    Returns:
      results: dict[backbone] -> {"ours": (mean_dIoU, mean_dBIoU), "sr": (...)}
              where means are equal-weight over datasets (simple average of dataset means).
    """
    assert len(ours_dirs) == len(baseline_dirs), "ours_dirs and baseline_dirs must have same length/order"

    # normalize and check
    ours_dirs = [d.rstrip(os.sep) for d in ours_dirs]
    baseline_dirs = [d.rstrip(os.sep) for d in baseline_dirs]
    backbones = list(backbones)

    # accumulate per-backbone dataset means
    per_bb_ours = {bb: [] for bb in backbones}
    per_bb_sr   = {bb: [] for bb in backbones}

    for ds_ours_dir, ds_base_dir in zip(ours_dirs, baseline_dirs):
        for bb in backbones:
            # OURS CSVs: <ds_ours_dir>/<bb>/<ours_method>/*.csv  (fallback to <bb>/*.csv)
            ours_bb = os.path.join(ds_ours_dir, bb)
            ours_method_dir = os.path.join(ours_bb, ours_method)
            ours_csvs = _list_csvs(ours_method_dir) if os.path.isdir(ours_method_dir) else _list_csvs(ours_bb)
            d_iou_o, d_bio_o = _load_and_pair_deltas(ours_csvs)

            # SR baseline CSVs: <ds_base_dir>/<bb>/*.csv
            sr_bb = os.path.join(ds_base_dir, bb)
            sr_csvs = _list_csvs(sr_bb)
            d_iou_sr, d_bio_sr = _load_and_pair_deltas(sr_csvs)

            if d_iou_o.size:
                per_bb_ours[bb].append((float(np.mean(d_iou_o)), float(np.mean(d_bio_o))))
            if d_iou_sr.size:
                per_bb_sr[bb].append((float(np.mean(d_iou_sr)), float(np.mean(d_bio_sr))))

    # equal-weight average over datasets
    def _avg(pairs):
        if not pairs:
            return (np.nan, np.nan)
        arr = np.array(pairs, dtype=float)  # [N,2]
        return (float(np.nanmean(arr[:,0])), float(np.nanmean(arr[:,1])))

    results = OrderedDict()
    # enforce order vit_b, vit_l, vit_h if present
    order = [bb for bb in ["vit_b","vit_l","vit_h"] if bb in backbones] + [bb for bb in backbones if bb not in {"vit_b","vit_l","vit_h"}]
    for bb in order:
        results[bb] = {"ours": _avg(per_bb_ours[bb]), "sr": _avg(per_bb_sr[bb])}
    return results, order


# ----------------------------
# Plotting (two-panel bar chart)
# ----------------------------
def plot_backbone_bars(results, backbones, out_path, dpi=180):
    """
    results: dict[backbone] -> {"ours": (dIoU%, dBIoU%), "sr": (dIoU%, dBIoU%)}
    """
    x = np.arange(len(backbones))
    ours_iou = [results[bb]["ours"][0] for bb in backbones]
    sr_iou   = [results[bb]["sr"][0]   for bb in backbones]
    ours_bio = [results[bb]["ours"][1] for bb in backbones]
    sr_bio   = [results[bb]["sr"][1]   for bb in backbones]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), constrained_layout=True)
    width = 0.35
    xticks = [bb.replace("vit_", "ViT-").upper().replace("VIT-", "ViT-") for bb in backbones]

    # Shared y-axis limit with a slightly larger headroom buffer for labels
    all_vals = np.array([*ours_iou, *sr_iou, *ours_bio, *sr_bio], dtype=float)
    ymax = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 1.0
    buffer = max(0.12 * ymax, 1.0)   # larger buffer so numbers fit comfortably
    y_top = ymax + buffer

    # ---------- Left: ΔIoU (Baseline left, Ours right) ----------
    ours_color_iou     = "#9AD29A"  # soft green
    baseline_color_iou = "#4D7F86"  # teal

    ax = axes[0]
    b_base = ax.bar(x - width/2, sr_iou,   width, label="SamRefiner", color=baseline_color_iou)
    b_ours = ax.bar(x + width/2, ours_iou, width, label="Ours",     color=ours_color_iou)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_ylabel("ΔIoU (%)", fontsize=14)
    ax.set_ylim(0, y_top)  # shared y-axis limit
    ax.legend(loc="upper left")
    ax.tick_params(axis='both', labelsize=12)

    for bars in (b_base, b_ours):
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.text(rect.get_x() + rect.get_width()/2.0, h,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=13)

    # ---------- Right: ΔBIoU (Baseline left, Ours right) ----------
    ours_color_biou     = "#F4C063"  # pastel amber
    baseline_color_biou = "#84CFC7"  # light teal

    ax = axes[1]
    b_base = ax.bar(x - width/2, sr_bio,   width, label="SamRefiner", color=baseline_color_biou)
    b_ours = ax.bar(x + width/2, ours_bio, width, label="Ours",     color=ours_color_biou)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_ylabel("ΔBIoU (%)", fontsize=14)
    ax.set_ylim(0, y_top)  # same y-axis limit as left panel
    ax.legend(loc="upper left")
    ax.tick_params(axis='both', labelsize=12)

    for bars in (b_base, b_ours):
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.text(rect.get_x() + rect.get_width()/2.0, h,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=13)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)






# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate paired ΔIoU/ΔBIoU (%) per backbone (Ours vs SR baseline) from explicit dataset directories and plot bar charts."
    )
    ap.add_argument("--ours_dirs", nargs="+", required=True,
                    help="List of dataset directories for Ours, e.g., .../backbone/BIG .../backbone/DAVIS585 ...")
    ap.add_argument("--baseline_dirs", nargs="+", required=True,
                    help="List of dataset directories for SR baseline in the SAME ORDER as --ours_dirs.")
    ap.add_argument("--backbones", nargs="*", default=["vit_b","vit_l","vit_h"],
                    help="Backbone subdirectory names to use (default: vit_b vit_l vit_h).")
    ap.add_argument("--ours_method", default="SamRefiner-MoE",
                    help="Subfolder under each backbone dir containing Ours CSVs (default: SamRefiner-MoE).")
    ap.add_argument("--outdir", default="../EVAL/", help="Output directory for figures.")
    ap.add_argument("--prefix", default="", help="Optional filename prefix.")
    ap.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results, ordered_bbs = gather_backbone_deltas_from_dirs(
        ours_dirs=args.ours_dirs,
        baseline_dirs=args.baseline_dirs,
        backbones=args.backbones,
        ours_method=args.ours_method
    )

    out_path = os.path.join(args.outdir, f"{args.prefix}dIoU_dBIoU.png")
    plot_backbone_bars(results, ordered_bbs, out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


    '''
    python plotAllBackbone.py \
    --ours_dirs \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/BIG/ \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/DAVIS585/ \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/ECSSD/ \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/VOC/ \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/MSRA-B/ \
    --baseline_dirs \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/ogSAMRefinerBackbone/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/ogSAMRefinerBackbone/DAVIS585 \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/ogSAMRefinerBackbone/ECSSD \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/ogSAMRefinerBackbone/MSRA-B \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/ogSAMRefinerBackbone/VOC \
    --backbones vit_b vit_l vit_h \
    --ours_method SamRefiner-MoE \
    --outdir /home/sprice/PromptMoE/paper_results/EVAL \
    --prefix backbone_ \
    --dpi 220

    '''