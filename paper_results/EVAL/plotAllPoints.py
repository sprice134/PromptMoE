#!/usr/bin/env python3
import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- Helpers to parse method names like: K1_S0.00 or K1_S0p00 -> (k=1, supp=0.00) ----
_K_SUPP_RE = re.compile(r"^K(?P<k>\d+)_S(?P<supp>\d+(?:[p.]\d+)?)$", re.IGNORECASE)

def parse_k_supp(method_name: str):
    """
    Parse strings like 'K3_S0.06' / 'k3_s0.06' or 'K3_S0p06' -> (3, 0.06)
    Returns (k:int, supp:float) or None if pattern doesn't match.
    """
    m = _K_SUPP_RE.search(method_name)
    if not m:
        return None
    k = int(m.group("k"))
    s_str = m.group("supp").replace("p", ".").replace("P", ".")
    s = float(s_str)
    return k, s


def find_method_dirs(base_dir):
    return sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))


def discover_base_models(method_dirs):
    """
    Collect ALL base models observed anywhere (csv filename stems), excluding 'U-2-Net'
    and ignoring any CSV whose filename contains 'weights' (case-insensitive).
    """
    all_base = set()
    for d in method_dirs:
        for p in glob.glob(os.path.join(d, "*.csv")):
            fname = os.path.basename(p)
            if "weights" in fname.lower():
                continue
            name = os.path.splitext(fname)[0]
            if name != "U-2-Net" and "weights" not in name.lower():
                all_base.add(name)
    return sorted(all_base)


def load_unrefined_for_model(model, method_dirs):
    """
    Find any CSV for this base model and compute unrefined means.
    """
    for d in method_dirs:
        cand = os.path.join(d, f"{model}.csv")
        if os.path.isfile(cand):
            df = pd.read_csv(cand)
            return df["unrefined_iou"].mean(), df["unrefined_boundary_iou"].mean()
    return np.nan, np.nan


def summarize_dataset_dir(base_dir):
    """
    Build final_df with rows = ['Unrefined'] + method names (subdir names),
    columns = MultiIndex[(base_model, 'IoU'/'BIoU')].

    CHANGE: For method rows, store (unref_mean + paired_delta_mean), where
            paired_delta_mean is mean(refined - unrefined) over rows with
            non-NaN values in all four columns. This way, when
            compute_cross_dataset_deltas later does (method - Unrefined),
            it recovers the paired Δ.
    """
    method_dirs = find_method_dirs(base_dir)
    if not method_dirs:
        raise RuntimeError(f"No method subdirectories found in: {base_dir}")

    methods = [os.path.basename(d) for d in method_dirs]
    base_models = discover_base_models(method_dirs)

    # Unrefined row + cache per-model unref means for reuse
    data = defaultdict(dict)
    unref_cache = {}
    for model in base_models:
        u_iou, u_biou = load_unrefined_for_model(model, method_dirs)
        unref_cache[model] = (u_iou, u_biou)
        data["Unrefined"][(model, "IoU")] = u_iou
        data["Unrefined"][(model, "BIoU")] = u_biou

    # Refined rows (paired per-image deltas)
    for method in methods:
        for model in base_models:
            path = os.path.join(base_dir, method, f"{model}.csv")
            if not os.path.isfile(path):
                data[method][(model, "IoU")] = np.nan
                data[method][(model, "BIoU")] = np.nan
            else:
                df = pd.read_csv(path)
                # Drop rows lacking any of the four required fields
                cols = ["unrefined_iou","refined_iou","unrefined_boundary_iou","refined_boundary_iou"]
                df = df.dropna(subset=cols)
                if df.shape[0] == 0:
                    data[method][(model, "IoU")] = np.nan
                    data[method][(model, "BIoU")] = np.nan
                else:
                    d_iou = (df["refined_iou"] - df["unrefined_iou"]).mean()
                    d_bio = (df["refined_boundary_iou"] - df["unrefined_boundary_iou"]).mean()
                    u_iou, u_bio = unref_cache.get(model, (np.nan, np.nan))
                    # Store unref_mean + paired_delta_mean so downstream subtraction yields paired Δ
                    data[method][(model, "IoU")] = u_iou + d_iou if np.isfinite(u_iou) else np.nan
                    data[method][(model, "BIoU")] = u_bio + d_bio if np.isfinite(u_bio) else np.nan

    final_df = pd.DataFrame.from_dict(data, orient="index")
    cols = pd.MultiIndex.from_product([base_models, ["IoU", "BIoU"]])
    final_df = final_df.reindex(columns=cols)
    return final_df, base_models, ["Unrefined"] + methods


def compute_cross_dataset_deltas(base_dirs):
    """
    For each dataset dir:
      - Compute final_df
      - Compute per-method mean ΔIoU / ΔBIoU vs Unrefined across base models

    (Unchanged consumer: since summarize_dataset_dir now stores method values as
     unref_mean + paired_delta_mean, the subtraction here yields paired Δ.)
    """
    agg_deltas = defaultdict(dict)
    dataset_order = []

    for bd in base_dirs:
        bd_clean = bd.rstrip(os.sep)
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                        if bn == "sota" else os.path.basename(bd_clean))

        final_df, base_models, methods = summarize_dataset_dir(bd_clean)
        io_cols = [(bm, "IoU") for bm in base_models]
        bi_cols = [(bm, "BIoU") for bm in base_models]

        if dataset_name not in dataset_order:
            dataset_order.append(dataset_name)

        for m in methods:
            if m == "Unrefined":
                continue
            d_iou = (final_df.loc[m, io_cols] - final_df.loc["Unrefined", io_cols]).mean() * 100.0
            d_bio = (final_df.loc[m, bi_cols] - final_df.loc["Unrefined", bi_cols]).mean() * 100.0
            agg_deltas[m][dataset_name] = (d_iou, d_bio)

    return agg_deltas, dataset_order


def build_grid(agg_deltas, dataset_order, which="iou"):
    """
    Build a grid (Y x X) where X = k, Y = suppression factor.
    Cell value = cross-dataset mean Δ (in %) for the chosen metric ('iou' or 'biou').

    Returns:
      grid (2D np.array with NaNs),
      y_vals (sorted suppression floats),
      x_vals (sorted K ints)
    """
    # Cross-dataset mean per method
    method_to_mean = {}
    for method, per_ds in agg_deltas.items():
        parsed = parse_k_supp(method)
        if not parsed:
            continue
        vals = []
        for ds in dataset_order:
            if ds in per_ds:
                diou, dbiou = per_ds[ds]
                vals.append(diou if which == "iou" else dbiou)
        if vals:
            method_to_mean[method] = np.nanmean(vals)

    # Collect axes
    ks = sorted({parse_k_supp(m)[0] for m in method_to_mean})
    sups = sorted({parse_k_supp(m)[1] for m in method_to_mean})

    grid = np.full((len(sups), len(ks)), np.nan, dtype=float)
    for method, val in method_to_mean.items():
        k, s = parse_k_supp(method)
        xi = ks.index(k)
        yi = sups.index(s)
        grid[yi, xi] = val

    return grid, sups, ks


def plot_heatmap(grid, y_vals, x_vals, title, out_path, annotate=True, dpi=180):
    """
    grid: (len(y_vals) x len(x_vals)) with NaNs allowed
    """
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(x_vals) + 2),
                                    max(4, 0.6 * len(y_vals) + 2)))

    # Symmetric diverging color range around zero for Δ
    vmax = np.nanmax(np.abs(grid)) * 1.1
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = np.nanmin(np.abs(grid)) * 0.9

    im = ax.imshow(grid, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

    # Axis labels/ticks
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(k) for k in x_vals])
    ax.set_xlabel("k (number of points)")

    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{s:.3f}".rstrip('0').rstrip('.') for s in y_vals])
    ax.set_ylabel("Suppression factor")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Δ (%) vs Unrefined")

    # Gridlines
    ax.set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    ax.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Annotate cells
    if annotate:
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                v = grid[yi, xi]
                if np.isfinite(v):
                    ax.text(xi, yi, f"{v:+.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_heatmap_pair(grid_iou, grid_bio, y_vals, x_vals, out_path, annotate=True, dpi=180):
    """
    Side-by-side heatmaps for IoU and BIoU with separate color scales and colorbars.

    Specs:
      • Per-plot color limits:
          vmin = (lowest reported cell) - 0.5
          vmax = (highest reported cell) + 0.5
        where "reported" = values rounded to 2 decimals.
      • Remove axis numbers on the BIoU subplot.
      • Titles use triangle Delta: "Mean ΔIoU", "Mean ΔBIoU".
      • Shorter figure height.
      • Single y-axis label ("Suppression factor") on the left subplot only.
      • White, bold, larger annotations.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def rounded_min_max(arr):
        if arr.size == 0 or np.all(~np.isfinite(arr)):
            return np.nan, np.nan
        r = np.round(arr.astype(float), 2)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return np.nan, np.nan
        return np.nanmin(r), np.nanmax(r)

    # Compute per-plot bounds
    min_iou, max_iou = rounded_min_max(grid_iou)
    min_bio, max_bio = rounded_min_max(grid_bio)

    if np.isfinite(min_iou) and np.isfinite(max_iou):
        vmin_iou = float(min_iou) - 0.5
        vmax_iou = float(max_iou) + 0.5
        if vmin_iou >= vmax_iou:
            center = 0.5 * (vmin_iou + vmax_iou); span = 1.0
            vmin_iou, vmax_iou = center - span, center + span
    else:
        vmin_iou, vmax_iou = -1.0, 1.0

    if np.isfinite(min_bio) and np.isfinite(max_bio):
        vmin_bio = float(min_bio) - 0.5
        vmax_bio = float(max_bio) + 0.5
        if vmin_bio >= vmax_bio:
            center = 0.5 * (vmin_bio + vmax_bio); span = 1.0
            vmin_bio, vmax_bio = center - span, center + span
    else:
        vmin_bio, vmax_bio = -1.0, 1.0

    # Even shorter figure height
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(8, 0.6 * len(x_vals) + 4), max(2.2, 0.34 * len(y_vals) + 0.8)),
        constrained_layout=True
    )

    # Left: IoU
    im1 = axes[0].imshow(grid_iou, aspect='auto', origin='lower', vmin=vmin_iou, vmax=vmax_iou)
    axes[0].set_title(r"Mean $\Delta$IoU")
    axes[0].set_xticks(range(len(x_vals)))
    axes[0].set_xticklabels([str(k) for k in x_vals])
    axes[0].set_xlabel("k (number of points)")
    axes[0].set_yticks(range(len(y_vals)))
    axes[0].set_yticklabels([f"{s:.3f}".rstrip('0').rstrip('.') for s in y_vals])
    axes[0].set_ylabel("Suppression factor")
    axes[0].set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    axes[0].set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    axes[0].grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.4)
    axes[0].tick_params(which='minor', bottom=False, left=False)

    # Right: BIoU (no axis numbers)
    im2 = axes[1].imshow(grid_bio, aspect='auto', origin='lower', vmin=vmin_bio, vmax=vmax_bio)
    axes[1].set_title(r"Mean $\Delta$BIoU")
    axes[1].set_xticks(range(len(x_vals)))
    axes[1].set_xticklabels([str(k) for k in x_vals])  # ensure same labels as left
    axes[1].set_xlabel("k (number of points)")
    axes[1].set_yticks(range(len(y_vals)))
    axes[1].set_yticklabels([f"{s:.3f}".rstrip('0').rstrip('.') for s in y_vals])  # match left
    axes[1].set_ylabel("")  # single suppression factor label on the left subplot only
    axes[1].set_xticks(np.arange(-0.5, len(x_vals), 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, len(y_vals), 1), minor=True)
    axes[1].grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.4)
    axes[1].tick_params(which='minor', bottom=False, left=False, labelbottom=False, labelleft=False)

    # White, larger annotations (NOT bold)
    if annotate:
        for ax, grid in zip(axes, (grid_iou, grid_bio)):
            for yi in range(len(y_vals)):
                for xi in range(len(x_vals)):
                    v = grid[yi, xi]
                    if np.isfinite(v):
                        ax.text(
                            xi, yi, f"{v:+.2f}",
                            ha="center", va="center",
                            fontsize=12, color='white'
                        )

    # Separate colorbars, one per subplot (on the right of each)
    cbar1 = fig.colorbar(im1, ax=axes[0], location='right', fraction=0.046, pad=0.04)
    cbar2 = fig.colorbar(im2, ax=axes[1], location='right', fraction=0.046, pad=0.04)
    cbar2.set_label("Mean Δ (%) vs Unrefined")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)





def main():
    p = argparse.ArgumentParser(
        description="Aggregate Δ vs Unrefined across datasets and save heatmaps for k vs suppression."
    )
    p.add_argument("base_dirs", nargs="+",
                   help="Dataset dirs; each contains method subdirectories with per-model CSVs.")
    p.add_argument("--outdir", default="../EVAL/", help="Output directory for heatmaps.")
    p.add_argument("--prefix", default="", help="Optional filename prefix.")
    p.add_argument("--no-annot", action="store_true", help="Disable cell annotations.")
    p.add_argument("--dpi", type=int, default=180, help="DPI for saved images.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    agg_deltas, dataset_order = compute_cross_dataset_deltas(args.base_dirs)

    # Build grids
    grid_iou, sups, ks = build_grid(agg_deltas, dataset_order, which="iou")
    grid_bio, _, _ = build_grid(agg_deltas, dataset_order, which="biou")

    # Save heatmaps
    hm_iou_path = os.path.join(args.outdir, f"{args.prefix}heatmap_dIoU.png")
    hm_bio_path = os.path.join(args.outdir, f"{args.prefix}heatmap_dBIoU.png")

    plot_heatmap(
        grid_iou, sups, ks,
        title="Mean ΔIoU (%) vs Unrefined — k × suppression",
        out_path=hm_iou_path,
        annotate=not args.no_annot,
        dpi=args.dpi
    )
    plot_heatmap(
        grid_bio, sups, ks,
        title="Mean ΔBIoU (%) vs Unrefined — k × suppression",
        out_path=hm_bio_path,
        annotate=not args.no_annot,
        dpi=args.dpi
    )
    
    print(f"Saved: {hm_iou_path}")
    print(f"Saved: {hm_bio_path}")
    # Third figure: side-by-side with shared colorbar/units
    hm_pair_path = os.path.join(args.outdir, f"{args.prefix}heatmap_pair_dIoU_dBIoU.png")
    plot_heatmap_pair(
        grid_iou, grid_bio, sups, ks,
        out_path=hm_pair_path,
        annotate=not args.no_annot,
        dpi=args.dpi
    )
    print(f"Saved: {hm_pair_path}")



if __name__ == "__main__":
    main()




'''
python plotAllPoints.py \
    /home/sprice/PromptMoE/paper_results/ablation/points/BIG \
    /home/sprice/PromptMoE/paper_results/ablation/points/DAVIS585/ \
    /home/sprice/PromptMoE/paper_results/ablation/points/ECSSD/ \
    /home/sprice/PromptMoE/paper_results/ablation/points/VOC/ \
    /home/sprice/PromptMoE/paper_results/ablation/points/MSRA-B/
'''