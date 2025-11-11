#!/usr/bin/env python3
import os
import glob
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
EXPERIMENT_ORDER = [
    "cascadePSP_fast",
    "cascadePSP_slow",
    "segrefiner_lr",
    "segrefiner_hr",
    "DualSight",
    "samRefiner",
    "SamRefiner-MoE",
    "vit_b",
    "vit_l",
    "vit_h",
]

# Initial IoU strata (on [0,1] scale). Last bin is inclusive on the right.
BINS = [
    (0.00, 0.20, "0–20"),
    (0.20, 0.40, "20–40"),
    (0.40, 0.60, "40–60"),
    (0.60, 0.80, "60–80"),
    (0.80, 1.01, "80–100"),  # 1.01 to include exactly 1.0
]

# ----------------------------
# Helpers
# ----------------------------
def _canon(s: str) -> str:
    """Loose string canonicalizer: lowercase and keep only alphanumerics."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _find_method_dir(dataset_dir: str, ours_name: str) -> str:
    """Find the subdir for our method, matching loosely."""
    want = _canon(ours_name)
    cands = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    for d in cands:
        if _canon(os.path.basename(d)) == want:
            return d
    exact = os.path.join(dataset_dir, ours_name)
    if os.path.isdir(exact):
        return exact
    raise RuntimeError(f"Could not find method directory for '{ours_name}' under: {dataset_dir}")

def _load_model_csv(csv_path: str) -> pd.DataFrame:
    """Load a per-model CSV and keep only fully-paired rows."""
    df = pd.read_csv(csv_path)
    cols = ["unrefined_iou", "refined_iou", "unrefined_boundary_iou", "refined_boundary_iou"]
    df = df.dropna(subset=cols)
    return df

def _bin_means_for_model(df: pd.DataFrame) -> dict:
    """
    For one model's DF, compute mean ΔIoU/ΔBIoU per initial IoU bin.
    Returns: {bin_label: (mean_dIoU, mean_dBIoU)}  (NaN if no rows in bin)
    """
    u_iou = df["unrefined_iou"].to_numpy(dtype=float)
    d_iou = (df["refined_iou"] - df["unrefined_iou"]).to_numpy(dtype=float)
    d_bio = (df["refined_boundary_iou"] - df["unrefined_boundary_iou"]).to_numpy(dtype=float)

    out = {}
    for lo, hi, label in BINS:
        if hi >= 1.0:
            mask = (u_iou >= lo) & (u_iou <= hi)  # inclusive for last bin
        else:
            mask = (u_iou >= lo) & (u_iou < hi)
        if not np.any(mask):
            out[label] = (np.nan, np.nan)
        else:
            out[label] = (float(np.mean(d_iou[mask])), float(np.mean(d_bio[mask])))
    return out

def _aggregate_bins_equal_weight(model_bin_means_list):
    """
    Equal-weight over models: average per-bin means ignoring NaNs.
    model_bin_means_list: [ {bin_label: (dIoU, dBIoU)}, ... ]
    Returns: {bin_label: (mean_dIoU, mean_dBIoU)}
    """
    labels = [lbl for _, _, lbl in BINS]
    agg = {}
    for lbl in labels:
        i_vals = [mb[lbl][0] for mb in model_bin_means_list if np.isfinite(mb[lbl][0])]
        b_vals = [mb[lbl][1] for mb in model_bin_means_list if np.isfinite(mb[lbl][1])]
        mi = float(np.mean(i_vals)) if i_vals else np.nan
        mb = float(np.mean(b_vals)) if b_vals else np.nan
        agg[lbl] = (mi, mb)
    return agg

def _fmt_pp(val: float) -> str:
    """Format a delta (in 0..1 units) as text in percentage points (no math mode, no color)."""
    if not np.isfinite(val):
        return "NaN"
    pp = val * 100.0
    if pp > 0:
        return f"+{pp:.2f}"
    return f"{pp:.2f}"

def _latex_table_strata(ds_to_bins: OrderedDict, label: str = "tab:strata_deltas") -> str:
    r"""
    Build one LaTeX table:
      • Rows: initial IoU bins (0–20, ..., 80–100)
      • Cols: datasets + Mean
      • Cell: ΔIoU / ΔBIoU in percentage points (no math mode)
      • Wrapped in \begin{adjustbox}{width=\linewidth}
    """
    datasets = list(ds_to_bins.keys())
    bin_labels = [lbl for _, _, lbl in BINS]

    # Precompute Mean across datasets, equal-weight (ignoring NaNs)
    mean_across_ds = {}
    for lbl in bin_labels:
        i_vals = [ds_to_bins[ds][lbl][0] for ds in datasets if np.isfinite(ds_to_bins[ds][lbl][0])]
        b_vals = [ds_to_bins[ds][lbl][1] for ds in datasets if np.isfinite(ds_to_bins[ds][lbl][1])]
        mi = float(np.mean(i_vals)) if i_vals else np.nan
        mb = float(np.mean(b_vals)) if b_vals else np.nan
        mean_across_ds[lbl] = (mi, mb)

    # Column spec with vertical line between second to last and last column
    colspec = "@{}l " + " ".join(["c"] * len(datasets)) + " | c @{}"

    lines = []
    lines.append("\\begin{table*}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Stratified evaluation of \\textbf{PromptMoE} highlighting refinement performance by initial IoU.}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(f"Initial IoU bin & " + " & ".join([ds.replace('_', '\\_') for ds in datasets] + ["Mean"]) + " \\\\")
    lines.append("\\midrule")

    for lbl in bin_labels:
        # Find max IoU and BIoU for this row (ignoring NaNs)
        i_vals = [ds_to_bins[ds][lbl][0] for ds in datasets if np.isfinite(ds_to_bins[ds][lbl][0])]
        b_vals = [ds_to_bins[ds][lbl][1] for ds in datasets if np.isfinite(ds_to_bins[ds][lbl][1])]
        max_i = max(i_vals) if i_vals else None
        max_b = max(b_vals) if b_vals else None

        cells = []
        for ds in datasets:
            di, db = ds_to_bins[ds][lbl]
            i_str = _fmt_pp(di)
            b_str = _fmt_pp(db)
            if max_i is not None and np.isfinite(di) and np.isclose(di, max_i):
                i_str = f"\\textbf{{{i_str}}}"
            if max_b is not None and np.isfinite(db) and np.isclose(db, max_b):
                b_str = f"\\textbf{{{b_str}}}"
            cells.append(f"{i_str} / {b_str}")
        mi, mb = mean_across_ds[lbl]
        mean_cell = f"{_fmt_pp(mi)} / {_fmt_pp(mb)}"
        lines.append(f"{lbl} & " + " & ".join(cells + [mean_cell]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stratified (by initial IoU) ΔIoU/ΔBIoU table for SamRefiner-MoE across datasets."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Each dir contains method subdirs with per-model CSVs.")
    parser.add_argument("--ours_name", type=str, default="SamRefiner-MoE",
                        help="Directory name for our method (default: SamRefiner-MoE).")
    parser.add_argument("--label", type=str, default="tab:stratified_moe",
                        help="LaTeX label for the table.")
    args = parser.parse_args()

    ds_to_bins = OrderedDict()  # dataset -> {bin_label: (mean_dIoU, mean_dBIoU)}

    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")

        # dataset name (keep your previous convention)
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                        if bn == "sota" else os.path.basename(bd_clean))

        # locate our method dir
        method_dir = _find_method_dir(bd_clean, args.ours_name)

        # collect models = CSV basenames (excluding weights files)
        model_csvs = sorted(glob.glob(os.path.join(method_dir, "*.csv")))
        models = []
        for p in model_csvs:
            base = os.path.splitext(os.path.basename(p))[0]
            if base in {"weights", "weights_full"}: continue
            if base.endswith("_weights") or base.endswith("_weights_full"): continue
            if base == "U-2-Net": continue
            models.append(base)
        models = sorted(set(models))

        if not models:
            raise RuntimeError(f"No model CSVs found under: {method_dir}")

        # per-model stratified means
        per_model_bin_means = []
        for m in models:
            csv_path = os.path.join(method_dir, f"{m}.csv")
            if not os.path.isfile(csv_path):
                continue
            df = _load_model_csv(csv_path)
            per_model_bin_means.append(_bin_means_for_model(df))

        if not per_model_bin_means:
            # if nothing loaded (unlikely), fill NaNs
            per_model_bin_means = [{lbl: (np.nan, np.nan)} for _, _, lbl in BINS]

        # dataset-level equal-weight average across models
        ds_to_bins[dataset_name] = _aggregate_bins_equal_weight(per_model_bin_means)

    # Emit ONE LaTeX table
    table = _latex_table_strata(ds_to_bins, label=args.label)
    print(table)

if __name__ == "__main__":
    main()

    '''
    python plotAllStratified.py \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/timeCheck/BIG/ \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/timeCheck/DAVIS585/ \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/timeCheck/ECSSD/ \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/timeCheck/VOC/ \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/timeCheck/MSRA-B/ \
    --ours_name PromptMoE


    '''