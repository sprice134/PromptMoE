#!/usr/bin/env python3
import os
import glob
import re
import argparse
from typing import List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------
_EXPERT_DIR_RE = re.compile(r'^E(0[1-9]|10)(?:_.+)?$')  # E01..E10, optional suffix like _Dp

def _is_single_expert_dir(name: str) -> bool:
    """True for directories named E01..E10, with or without a suffix (e.g., E01, E01_Dp)."""
    return isinstance(name, str) and _EXPERT_DIR_RE.match(name) is not None

def _collect_e01_e10_dirs(base_dir: str) -> List[str]:
    """List present E01..E10 dirs (sorted lexicographically)."""
    ds = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and _is_single_expert_dir(d)]
    return sorted(ds)

def _collect_models_from_expert_dirs(base_dir: str, expert_dirs: List[str]) -> List[str]:
    """Union of CSV basenames across expert dirs (ignores *_weights*.csv)."""
    names = set()
    for ed in expert_dirs:
        for p in glob.glob(os.path.join(base_dir, ed, "*.csv")):
            base = os.path.splitext(os.path.basename(p))[0]
            if base in {"weights", "weights_full"} or base.endswith("_weights") or base.endswith("_weights_full"):
                continue
            names.add(base)
    return sorted(names)

def _load_per_model_matrix(base_dir: str, expert_dirs: List[str], model: str, metric: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load per-image refined metric for all experts for a given model, aligned by 'subdir'.
    Returns (df with columns ['subdir', <experts...>], expert_cols). Missing experts/images become NaN.
    """
    assert metric in ("IoU", "BIoU")
    usecol = "refined_iou" if metric == "IoU" else "refined_boundary_iou"

    dfs = []
    for ed in expert_dirs:
        p = os.path.join(base_dir, ed, f"{model}.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p, usecols=["subdir", usecol]).rename(columns={usecol: ed})
            dfs.append(df)

    if not dfs:
        cols = ["subdir"] + expert_dirs
        return pd.DataFrame(columns=cols), expert_dirs

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="subdir", how="outer")

    for ed in expert_dirs:
        if ed not in merged.columns:
            merged[ed] = np.nan

    keep_cols = ["subdir"] + expert_dirs
    return merged[keep_cols], expert_dirs

def _best_expert_distribution_from_matrix(df: pd.DataFrame, expert_cols: List[str]) -> Dict[str, float]:
    """
    For each image (row), find the max across experts (ignoring NaNs).
    If k experts tie, each gets 1/k credit. Skip rows where all experts are NaN.
    Returns per-expert fraction in [0,1] summing ~1 over valid rows.
    """
    if df.empty or not expert_cols:
        return {e: np.nan for e in expert_cols}

    M = df[expert_cols].to_numpy(dtype=float)
    valid = ~np.all(np.isnan(M), axis=1)
    if not np.any(valid):
        return {e: np.nan for e in expert_cols}

    Mv = M[valid]
    with np.errstate(all="ignore"):
        row_max = np.nanmax(Mv, axis=1)

    winners = (Mv == row_max[:, None]) & np.isfinite(Mv)
    tie_counts = winners.sum(axis=1).astype(float)
    contrib = winners / tie_counts[:, None]

    wins = np.nansum(contrib, axis=0)
    frac = wins / float(Mv.shape[0])
    return {e: float(frac[i]) for i, e in enumerate(expert_cols)}

def _aggregate_over_models(distributions: List[Dict[str, float]], expert_cols: List[str]) -> Dict[str, float]:
    """Equal-weight over models: average per-model fractions (ignoring NaNs)."""
    out = {}
    for e in expert_cols:
        vals = [d.get(e, np.nan) for d in distributions]
        vals = [v for v in vals if np.isfinite(v)]
        out[e] = float(np.mean(vals)) if vals else np.nan
    return out

def _latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for table text."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}"))

# ----------------------------
# Dataset evaluation
# ----------------------------
def evaluate_dataset_best_share(base_dir: str, metric: str) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    For a dataset dir:
      - find expert dirs matching E01..E10 (suffix allowed, e.g., E01_Dp)
      - find models in any expert dir
      - compute per-model best-expert distributions
      - return equal-weight average across models
    Returns (expert_order, aggregated_distribution, models_used)
    """
    expert_dirs = _collect_e01_e10_dirs(base_dir)
    if not expert_dirs:
        raise RuntimeError(f"No E01..E10 expert dirs found in: {base_dir}")

    models = _collect_models_from_expert_dirs(base_dir, expert_dirs)
    if not models:
        raise RuntimeError(f"No model CSVs found under experts in: {base_dir}")

    per_model = []
    for model in models:
        df, expert_cols = _load_per_model_matrix(base_dir, expert_dirs, model, metric)
        dist = _best_expert_distribution_from_matrix(df, expert_cols)
        per_model.append(dist)

    agg = _aggregate_over_models(per_model, expert_dirs)
    return expert_dirs, agg, models

# ----------------------------
# LaTeX table builder (single-column with adjustbox width=\linewidth)
# ----------------------------
def latex_best_expert_table(all_ds_best: Dict[str, Dict[str, float]],
                            metric: str,
                            caption_note: str | None = None,
                            label: str = "tab:bestSingle") -> str:
    r"""
    Builds a LaTeX table:
      • Rows: expert directories present (e.g., E01\_Dp)
      • Cols: datasets + Mean
      • Cell: % of images where the expert is the best single performer (equal-weight over models)
      • Bold: max per dataset and the best Mean (uses \textbf, not math mode)
      • Wrapped in \begin{adjustbox}{width=\linewidth} for single-column fit.
    """
    datasets = list(all_ds_best.keys())
    datasets_esc = [_latex_escape(d) for d in datasets]

    # Union of experts across datasets, stable lexicographic order
    expert_union = []
    seen = set()
    for ds in datasets:
        for e in sorted(all_ds_best[ds].keys()):
            if e not in seen:
                seen.add(e)
                expert_union.append(e)

    if caption_note is None:
        caption_note = (f"Distribution of single-expert winners for \\textbf{{{metric}}}: "
                        "percentage of images where each expert (E01..E10 with suffix) is the top performer "
                        "(ties split evenly). Values are averaged equally across base models.")

    # Best per dataset (for bolding)
    best_per_ds = {}
    for ds in datasets:
        vals = [all_ds_best[ds].get(e, np.nan) for e in expert_union]
        vals = [v for v in vals if np.isfinite(v)]
        best_per_ds[ds] = (max(vals) if vals else np.nan)

    # Best Mean across experts
    mean_per_expert = {}
    for e in expert_union:
        vals = [all_ds_best[ds].get(e, np.nan) for ds in datasets]
        vals = [v for v in vals if np.isfinite(v)]
        mean_per_expert[e] = (float(np.mean(vals)) if vals else np.nan)
    best_mean = max([v for v in mean_per_expert.values() if np.isfinite(v)], default=np.nan)

    colspec = "@{}l " + " ".join(["r"] * (len(datasets) + 1)) + " @{}"

    lines = []
    lines.append("\\begin{table}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption_note}}}")
    lines.append(f"\\label{{{label}_{metric}}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(f" & \\multicolumn{{{len(datasets)+1}}}{{c}}{{Dataset}} \\\\")
    lines.append(f"\\cmidrule(lr){{2-{len(datasets)+1}}}")
    header = "Expert & " + " & ".join(datasets_esc + ["Mean"]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for e in expert_union:
        e_esc = _latex_escape(e)
        cells = []
        vals_for_mean = []
        for ds in datasets:
            v = all_ds_best[ds].get(e, np.nan)
            if not np.isfinite(v):
                cells.append("NaN")
            else:
                s = f"{v*100.0:.2f}\\%"
                if np.isfinite(best_per_ds[ds]) and abs(v - best_per_ds[ds]) <= 1e-12:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
                vals_for_mean.append(v)

        if vals_for_mean:
            m = float(np.mean(vals_for_mean))
            mcell = f"{m*100.0:.2f}\\%"
            if np.isfinite(best_mean) and abs(m - best_mean) <= 1e-12:
                mcell = f"\\textbf{{{mcell}}}"
        else:
            mcell = "NaN"

        lines.append(f"{e_esc} & " + " & ".join(cells + [mcell]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")

    return "\n".join(lines)

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Produce LaTeX table(s) of best single-expert (E01..E10[_suffix]) winner distribution per dataset."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Each dir is a dataset root containing expert dirs like E01_Dp, E02_Gd, ... with per-model CSVs.")
    parser.add_argument("--metrics", choices=["IoU", "BIoU", "both"], default="both",
                        help="Which metric(s) to evaluate as 'best single' (default: both).")
    parser.add_argument("--label_base", type=str, default="tab:bestSingle",
                        help="Base LaTeX label; metric is appended (e.g., tab:bestSingle_IoU).")
    args = parser.parse_args()

    metrics = ["IoU", "BIoU"] if args.metrics == "both" else [args.metrics]

    dataset_names = []
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {m: OrderedDict() for m in metrics}

    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")
        bn = os.path.basename(bd_clean).lower()
        ds_name = (os.path.basename(os.path.dirname(bd_clean)) if bn == "sota" else os.path.basename(bd_clean))
        dataset_names.append(ds_name)

        for metric in metrics:
            experts, agg_dist, models = evaluate_dataset_best_share(bd_clean, metric)
            all_results[metric][ds_name] = agg_dist

    # Emit one LaTeX table per metric in the CLI dataset order
    for metric in metrics:
        ordered = OrderedDict((ds, all_results[metric][ds]) for ds in dataset_names)
        table = latex_best_expert_table(ordered, metric, label=args.label_base)
        print(table)
        print()  # spacer

if __name__ == "__main__":
    main()


    '''
    python plotAllSingleExpertsDistribution.py \
        /home/sprice/PromptMoE/paper_results/ablation/experts/BIG/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/DAVIS585/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/ECSSD/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/VOC/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/MSRA-B/ \
        --metrics IoU 
    '''