#!/usr/bin/env python3
import os
import glob
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def _canon(s: str) -> str:
    """Case/char-normalize a name for loose matching."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _find_method_dir(dataset_dir: str, ours_name: str) -> str:
    """Find the subdirectory for our method (SamRefiner-MoE by default)."""
    want = _canon(ours_name)
    candidates = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    for d in candidates:
        if _canon(os.path.basename(d)) == want:
            return d
    exact = os.path.join(dataset_dir, ours_name)
    if os.path.isdir(exact):
        return exact
    raise RuntimeError(f"Could not find method directory for '{ours_name}' under: {dataset_dir}")

def _expert_columns(df: pd.DataFrame) -> list[str]:
    """Expert columns = all except 'subdir'."""
    return [c for c in df.columns if c not in {"subdir"}]

def _load_weights_csv(csv_path: str, eps: float = 0.0) -> tuple[dict, int]:
    """
    Read a *_weights.csv and compute per-expert concentration:
      concentration = mean(weight > eps) over rows.
    Returns (expert->concentration, n_rows).
    """
    df = pd.read_csv(csv_path)
    cols = _expert_columns(df)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    n = len(df)
    if n == 0:
        return {c: np.nan for c in cols}, 0
    conc = {c: float((df[c].to_numpy() > eps).mean()) for c in cols}
    return conc, n

def _summarize_dataset(dataset_dir: str, dataset_name: str, ours_name: str, eps: float):
    """
    For one dataset:
      - locate SamRefiner-MoE dir
      - read every *_weights.csv (each = a base model)
      - compute per-model expert concentration
      - return equal-weight mean across models
    """
    method_dir = _find_method_dir(dataset_dir, ours_name)
    weight_csvs = sorted(glob.glob(os.path.join(method_dir, "*_weights.csv")))
    if not weight_csvs:
        raise RuntimeError(f"No '*_weights.csv' files found in: {method_dir}")

    per_model_conc = OrderedDict()
    expert_order = []
    for csv_path in weight_csvs:
        model = os.path.basename(csv_path)[:-len("_weights.csv")]
        conc, _n = _load_weights_csv(csv_path, eps=eps)
        if not expert_order:
            expert_order = list(conc.keys())
        else:
            for e in conc.keys():
                if e not in expert_order:
                    expert_order.append(e)
        per_model_conc[model] = conc

    # equal-weight mean across models (ignoring NaNs)
    ds_mean_conc = {}
    for e in expert_order:
        vals = [per_model_conc[m].get(e, np.nan) for m in per_model_conc.keys()]
        ds_mean_conc[e] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan

    return ds_mean_conc

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
# LaTeX table builder (single-column with adjustbox width=\linewidth)
# ----------------------------
def latex_expert_concentration_table(all_ds_conc: dict, caption_note: str | None = None, label: str = "tab:expertConcentration"):
    r"""
    Builds a LaTeX table:
      • Rows: experts
      • Cols: datasets + Mean
      • Cell: % of images where expert's weight > eps
      • Bold the max value per dataset (and in the Mean column) with \textbf (no math mode).
      • Wrapped in \begin{adjustbox}{width=\linewidth} inside a single-column table.
    """
    datasets = list(all_ds_conc.keys())
    datasets_esc = [_latex_escape(d) for d in datasets]

    # collect expert order (first-seen), then union
    expert_order = []
    seen = set()
    for ds in datasets:
        for e in all_ds_conc[ds].keys():
            if e not in seen:
                seen.add(e)
                expert_order.append(e)

    if caption_note is None:
        caption_note = ("Expert concentration (percentage of images with non-zero expert weight) "
                        "for \\textbf{SamRefiner-MoE} across datasets. Values are averaged equally across base models.")

    # best per dataset (for bolding)
    best_per_ds = {}
    for ds in datasets:
        vals = [100.0 * all_ds_conc[ds].get(e, np.nan) for e in expert_order]
        best_per_ds[ds] = np.nanmax(vals) if len(vals) else np.nan

    # best over Mean column (per expert mean across datasets)
    mean_per_expert = {}
    for e in expert_order:
        pv = [100.0 * all_ds_conc[ds].get(e, np.nan) for ds in datasets]
        mean_per_expert[e] = float(np.nanmean(pv)) if np.any(np.isfinite(pv)) else np.nan
    best_mean = np.nanmax(list(mean_per_expert.values())) if mean_per_expert else np.nan

    # column spec: left for expert names, right-aligned numeric columns
    colspec = "@{}l " + " ".join(["r"] * (len(datasets) + 1)) + " @{}"

    lines = []
    lines.append("\\begin{table}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption_note}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{adjustbox}{width=\\linewidth}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")  # <-- escaped {tabular}
    lines.append("\\toprule")
    lines.append(f" & \\multicolumn{{{len(datasets)+1}}}{{c}}{{Dataset}} \\\\")
    lines.append(f"\\cmidrule(lr){{2-{len(datasets)+1}}}")
    header = "Expert & " + " & ".join(datasets_esc + ["Mean"]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for e in expert_order:
        e_esc = _latex_escape(e)
        cells = []
        vals_for_mean = []
        for ds in datasets:
            v = all_ds_conc[ds].get(e, np.nan)
            if np.isnan(v):
                cells.append("NaN")
            else:
                pv = 100.0 * v
                s = f"{pv:.2f}\\%"
                if np.isfinite(best_per_ds[ds]) and abs(pv - best_per_ds[ds]) <= 1e-12:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
                vals_for_mean.append(pv)

        if vals_for_mean:
            m = float(np.nanmean(vals_for_mean))
            mcell = f"{m:.2f}\\%"
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
        description="Output a single LaTeX table summarizing expert concentration for SamRefiner-MoE."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Each dir is a dataset root containing a SamRefiner-MoE subdir with *_weights.csv files.")
    parser.add_argument("--ours_name", type=str, default="SamRefiner-MoE",
                        help="Directory name for our method (default: SamRefiner-MoE).")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="Threshold for 'non-zero' expert weight (default: > 0).")
    parser.add_argument("--label", type=str, default="tab:expertConcentration",
                        help="LaTeX label for the table.")
    args = parser.parse_args()

    all_ds_conc = OrderedDict()  # dataset -> {expert -> concentration in [0,1]}

    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                        if bn == "sota" else os.path.basename(bd_clean))

        ds_mean_conc = _summarize_dataset(bd_clean, dataset_name, ours_name=args.ours_name, eps=args.eps)
        all_ds_conc[dataset_name] = ds_mean_conc

    # LaTeX: expert concentration table (single table only)
    conc_table = latex_expert_concentration_table(all_ds_conc, label=args.label)

    # Print ONLY this LaTeX table
    print(conc_table)

if __name__ == "__main__":
    main()


    '''
    python plotAllExpertsDistribution.py \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/DAVIS585 \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/ECSSD \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/MSRA-B \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/VOC \
        --ours_name PromptMoE
    '''