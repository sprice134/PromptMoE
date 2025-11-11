
#!/usr/bin/env python3
import os
import glob
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

# ----------------------------
# Display names / ordering
# ----------------------------
METHOD_DISPLAY = {
    "Exp1": "1 PP",
    "Exp2": "1 PP + B",
    "Exp3": "1 PP + B + M",
    "Exp4": "1 PP + 1 NP + B + M",
    "Exp5": "5 PP + 1 NP + B + M",
    "Exp6": "5 PP + 1 NP + B + M + PPE",
    "Exp7": "5 PP + 1 NP + B + M + PPE + DES",

    # Base model renames (for header readability only)
    "ZQL9711_RMBG-2-Matting":       "RMBG-2",
    "briaai_RMBG-1.4":              "RMBG-1.4",
    "ZhengPeng7_BiRefNet-matting":  "BiRefNet",
}

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
    "vit_h"
]

# ----------------------------
# Helpers
# ----------------------------
def _disp(name: str) -> str:
    return METHOD_DISPLAY.get(name, METHOD_DISPLAY.get(name.lower(), name))

def discover_methods_and_models(base_dir):
    """Return (methods_in_order, base_models_sorted)."""
    method_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
    if not method_dirs:
        raise RuntimeError(f"No method subdirectories found in: {base_dir}")

    present = [os.path.basename(d) for d in method_dirs]
    ordered = [m for m in EXPERIMENT_ORDER if m in present]
    others  = [m for m in present if m not in EXPERIMENT_ORDER]
    methods = others + ordered

    all_base = set()
    for d in method_dirs:
        for p in glob.glob(os.path.join(d, "*.csv")):
            bn = os.path.splitext(os.path.basename(p))[0]
            if bn in {"weights", "weights_full"}: continue
            if bn.endswith("_weights") or bn.endswith("_weights_full"): continue
            if bn == "U-2-Net": continue
            all_base.add(bn)

    return methods, sorted(all_base)

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # keep only fully-paired rows
    cols = ["unrefined_iou","refined_iou","unrefined_boundary_iou","refined_boundary_iou"]
    df = df.dropna(subset=cols)
    return df

def paired_deltas_from_df(df):
    d_iou = (df["refined_iou"].to_numpy() - df["unrefined_iou"].to_numpy())
    d_bio = (df["refined_boundary_iou"].to_numpy() - df["unrefined_boundary_iou"].to_numpy())
    return d_iou, d_bio

def unref_means_from_df(df):
    return float(df["unrefined_iou"].mean()), float(df["unrefined_boundary_iou"].mean())

# Bootstrap utilities for head-to-head CI (datasets -> models -> images)
def nested_boot_dataset_ci(model_to_vals, n_boot=3000, alpha=0.05, rng=None):
    """Hierarchical bootstrap for dataset-level mean (equal-weight over models)."""
    if rng is None:
        rng = np.random.default_rng(123)
    models = [m for m,a in model_to_vals.items() if a.size > 0]
    if not models:
        return (np.nan, np.nan)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(models, size=len(models), replace=True)
        means = []
        for m in picked:
            arr = model_to_vals[m]
            n = arr.size
            idx = rng.integers(0, n, size=n)
            means.append(arr[idx].mean())
        boots[b] = np.mean(means) if means else np.nan
    boots = boots[~np.isnan(boots)]
    if boots.size == 0: return (np.nan, np.nan)
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return (lo, hi)

# ----------------------------
# Per-dataset summarization
# ----------------------------
def summarize_dataset(dataset_dir, dataset_name, exclude_names=None):
    """
    Returns:
      ds_table_rows: list of dicts per method with dataset-level mean Δ (no CI)
      ds_unref_means: dict with dataset-level mean unref IoU/BIoU (equal-weight over models)
      strata: dict[method] -> {'IoU': model->Δarray, 'BIoU': model->Δarray}
              also includes 'UNREF' -> {'IoU': model->array([mean]), 'BIoU': ...}
    """
    methods, base_models = discover_methods_and_models(dataset_dir)

    # Optionally exclude a specific method by raw or display name
    exclude_names_set = set(exclude_names or [])
    if exclude_names_set:
        methods = [m for m in methods if m not in exclude_names_set and _disp(m) not in exclude_names_set]

    # Collect paired deltas by method/model, and unref means by model
    deltas = defaultdict(lambda: defaultdict(lambda: (np.array([]), np.array([]))))
    unref_means = defaultdict(lambda: defaultdict(lambda: (np.nan, np.nan)))  # method->model->(IoU, BIoU)

    for method in methods:
        method_dir = os.path.join(dataset_dir, method)
        for model in base_models:
            csv_path = os.path.join(method_dir, f"{model}.csv")
            if not os.path.isfile(csv_path):
                continue
            df = load_csv(csv_path)
            d_iou, d_bio = paired_deltas_from_df(df)
            deltas[method][model] = (d_iou, d_bio)
            uiou, ubio = unref_means_from_df(df)
            unref_means[method][model] = (uiou, ubio)

    # Build dataset-level table (equal-weight over base models) — Δ only
    rows = []
    strata = {}

    # Compute dataset-level unref means (equal-weight over models)
    model_to_unref = OrderedDict()
    for model in base_models:
        for method in methods:
            if model in unref_means[method] and not np.isnan(unref_means[method][model][0]):
                ui, ub = unref_means[method][model]
                model_to_unref[model] = (ui, ub)
                break

    if model_to_unref:
        ds_unref_iou = float(np.mean([v[0] for v in model_to_unref.values()]))
        ds_unref_bio = float(np.mean([v[1] for v in model_to_unref.values()]))
    else:
        ds_unref_iou = np.nan
        ds_unref_bio = np.nan

    # record unref strata as scalar arrays
    strata.setdefault("UNREF", {"IoU": {}, "BIoU": {}})
    for m, (ui, ub) in model_to_unref.items():
        strata["UNREF"]["IoU"][m]  = np.array([ui], dtype=float)
        strata["UNREF"]["BIoU"][m] = np.array([ub], dtype=float)

    for method in methods:
        model_means_iou, model_means_bio = [], []
        strata.setdefault(_disp(method), {"IoU": {}, "BIoU": {}})
        n_models, n_imgs = 0, 0
        for model in base_models:
            d_iou, d_bio = deltas[method][model]
            if d_iou.size == 0:
                continue
            n_models += 1
            n_imgs   += d_iou.size
            model_means_iou.append(float(d_iou.mean()))
            model_means_bio.append(float(d_bio.mean()))
            strata[_disp(method)]["IoU"][model]  = d_iou
            strata[_disp(method)]["BIoU"][model] = d_bio

        if n_models == 0:
            continue
        ds_mean_iou  = float(np.mean(model_means_iou))
        ds_mean_bio  = float(np.mean(model_means_bio))
        rows.append({
            "Dataset": dataset_name,
            "Method": _disp(method),
            "ΔIoU_mean": ds_mean_iou,
            "ΔBIoU_mean": ds_mean_bio,
            "#Models": n_models,
            "#Images(total)": n_imgs,
        })

    # order rows by preferred method ordering
    def method_sort_key(row):
        name = row["Method"]
        try:
            idx = [_disp(x) for x in EXPERIMENT_ORDER].index(name)
            return (0, idx, name)
        except ValueError:
            return (1, name, name)

    rows.sort(key=method_sort_key)

    ds_unref_means = {"IoU": ds_unref_iou, "BIoU": ds_unref_bio}
    return rows, ds_unref_means, strata

# ----------------------------
# Name prettifier for LaTeX left column
# ----------------------------
_PRETTY_METHOD = {
    "samrefiner": "SamRefiner \\cite{lin2025samrefiner}",
    "segrefiner-hr": "SegRefiner-HR \\cite{wang2023segrefiner}",
    "segrefiner-lr": "SegRefiner-LR \\cite{wang2023segrefiner}",
    "cascadepsp-slow": "CascadePSP-Slow \\cite{cheng2020cascadepsp}",
    "cascadepsp-fast": "CascadePSP-Fast \\cite{cheng2020cascadepsp}",
    "dualsight": "DualSight \\cite{price2025dualsight}",
}
def _hyphen(s: str) -> str:
    return s.replace("_", "-")
def _pretty_method(name: str) -> str:
    key = _hyphen(name).lower()
    return _PRETTY_METHOD.get(key, _hyphen(name))

# ----------------------------
# Cross-dataset latex tables (head-to-head CIs)
# ----------------------------
def latex_head_to_head_ci_tables(all_strata_by_method, datasets, ours_name="SamRefiner-MoE", n_boot=3000, alpha=0.05):
    """
    Build two LaTeX table strings:
      1) IoU ΔΔ 95% CI brackets per dataset (+ Mean column)
      2) BIoU ΔΔ 95% CI brackets per dataset (+ Mean column)

    Coloring rule (per-number):
      • Each bound is colored independently (green if >0, red if <0, black if ==0).
      • Brackets and commas remain black.
    """
    def color_num(val):
        if np.isnan(val):
            return "NaN"
        if val > 0:
            return f"+{val:.2f}"
        if val < 0:
            return f"{val:.2f}"
        return f"{val:.2f}"

    def colored_bracket(lo_pp, hi_pp):
        if np.isnan(lo_pp) or np.isnan(hi_pp):
            return "[NaN, NaN]"
        lo_txt = color_num(lo_pp)
        hi_txt = color_num(hi_pp)
        # brackets and comma uncolored (black)
        return f"[{lo_txt}, {hi_txt}]"

    ours_disp = _disp(ours_name)
    rng = np.random.default_rng(123)
    comparators = [m for m in all_strata_by_method.keys() if m not in {"UNREF", ours_disp}]

    # Sort comparators by preferred display order when applicable
    display_order = [_disp(x) for x in EXPERIMENT_ORDER]
    comparators_sorted = sorted(
        comparators,
        key=lambda name: (0, display_order.index(name)) if name in display_order else (1, name)
    )

    def ci_per_dataset_for(comp, metric_key):
        """Return (cells_per_dataset, lohi_per_dataset) where lohi_per_dataset is list of (lo,hi) floats."""
        cells = []
        lohi = []
        for ds in datasets:
            if ours_disp not in all_strata_by_method or ds not in all_strata_by_method[ours_disp]:
                cells.append("[NaN, NaN]"); lohi.append((np.nan, np.nan)); continue
            if comp not in all_strata_by_method or ds not in all_strata_by_method[comp]:
                cells.append("[NaN, NaN]"); lohi.append((np.nan, np.nan)); continue

            ours_by_model = all_strata_by_method[ours_disp][ds][metric_key]
            comp_by_model = all_strata_by_method[comp][ds][metric_key]
            common_models = sorted(set(ours_by_model.keys()) & set(comp_by_model.keys()))
            if not common_models:
                cells.append("[NaN, NaN]"); lohi.append((np.nan, np.nan)); continue

            model_dd = OrderedDict()
            for m in common_models:
                a = ours_by_model[m]
                b = comp_by_model[m]
                n = min(a.size, b.size)
                if n == 0:
                    continue
                model_dd[m] = a[:n] - b[:n]

            if not model_dd:
                cells.append("[NaN, NaN]"); lohi.append((np.nan, np.nan)); continue

            lo, hi = nested_boot_dataset_ci(model_dd, n_boot=n_boot, alpha=alpha, rng=rng)
            lo_pp, hi_pp = 100*lo, 100*hi
            cells.append(colored_bracket(lo_pp, hi_pp))
            lohi.append((lo_pp, hi_pp))
        return cells, lohi

    def mean_interval(lohi_list):
        """Average lo and hi across datasets where both are finite; color each bound independently."""
        fin = [(lo,hi) for (lo,hi) in lohi_list if np.isfinite(lo) and np.isfinite(hi)]
        if not fin:
            return "[NaN, NaN]"
        arr = np.array(fin, dtype=float)
        lo_m = float(np.mean(arr[:,0]))
        hi_m = float(np.mean(arr[:,1]))
        return colored_bracket(lo_m, hi_m)

    # Build IoU table
    iou_lines = []
    iou_lines.append("\\begin{table*}[!h]")
    iou_lines.append("\\centering")
    iou_lines.append("\\small")
    iou_lines.append(
        f"\\caption{{Head-to-head 95\\% confidence intervals for $\\Delta\\Delta$IoU (percentage points), "
        f"comparing \\textbf{{{_hyphen(ours_disp)}}} against each comparator (row) across datasets (columns). "
        f"$\\Delta\\Delta$ = $\\Delta$Ours $-$ $\\Delta$Comparator.}}"
    )
    iou_lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    iou_lines.append("\\begin{tabular}{@{\\extracolsep{\\fill}}l " + " ".join(["c"]*(len(datasets)+1)) + "}")
    iou_lines.append("\\toprule")
    iou_lines.append("Comparator " + " & " + " & ".join([_hyphen(d) for d in datasets] + ["Mean"]) + " \\\\")
    iou_lines.append("\\midrule")
    for comp in comparators_sorted:
        cells, lohi = ci_per_dataset_for(comp, "IoU")
        cells.append(mean_interval(lohi))
        iou_lines.append(f"{_pretty_method(comp)} " + " & " + " & ".join(cells) + " \\\\")
    iou_lines.append("\\bottomrule")
    iou_lines.append("\\end{tabular}")
    iou_lines.append("\\end{adjustbox}")
    iou_lines.append("\\end{table*}")
    iou_table = "\n".join(iou_lines)

    # Build BIoU table
    bio_lines = []
    bio_lines.append("\\begin{table*}[t]")
    bio_lines.append("\\centering")
    bio_lines.append("\\small")
    bio_lines.append(
        f"\\caption{{Head-to-head 95\\% confidence intervals for $\\Delta\\Delta$BIoU (percentage points), "
        f"comparing \\textbf{{{_hyphen(ours_disp)}}} against each comparator (row) across datasets (columns). "
        f"$\\Delta\\Delta$ = $\\Delta$Ours $-$ $\\Delta$Comparator.}}"
    )
    bio_lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    bio_lines.append("\\begin{tabular}{@{\\extracolsep{\\fill}}l " + " ".join(["c"]*(len(datasets)+1)) + "}")
    bio_lines.append("\\toprule")
    bio_lines.append("Comparator " + " & " + " & ".join([_hyphen(d) for d in datasets] + ["Mean"]) + " \\\\")
    bio_lines.append("\\midrule")
    for comp in comparators_sorted:
        cells, lohi = ci_per_dataset_for(comp, "BIoU")
        cells.append(mean_interval(lohi))
        bio_lines.append(f"{_pretty_method(comp)} " + " & " + " & ".join(cells) + " \\\\")
    bio_lines.append("\\bottomrule")
    bio_lines.append("\\end{tabular}")
    bio_lines.append("\\end{adjustbox}")
    bio_lines.append("\\end{table*}")
    bio_table = "\n".join(bio_lines)

    return iou_table, bio_table

# ----------------------------
# Cross-dataset summary latex (colored, with Unrefined row)
# ----------------------------
def latex_cross_dataset_summary_table_colored(all_rows_by_dataset, all_strata_by_method, caption_note=None, ours_name="SamRefiner-MoE"):
    r"""
    Build a LaTeX table:
      • First row: Unrefined (mean IoU/BIoU per dataset, NOT deltas), uncolored.
      • Then \midrule, then methods.
      • For methods: show ΔIoU / ΔBIoU in pp, green if >0, red if <0, zero plain.
      • Bold the best (max) Δ per dataset for IoU and BIoU separately (in math mode).
      • Also bold the best method in the Mean column (IoU and BIoU separately).
      • Highlight the SAMRefiner-MoE row with \rowcolor{ForestGreen!20}.
      • Use hyphens instead of underscores in names, and apply pretty mapping for selected methods.
    """
    datasets = list(all_rows_by_dataset.keys())

    # method -> {dataset: (ΔIoU_mean, ΔBIoU_mean)}
    method_to_dsmeans = defaultdict(dict)
    for ds, rows in all_rows_by_dataset.items():
        for r in rows:
            method_to_dsmeans[r["Method"]][ds] = (r["ΔIoU_mean"], r["ΔBIoU_mean"])

    # collect Unrefined means from strata 'UNREF'
    unref_means = {}
    if "UNREF" in all_strata_by_method:
        for ds in datasets:
            iu = []
            bu = []
            for _, arr in all_strata_by_method["UNREF"].get(ds, {}).get("IoU", {}).items():
                iu.append(float(arr.mean()))
            for _, arr in all_strata_by_method["UNREF"].get(ds, {}).get("BIoU", {}).items():
                bu.append(float(arr.mean()))
            unref_means[ds] = (float(np.mean(iu)) if iu else np.nan,
                               float(np.mean(bu)) if bu else np.nan)
    else:
        for ds in datasets:
            unref_means[ds] = (np.nan, np.nan)

    # Determine display order
    display_order = [_disp(x) for x in EXPERIMENT_ORDER]
    methods_sorted = sorted(
        method_to_dsmeans.keys(),
        key=lambda n: (0, display_order.index(n)) if n in display_order else (1, n)
    )

    # Identify best per-dataset values
    best_iou = {ds: -np.inf for ds in datasets}
    best_bio = {ds: -np.inf for ds in datasets}
    for method, dsmap in method_to_dsmeans.items():
        for ds in datasets:
            if ds in dsmap:
                mi = 100 * dsmap[ds][0]
                mb = 100 * dsmap[ds][1]
                if mi > best_iou[ds]:
                    best_iou[ds] = mi
                if mb > best_bio[ds]:
                    best_bio[ds] = mb

    # Identify best MEAN across methods (exclude methods missing all datasets)
    mean_iou_per_method = {}
    mean_bio_per_method = {}
    for method, dsmap in method_to_dsmeans.items():
        vals = [(100*dsmap[ds][0], 100*dsmap[ds][1]) for ds in datasets if ds in dsmap]
        if vals:
            arr = np.array(vals, dtype=float)
            mean_iou_per_method[method] = float(np.mean(arr[:,0]))
            mean_bio_per_method[method] = float(np.mean(arr[:,1]))
    best_mean_iou = max(mean_iou_per_method.values()) if mean_iou_per_method else -np.inf
    best_mean_bio = max(mean_bio_per_method.values()) if mean_bio_per_method else -np.inf

    def fmt_delta_text(val):
        """Return LaTeX (no surrounding $) for a single value in percentage points."""
        if np.isnan(val):
            return "NaN"
        if val > 0:
            return f"+{val:.2f}"
        if val < 0:
            return f"{val:.2f}"
        return f"{val:.2f}"

    def cell_str(mi, mb, is_best_i, is_best_b):
        s_i = fmt_delta_text(mi)
        s_b = fmt_delta_text(mb)
        if is_best_i:
            s_i = f"\\mathbf{{{s_i}}}"
        if is_best_b:
            s_b = f"\\mathbf{{{s_b}}}"
        return f"${s_i}$ / ${s_b}$"

    # Build LaTeX lines
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    if caption_note is None:
        caption_note = ("Comparative study with state-of-the-art refinement methods across 5 benchmark datasets. "
                        "Metrics are mean improvement ($\\Delta$) in Intersection-over-Union (IoU) and Boundary IoU over the unrefined base masks. "
                        "Positive values indicate improvement and negative indicate degradation of mask quality. Our method is highlighted in green.")
    lines.append(f"\\caption{{{caption_note}}}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    header_cols = " & ".join([_hyphen(d) for d in datasets] + ["Mean $\\Delta$IoU / $\\Delta$BIoU"])
    lines.append("\\begin{tabular}{@{\\extracolsep{\\fill}}l " + " ".join(["c"]*len(datasets)) + " | c}")
    lines.append("\\toprule")
    lines.append(f"Method & {header_cols} \\\\")
    lines.append("\\midrule")

    # Unrefined row (raw means, not deltas)
    unref_cells = []
    for ds in datasets:
        u = unref_means.get(ds, (np.nan, np.nan))
        if np.isnan(u[0]) or np.isnan(u[1]):
            unref_cells.append("NaN / NaN")
        else:
            unref_cells.append(f"{100*u[0]:.2f} / {100*u[1]:.2f}")
    # overall unref equal-weight
    over_unref_i = np.nanmean([u[0] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_b = np.nanmean([u[1] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_cell = f"{100*over_unref_i:.2f} / {100*over_unref_b:.2f}" if np.isfinite(over_unref_i) and np.isfinite(over_unref_b) else "NaN / NaN"
    lines.append("Unrefined & " + " & ".join(unref_cells + [over_unref_cell]) + " \\\\")
    lines.append("\\midrule")

    ours_disp = _disp(ours_name)

    # Methods (with coloring, bolding bests and bolding best mean)
    for method in methods_sorted:
        ds_vals = []
        cells = []
        for ds in datasets:
            if ds in method_to_dsmeans[method]:
                mi = 100 * method_to_dsmeans[method][ds][0]
                mb = 100 * method_to_dsmeans[method][ds][1]
                ds_vals.append((mi, mb))
                cells.append(cell_str(mi, mb, mi >= best_iou[ds] - 1e-9, mb >= best_bio[ds] - 1e-9))
            else:
                cells.append("NaN / NaN")

        if ds_vals:
            overall_iou  = float(np.mean([v[0] for v in ds_vals]))
            overall_bio  = float(np.mean([v[1] for v in ds_vals]))
            mean_cell = cell_str(overall_iou, overall_bio,
                                 overall_iou >= best_mean_iou - 1e-9,
                                 overall_bio >= best_mean_bio - 1e-9)
        else:
            mean_cell = "NaN / NaN"

        disp_name = _pretty_method(method)
        if method == ours_disp:
            lines.append("\\rowcolor{ForestGreen!20}")
            disp_name += " (Ours)"
        lines.append(f"{disp_name} & " + " & ".join(cells + [mean_cell]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")

    return "\n".join(lines)

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="(Cross-dataset) Output LaTeX tables: head-to-head ΔΔIoU/ΔΔBIoU 95% CI tables (with mean column), and a colored cross-dataset summary table with Unrefined row."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Each dir contains method subdirs with per-model CSVs.")
    parser.add_argument("--ours_name", type=str, default="SamRefiner-MoE",
                        help="Display/dir name for our method (default: SamRefiner-MoE).")
    parser.add_argument("--boots", type=int, default=3000, help="Bootstrap samples for head-to-head CIs.")
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - CI level for head-to-head (default 0.05 => 95%% CI).")
    parser.add_argument("--exclude_names", nargs="+", default=[],
                        help="List of method names to exclude from evaluation (match raw or display name).")
    args = parser.parse_args()

    all_rows_by_dataset = OrderedDict()         # ds -> [rows]
    all_strata_by_method = defaultdict(dict)    # method -> ds -> {'IoU': model->arr, 'BIoU': model->arr}
                                                # also 'UNREF' -> ds -> {'IoU': model->arr([mean]), 'BIoU': ...}

    datasets = []
    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                        if bn == "sota" else os.path.basename(bd_clean))
        datasets.append(dataset_name)

        rows, ds_unref_means, strata = summarize_dataset(bd_clean, dataset_name, exclude_names=args.exclude_names)
        all_rows_by_dataset[dataset_name] = rows

        # merge strata
        for method, v in strata.items():
            all_strata_by_method[method].setdefault(dataset_name, {})
            all_strata_by_method[method][dataset_name] = v

    # LaTeX: head-to-head CI tables (with per-number coloring and mean column)
    iou_table, bio_table = latex_head_to_head_ci_tables(
        all_strata_by_method, datasets, ours_name=args.ours_name, n_boot=args.boots, alpha=args.alpha
    )

    # LaTeX: colored cross-dataset summary with Unrefined row (updated caption, bold mean column)
    summary_caption = ("Comparative study with state-of-the-art refinement methods across 5 benchmark datasets. "
                       "Metrics are mean improvement ($\\Delta$) in Intersection-over-Union (IoU) and Boundary IoU over the unrefined base masks. "
                       "Positive values indicate improvement and negative indicate degradation of mask quality. Our method is highlighted in green.")
    summary_table = latex_cross_dataset_summary_table_colored(
        all_rows_by_dataset, all_strata_by_method, caption_note=summary_caption, ours_name=args.ours_name
    )

    # Print ONLY these three LaTeX tables
    print(iou_table)
    print()
    print(bio_table)
    print()
    print(summary_table)

if __name__ == "__main__":
    main()



"""

python plotAllEval.py \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/BIG \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/DAVIS585 \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/ECSSD \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/MSRA-B \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/VOC \
        --ours_name PromptMoE --exclude_name PromptMoE-All PromptMoE-Lite PromptMoE-HQ


python plotAllEval.py \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/BIG \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/DAVIS585 \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/ECSSD \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/MSRA-B \
        /home/sprice/PromptMoE/PromptMoE_Paper_Results/ablation/SOTA_COMPARE/VOC \
        --ours_name PromptMoE-HQ --exclude_name PromptMoE-All PromptMoE-Lite

"""



