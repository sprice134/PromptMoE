#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

# ---------------------------------------------------------------------
# Display names / ordering (updated to match your console summary)
# ---------------------------------------------------------------------
METHOD_DISPLAY = {
    "Exp1": "1 PP",
    "Exp2": "1 PP + B",
    "Exp3": "1 PP + B + M",
    "Exp4": "1 PP + 1 NP + B + M",
    "Exp5": "5 PP + 1 NP + B + M",
    "Exp6": "5 PP + 1 NP + B + M + PPE",
    "Exp7": "5 PP + 1 NP + B + M + PPE + DES",

    # Base model renames (if any appear)
    "ZQL9711_RMBG-2-Matting":       "RMBG-2",
    "briaai_RMBG-1.4":              "RMBG-1.4",
    "ZhengPeng7_BiRefNet-matting":  "BiRefNet",
}

# Experiments (forced order)
EXPERIMENT_ORDER = ["Exp1","Exp2","Exp3","Exp4","Exp5","Exp6","Exp7"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _disp(name: str) -> str:
    return METHOD_DISPLAY.get(name, METHOD_DISPLAY.get(name.lower(), name))

def _list_subdirs(p):
    return sorted(d for d in glob.glob(os.path.join(p, "*")) if os.path.isdir(d))

def discover_methods_and_models(base_dir):
    """
    Return (methods_in_order, base_models_sorted).
    Methods are FORCED to Exp1..Exp7 if present.
    """
    present = [os.path.basename(d) for d in _list_subdirs(base_dir)]
    methods = [m for m in EXPERIMENT_ORDER if m in present]
    if not methods:
        raise RuntimeError(f"No Exp1–Exp7 subdirectories found in: {base_dir}")

    # collect base models from any method folder
    all_base = set()
    for m in methods:
        for p in glob.glob(os.path.join(base_dir, m, "*.csv")):
            bn = os.path.splitext(os.path.basename(p))[0]
            if bn in {"weights", "weights_full"}: 
                continue
            if bn.endswith("_weights") or bn.endswith("_weights_full"): 
                continue
            if bn == "U-2-Net":
                continue
            all_base.add(bn)

    if not all_base:
        raise RuntimeError(f"No base-model CSVs found under Exp1–Exp7 in: {base_dir}")

    return methods, sorted(all_base)

def _get_series(df, possible_names):
    """Case-insensitive, whitespace-insensitive column fetch."""
    cols_norm = {c.strip().lower(): c for c in df.columns}
    for nm in possible_names:
        hit = cols_norm.get(nm.strip().lower(), None)
        if hit is not None:
            return pd.to_numeric(df[hit], errors="coerce")
    return None

def paired_deltas_from_df(df):
    """
    Robust delta extraction:
      1) If delta columns exist (delta_iou / delta_biou OR d_iou / d_biou), use them.
      2) Else, if (refined - unrefined) columns exist, compute per-image deltas.
      3) Drop NaNs row-wise.
    Returns arrays in FRACTION (0–1), not percent.
    """
    # Direct delta columns
    d_iou = _get_series(df, ["delta_iou", "d_iou", "delta-iou", "deltaiou", "diou"])
    d_bio = _get_series(df, [
        "delta_biou","d_biou","delta-biou","deltabiou","delta_boundary_iou","delta_boundaryiou"
    ])
    if d_iou is not None and d_bio is not None:
        tmp = pd.DataFrame({"diou": d_iou, "dbio": d_bio}).dropna()
        if tmp.empty:
            return np.array([], float), np.array([], float)
        di = tmp["diou"].to_numpy(float)
        db = tmp["dbio"].to_numpy(float)
        # If values look like percents, convert to fraction
        if (np.nanmax(np.abs(di)) > 1.5) or (np.nanmax(np.abs(db)) > 1.5):
            di = di / 100.0
            db = db / 100.0
        return di, db

    # Compute from refined/unrefined
    u_iou = _get_series(df, ["unrefined_iou", "unref_iou", "iou_unrefined"])
    r_iou = _get_series(df, ["refined_iou", "iou_refined"])
    u_bio = _get_series(df, ["unrefined_boundary_iou", "unref_biou", "boundary_iou_unrefined", "unrefined_biou"])
    r_bio = _get_series(df, ["refined_boundary_iou", "ref_biou", "boundary_iou_refined", "refined_biou"])

    if u_iou is not None and r_iou is not None and u_bio is not None and r_bio is not None:
        tmp = pd.DataFrame({
            "u_i": u_iou, "r_i": r_iou,
            "u_b": u_bio, "r_b": r_bio
        }).dropna()
        if tmp.empty:
            return np.array([], float), np.array([], float)
        di = (tmp["r_i"].to_numpy(float) - tmp["u_i"].to_numpy(float))
        db = (tmp["r_b"].to_numpy(float) - tmp["u_b"].to_numpy(float))
        return di, db

    return np.array([], float), np.array([], float)

def unref_means_from_df(df):
    u_iou = _get_series(df, ["unrefined_iou", "unref_iou", "iou_unrefined"])
    u_bio = _get_series(df, ["unrefined_boundary_iou", "unref_biou", "boundary_iou_unrefined", "unrefined_biou"])
    if u_iou is not None and u_bio is not None:
        return float(u_iou.mean()), float(u_bio.mean())
    return np.nan, np.nan

# ---------------------------------------------------------------------
# Per-dataset summarize
# ---------------------------------------------------------------------
def summarize_dataset(dataset_dir, dataset_name):
    """
    Returns:
      rows: list of dicts per method with dataset-level mean Δ (no CI)
      ds_unref_means: {'IoU': mean_unref, 'BIoU': mean_unref}
      strata: dict[method] -> {'IoU': model->Δarray, 'BIoU': model->Δarray}
              and 'UNREF' -> {'IoU': model->array([mean]), 'BIoU': ...}
    """
    methods, base_models = discover_methods_and_models(dataset_dir)

    deltas = defaultdict(lambda: defaultdict(lambda: (np.array([]), np.array([]))))
    unref_means = defaultdict(lambda: defaultdict(lambda: (np.nan, np.nan)))

    for method in methods:
        mdir = os.path.join(dataset_dir, method)
        for model in base_models:
            csv_path = os.path.join(mdir, f"{model}.csv")
            if not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            d_iou, d_bio = paired_deltas_from_df(df)
            deltas[method][model] = (d_iou, d_bio)
            uiou, ubio = unref_means_from_df(df)
            unref_means[method][model] = (uiou, ubio)

    # dataset-level unref means (equal-weight over models)
    model_to_unref = OrderedDict()
    for model in base_models:
        for method in methods:
            ui, ub = unref_means[method][model]
            if not np.isnan(ui) and not np.isnan(ub):
                model_to_unref[model] = (ui, ub)
                break

    if model_to_unref:
        ds_unref_iou = float(np.mean([v[0] for v in model_to_unref.values()]))
        ds_unref_bio = float(np.mean([v[1] for v in model_to_unref.values()]))
    else:
        ds_unref_iou = np.nan
        ds_unref_bio = np.nan

    strata = {"UNREF": {"IoU": {}, "BIoU": {}}}
    for m, (ui, ub) in model_to_unref.items():
        strata["UNREF"]["IoU"][m]  = np.array([ui], float)
        strata["UNREF"]["BIoU"][m] = np.array([ub], float)

    rows = []
    for method in methods:
        model_means_iou, model_means_bio = [], []
        strata.setdefault(_disp(method), {"IoU": {}, "BIoU": {}})
        n_models = 0
        for model in base_models:
            d_iou, d_bio = deltas[method][model]
            if d_iou.size == 0:
                continue
            n_models += 1
            model_means_iou.append(float(d_iou.mean()))
            model_means_bio.append(float(d_bio.mean()))
            strata[_disp(method)]["IoU"][model]  = d_iou
            strata[_disp(method)]["BIoU"][model] = d_bio
        if n_models == 0:
            continue
        rows.append({
            "Dataset": dataset_name,
            "Method": method,  # keep raw method key here; display later
            "ΔIoU_mean": float(np.mean(model_means_iou)),
            "ΔBIoU_mean": float(np.mean(model_means_bio)),
        })

    rows.sort(key=lambda r: EXPERIMENT_ORDER.index(r["Method"]) if r["Method"] in EXPERIMENT_ORDER else 999)

    return rows, {"IoU": ds_unref_iou, "BIoU": ds_unref_bio}, strata

# ---------------------------------------------------------------------
# LaTeX cross-dataset summary (colored, with Unrefined row)
# ---------------------------------------------------------------------
def latex_cross_dataset_summary_table_colored(all_rows_by_dataset, all_strata_by_method, caption_note=None):
    datasets = list(all_rows_by_dataset.keys())

    # method(raw) -> {dataset: (ΔIoU_mean, ΔBIoU_mean)}
    method_to_dsmeans = defaultdict(dict)
    for ds, rows in all_rows_by_dataset.items():
        for r in rows:
            method_to_dsmeans[r["Method"]][ds] = (r["ΔIoU_mean"], r["ΔBIoU_mean"])

    # Unref means
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

    # only Exp1..Exp7 (present)
    methods_sorted = [m for m in EXPERIMENT_ORDER if m in method_to_dsmeans]

    # best per-dataset
    best_iou = {ds: -np.inf for ds in datasets}
    best_bio = {ds: -np.inf for ds in datasets}
    for method, dsmap in method_to_dsmeans.items():
        for ds in datasets:
            if ds in dsmap:
                mi = 100 * dsmap[ds][0]
                mb = 100 * dsmap[ds][1]
                if mi > best_iou[ds]: best_iou[ds] = mi
                if mb > best_bio[ds]: best_bio[ds] = mb

    # best means
    mean_iou_per_method = {}
    mean_bio_per_method = {}
    for method, dsmap in method_to_dsmeans.items():
        vals = [(100*dsmap[ds][0], 100*dsmap[ds][1]) for ds in datasets if ds in dsmap]
        if vals:
            arr = np.array(vals, float)
            mean_iou_per_method[method] = float(np.mean(arr[:,0]))
            mean_bio_per_method[method] = float(np.mean(arr[:,1]))
    best_mean_iou = max(mean_iou_per_method.values()) if mean_iou_per_method else -np.inf
    best_mean_bio = max(mean_bio_per_method.values()) if mean_bio_per_method else -np.inf

    def fmt_delta_text(val):
        if np.isnan(val):
            return "NaN"
        if val > 0:
            return f"\\textcolor{{ForestGreen}}{{+{val:.2f}}}"
        if val < 0:
            return f"\\textcolor{{red}}{{{val:.2f}}}"
        return f"{val:.2f}"

    def cell_str(mi, mb, is_best_i, is_best_b):
        s_i = fmt_delta_text(mi)
        s_b = fmt_delta_text(mb)
        if is_best_i: s_i = f"\\mathbf{{{s_i}}}"
        if is_best_b: s_b = f"\\mathbf{{{s_b}}}"
        return f"${s_i}$ / ${s_b}$"

    # Build LaTeX
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    if caption_note is None:
        caption_note = ("Quantitative evaluation of \\textbf{PromptMoE} ablation, highlighting the individual impact of Dynamic Expert Selection (DES), Point-Placement Exploration (PPE), the addition of multiple inclusion points, and overall prompt composition. We report the mean improvement ($\Delta$) in Intersection-over-Union (IoU) and Boundary IoU relative to the unrefined base masks for each dataset. Positive values indicate improvement (green), negative indicate degradation (red). The rightmost column reports a macro-average, equally weighting all five datasets.")
    lines.append(f"\\caption{{{caption_note}}}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    header_cols = " & ".join(datasets + ["Mean $\\Delta$IoU / $\\Delta$BIoU"])
    lines.append("\\begin{tabular}{@{\\extracolsep{\\fill}}l " + " ".join(["c"]*(len(datasets)+1)) + "}")
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
    over_unref_i = np.nanmean([u[0] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_b = np.nanmean([u[1] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_cell = f"{100*over_unref_i:.2f} / {100*over_unref_b:.2f}" if np.isfinite(over_unref_i) and np.isfinite(over_unref_b) else "NaN / NaN"
    lines.append("Unrefined & " + " & ".join(unref_cells + [over_unref_cell]) + " \\\\")
    lines.append("\\midrule")

    # Methods
    for method in methods_sorted:
        ds_vals = []
        cells = []
        for ds in datasets:
            if ds in method_to_dsmeans[method]:
                mi = 100 * method_to_dsmeans[method][ds][0]
                mb = 100 * method_to_dsmeans[method][ds][1]
                ds_vals.append((mi, mb))
                cells.append(cell_str(mi, mb, mi >= best_iou[ds]-1e-9, mb >= best_bio[ds]-1e-9))
            else:
                cells.append("NaN / NaN")

        if ds_vals:
            overall_iou  = float(np.mean([v[0] for v in ds_vals]))
            overall_bio  = float(np.mean([v[1] for v in ds_vals]))
            mean_cell = cell_str(overall_iou, overall_bio,
                                 overall_iou >= best_mean_iou-1e-9,
                                 overall_bio >= best_mean_bio-1e-9)
        else:
            mean_cell = "NaN / NaN"

        lines.append(f"{_disp(method)} & " + " & ".join(cells + [mean_cell]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")

    return "\n".join(lines)

# ---------------------------------------------------------------------
# Pretty console summary (like your working printout)
# ---------------------------------------------------------------------
def print_console_summary(agg_deltas, dataset_order, method_order):
    print("\n=== Cross-Dataset Δ (vs Unrefined) Summary ===\n")
    method_col_w = 28
    cell_w = 18
    final_w = 22

    hdr = f"{'Method':<{method_col_w}}"
    for ds in dataset_order:
        hdr += f"{ds:>{cell_w}}"
    hdr += f"{'Mean ΔIoU / ΔBIoU':>{final_w}}"
    print(hdr)
    print("-" * len(hdr))

    for method in method_order:
        disp = _disp(method)
        row = f"{disp:<{method_col_w}}"
        mi_vals, mb_vals = [], []
        for ds in dataset_order:
            mi, mb = agg_deltas.get(method, {}).get(ds, (np.nan, np.nan))
            mi_vals.append(mi); mb_vals.append(mb)
            if np.isnan(mi) or np.isnan(mb):
                cell = "   NaN / NaN   "
            else:
                cell = f"{mi:+6.2f} / {mb:+6.2f}"
            row += f"{cell:>{cell_w}}"
        mi_mean = pd.Series(mi_vals).mean()
        mb_mean = pd.Series(mb_vals).mean()
        if np.isnan(mi_mean) or np.isnan(mb_mean):
            overall = "    NaN / NaN    "
        else:
            overall = f"{mi_mean:+7.2f} / {mb_mean:+7.2f}"
        row += f"{overall:>{final_w}}"
        print(row)
    print()

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Output console summary and a LaTeX cross-dataset summary table (Exp1–Exp7 ablations only)."
    )
    parser.add_argument("--base_dirs", nargs="+",
                        default=[
                            "/home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/BIG",
                            "/home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/DAVIS585",
                            "/home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/ECSSD",
                            "/home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/VOC",
                            "/home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/MSRA-B",
                        ],
                        help="Dataset directories (each contains Exp1..Exp7 with per-model CSVs).")
    args = parser.parse_args()

    all_rows_by_dataset = OrderedDict()
    all_strata_by_method = defaultdict(dict)

    dataset_order = []
    agg_deltas = defaultdict(dict)  # raw method key -> {ds: (ΔIoU_pp, ΔBIoU_pp)}
    method_order = []               # keep Exp1..Exp7 in order if present

    for bd in args.base_dirs:
        bd = bd.rstrip(os.sep)
        ds_name = os.path.basename(bd)
        if ds_name not in dataset_order:
            dataset_order.append(ds_name)

        rows, ds_unref_means, strata = summarize_dataset(bd, ds_name)
        all_rows_by_dataset[ds_name] = rows

        # merge strata
        for method, v in strata.items():
            all_strata_by_method[method].setdefault(ds_name, {})
            all_strata_by_method[method][ds_name] = v

        # accumulate Δ means vs Unrefined for this dataset
        # method keys in rows are raw methods (Exp1..Exp7)
        for r in rows:
            method = r["Method"]
            mi = r["ΔIoU_mean"] * 100.0
            mb = r["ΔBIoU_mean"] * 100.0
            agg_deltas[method][ds_name] = (mi, mb)
            if method not in method_order and method in EXPERIMENT_ORDER:
                method_order.append(method)

    # Console summary like your “found them” printout
    if method_order:
        print_console_summary(agg_deltas, dataset_order, method_order)

    # LaTeX table
    summary_table = latex_cross_dataset_summary_table_colored(
        all_rows_by_dataset, all_strata_by_method
    )
    print(summary_table)

if __name__ == "__main__":
    main()


    '''
    python plotAllComponents.py \
    --base_dirs \
        /home/sprice/PromptMoE/results/ABLATION/components/BIG/ \
        /home/sprice/PromptMoE/results/ABLATION/components//DAVIS585/ \
        /home/sprice/PromptMoE/results/ABLATION/components//ECSSD/ \
        /home/sprice/PromptMoE/results/ABLATION/components//VOC/ \
        /home/sprice/PromptMoE/results/ABLATION/components//MSRA-B/ 

    '''
