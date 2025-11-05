#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import re
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
    "Exp4": "1 PP + B + M + G",
    "Exp5": "1 PP + 1 NP + B + M + G",
    "Exp6": "3 PP + 1 NP + B + M + G",
    "Exp7": "3 PP + 1 NP + B + M + G + SF",

    # ADDED FOR ROUTERS
    "E14_Router-T": "Router (Triplet)",
    "E15_Router-Q": "Router (Quadruplet)",

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
    # ADDED FOR ROUTERS
    "E14_Router-T",
    "E15_Router-Q",
    # END ADDITIONS
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

# ----------------------------
# Per-dataset summarization (table 1)
# ----------------------------
def summarize_dataset(dataset_dir, dataset_name):
    """
    Returns:
      ds_table_rows: list of dicts per method with dataset-level mean Δ (no CI)
      ds_unref_means: dict with dataset-level mean unref IoU/BIoU (equal-weight over models)
      strata: dict[method] -> {'IoU': model->Δarray, 'BIoU': model->Δarray}
              also includes 'UNREF' -> {'IoU': model->array([mean]), 'BIoU': ...}
    """
    methods, base_models = discover_methods_and_models(dataset_dir)

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
    "samrefiner": "SamRefiner",
    "segrefiner-hr": "SegRefiner-HR",
    "segrefiner-lr": "SegRefiner-LR",
    "cascadepsp-slow": "CascadePSP-Slow",
    "cascadepsp-fast": "CascadePSP-Fast",
    # ADDED FOR ROUTERS
    "e14-router-t": "Router (Triplet)",
    "e15-router-q": "Router (Quadruplet)",
}
def _hyphen(s: str) -> str:
    return s.replace("_", "-")
def _pretty_method(name: str) -> str:
    key = _hyphen(name).lower()
    return _PRETTY_METHOD.get(key, _hyphen(name))

# ----------------------------
# Cross-dataset summary latex (no colors, with Unrefined row)
# ----------------------------
def latex_cross_dataset_summary_table(all_rows_by_dataset, all_strata_by_method, caption_note=None, ours_name="SamRefiner-MoE"):
    r"""
    Build a LaTeX table:
      • First row: Unrefined (mean IoU/BIoU per dataset, NOT deltas).
      • Then \midrule, then methods.
      • For methods: show ΔIoU / ΔBIoU in pp, with +/- signs; no colors.
      • Bold the best (max) Δ per dataset for IoU and BIoU separately (in math mode).
      • Also bold the best method in the Mean column (IoU and BIoU separately).
      • No row color highlighting.
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

    # Identify best MEAN across methods
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
        """Return string for a single value in percentage points with sign."""
        if np.isnan(val):
            return "NaN"
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.2f}"

    def cell_str(mi, mb, is_best_i, is_best_b):
        s_i = fmt_delta_text(mi)
        s_b = fmt_delta_text(mb)
        if is_best_i:
            s_i = f"\\textbf{{{s_i}}}"
        if is_best_b:
            s_b = f"\\textbf{{{s_b}}}"
        return f"${s_i}$ / ${s_b}$"

    # Build LaTeX lines
    lines = []
    lines.append("\\begin{table*}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    if caption_note is None:
        caption_note = ("Quantitative comparison with state-of-the-art refinement methods across benchmark datasets. "
                        "We report the mean improvement ($\\Delta$) in Intersection-over-Union (IoU) and Boundary IoU over the unrefined base masks. "
                        "Values are percentage points with sign. The rightmost column averages equally across datasets.")
    lines.append(f"\\caption{{{caption_note}}}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    header_cols = " & ".join([_hyphen(d) for d in datasets] + ["Mean $\\Delta$IoU / $\\Delta$BIoU"])
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
    # overall unref equal-weight
    over_unref_i = np.nanmean([u[0] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_b = np.nanmean([u[1] for u in unref_means.values()]) if unref_means else np.nan
    over_unref_cell = f"{100*over_unref_i:.2f} / {100*over_unref_b:.2f}" if np.isfinite(over_unref_i) and np.isfinite(over_unref_b) else "NaN / NaN"
    lines.append("Unrefined & " + " & ".join(unref_cells + [over_unref_cell]) + " \\\\")
    lines.append("\\midrule")

    # Methods (bold bests; no colors)
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
        lines.append(f"{disp_name} & " + " & ".join(cells + [mean_cell]) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")

    return "\n".join(lines)

# ----------------------------
# Oracle / Average / Random (avg-of-5) / All / Routers stats per dataset
# ----------------------------
def compute_oracle_random_stats_for_dataset(dataset_dir: str):
    """
    Returns dict with keys: 'unref','worst','avg','best','random_avg5','all','router_single','router_pair', 'router_triplet', 'router_quad'
    Each non-'unref' value maps to (ΔIoU_mean, ΔBIoU_mean) as decimals (e.g., 0.0612).
    'unref' maps to raw means (IoU_mean, BIoU_mean) in decimals.
    """
    # Discover methods and base models
    methods, base_models = discover_methods_and_models(dataset_dir)

    # Identify the 10 single-expert methods: E01..E10 with allowed suffixes
    allowed_suffixes = {"Dp","Gd","Gc","sp","ctr","bri","tex","frac","Sm","prompt"}
    expert_methods = []
    for m in methods:
        base = os.path.basename(m)
        mobj = re.match(r"^E(0?[1-9]|10)_([A-Za-z]+)$", base)
        if not mobj:
            continue
        suffix = mobj.group(2)
        if suffix in allowed_suffixes:
            expert_methods.append(m)
    expert_methods = sorted(expert_methods)
    if len(expert_methods) != 10:
        raise RuntimeError(
            f"Expected exactly 10 single-expert methods in '{dataset_dir}' "
            f"but found {len(expert_methods)}: {expert_methods}"
        )

    # Directories for "macro" rows
    dir_all    = "E11_AllExperts"
    dir_rs     = "E12_Router-S"
    dir_rp     = "E13_Router-P"
    dir_rt     = "E14_Router-T"  # NEW
    dir_rq     = "E15_Router-Q"  # NEW

    # Include a base model only if all 10 expert CSVs exist for it
    model_iou_mats, model_bio_mats, included_models = {}, {}, []
    # Also collect unref means per model (we can read from the first expert)
    model_unref = {}  # model -> (IoU_mean, BIoU_mean)

    for model in base_models:
        expert_iou_lists, expert_bio_lists = [], []
        ok = True
        for em in expert_methods:
            csv_path = os.path.join(dataset_dir, em, f"{model}.csv")
            if not os.path.isfile(csv_path):
                ok = False
                break
            df = load_csv(csv_path)
            d_iou, d_bio = paired_deltas_from_df(df)
            expert_iou_lists.append(d_iou.astype(float))
            expert_bio_lists.append(d_bio.astype(float))
        if not ok or len(expert_iou_lists) != 10:
            continue
        min_len = min(arr.size for arr in expert_iou_lists)
        if min_len == 0:
            continue
        iou_mat = np.stack([arr[:min_len] for arr in expert_iou_lists], axis=0)   # (E, N)
        bio_mat = np.stack([arr[:min_len] for arr in expert_bio_lists], axis=0)   # (E, N)
        model_iou_mats[model] = iou_mat
        model_bio_mats[model] = bio_mat
        included_models.append(model)

        # compute unref means for this model using the same CSV (first expert)
        df_unref = load_csv(os.path.join(dataset_dir, expert_methods[0], f"{model}.csv"))
        uiou, ubio = unref_means_from_df(df_unref)
        model_unref[model] = (uiou, ubio)

    if not included_models:
        raise RuntimeError("No base models had complete (10/10) expert CSVs; cannot compute oracles/average/random.")

    def mean_over_models(per_model_vals):
        vals = [float(v) for v in per_model_vals]
        return float(np.mean(vals)) if vals else float("nan")

    # Unrefined (raw)
    unref_iou = mean_over_models([model_unref[m][0] for m in included_models if m in model_unref])
    unref_bio = mean_over_models([model_unref[m][1] for m in included_models if m in model_unref])

    # Worst / Average / Best (per model)
    worst_iou_per_model, avg_iou_per_model, best_iou_per_model = [], [], []
    worst_bio_per_model, avg_bio_per_model, best_bio_per_model = [], [], []
    for m in included_models:
        iou_mat = model_iou_mats[m]
        bio_mat = model_bio_mats[m]
        worst_iou_per_model.append(float(np.min(iou_mat, axis=0).mean()))
        avg_iou_per_model.append(float(np.mean(iou_mat, axis=0).mean()))
        best_iou_per_model.append(float(np.max(iou_mat, axis=0).mean()))
        worst_bio_per_model.append(float(np.min(bio_mat, axis=0).mean()))
        avg_bio_per_model.append(float(np.mean(bio_mat, axis=0).mean()))
        best_bio_per_model.append(float(np.max(bio_mat, axis=0).mean()))

    worst_iou = mean_over_models(worst_iou_per_model)
    avg_iou   = mean_over_models(avg_iou_per_model)
    best_iou  = mean_over_models(best_iou_per_model)
    worst_bio = mean_over_models(worst_bio_per_model)
    avg_bio   = mean_over_models(avg_bio_per_model)
    best_bio  = mean_over_models(best_bio_per_model)

    # Random (avg of 5 trials)
    rng = np.random.default_rng(123)
    rnd_iou_trials, rnd_bio_trials = [], []
    for _ in range(5):
        trial_iou_per_model, trial_bio_per_model = [], []
        for m in included_models:
            iou_mat = model_iou_mats[m]
            bio_mat = model_bio_mats[m]
            E, N = iou_mat.shape
            idx = rng.integers(0, E, size=N)
            iou_sel = iou_mat[idx, np.arange(N)]
            bio_sel = bio_mat[idx, np.arange(N)]
            trial_iou_per_model.append(float(iou_sel.mean()))
            trial_bio_per_model.append(float(bio_sel.mean()))
        rnd_iou_trials.append(mean_over_models(trial_iou_per_model))
        rnd_bio_trials.append(mean_over_models(trial_bio_per_model))
    rnd_iou = float(np.mean(rnd_iou_trials))
    rnd_bio = float(np.mean(rnd_bio_trials))

    # Helper: mean Δ for a method dir (equal-weight over models with data)
    def mean_delta_for_dir(dir_name: str):
        vals_i, vals_b = [], []
        for m in included_models:
            p = os.path.join(dataset_dir, dir_name, f"{m}.csv")
            if not os.path.isfile(p):
                continue
            df = load_csv(p)
            d_iou, d_bio = paired_deltas_from_df(df)
            if d_iou.size == 0:
                continue
            vals_i.append(float(d_iou.mean()))
            vals_b.append(float(d_bio.mean()))
        if not vals_i or not vals_b:
            return (float("nan"), float("nan"))
        return (float(np.mean(vals_i)), float(np.mean(vals_b)))

    all_iou, all_bio         = mean_delta_for_dir("E11_AllExperts")
    routerS_iou, routerS_bio = mean_delta_for_dir("E12_Router-S")
    routerP_iou, routerP_bio = mean_delta_for_dir("E13_Router-P")
    routerT_iou, routerT_bio = mean_delta_for_dir("E14_Router-T")  # NEW
    routerQ_iou, routerQ_bio = mean_delta_for_dir("E15_Router-Q")  # NEW

    return {
        "unref": (unref_iou, unref_bio),
        "worst": (worst_iou, worst_bio),
        "avg": (avg_iou, avg_bio),
        "best": (best_iou, best_bio),
        "random_avg5": (rnd_iou, rnd_bio),
        "all": (all_iou, all_bio),
        "router_single": (routerS_iou, routerS_bio),
        "router_pair": (routerP_iou, routerP_bio),
        "router_triplet": (routerT_iou, routerT_bio),  # NEW KEY
        "router_quad": (routerQ_iou, routerQ_bio),    # NEW KEY
    }

# ----------------------------
# LaTeX: cross-dataset table for Unref + Oracle/Avg/Random/All/Routers
# (no colors; bolding excludes Best-oracle)
# ----------------------------
def latex_oracle_random_cross_dataset_table(
    dataset_dirs, dataset_names
) -> str:
    """
    Builds one cross-dataset LaTeX table with rows (in this exact order):
      Unrefined
      \midrule
      Worst single (oracle)
      \midrule
      Random (avg×5)
      Average single
      All experts
      Router (single)
      Router (pair)
      Router (Triplet)
      Router (Quadruplet)
      \midrule
      Best single (oracle)

    Columns: each dataset + Mean column.
    Unrefined row shows raw IoU/BIoU means (percent). All other rows show ΔIoU/ΔBIoU (pp, signed).
    Bolding ignores the Best-oracle row when determining “best”.
    """
    # Collect stats per dataset, skipping those that fail
    ds_ok, stats_by_ds = [], {}
    for ddir, dname in zip(dataset_dirs, dataset_names):
        try:
            stats = compute_oracle_random_stats_for_dataset(ddir)
            ds_ok.append(dname)
            stats_by_ds[dname] = stats
        except RuntimeError as e:
            print(f"% Skipping oracle/random stats for {dname}: {e}")

    if not ds_ok:
        raise RuntimeError("No datasets produced oracle/random stats.")

    # Row spec and keys
    ordered_sections = [
        [("Unrefined", "unref")],
        [("Worst single (oracle)", "worst")],
        [
            ("Random (avg×5)", "random_avg5"),
            ("Average single", "avg"),
            ("All experts", "all"),
            ("Router (single)", "router_single"),
            ("Router (pair)", "router_pair"),
            ("Router (Triplet)", "router_triplet"),  # UPDATED
            ("Router (Quadruplet)", "router_quad"),    # UPDATED
        ],
        [("Best single (oracle)", "best")],
    ]
    flat_rows = [item for sec in ordered_sections for item in sec]

    # Determine bolding targets (exclude 'best' row)
    eligible_keys = [key for (label, key) in flat_rows if key not in {"best", "unref"}]

    # Prepare best-per-dataset and best-mean (excluding best)
    best_iou = {ds: -np.inf for ds in ds_ok}
    best_bio = {ds: -np.inf for ds in ds_ok}
    mean_iou_per_key = {}
    mean_bio_per_key = {}

    for label, key in flat_rows:
        if key == "unref":
            continue
        # per-dataset maxima (exclude 'best')
        if key in eligible_keys:
            for ds in ds_ok:
                mi = 100.0 * stats_by_ds[ds][key][0]
                mb = 100.0 * stats_by_ds[ds][key][1]
                if mi > best_iou[ds]:
                    best_iou[ds] = mi
                if mb > best_bio[ds]:
                    best_bio[ds] = mb
        # mean across datasets
        vals = [(100.0*stats_by_ds[ds][key][0], 100.0*stats_by_ds[ds][key][1]) for ds in ds_ok]
        arr = np.array(vals, dtype=float)
        mean_iou_per_key[key] = float(np.nanmean(arr[:,0]))
        mean_bio_per_key[key] = float(np.nanmean(arr[:,1]))

    # Best means across eligible rows
    best_mean_iou = max([mean_iou_per_key[k] for k in eligible_keys]) if eligible_keys else -np.inf
    best_mean_bio = max([mean_bio_per_key[k] for k in eligible_keys]) if eligible_keys else -np.inf

    def fmt_signed_pp(x: float) -> str:
        if np.isnan(x): return "NaN"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.2f}"

    def cell_delta(mi, mb, bold_i, bold_b):
        s_i = fmt_signed_pp(mi)
        s_b = fmt_signed_pp(mb)
        if bold_i: s_i = f"\\textbf{{{s_i}}}"
        if bold_b: s_b = f"\\textbf{{{s_b}}}"
        return f"${s_i}$ / ${s_b}$"

    # Build LaTeX
    lines = []
    lines.append("\\begin{table*}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Unrefined vs. oracle/average/random/all/routers across datasets. "
                 "Unrefined shows raw IoU/BIoU means (\\%). All other rows report mean $\\Delta$IoU / $\\Delta$BIoU (pp, signed), "
                 "equal-weight over base models per dataset. Bolding ignores the Best-oracle row.}")
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    header_cols = " & ".join([_hyphen(d) for d in ds_ok] + ["Mean $\\Delta$IoU / $\\Delta$BIoU"])
    lines.append("\\begin{tabular}{@{\\extracolsep{\\fill}}l " + " ".join(["c"]*(len(ds_ok)+1)) + "}")
    lines.append("\\toprule")
    lines.append(f"Method & {header_cols} \\\\")
    lines.append("\\midrule")

    for sec_idx, section in enumerate(ordered_sections):
        for label, key in section:
            if key == "unref":
                # Raw percentages
                cells = []
                for ds in ds_ok:
                    ui, ub = stats_by_ds[ds]["unref"]
                    cells.append(f"{100.0*ui:.2f} / {100.0*ub:.2f}")
                mean_ui = float(np.nanmean([stats_by_ds[ds]['unref'][0] for ds in ds_ok]))
                mean_ub = float(np.nanmean([stats_by_ds[ds]['unref'][1] for ds in ds_ok]))
                mean_cell = f"{100.0*mean_ui:.2f} / {100.0*mean_ub:.2f}"
                lines.append(f"{label} & " + " & ".join(cells + [mean_cell]) + " \\\\")
            else:
                # Deltas with +/- signs; bolding excludes 'best'
                cells = []
                for ds in ds_ok:
                    mi = 100.0 * stats_by_ds[ds][key][0]
                    mb = 100.0 * stats_by_ds[ds][key][1]
                    bold_i = (key in eligible_keys) and (mi >= best_iou[ds] - 1e-9)
                    bold_b = (key in eligible_keys) and (mb >= best_bio[ds] - 1e-9)
                    cells.append(cell_delta(mi, mb, bold_i, bold_b))
                m_i = mean_iou_per_key[key]
                m_b = mean_bio_per_key[key]
                bold_mean_i = (key in eligible_keys) and (m_i >= best_mean_iou - 1e-9)
                bold_mean_b = (key in eligible_keys) and (m_b >= best_mean_bio - 1e-9)
                mean_cell = cell_delta(m_i, m_b, bold_mean_i, bold_mean_b)
                lines.append(f"{label} & " + " & ".join(cells + [mean_cell]) + " \\\\")
        if sec_idx < len(ordered_sections) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

# ----------------------------
# NEW: Mean-only (three-column) table with separate IoU/BIoU columns
# (no colors; bolding excludes Best-oracle; no per-dataset columns)
# ----------------------------
def latex_oracle_random_mean_only_table(dataset_dirs, dataset_names) -> str:
    """
    Builds a compact LaTeX table with three columns:
      Method & Mean ΔIoU & Mean ΔBIoU

    • One row per method (like the ablation example).
    • Unrefined row shows raw means (%) for IoU and BIoU.
    • All other rows report mean Δ (pp, signed).
    • Bolding for each column ignores the Best-oracle row; Unrefined is never bolded.
    """
    # Collect stats across datasets
    ds_ok, stats_by_ds = [], {}
    for ddir, dname in zip(dataset_dirs, dataset_names):
        try:
            stats = compute_oracle_random_stats_for_dataset(ddir)
            ds_ok.append(dname)
            stats_by_ds[dname] = stats
        except RuntimeError as e:
            print(f"% Skipping oracle/random stats for {dname}: {e}")

    if not ds_ok:
        raise RuntimeError("No datasets produced oracle/random stats.")

    # Row spec/order matches the larger cross-dataset table
    ordered_sections = [
        [("Unrefined", "unref")],
        [("Worst single (oracle)", "worst")],
        [
            ("Random (avg×5)", "random_avg5"),
            ("Average single", "avg"),
            ("All experts", "all"),
            ("Router (Sing)", "router_single"),
            ("Router (Pair)", "router_pair"),
            ("Router (Triplet)", "router_triplet"),  # UPDATED
            ("Router (Quadruplet)", "router_quad"),    # UPDATED
        ],
        [("Best single (oracle)", "best")],
    ]
    flat_rows = [item for sec in ordered_sections for item in sec]

    # Mean over datasets for each key
    mean_iou_per_key = {}
    mean_bio_per_key = {}
    for _, key in flat_rows:
        vals = [(100.0*stats_by_ds[ds][key][0], 100.0*stats_by_ds[ds][key][1]) for ds in ds_ok]
        arr = np.array(vals, dtype=float)
        mean_iou_per_key[key] = float(np.nanmean(arr[:,0]))
        mean_bio_per_key[key] = float(np.nanmean(arr[:,1]))

    # Determine bolding targets (exclude best, unref) per metric
    eligible_keys = [key for (_, key) in flat_rows if key not in {"best","unref"}]
    best_mean_iou = max([mean_iou_per_key[k] for k in eligible_keys]) if eligible_keys else -np.inf
    best_mean_bio = max([mean_bio_per_key[k] for k in eligible_keys]) if eligible_keys else -np.inf

    def fmt_signed_pp(x: float) -> str:
        if np.isnan(x): return "NaN"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.2f}"

    # Build LaTeX (ablation-style: lrr with separate columns)
    lines = []
    lines.append("\\begin{table}[!h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Mean-only summary across datasets: one row per method with separate columns for IoU and BIoU. "
                 "Unrefined shows raw means (\\%); others are mean $\\Delta$ (pp, signed). "
                 "Bolding ignores the Best-oracle row.}")
    lines.append("\\begin{adjustbox}{max width=0.9\\textwidth}")
    lines.append("\\begin{tabular}{@{}lrr@{}}")
    lines.append("\\toprule")
    lines.append("Method & Mean $\\Delta$IoU & Mean $\\Delta$BIoU \\\\")
    lines.append("\\midrule")

    for sec_idx, section in enumerate(ordered_sections):
        for label, key in section:
            if key == "unref":
                # Raw percentages, both columns
                mean_ui = float(np.nanmean([stats_by_ds[ds]['unref'][0] for ds in ds_ok]))
                mean_ub = float(np.nanmean([stats_by_ds[ds]['unref'][1] for ds in ds_ok]))
                lines.append(f"{label} & {100.0*mean_ui:.2f} & {100.0*mean_ub:.2f} \\\\")
            else:
                mi = mean_iou_per_key[key]
                mb = mean_bio_per_key[key]
                s_i = fmt_signed_pp(mi)
                s_b = fmt_signed_pp(mb)
                if key in eligible_keys and (mi >= best_mean_iou - 1e-9):
                    s_i = f"\\textbf{{{s_i}}}"
                if key in eligible_keys and (mb >= best_mean_bio - 1e-9):
                    s_b = f"\\textbf{{{s_b}}}"
                lines.append(f"{label} & {s_i} & {s_b} \\\\")
        if sec_idx < len(ordered_sections) - 1:
            lines.append("\\midrule")

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
        description="Output LaTeX tables: (1) cross-dataset summary (with Unrefined row), and (2) cross-dataset Unrefined + oracle/avg/random/all/routers table (no colors)."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Each dir contains method subdirs with per-model CSVs.")
    parser.add_argument("--ours_name", type=str, default="SamRefiner-MoE",
        help="(Unused in the second/third tables now) Display/dir name for our method.")
    args = parser.parse_args()

    all_rows_by_dataset = OrderedDict()         # ds -> [rows]
    all_strata_by_method = defaultdict(dict)    # method -> ds -> {'IoU': model->arr, 'BIoU': model->arr}
                                                # also 'UNREF' -> ds -> {'IoU': model->arr([mean]), 'BIoU': ...}

    datasets = []
    dataset_dirs = []
    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                        if bn == "sota" else os.path.basename(bd_clean))
        datasets.append(dataset_name)
        dataset_dirs.append(bd_clean)

        rows, ds_unref_means, strata = summarize_dataset(bd_clean, dataset_name)
        all_rows_by_dataset[dataset_name] = rows

        # merge strata
        for method, v in strata.items():
            all_strata_by_method[method].setdefault(dataset_name, {})
            all_strata_by_method[method][dataset_name] = v

    # (1) Cross-dataset summary with Unrefined row (no colors)
    summary_caption = ("Quantitative comparison with state-of-the-art refinement methods across benchmark datasets. "
                       "We report the mean improvement ($\\Delta$) in Intersection-over-Union (IoU) and Boundary IoU over the unrefined base masks "
                       "as percentage points (signed). The rightmost column averages equally across datasets.")
    summary_table = latex_cross_dataset_summary_table(
        all_rows_by_dataset, all_strata_by_method, caption_note=summary_caption, ours_name=args.ours_name
    )
    print(summary_table)
    print()

    # (2) Cross-dataset Unrefined + oracle/avg/random/all/routers table (no colors; bolding excludes Best-oracle)
    cross_oracle_table = latex_oracle_random_cross_dataset_table(
        dataset_dirs, datasets
    )
    print(cross_oracle_table)
    print()

    # (3) Mean-only (ablation-style columns) table for the same oracle/avg/random/all/routers set
    mean_only_table = latex_oracle_random_mean_only_table(
        dataset_dirs, datasets
    )
    print(mean_only_table)
    print()

if __name__ == "__main__":
    main()






 
    '''
    python plotAllExperts.py \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/experts/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/experts/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/experts/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/experts/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/experts/MSRA-B/ \
    --ours_name E13_Router-P

    python plotAllExperts.py \
        /home/sprice/PromptMoE/paper_results/ablation/experts/BIG/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/DAVIS585/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/ECSSD/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/VOC/ \
        /home/sprice/PromptMoE/paper_results/ablation/experts/MSRA-B/ \
        --ours_name E13_Router-P
    '''
