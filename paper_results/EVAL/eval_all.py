#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
from collections import defaultdict

# map on-disk names to display names (display only; never renames files)
METHOD_DISPLAY = {
    "Exp1": "1 PP",
    "Exp2": "1 PP + B",
    "Exp3": "1 PP + B + M",
    "Exp4": "1 PP + 1 NP + B + M",
    "Exp5": "3 PP + 1 NP + B + M",
    "Exp6": "3 PP + 1 NP + B + M + SF",
    "Exp7": "3 PP + 1 NP + B + M + SF + G",

    # Original mappings for base models if they appear
    "ZQL9711_RMBG-2-Matting":       "RMBG-2",
    "briaai_RMBG-1.4":              "RMBG-1.4",
    "ZhengPeng7_BiRefNet-matting":  "BiRefNet",
}

# Define the exact reverse cascading order for the experiments as a global constant
EXPERIMENT_ORDER = [
    "cascadePSP_fast",
    "cascadePSP_slow",
    "segrefiner_lr",
    "segrefiner_hr",
    "DualSight", 
    "samRefiner",
    "SamRefiner-MoE",
    "PromptMoE",
    "vit_b",
    "vit_l",
    "vit_h"
]

def summarize_directory(base_dir):
    method_dirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
    if not method_dirs:
        raise RuntimeError(f"No method subdirectories found in: {base_dir}")

    orig_methods = [os.path.basename(d) for d in method_dirs]

    # Keep your preferred experiment ordering but include any extra dirs present
    methods = [m for m in EXPERIMENT_ORDER if m in orig_methods]
    other_methods = [m for m in orig_methods if m not in EXPERIMENT_ORDER]
    methods = other_methods + methods

    # --- NEW: collect ALL base models seen anywhere (not just the first dir)
    # Skip router/weight dumps like weights.csv, weights_full.csv, *_weights.csv, *_weights_full.csv
    all_base_models = set()
    for d in method_dirs:
        for p in glob.glob(os.path.join(d, "*.csv")):
            base = os.path.splitext(os.path.basename(p))[0]
            # ignore known non-model CSVs
            if base == "U-2-Net":
                continue
            if base in {"weights", "weights_full"}:
                continue
            if base.endswith("_weights") or base.endswith("_weights_full"):
                continue
            all_base_models.add(base)
    base_models = sorted(all_base_models)

    data = defaultdict(dict)

    # --- NEW: Unrefined baselines — find a dir that has the CSV for each model
    for model in base_models:
        path = None
        for d in method_dirs:
            cand = os.path.join(d, f"{model}.csv")
            if os.path.isfile(cand):
                path = cand
                break
        if path is not None:
            df = pd.read_csv(path)
            data["Unrefined"][(model, "IoU")] = df["unrefined_iou"].mean()
            data["Unrefined"][(model, "BIoU")] = df["unrefined_boundary_iou"].mean()
        else:
            data["Unrefined"][(model, "IoU")] = float("nan")
            data["Unrefined"][(model, "BIoU")] = float("nan")

    # Refined methods (unchanged logic)
    for method in methods:
        for model in base_models:
            path = os.path.join(base_dir, method, f"{model}.csv")
            if not os.path.isfile(path):
                data[method][(model, "IoU")] = float("nan")
                data[method][(model, "BIoU")] = float("nan")
            else:
                df = pd.read_csv(path)
                data[method][(model, "IoU")] = df["refined_iou"].mean()
                data[method][(model, "BIoU")] = df["refined_boundary_iou"].mean()

    final_df = pd.DataFrame.from_dict(data, orient="index")
    cols = pd.MultiIndex.from_product([base_models, ["IoU", "BIoU"]])
    final_df = final_df.reindex(columns=cols)

    unref = final_df.loc["Unrefined"]
    delta_df = final_df.subtract(unref, axis=1)

    rank_df = final_df.drop("Unrefined").rank(ascending=False, method="min")
    rank_df.loc["Unrefined"] = float("nan")
    rank_df = rank_df.reindex(final_df.index)

    return final_df, delta_df, rank_df, base_models, ["Unrefined"] + methods


def _disp(method: str) -> str:
    # Case-insensitive display mapping for the first column
    return METHOD_DISPLAY.get(method, METHOD_DISPLAY.get(method.lower(), method))

def print_combined_table(final_df, delta_df, rank_df, base_models, methods, dataset_name):
    method_w = 28 # Increased width for longer names
    cell_w = 20
    sep = " |"

    print(f"\n\n=== Dataset: {dataset_name} ===\n")
    # Header
    h1 = f"{'Method':<{method_w}}"
    h2 = " " * method_w

    # Remap the column headers based on METHOD_DISPLAY (only if present)
    for m in base_models:
        mapped_name = METHOD_DISPLAY.get(m, METHOD_DISPLAY.get(m.lower(), m))
        span = cell_w * 2 + len(sep)
        h1 += f"{mapped_name:^{span}}"
        h2 += f"{'IoU':>{cell_w}}{'BIoU':>{cell_w}}{sep}"

    print(h1); print(h2)
    print("-" * len(h1))

    # Rows
    for method in methods:
        disp = _disp(method)
        is_unref = (method == "Unrefined")
        row = f"{disp:<{method_w}}"
        for m in base_models:
            v_iou = final_df.loc[method, (m, "IoU")]
            v_biou = final_df.loc[method, (m, "BIoU")]
            d_iou = delta_df.loc[method, (m, "IoU")]
            d_biou = delta_df.loc[method, (m, "BIoU")]
            r_iou = rank_df.loc[method, (m, "IoU")]
            r_biou = rank_df.loc[method, (m, "BIoU")]

            def fmt(v, dv, r, no_rank=False):
                if pd.isna(v):
                    return f"{'NaN':>{cell_w}}"
                pct = v * 100
                if no_rank:
                    return f"{pct:6.2f}".rjust(cell_w)
                delta = dv * 100
                sign = '+' if delta >= 0 else '-'
                ds = f"{sign}{abs(delta):5.2f}"
                return f"{int(r)}. {pct:5.2f} ({ds})".rjust(cell_w)

            row += fmt(v_iou, d_iou, r_iou, is_unref)
            row += fmt(v_biou, d_biou, r_biou, is_unref)
            row += sep
        print(row)
        if is_unref:
            print("-" * len(row))

    # Mean Rank Summary
    methods_to_sum = [m for m in methods if m != "Unrefined"]
    print("\nMean Rank Summary (lower is better):\n")
    summary = []
    for m in methods_to_sum:
        disp = _disp(m)
        mean_iou_rank = rank_df.loc[m, [(bm, "IoU") for bm in base_models]].mean()
        mean_biou_rank = rank_df.loc[m, [(bm, "BIoU") for bm in base_models]].mean()
        summary.append((disp, mean_iou_rank, mean_biou_rank))
    sum_df = pd.DataFrame(summary, columns=["Method", "Mean IoU Rank", "Mean BIoU Rank"])
    col_w = method_w
    print(f"{'Method':<{col_w}}{'Mean IoU Rank':>15}{'Mean BIoU Rank':>18}")
    print("-" * (col_w + 15 + 18))
    for _, r in sum_df.iterrows():
        print(f"{r['Method']:<{col_w}}{r['Mean IoU Rank']:15.2f}{r['Mean BIoU Rank']:18.2f}")

    # Differential lines vs Unrefined
    def _find_row_key(target_lower: str):
        for k in final_df.index:
            if isinstance(k, str) and k.lower() == target_lower:
                return k
        return None
    print("")
    base_key = _find_row_key("unrefined")
    for comp in EXPERIMENT_ORDER: # Use the ordered list for comparison
        comp_key = _find_row_key(comp)
        if base_key and comp_key:
            diffs_iou = final_df.loc[comp_key, [(bm, "IoU") for bm in base_models]] \
                        - final_df.loc[base_key, [(bm, "IoU") for bm in base_models]]
            diffs_biou = final_df.loc[comp_key, [(bm, "BIoU") for bm in base_models]] \
                         - final_df.loc[base_key, [(bm, "BIoU") for bm in base_models]]
            mi = diffs_iou.mean() * 100
            mb = diffs_biou.mean() * 100
            print(f"{_disp(comp_key)} vs {_disp('Unrefined')} mean Δ: IoU {mi:+.2f}, BIoU {mb:+.2f}")

    print()  # trailing newline

def main():
    parser = argparse.ArgumentParser(
        description="Summarize results with % scores, deltas, ranks, summary, and ΔsamRefiner."
    )
    parser.add_argument("base_dirs", nargs="+",
                        help="Dirs each containing method subdirectories")
    args = parser.parse_args()

    # --- NEW: accumulate per-dataset mean deltas for a final summary ---
    dataset_order = []
    agg_deltas = defaultdict(dict)   # disp_method -> {dataset_name: (mean_dIoU, mean_dBIoU)}
    method_order_global = []         # preserve first-seen order of methods

    for bd in args.base_dirs:
        bd_clean = bd.rstrip("/")
        bn = os.path.basename(bd_clean).lower()
        dataset_name = (os.path.basename(os.path.dirname(bd_clean))
                       if bn == "sota" else os.path.basename(bd_clean))

        final_df, delta_df, rank_df, models, methods = summarize_directory(bd_clean)
        print_combined_table(final_df, delta_df, rank_df, models, methods, dataset_name)

        # ---- accumulate Δ means vs Unrefined for each method on this dataset (NEW) ----
        if dataset_name not in dataset_order:
            dataset_order.append(dataset_name)

        io_cols  = [(bm, "IoU")  for bm in models]
        bi_cols  = [(bm, "BIoU") for bm in models]

        for m in methods:
            if m == "Unrefined":
                continue
            disp = _disp(m)
            # compute diffs vs Unrefined; means (skip NaNs) then scale to %
            diffs_iou  = final_df.loc[m, io_cols]  - final_df.loc["Unrefined", io_cols]
            diffs_biou = final_df.loc[m, bi_cols]  - final_df.loc["Unrefined", bi_cols]
            mi = diffs_iou.mean()  * 100
            mb = diffs_biou.mean() * 100
            agg_deltas[disp][dataset_name] = (mi, mb)
            if disp not in method_order_global:
                method_order_global.append(disp)

    # ---- print the new cross-dataset summary at the very end (NEW) ----
    if dataset_order:
        print("\n=== Cross-Dataset Δ (vs Unrefined) Summary ===\n")
        method_col_w = 28 # Increased width
        cell_w = 18
        final_w = 22

        # Header
        hdr = f"{'Method':<{method_col_w}}"
        for ds in dataset_order:
            hdr += f"{ds:>{cell_w}}"
        hdr += f"{'Mean ΔIoU / ΔBIoU':>{final_w}}"
        print(hdr)
        print("-" * len(hdr))

        # Rows
        for disp in method_order_global:
            row = f"{disp:<{method_col_w}}"
            mi_vals, mb_vals = [], []
            for ds in dataset_order:
                pair = agg_deltas.get(disp, {}).get(ds, (float('nan'), float('nan')))
                mi, mb = pair
                mi_vals.append(mi)
                mb_vals.append(mb)
                if pd.isna(mi) or pd.isna(mb):
                    cell = "   NaN / NaN   "
                else:
                    cell = f"{mi:+6.2f} / {mb:+6.2f}"
                row += f"{cell:>{cell_w}}"
            # Overall mean across datasets (skip NaNs)
            mi_mean = pd.Series(mi_vals).mean()
            mb_mean = pd.Series(mb_vals).mean()
            if pd.isna(mi_mean) or pd.isna(mb_mean):
                overall = "    NaN / NaN    "
            else:
                overall = f"{mi_mean:+7.2f} / {mb_mean:+7.2f}"
            row += f"{overall:>{final_w}}"
            print(row)

        print()  # trailing newline

if __name__ == "__main__":
    main()





    '''
    # Overall
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/ 

        
    # BACKBONE
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/backbone/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/backbone/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/backbone/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/backbone/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/backbone/MSRA-B/ 

    # Component
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/component/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/component/ECSSD \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/component/DAVIS585 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/component/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/component/MSRA-B/ 

    # Experts
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/experts/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/experts/ECSSD \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/experts/DAVIS585 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/experts/MSRA-B 

    # Points
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/points/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/points/ECSSD \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/points/DAVIS585 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/points/MSRA-B \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/ABLATIONS/points/VOC


    # 9-Expert
    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/funkyBS/SOTA/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/funkyBS/SOTA/DAVIS585

    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/DAVIS585 \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/components/MSRA-B/

    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/experts/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/experts/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/experts/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/experts/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/experts/MSRA-B/ 


    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/backbone/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/backbone/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/backbone/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/backbone/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Sep27/ABLATIONS/backbone/MSRA-B/ 

    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/SOTA/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/SOTA/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/SOTA/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/SOTA/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/SOTA/MSRA-B/ 

    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/ABLATIONS/backbone/BIG/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/ABLATIONS/backbone/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/ABLATIONS/backbone/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/ABLATIONS/backbone/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct13/ABLATIONS/backbone/MSRA-B/ 

    python eval_all.py \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/points/BIG \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/points/DAVIS585/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/points/ECSSD/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/points/VOC/ \
        /home/sprice/CVPR26_Dataset/final_results_Oct19/ABLATIONS/points/MSRA-B/
        

    python eval_all.py \
        /home/sprice/PromptMoE/SOTA/BIG \
        /home/sprice/PromptMoE/SOTA/DAVIS585/ \
        /home/sprice/PromptMoE/SOTA/ECSSD/ \
        /home/sprice/PromptMoE/SOTA/VOC/ \
        /home/sprice/PromptMoE/SOTA/MSRA-B/

    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/lite/DAVIS585 \
        /home/sprice/PromptMoE/paper_results/ablation/lite/BIG \
        /home/sprice/PromptMoE/paper_results/ablation/lite/ECSSD \
        /home/sprice/PromptMoE/paper_results/ablation/lite/MSRA-B \
        /home/sprice/PromptMoE/paper_results/ablation/lite/VOC 

        




    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/_PromptMoE/VOC/


    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/backbone/VOC

    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/components/VOC

    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/experts/VOC

    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/latencyStepEval/VOC

    python eval_all.py \
        /home/sprice/PromptMoE/paper_results/ablation/timeCheck/VOC
    '''