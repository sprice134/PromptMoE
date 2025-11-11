#!/usr/bin/env python3
"""
conda activate segrefiner
    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/cascadePSP_fast \
      --fast True

    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/cascadePSP_slow \
      --fast False
      

    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/cascadePSP_fast \
      --fast True

    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/cascadePSP_slow \
      --fast False

    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/SP \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/STM \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/cascadePSP_slow \
      --fast False

    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/SP \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/STM \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/cascadePSP_fast \
      --fast True

      
    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/cascadePSP_fast \
      --fast True


    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/cascadePSP_slow \
      --fast False


    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/cascadePSP_fast \
      --fast True


    python cascadePSP.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/cascadePSP_slow \
      --fast False
    
    
This script:
 1. Loads the segmentation_refinement Refiner.
 2. For each pred_root in --pred_roots:
    - Infers model name from basename(pred_root)
    - For each subdir under input_root:
        • Reads raw.png, <subdir>_pred.png (from pred_root), gt.png
        • Runs Refiner.refine() (fast/slow)
        • Computes IoU & boundary-IoU before/after
    - Writes: output_dir/<model_name>.csv
      with columns: subdir, unrefined_iou, unrefined_boundary_iou,
                    refined_iou, refined_boundary_iou
"""
import os
import argparse
import csv
import time

import cv2
import numpy as np
from tqdm import tqdm
import segmentation_refinement as refine

def load_model(device):
    """Instantiate the Refiner."""
    return refine.Refiner(device=device)

def refine_mask(image, init_mask, refiner, fast, L=900):
    """Run one call to Refiner.refine and binarize to 0/255."""
    out = refiner.refine(image, init_mask, fast=fast, L=L)
    return (out > 0).astype(np.uint8) * 255

def compute_iou(mask1, mask2):
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter)/union if union>0 else 0.0

def compute_boundary_iou(mask1, mask2, dilation_ratio=0.02):
    h, w = mask1.shape[:2]
    d = max(1, int(np.sqrt(h*h + w*w)*dilation_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*d+1, 2*d+1))
    def boundary(mask):
        bw  = (mask>0).astype(np.uint8)
        return np.logical_xor(cv2.dilate(bw,kernel),
                              cv2.erode (bw,kernel))
    b1 = boundary(mask1)
    b2 = boundary(mask2)
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter)/union if union>0 else 0.0

def str2bool(v):
    if isinstance(v, bool): return v
    vl = v.lower()
    if vl in ('yes','true','t','1'):  return True
    if vl in ('no','false','f','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def process_one(input_root, pred_root, refiner, fast, output_csv):
    subdirs = [d for d in sorted(os.listdir(input_root))
               if os.path.isdir(os.path.join(input_root, d))]
    rows = []
    total_refine_time = 0.0
    refine_calls = 0

    for sub in tqdm(subdirs, desc=f"{os.path.basename(pred_root)}"):
        raw_p  = os.path.join(input_root, sub, "raw.png")
        gt_p   = os.path.join(input_root, sub, "gt.png")
        pred_p = os.path.join(pred_root,    f"{sub}_pred.png")
        if not (os.path.isfile(raw_p) 
                and os.path.isfile(gt_p) 
                and os.path.isfile(pred_p)):
            continue

        img  = cv2.imread(raw_p, cv2.IMREAD_COLOR)
        pm   = cv2.imread(pred_p, cv2.IMREAD_GRAYSCALE)
        gtm  = cv2.imread(gt_p,    cv2.IMREAD_GRAYSCALE)
        if img is None or pm is None or gtm is None:
            continue

        pm8 = (pm>0).astype(np.uint8)*255
        gt8 = (gtm>0).astype(np.uint8)*255

        u_iou  = compute_iou(pm8, gt8)
        u_biou = compute_boundary_iou(pm8, gt8)

        # time the refinement step
        t0 = time.monotonic()
        rf   = refine_mask(img, pm, refiner, fast=fast)
        t1 = time.monotonic()
        total_refine_time += (t1 - t0)
        refine_calls += 1

        r_iou  = compute_iou(rf, gt8)
        r_biou = compute_boundary_iou(rf, gt8)

        rows.append((sub, u_iou, u_biou, r_iou, r_biou))

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'subdir',
            'unrefined_iou',
            'unrefined_boundary_iou',
            'refined_iou',
            'refined_boundary_iou'
        ])
        w.writerows(rows)
    print(f"→ saved {output_csv}")

    return total_refine_time, refine_calls

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_root',  required=True,
                   help='Root dir with raw/gt per-subfolder')
    p.add_argument('--pred_roots',  required=True, nargs='+',
                   help='One or more dirs with <subdir>_pred.png files')
    p.add_argument('--output_dir',  required=True,
                   help='Where to dump each model’s CSV and timing .txt')
    p.add_argument('--device',      default='cuda:0',
                   help='Refiner device ("cuda:0" or "cpu")')
    p.add_argument('--fast',        type=str2bool, required=True,
                   help='fast mode? True or False')
    args = p.parse_args()

    refiner = load_model(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip('/'))
        out_csv    = os.path.join(args.output_dir, f"{model_name}.csv")
        total_time, calls = process_one(
            args.input_root, pred_root, refiner, args.fast, out_csv
        )

        # write timing info
        time_txt = os.path.join(args.output_dir, f"{model_name}_refine_time.txt")
        avg_time = total_time / calls if calls > 0 else 0.0
        with open(time_txt, 'w') as tf:
            tf.write(f"Total refinement time: {total_time:.4f} seconds\n")
            tf.write(f"Number of refinements: {calls}\n")
            tf.write(f"Average per image: {avg_time:.4f} seconds\n")
        print(f"→ saved timing info to {time_txt}")

if __name__ == "__main__":
    main()
