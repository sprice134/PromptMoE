"""
This script:
 1. Loads the SAM backbone and SAMRefiner.
 2. For each pred_root:
    - Infers model_name = basename(pred_root)
    - For each subdir under input_root:
        • Reads raw.png, <subdir>_pred.png, gt.png
        • Refines <subdir>_pred.png via SAMRefiner
        • Computes IoU & boundary-IoU before and after refinement
    - Writes: output_dir/<model_name>.csv
      with columns: subdir, unrefined_iou, unrefined_boundary_iou,
                    refined_iou, refined_boundary_iou
"""
import os
import sys
import argparse
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

# point to your SAMRefiner code
sys.path.insert(0, "/home/sprice/SAMRefiner")
from sam_refiner import sam_refiner
from segment_anything import sam_model_registry

def load_sam(checkpoint_path, device, model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return sam

def refine_mask(img_path, init_mask, sam, device):
    """
    img_path: path to raw image (for SAM to read RGB)
    init_mask: 2D uint8 mask (0 or 255)
    returns: 2D uint8 binary mask (0 or 255)
    """
    # normalize to [0,1]
    m = init_mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0

    # sam_refiner wants a list of masks
    out_masks = sam_refiner(img_path, [m], sam)
    refined = out_masks[0]
    # squeeze channel if needed
    if refined.ndim==3 and refined.shape[0]==1:
        refined = refined[0]
    # threshold
    return ((refined > 0.5).astype(np.uint8) * 255)

def compute_iou(mask1, mask2):
    m1 = mask1>0
    m2 = mask2>0
    inter = np.logical_and(m1,m2).sum()
    union = np.logical_or(m1,m2).sum()
    return float(inter)/union if union>0 else 0.0

def compute_boundary_iou(mask1, mask2, dilation_ratio=0.02):
    h, w = mask1.shape[:2]
    d = max(1, int(np.sqrt(h*h + w*w)*dilation_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*d+1, 2*d+1))

    def boundary(bw):
        bw_u = (bw>0).astype(np.uint8)
        dil = cv2.dilate(bw_u, kernel)
        er  = cv2.erode (bw_u, kernel)
        return np.logical_xor(dil, er)

    b1 = boundary(mask1)
    b2 = boundary(mask2)
    inter = np.logical_and(b1,b2).sum()
    union = np.logical_or(b1,b2).sum()
    return float(inter)/union if union>0 else 0.0

def process_model(input_root, pred_root, sam, device, output_csv):
    subdirs = sorted(d for d in os.listdir(input_root)
                     if os.path.isdir(os.path.join(input_root, d)))
    rows = []

    for sub in tqdm(subdirs, desc=os.path.basename(pred_root)):
        raw_p  = os.path.join(input_root, sub, "raw.png")
        gt_p   = os.path.join(input_root, sub, "gt.png")
        pred_p = os.path.join(pred_root,    f"{sub}_pred.png")
        if not (os.path.isfile(raw_p) and os.path.isfile(gt_p)
                and os.path.isfile(pred_p)):
            continue

        # read
        pm = cv2.imread(pred_p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_p,   cv2.IMREAD_GRAYSCALE)
        if pm is None or gt is None:
            continue

        pm8 = ((pm>0).astype(np.uint8)*255)
        gt8 = ((gt>0).astype(np.uint8)*255)

        u_iou  = compute_iou(pm8, gt8)
        u_biou = compute_boundary_iou(pm8, gt8)

        rf8 = refine_mask(raw_p, pm8, sam, device)
        r_iou  = compute_iou(rf8, gt8)
        r_biou = compute_boundary_iou(rf8, gt8)

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='path to SAM checkpoint (e.g., vit_h.pth / vit_l.pth / vit_b.pth)')
    p.add_argument('--input_root', required=True,
                   help='Root dir with raw/gt per-subfolder')
    p.add_argument('--pred_roots', required=True, nargs='+',
                   help='One or more dirs with <subdir>_pred.png files')
    p.add_argument('--output_dir', required=True,
                   help='Where to dump each model’s CSV')
    p.add_argument('--device', default='cuda:0',
                   help='Torch device ("cuda:0" or "cpu")')
    # NEW: backbone selector
    p.add_argument('--model_type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                   help='SAM backbone type (default: vit_h)')
    args = p.parse_args()

    # normalize hyphen input like "vit-h"
    model_type = args.model_type.replace('-', '_')

    # pass through to loader
    sam = load_sam(args.checkpoint, args.device, model_type=model_type)

    for pred_root in args.pred_roots:
        name     = os.path.basename(pred_root.rstrip('/'))
        out_csv  = os.path.join(args.output_dir, f"{name}.csv")
        process_model(args.input_root, pred_root, sam, args.device, out_csv)


if __name__ == "__main__":
    main()
