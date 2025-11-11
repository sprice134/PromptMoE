#!/usr/bin/env python3
"""
samPromptRefiner_batch.py

What it does (same high-level behavior as your original script):
  • Loads a SAM backbone once.
  • For each --pred_root, iterates subfolders under --input_root:
      - reads raw.png, gt.png, and <subdir>_pred.png
      - REFINES <subdir>_pred.png using SAM with prompts derived from the init mask:
          * an expanded bounding box around the init mask
          * K positive points sampled inside the init mask (via distance-transform peaks)
          * (optional) a small number of negative points near the mask boundary
      - Computes IoU and boundary-IoU before / after
  • Writes one CSV per --pred_root to --output_dir: <model_name>.csv

Usage example:

    python samPromptRefiner_batch.py \
      --checkpoint /path/to/sam_vit_l_0b3195.pth \
      --input_root /path/to/dataset/ECSSD/images \
      --pred_roots \
        /path/to/dataset/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /path/to/dataset/ECSSD/outputs/briaai_RMBG-1.4 \
        /path/to/dataset/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      --output_dir /path/to/output/ECSSD/samPromptRefiner

Notes:
  • This script uses *only* the binary masks from your <subdir>_pred.png as prompts.
    No YOLO. The “prompt generation logic” (points + expanded box) mirrors the Colab demo idea.
  • Default SAM is vit_l, but you can change via --model_type.
"""

import os
import sys
import argparse
import csv
import math
import random
import subprocess

import numpy as np
import cv2
import torch

# --- Optional Colab-friendly bootstrap ----------------------------------------------------------
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

def _maybe_colab_install():
    """Lightweight, safe installs when running inside Colab."""
    if not IN_COLAB:
        return
    pkgs = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "git+https://github.com/facebookresearch/segment-anything.git",
        "opencv-python pycocotools matplotlib ipykernel",
    ]
    for p in pkgs:
        subprocess.run(["pip", "install", "-q", *p.split()], check=False)
    os.makedirs("models", exist_ok=True)

_maybe_colab_install()

# After possible install:
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# --- Metrics ------------------------------------------------------------------------------------

def compute_iou(mask1, mask2):
    """mask1, mask2: uint8 {0,255} arrays."""
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / union if union > 0 else 0.0

def compute_boundary_iou(mask1, mask2, dilation_ratio=0.02):
    h, w = mask1.shape[:2]
    d = max(1, int(np.sqrt(h * h + w * w) * dilation_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))

    def boundary(bw):
        bw_u = (bw > 0).astype(np.uint8)
        dil = cv2.dilate(bw_u, kernel)
        er  = cv2.erode (bw_u, kernel)
        return np.logical_xor(dil, er)

    b1 = boundary(mask1)
    b2 = boundary(mask2)
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter) / union if union > 0 else 0.0

# --- Prompting utilities ------------------------------------------------------------------------

def bbox_from_mask(mask):
    """Return (x0,y0,x1,y1) in image coords. Returns None if mask empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return float(x0), float(y0), float(x1), float(y1)

def expand_bbox(x0, y0, x1, y1, W, H, expansion=0.10):
    """Expand bbox by % of its size, clamped to image."""
    w = x1 - x0 + 1e-6
    h = y1 - y0 + 1e-6
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    nw = w * (1.0 + expansion)
    nh = h * (1.0 + expansion)
    nx0 = max(0.0, cx - nw / 2.0)
    ny0 = max(0.0, cy - nh / 2.0)
    nx1 = min(float(W - 1), cx + nw / 2.0)
    ny1 = min(float(H - 1), cy + nh / 2.0)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

def topk_distance_peaks(mask, k=3, min_radius=10):
    """
    Pick up to k interior points using a distance transform peak finder with NMS.
    Returns list of (x,y) float coords in image space.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []

    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    d = dist.copy()
    H, W = d.shape
    pts = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(d), d.shape)
        y, x = int(idx[0]), int(idx[1])
        if d[y, x] <= 0:
            break
        pts.append((float(x), float(y)))
        # non-max suppression: zero a disk around chosen peak
        rr, cc = np.ogrid[:H, :W]
        mask_nms = (rr - y) ** 2 + (cc - x) ** 2 <= (min_radius ** 2)
        d[mask_nms] = 0.0
    return pts

def sample_ring_negatives(mask, n=1, ring_px=8):
    """
    Sample up to n negative points just outside the mask boundary.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    dil = cv2.dilate(m, kernel)
    ring = np.logical_and(dil > 0, m == 0)

    ys, xs = np.where(ring)
    if xs.size == 0:
        return []
    idxs = np.random.choice(xs.size, size=min(n, xs.size), replace=False)
    return [(float(xs[i]), float(ys[i])) for i in idxs]

# --- SAM refinement with prompts ----------------------------------------------------------------

@torch.no_grad()
def refine_with_sam_prompts(img_bgr, init_mask_u8, predictor: SamPredictor,
                            k_pos=3, k_neg=1, box_expand_ratio=0.10,
                            choose_by="iou_to_init"):
    """
    img_bgr: HxWx3 uint8 BGR
    init_mask_u8: HxW uint8 (0/255)

    Strategy:
      1) Expanded bbox around init mask
      2) K positive points (distance-transform peaks inside mask)
      3) (optional) K negative points just outside boundary
      4) Run SAM predictor with multimask_output=True
      5) Choose best mask (by IoU to the init mask or by SAM score)

    Returns uint8 refined mask (0/255)
    """
    H, W = init_mask_u8.shape[:2]

    # Prepare prompts
    bbox = bbox_from_mask(init_mask_u8)
    if bbox is None:
        return (init_mask_u8 > 0).astype(np.uint8) * 255  # nothing to do

    x0, y0, x1, y1 = bbox
    box = expand_bbox(x0, y0, x1, y1, W, H, expansion=box_expand_ratio)

    pos_pts = topk_distance_peaks(init_mask_u8, k=k_pos, min_radius=max(5, int(0.01 * max(H, W))))
    neg_pts = sample_ring_negatives(init_mask_u8, n=k_neg, ring_px=max(3, int(0.005 * max(H, W))))

    # Build predictor inputs
    point_coords = np.array(pos_pts + neg_pts, dtype=np.float32) if (pos_pts or neg_pts) else None
    point_labels = None
    if point_coords is not None:
        point_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)

    # SAM expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords if point_coords is not None else None,
        point_labels=point_labels if point_coords is not None else None,
        box=box[None, :],
        multimask_output=True,
    )

    # Choose the best candidate
    if masks is None or len(masks) == 0:
        return (init_mask_u8 > 0).astype(np.uint8) * 255

    if choose_by == "iou_to_init":
        init_bool = init_mask_u8 > 0
        ious = [(m.astype(bool) & init_bool).sum() / max(1, (m.astype(bool) | init_bool).sum()) for m in masks]
        best_idx = int(np.argmax(np.array(ious)))
    else:
        best_idx = int(np.argmax(scores))  # SAM's internal score

    best = masks[best_idx].astype(np.uint8)  # {0,1}
    return best * 255

# --- Main processing ----------------------------------------------------------------------------

def process_model(input_root, pred_root, predictor, output_csv, device_str):
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

        pm = cv2.imread(pred_p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_p,   cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.imread(raw_p, cv2.IMREAD_COLOR)
        if pm is None or gt is None or img_bgr is None:
            continue

        pm8 = ((pm > 0).astype(np.uint8) * 255)
        gt8 = ((gt > 0).astype(np.uint8) * 255)

        u_iou  = compute_iou(pm8, gt8)
        u_biou = compute_boundary_iou(pm8, gt8)

        rf8 = refine_with_sam_prompts(
            img_bgr, pm8, predictor,
            k_pos=3,      # number of positive interior points
            k_neg=1,      # a single negative just outside the boundary
            box_expand_ratio=0.10,
            choose_by="iou_to_init"  # or "sam_score"
        )
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

def load_sam_predictor(checkpoint_path, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='path to SAM checkpoint (e.g., sam_vit_l_0b3195.pth)')
    p.add_argument('--model_type', default='vit_l',
                   help='SAM model type (vit_h | vit_l | vit_b)')
    p.add_argument('--input_root', required=True,
                   help='Root dir with raw/gt per-subfolder')
    p.add_argument('--pred_roots', required=True, nargs='+',
                   help='One or more dirs with <subdir>_pred.png files')
    p.add_argument('--output_dir', required=True,
                   help='Where to dump each model’s CSV')
    p.add_argument('--device', default=None,
                   help='Torch device ("cuda:0" or "cpu"). Default: auto')
    args = p.parse_args()

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    predictor = load_sam_predictor(args.checkpoint, args.model_type, device)

    for pred_root in args.pred_roots:
        name     = os.path.basename(pred_root.rstrip('/'))
        out_csv  = os.path.join(args.output_dir, f"{name}.csv")
        process_model(args.input_root, pred_root, predictor, out_csv, device)

if __name__ == "__main__":
    main()
    '''
    python DualSight.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
    --model_type vit_h \
    --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/images \
    --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
    --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/DualSight

    
    python DualSight.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
  --model_type vit_h \
  --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/images \
  --pred_roots \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/DeepLabV3 \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/FCN \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/LR-ASPP \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/DualSight

    python DualSight.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
  --model_type vit_h \
  --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/images \
  --pred_roots \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/SP \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/STM \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/DualSight

    python DualSight.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
  --model_type vit_h \
  --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/images \
  --pred_roots \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/DeepLabV3 \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/FCN \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/LR-ASPP \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/DualSight

    python DualSight.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
  --model_type vit_h \
  --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/images \
  --pred_roots \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/briaai_RMBG-1.4 \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
    /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/DualSight

    '''