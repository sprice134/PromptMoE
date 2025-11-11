"""   
    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/segrefiner_lr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth


      python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/ECSSD/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth
    
    
      
    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/segrefiner_lr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth


      python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/BIG/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth
    
    
    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/segrefiner_lr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth


    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/briaai_RMBG-1.4 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/MSRA-B/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth


    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/SP \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/STM \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/segrefiner_lr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth


    python segrefiner.py \
     --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/SP \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/DAVIS585/outputs/STM \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/DAVIS585/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth


    
    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/segrefiner_lr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth

    python segrefiner.py \
      --input_root /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/images \
      --pred_roots \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/DeepLabV3 \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/FCN \
        /home/sprice/CVPR26_Dataset/final_results_Sep26/datasets/VOC/outputs/LR-ASPP \
      --output_dir /home/sprice/CVPR26_Dataset/final_results_Sep26/SOTA/VOC/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth

    

"""
import os
import argparse
import csv

import cv2
import numpy as np
import torch
import mmcv
from tqdm import tqdm
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks

def load_segrefiner_model(cfg_path, ckpt_path, device):
    """
    Loads the SegRefiner model and its configuration.
    """
    cfg = Config.fromfile(cfg_path)
    cfg = replace_cfg_vals(cfg)
    if hasattr(cfg.model, 'pretrained'):
        cfg.model.pretrained = None
    cfg.device = device
    cfg.gpu_ids = [0] if 'cuda' in device else []
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=True)
    model.to(device).eval()
    return model, cfg

def refine_with_segrefiner(image_bgr, coarse_mask, model, cfg, device):
    """
    Runs one call to the SegRefiner model, with padding to handle arbitrary sizes.
    """
    # Get original image dimensions
    h, w = image_bgr.shape[:2]

    # Handle cases where the mask is None or completely empty from the start.
    if coarse_mask is None or not np.any(coarse_mask):
        return np.zeros((h, w), dtype=np.uint8)

    # Define the stride the model requires (32 is common for UNet-like models)
    stride = 32

    # Calculate padding needed to make dimensions divisible by the stride
    new_h = (h + stride - 1) // stride * stride
    new_w = (w + stride - 1) // stride * stride
    pad_h = new_h - h
    pad_w = new_w - w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # Pad the image using reflection to avoid harsh borders
    img_padded = cv2.copyMakeBorder(image_bgr, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # Pad the coarse mask with zeros
    if coarse_mask.ndim == 3:
        coarse_mask = cv2.cvtColor(coarse_mask, cv2.COLOR_BGR2GRAY)
    mask_padded = cv2.copyMakeBorder(coarse_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Normalize the PADDED image
    img = img_padded
    if hasattr(cfg, 'img_norm_cfg'):
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std  = np.array(cfg.img_norm_cfg['std'],  dtype=np.float32)
        if cfg.img_norm_cfg.get('to_rgb', True):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=False)
    img_t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device)

    # Prepare the PADDED mask
    coarse_8_padded = (mask_padded > 0).astype(np.uint8)
    h_pad, w_pad = coarse_8_padded.shape
    ids = np.unique(coarse_8_padded)
    ids = ids[ids > 0]

    # Initialize the output mask with the ORIGINAL dimensions
    out_mask = np.zeros((h, w), dtype=np.uint8)
    metas = {
        'ori_filename': 'input.png',
        'img_shape': (h_pad, w_pad, 3),
        'ori_shape':  (h, w, 3),
        'pad_shape':  (h_pad, w_pad, 3),
        'scale_factor': 1.0,
        'flip': False,
    }

    for obj_id in ids:
        bm_arr = (coarse_8_padded == obj_id).astype(np.uint8)[None, ...]
        bm = BitmapMasks(bm_arr, height=h_pad, width=w_pad)
        data = {'img': img_t, 'img_metas': [metas], 'coarse_masks': [bm]}

        try:
            with torch.no_grad():
                res = model(return_loss=False, rescale=True, **data)

            out = res[0][0]
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            if out.ndim == 3 and out.shape[0] == 3:
                out = out.transpose(1,2,0)

            # --- NEW FIX ---
            # Manually crop the output mask from its padded size back to the
            # original image size to prevent the IndexError.
            out_h, out_w = out.shape[:2]
            if out_h != h or out_w != w:
                out = out[top:top+h, left:left+w]
            # --- END NEW FIX ---

            binm = (out > 0).astype(np.uint8)
            # Now `out_mask` and `binm` are guaranteed to have the same shape.
            out_mask[binm > 0] = 255

        except RuntimeError as e:
            if "Sizes of tensors must match except in dimension 1" in str(e):
                pass # Silently skip degenerate masks
            else:
                raise e # Re-raise any other runtime error

    return out_mask

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

def process_one_pred_root(input_root, pred_root, model, cfg, device, output_csv):
    subdirs = [d for d in sorted(os.listdir(input_root))
               if os.path.isdir(os.path.join(input_root, d))]
    rows = []
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

        rf   = refine_with_segrefiner(img, pm, model, cfg, device)
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

def main():
    p = argparse.ArgumentParser(
        description="Batch process initial segmentations with SegRefiner.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument('--input_root',  required=True,
                   help='Root dir with raw/gt per-subfolder')
    p.add_argument('--pred_roots',  required=True, nargs='+',
                   help='One or more dirs with <subdir>_pred.png files')
    p.add_argument('--output_dir',  required=True,
                   help='Where to dump each model’s CSV')
    p.add_argument('--config',     required=True,
                   help='SegRefiner inference config .py')
    p.add_argument('--checkpoint', required=True,
                   help='SegRefiner checkpoint .pth')
    p.add_argument('--device',      default='cuda:0',
                   help='Refiner device ("cuda:0" or "cpu")')
    args = p.parse_args()

    model, cfg = load_segrefiner_model(args.config, args.checkpoint, args.device)

    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip('/'))
        out_csv    = os.path.join(args.output_dir, f"{model_name}.csv")
        process_one_pred_root(
            args.input_root, pred_root, model, cfg, args.device, out_csv
        )

if __name__ == "__main__":
    main()