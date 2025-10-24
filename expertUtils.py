#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute all 10 experts over a dataset with strict per-expert timing.

Warm start:
  For each model dir, first run the *last* image once in dry-run mode
  (no saves, not in CSV/console) to warm caches/compilation, then process
  *all* images normally, including the last image again.

Times recorded per image:
  - pre_total : image-only feature extraction (no mask)
  - eval_total: mask-relative expert map construction
  - io_total  : I/O (read raw+mask, write experts)
  - total     : pre_total + eval_total + io_total

Per-expert times (pre/eval for each):
  Dp (spatial), Gd (depth), Gc (color/Lab), sp (superpixel),
  ctr (contrast), bri (brightness), tex (texture),
  frac (SAM coverage), Sm (SAM weighted coverage), prompt (SAM box-prompted)

Saves per (model, base):
  <cache_root>/<model>/<base>/experts/
    Dp.png, Gd.png, Gc.png, sp.png, ctr.png, bri.png, tex.png,
    frac.png, Sm.png, prompt.png

CSV (append-safe):
  model,base,pre_total,eval_total,io_total,total,
  dp_pre,dp_eval,gd_pre,gd_eval,gc_pre,gc_eval,
  sp_pre,sp_eval,ctr_pre,ctr_eval,bri_pre,bri_eval,
  tex_pre,tex_eval,frac_pre,frac_eval,Sm_pre,Sm_eval,prompt_pre,prompt_eval
"""

import os
import sys
import csv
import time
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from skimage.segmentation import slic
import torchvision.transforms as T
from matplotlib import colormaps

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide



# ── Hard-coded Marigold root ─────────────────────────────────────────────────
MARIGOLD_ROOT = "/home/sprice/CVPR26_Dataset/Marigold"
if MARIGOLD_ROOT and MARIGOLD_ROOT not in sys.path:
    sys.path.insert(0, MARIGOLD_ROOT)
try:
    from marigold import MarigoldDepthPipeline  # noqa: F401
except Exception as e:
    raise SystemExit(f"[Marigold] Could not import from {MARIGOLD_ROOT}: {e}")




# ── Marigold runner ──────────────────────────────────────────────────────────
class MarigoldRunner:
    def __init__(self, checkpoint: str, device: torch.device, half: bool):
        from marigold import MarigoldDepthPipeline
        dtype = torch.float16 if half else torch.float32
        variant = "fp16" if half else None
        self.pipe = MarigoldDepthPipeline.from_pretrained(checkpoint, variant=variant, torch_dtype=dtype)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe = self.pipe.to(device)
        self.device = device

    @torch.no_grad()
    def infer_depth01(self, image_path: str,
                      denoise_steps: int | None,
                      processing_res: int | None,
                      ensemble_size: int,
                      seed: int | None) -> torch.Tensor:
        """Return torch float32 HxW depth in [0..1] on CPU."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)

        # Pass a PIL.Image to the pipeline (not a string path)
        input_image = Image.open(image_path).convert("RGB")

        out = self.pipe(
            input_image,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=True,
            batch_size=0,
            color_map="Spectral",
            show_progress_bar=False,
            resample_method="bilinear",
            generator=gen,
        )
        depth_np = out.depth_np.astype("float32")  # [0..1]
        return torch.from_numpy(depth_np)  # CPU


# ── IO & helpers ─────────────────────────────────────────────────────────────
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = T.ToTensor()(img) * 255.0
    return t.float()

def load_mask(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy((arr > 127).astype(np.uint8))

def to_uint8_numpy_rgb(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0,255).byte().permute(1,2,0).cpu().numpy()

def apply_colormap(t: torch.Tensor, cmap_name="viridis") -> np.ndarray:
    arr = t.detach().cpu().numpy()
    mn, mx = float(arr.min()), float(arr.max())
    norm = (arr - mn) / (mx - mn + 1e-8)
    rgb = colormaps[cmap_name](norm)[..., :3]
    return (rgb * 255).astype(np.uint8)

def tensor_to_color_pil(t: torch.Tensor, cmap_name="viridis") -> Image.Image:
    return Image.fromarray(apply_colormap(t, cmap_name), mode="RGB")

def _resize_2d(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if t.shape == (H, W):
        return t
    return F.interpolate(t[None,None], size=(H,W), mode='bilinear', align_corners=False)[0,0]


# ── PRE: image-only features (timed) ─────────────────────────────────────────
def pre_depth(mg: MarigoldRunner, raw_path: str,
              denoise: int|None, proc_res: int|None, ens: int, seed: int|None) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    d01 = mg.infer_depth01(raw_path, denoise, proc_res, ens, seed)     # CPU [0..1]
    d255 = (d01 * 255.0).clamp(0,255).float()                          # CPU [0..255]
    t1 = time.perf_counter()
    return d255, (t1 - t0)

def pre_lab(img_rgb: torch.Tensor) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    lab = cv2.cvtColor(to_uint8_numpy_rgb(img_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    t1 = time.perf_counter()
    return lab, (t1 - t0)

def pre_luminance(img: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    r,g,b = img[0], img[1], img[2]
    lum = 0.299*r + 0.587*g + 0.114*b
    t1 = time.perf_counter()
    return lum, (t1 - t0)

def pre_local_mean(lum: torch.Tensor, k: int = 7) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    lm = F.avg_pool2d(lum[None,None], kernel_size=k, stride=1, padding=k//2)[0,0]
    t1 = time.perf_counter()
    return lm, (t1 - t0)

def pre_grad_mag(lum: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
    sobel_y = sobel_x.t()
    gx = F.conv2d(lum[None,None], sobel_x.view(1,1,3,3), padding=1)[0,0]
    gy = F.conv2d(lum[None,None], sobel_y.view(1,1,3,3), padding=1)[0,0]
    grad = torch.sqrt(gx*gx + gy*gy)
    t1 = time.perf_counter()
    return grad, (t1 - t0)

def pre_slic(img: torch.Tensor, n_segments=200, compactness=10.0, max_side=1024) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    H, W = img.shape[1:]
    long_side = max(H, W)
    if long_side > max_side:
        scale = max_side / long_side
        H2 = int(round(H*scale)); W2 = int(round(W*scale))
        img_small = F.interpolate(img[None].float(), size=(H2,W2), mode='bilinear', align_corners=False)[0]
    else:
        img_small = img
    seg_small = slic(img_small.permute(1,2,0).cpu().numpy()/255.0,
                     n_segments=n_segments, compactness=compactness, start_label=0).astype(np.int32)
    if img_small.shape[-2:] != (H, W):
        seg = cv2.resize(seg_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    else:
        seg = seg_small
    t1 = time.perf_counter()
    return seg, (t1 - t0)

def pre_sam(img: torch.Tensor, mask_gen: SamAutomaticMaskGenerator, resize_t: ResizeLongestSide) -> Tuple[Dict, float, np.ndarray]:
    t0 = time.perf_counter()
    img_np = to_uint8_numpy_rgb(img)
    img_rs = resize_t.apply_image(img_np)
    sam_masks = mask_gen.generate(img_rs)
    H, W = img_np.shape[:2]
    seg_full = np.full((H, W), -1, dtype=np.int32)
    for idx, m in enumerate(sam_masks):
        small = m["segmentation"].astype(np.uint8)
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        new_px = up & (seg_full == -1)
        if new_px.any():
            seg_full[new_px] = idx
    t1 = time.perf_counter()
    return {"sam_masks": sam_masks, "img_rs": img_rs}, (t1 - t0), seg_full


# ── EVAL: mask-relative expert maps (timed) ──────────────────────────────────
def eval_dp(mask: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    m_np = mask.cpu().numpy().astype(np.uint8) * 255
    dist = cv2.distanceTransform(m_np, cv2.DIST_L2, 5).astype(np.float32)
    dist /= (dist.max() + 1e-8)
    t1 = time.perf_counter()
    return torch.from_numpy(dist).to(device), (t1 - t0)

def eval_gd(d255: torch.Tensor, fg: torch.Tensor, omega: float, gamma: float) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    inv = 255.0 - d255
    if fg.any():
        center = inv[fg].mean()
        delta  = (inv - center).abs()
        maxd   = delta[fg].max()
    else:
        center, maxd = 0.0, torch.tensor(1.0, dtype=inv.dtype)
        delta = (inv - center).abs()
    dn = delta / (maxd + 1e-8)
    Gd = omega * (gamma ** dn)
    Gd = Gd / (Gd.max() + 1e-8)
    t1 = time.perf_counter()
    return Gd, (t1 - t0)

def eval_gc_lab(lab_f: np.ndarray, fg: torch.Tensor, omega: float, gamma: float, device: torch.device) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W, _ = lab_f.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None,None], size=(H,W), mode='nearest')[0,0].bool()
    if not fg.any():
        out = torch.zeros((H,W), dtype=torch.float32, device=device)
        return out, 0.0
    lab = torch.from_numpy(lab_f).to(device)  # (H,W,3) float
    mean_lab = lab[fg].mean(dim=0)
    delta = torch.linalg.norm(lab - mean_lab, dim=2)
    maxd = delta[fg].max().clamp_min(1e-8)
    dn = delta / maxd
    Gc = omega * torch.pow(gamma, dn)
    Gc = Gc / Gc.max().clamp_min(1e-8)
    t1 = time.perf_counter()
    return Gc, (t1 - t0)

def eval_brightness(lum: torch.Tensor, fg: torch.Tensor, omega: float, gamma: float) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W = lum.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None,None], size=(H,W), mode='nearest')[0,0].bool()
    if not fg.any():
        out = torch.zeros((H,W), dtype=torch.float32, device=lum.device)
        return out, 0.0
    mean_b = lum[fg].mean()
    delta  = (lum - mean_b).abs()
    maxd   = delta[fg].max().clamp_min(1e-8)
    dn = delta / maxd
    Gb = omega * torch.pow(gamma, dn)
    Gb = Gb / Gb.max().clamp_min(1e-8)
    t1 = time.perf_counter()
    return Gb, (t1 - t0)

def eval_contrast(lum: torch.Tensor, local_mean: torch.Tensor, fg: torch.Tensor,
                  omega: float, gamma: float, clip_percentile: float = 0.98) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W = lum.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None,None], size=(H,W), mode='nearest')[0,0].bool()
    if not fg.any():
        out = torch.zeros((H,W), dtype=torch.float32, device=lum.device)
        return out, 0.0
    contrast_raw = (lum - local_mean).abs()
    mean_c = contrast_raw[fg].mean()
    delta  = (contrast_raw - mean_c).abs()
    vals = delta[fg].flatten()
    thresh = torch.quantile(vals, clip_percentile).clamp_min(1e-8) if vals.numel() > 1 else vals.max().clamp_min(1e-8)
    dn = (delta / thresh).clamp(0,1)
    Gc = omega * torch.pow(gamma, dn)
    Gc = Gc / Gc.max().clamp_min(1e-8)
    t1 = time.perf_counter()
    return Gc, (t1 - t0)

def eval_texture(gradmag: torch.Tensor, fg: torch.Tensor,
                 omega: float, gamma: float, clip_percentile: float = 0.98) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W = gradmag.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None,None], size=(H,W), mode='nearest')[0,0].bool()
    if not fg.any():
        out = torch.zeros((H,W), dtype=torch.float32, device=gradmag.device)
        return out, 0.0
    mean_t = gradmag[fg].mean()
    delta  = (gradmag - mean_t).abs()
    vals = delta[fg].flatten()
    thresh = torch.quantile(vals, clip_percentile).clamp_min(1e-8) if vals.numel() > 1 else vals.max().clamp_min(1e-8)
    dn = (delta / thresh).clamp(0,1)
    Gt = omega * torch.pow(gamma, dn)
    Gt = Gt / Gt.max().clamp_min(1e-8)
    t1 = time.perf_counter()
    return Gt, (t1 - t0)

def eval_superpixel(seg: np.ndarray, fg: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W = seg.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None,None], size=(H,W), mode='nearest')[0,0].bool()
    seg_flat = seg.reshape(-1)
    fg_flat  = fg.view(-1).float().cpu().numpy()
    max_id   = seg_flat.max() + 1 if seg_flat.size else 0
    fg_per   = np.bincount(seg_flat, weights=fg_flat, minlength=max_id)
    pix_per  = np.bincount(seg_flat, minlength=max_id).astype(np.float32)
    frac     = fg_per / np.clip(pix_per, 1, None)
    cov = frac[seg_flat].reshape(H, W).astype(np.float32)
    t1 = time.perf_counter()
    return torch.from_numpy(cov).to(fg.device), (t1 - t0)

def eval_sam_frac_sm(seg_full: np.ndarray, fg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    t0 = time.perf_counter()
    seg_full = torch.from_numpy(seg_full).to(fg.device)
    H, W = seg_full.shape

    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None, None], size=(H, W), mode='nearest')[0, 0].bool()

    frac = torch.zeros((H, W), dtype=torch.float32, device=fg.device)
    Sm = torch.zeros((H, W), dtype=torch.float32, device=fg.device)
    if not (seg_full >= 0).any():
        return frac, Sm, 0.0
    mask_bool = fg.bool()

    pred_total = mask_bool.sum().float()

    if pred_total <= 0:
        return frac, Sm, 0.0

    unique_region_ids = seg_full.unique()
    region_one_hot = torch.stack([seg_full == region_id for region_id in unique_region_ids], dim=0)
    overlap = (region_one_hot & mask_bool).sum(dim=(1, 2)).float()  # Overlap for each region (sum of boolean values)
    size = region_one_hot.sum(dim=(1, 2)).float()  # Size of each region
    frac_vals = overlap / size  # Calculate frac value for each region
    weight_vals = overlap / pred_total  # Weight for each region
    weighted_vals = frac_vals * weight_vals  # Weighted value for Sm

    for idx, region_id in enumerate(unique_region_ids):
        region_mask = (seg_full == region_id)
        frac[region_mask] = frac_vals[idx]
        Sm[region_mask] = weighted_vals[idx]

    maxw = Sm.max()
    if maxw > 0:
        Sm /= maxw

    t1 = time.perf_counter()
    return frac, Sm, (t1 - t0)


def eval_sam_prompt(predictor: SamPredictor, resize_t: ResizeLongestSide, fg: torch.Tensor) -> Tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    H, W = fg.shape
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        out = torch.zeros((H,W), dtype=torch.float32, device=fg.device)
        return out, 0.0
    y0, y1 = ys.min().item(), ys.max().item()
    x0, x1 = xs.min().item(), xs.max().item()
    input_box = np.array([x0, y0, x1, y1])
    box = resize_t.apply_boxes(input_box, (H, W))
    masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None, box=box[None,:], multimask_output=True
    )
    combined = np.logical_or.reduce(masks, axis=0).astype(np.uint8)
    up = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    # IoU fallback
    pred = fg.cpu().numpy().astype(bool)
    pm   = up.astype(bool)
    inter = float((pred & pm).sum()); union = float((pred | pm).sum())
    if union > 0 and (inter / union) < 0.5:
        up = pred.astype(np.float32)
    t1 = time.perf_counter()
    return torch.from_numpy(up).to(fg.device), (t1 - t0)


