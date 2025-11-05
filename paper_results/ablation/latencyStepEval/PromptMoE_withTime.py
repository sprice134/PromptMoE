from __future__ import annotations
import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import time
from collections import defaultdict
from skimage.segmentation import slic

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide


# ============================================================================
# 1) CONSTANTS
# ============================================================================
EXPERT_NAMES = ["Dp","Gd","Gc","sp","ctr","bri","tex","frac","Sm","prompt"]


# ============================================================================
# 2) ROUTER: MLP, PairRouter, load_router, gating helpers
# ============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class PairRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = type("Cfg", (), cfg) if isinstance(cfg, dict) else cfg
        E = self.cfg.E
        self.v = nn.Embedding(E, self.cfg.d_hidden)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.02)
        self.g_enc = MLP(self.cfg.d_ctx, self.cfg.d_hidden, self.cfg.d_hidden,
                         n_layers=self.cfg.n_layers, dropout=self.cfg.dropout)
        self.mlp_U = MLP(self.cfg.d_hidden*3, self.cfg.d_hidden, 1,
                         n_layers=self.cfg.n_layers, dropout=self.cfg.dropout)
        self.mlp_I = MLP(self.cfg.d_hidden*4, self.cfg.d_hidden, 1,
                         n_layers=self.cfg.n_layers, dropout=self.cfg.dropout)

    def forward_utilities(self, g):
        B = g.shape[0]; H = self.cfg.d_hidden; E = self.cfg.E
        enc_g = self.g_enc(g)
        v = self.v.weight
        g_exp = enc_g.unsqueeze(1).expand(B, E, H)
        v_exp = v.unsqueeze(0).expand(B, E, H)
        feats = torch.cat([g_exp * v_exp, g_exp, v_exp], dim=-1)
        U = self.mlp_U(feats).squeeze(-1)
        return U, enc_g

    def forward_interactions(self, enc_g):
        B = enc_g.shape[0]; H = self.cfg.d_hidden; E = self.cfg.E
        v = self.v.weight
        ve = v.unsqueeze(0).unsqueeze(2).expand(B, E, E, H)
        vf = v.unsqueeze(0).unsqueeze(1).expand(B, E, E, H)
        feats = torch.cat([ve+vf, ve*vf, torch.abs(ve-vf),
                           enc_g.unsqueeze(1).unsqueeze(1).expand(B, E, E, H)], dim=-1)
        I = self.mlp_I(feats).squeeze(-1)
        I = 0.5 * (I + I.transpose(1, 2))
        I = I - torch.diag_embed(torch.diagonal(I, dim1=1, dim2=2))
        return I

    def forward_scores(self, g):
        U, enc_g = self.forward_utilities(g)
        I = self.forward_interactions(enc_g)
        S = (U.unsqueeze(2) + U.unsqueeze(1)) + I
        S = S - torch.diag_embed(torch.diagonal(S, dim1=1, dim2=2))
        return U, I, S

    @torch.no_grad()
    def gating_weights(self, g, method="pair_marginal", T=1.0, T_pair=1.0, topk=2):
        if g.dim() == 1: g = g.unsqueeze(0)
        U, _, S = self.forward_scores(g)
        if method == "utility":
            logits = U / max(T, 1e-8)
        else:
            B, E, _ = S.shape
            diag_mask = torch.eye(E, device=S.device, dtype=torch.bool).unsqueeze(0)
            S_masked = S.masked_fill(diag_mask, float("-inf"))
            pair_logits = torch.logsumexp(S_masked / max(T_pair, 1e-8), dim=2)
            logits = pair_logits / max(T, 1e-8)
        w = torch.softmax(logits, dim=1)
        if topk is not None and 0 < topk < w.size(1):
            vals, idx = torch.topk(w, topk, dim=1)
            mask = torch.zeros_like(w); mask.scatter_(1, idx, 1.0)
            w = w * mask
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return w[0]

@torch.no_grad()
def get_weights_from_pair_router(router: PairRouter, context_g: torch.Tensor, topk: int) -> Dict[str, float]:
    method = "utility" if topk == 1 else "pair_marginal"
    w = router.gating_weights(context_g, method=method, topk=topk)
    selected = {name: float(w[i].item()) for i, name in enumerate(EXPERT_NAMES) if w[i].item() > 1e-6}
    s = sum(selected.values())
    if s > 0:
        for k in selected: selected[k] /= s
    return selected

def load_router(router_ckpt: Optional[str], device: torch.device) -> Tuple[Optional[PairRouter], Optional[dict]]:
    if router_ckpt is None:
        return None, None
    if not os.path.exists(router_ckpt):
        print(f"[Router] Warning: checkpoint not found at {router_ckpt}. Continuing without router.")
        return None, None
    payload = torch.load(router_ckpt, map_location=device)
    cfg = payload.get("cfg", {})
    model = PairRouter(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[Router] Loaded PairRouter; d_ctx={cfg.get('d_ctx')}, E={cfg.get('E')}")
    return model, cfg


# ============================================================================
# 3) EXPERT PRE/EVAL UTILITIES (inlined) + optional Marigold depth runner
# ============================================================================
# Optional Marigold import — depth expert 'Gd' will be skipped if not provided.
import sys, os
LOCAL_MARIGOLD = "/home/sprice/PromptMoE/Marigold"
if os.path.isdir(os.path.join(LOCAL_MARIGOLD, "marigold")):
    sys.path.insert(0, LOCAL_MARIGOLD)

try:
    import sys
    from marigold import MarigoldDepthPipeline  # noqa: F401
    _MARIGOLD_AVAILABLE = True
except Exception:
    _MARIGOLD_AVAILABLE = False

class MarigoldRunner:
    def __init__(self, checkpoint: str, device: torch.device, half: bool):
        if not _MARIGOLD_AVAILABLE:
            raise SystemExit("[Marigold] marigold package not available but --marigold_checkpoint provided.")
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
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
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

class LatencyTimer:
    def __init__(self, device, timers_dict):
        self.device = device
        self.timers = timers_dict
        self.start_times = {}

    def start(self, name):
        torch.cuda.synchronize(self.device)
        self.start_times[name] = time.time()

    def stop(self, name):
        torch.cuda.synchronize(self.device)
        self.timers[name] += (time.time() - self.start_times[name])

# ---- helpers shared with pre/eval ----
def to_uint8_numpy_rgb(t: torch.Tensor) -> np.ndarray:
    return t.clamp(0,255).byte().permute(1,2,0).cpu().numpy()

def _resize_2d(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if t.shape == (H, W): return t
    return F.interpolate(t[None,None], size=(H,W), mode='bilinear', align_corners=False)[0,0]

# ---- PRE (image-only) ----
def pre_lab(img_rgb: torch.Tensor):
    lab = cv2.cvtColor(to_uint8_numpy_rgb(img_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab, 0.0

def pre_luminance(img: torch.Tensor):
    r,g,b = img[0], img[1], img[2]
    lum = 0.299*r + 0.587*g + 0.114*b
    return lum, 0.0

def pre_local_mean(lum: torch.Tensor, k: int = 7):
    lm = F.avg_pool2d(lum[None,None], kernel_size=k, stride=1, padding=k//2)[0,0]
    return lm, 0.0

def pre_grad_mag(lum: torch.Tensor):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
    sobel_y = sobel_x.t()
    gx = F.conv2d(lum[None,None], sobel_x.view(1,1,3,3), padding=1)[0,0]
    gy = F.conv2d(lum[None,None], sobel_y.view(1,1,3,3), padding=1)[0,0]
    grad = torch.sqrt(gx*gx + gy*gy)
    return grad, 0.0

def pre_slic(img: torch.Tensor, n_segments=200, compactness=10.0, max_side=1024):
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
    return seg, 0.0

def pre_sam(img: torch.Tensor, mask_gen: SamAutomaticMaskGenerator, resize_t: ResizeLongestSide):
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
    return {"sam_masks": sam_masks, "img_rs": img_rs}, 0.0, seg_full

# ---- EVAL (mask-relative) ----
def eval_dp(mask: torch.Tensor, device: torch.device):
    m_np = mask.cpu().numpy().astype(np.uint8) * 255
    dist = cv2.distanceTransform(m_np, cv2.DIST_L2, 5).astype(np.float32)
    dist /= (dist.max() + 1e-8)
    return torch.from_numpy(dist).to(device), 0.0

def eval_gd(d255: torch.Tensor, fg: torch.Tensor, omega: float, gamma: float):
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
    return Gd, 0.0

def eval_gc_lab(lab_f: np.ndarray, fg: torch.Tensor, omega: float, gamma: float, device: torch.device):
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
    return Gc, 0.0

def eval_brightness(lum: torch.Tensor, fg: torch.Tensor, omega: float, gamma: float):
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
    return Gb, 0.0

def eval_contrast(lum: torch.Tensor, local_mean: torch.Tensor, fg: torch.Tensor,
                  omega: float, gamma: float, clip_percentile: float = 0.98):
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
    return Gc, 0.0

def eval_texture(gradmag: torch.Tensor, fg: torch.Tensor,
                 omega: float, gamma: float, clip_percentile: float = 0.98):
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
    return Gt, 0.0

def eval_superpixel(seg: np.ndarray, fg: torch.Tensor):
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
    return torch.from_numpy(cov).to(fg.device), 0.0

def eval_sam_frac_sm(seg_full: np.ndarray, fg: torch.Tensor):
    seg_full = torch.from_numpy(seg_full).to(fg.device)
    H, W = seg_full.shape
    if fg.shape != (H, W):
        fg = F.interpolate(fg.float()[None, None], size=(H, W), mode='nearest')[0, 0].bool()
    frac = torch.zeros((H, W), dtype=torch.float32, device=fg.device)
    Sm   = torch.zeros((H, W), dtype=torch.float32, device=fg.device)
    if not (seg_full >= 0).any():
        return frac, Sm, 0.0
    mask_bool = fg.bool()
    pred_total = mask_bool.sum().float()
    if pred_total <= 0:
        return frac, Sm, 0.0
    unique_region_ids = seg_full.unique()
    region_one_hot = torch.stack([seg_full == region_id for region_id in unique_region_ids], dim=0)
    overlap = (region_one_hot & mask_bool).sum(dim=(1, 2)).float()
    size    = region_one_hot.sum(dim=(1, 2)).float()
    frac_vals   = overlap / size
    weight_vals = overlap / pred_total
    weighted_vals = frac_vals * weight_vals
    for idx, region_id in enumerate(unique_region_ids):
        region_mask = (seg_full == region_id)
        frac[region_mask] = frac_vals[idx]
        Sm[region_mask]   = weighted_vals[idx]
    maxw = Sm.max()
    if maxw > 0: Sm /= maxw
    return frac, Sm, 0.0

def eval_sam_prompt(predictor: SamPredictor, resize_t: ResizeLongestSide, fg: torch.Tensor):
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
    # IoU fallback: if SAM box-prompt disagrees wildly, use fg mask itself
    pred = fg.cpu().numpy().astype(bool)
    pm   = up.astype(bool)
    inter = float((pred & pm).sum()); union = float((pred | pm).sum())
    if union > 0 and (inter / union) < 0.5:
        up = pred.astype(np.float32)
    return torch.from_numpy(up).to(fg.device), 0.0


# ============================================================================
# 4) SALIENCY: compute only selected experts, lazy prereqs, depth sentinel
# ============================================================================
def _minmax01(t: torch.Tensor) -> torch.Tensor:
    lo = t.min(); hi = t.max()
    if float(hi - lo) < 1e-8:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo)

@torch.no_grad()
def compute_selected_experts_maps(
    img_rgb: torch.Tensor,
    fg: torch.Tensor,
    selected: Dict[str, float],
    predictor: SamPredictor,                   # single predictor
    resize_tf: ResizeLongestSide,
    mg: Optional[MarigoldRunner],
    device: torch.device,
    mask_gen: SamAutomaticMaskGenerator,
) -> Dict[str, torch.Tensor]:
    """
    Compute only the selected expert maps, running minimal prerequisites.
    Depth 'Gd' returns a sentinel scalar tensor (-1) here; it's filled later.
    """
    H, W = fg.shape
    fg_bool = fg.bool().to(device)
    cache = {}

    def need(*keys): return any(k in selected for k in keys)

    if need("bri", "ctr", "tex", "Gc"):
        lum, _ = pre_luminance(img_rgb); cache["lum"] = lum
    if need("ctr"):
        local_mean, _ = pre_local_mean(cache["lum"], k=7); cache["local_mean"] = local_mean
    if need("tex"):
        gradmag, _ = pre_grad_mag(cache["lum"]); cache["gradmag"] = gradmag
    if need("Gc"):
        lab_f, _ = pre_lab(img_rgb); cache["lab_f"] = lab_f
    if need("sp"):
        seg_slic, _ = pre_slic(img_rgb, n_segments=200, compactness=10.0, max_side=1024); cache["seg_slic"] = seg_slic
    if need("prompt"):
        cache["sam_info"] = {"predictor": predictor}  # predictor already has set_image()
    if need("frac", "Sm"):
        _, _, seg_sam_full = pre_sam(img_rgb, mask_gen=mask_gen, resize_t=resize_tf)
        cache["seg_sam_full"] = seg_sam_full

    out: Dict[str, torch.Tensor] = {}
    for name in selected.keys():
        if name == "Dp":
            dp_map, _ = eval_dp(fg, device)
            out["Dp"] = _minmax01(dp_map.to(device))

        elif name == "Gd":
            out["Gd"] = torch.tensor(-1.0, device=device)  # sentinel; filled in build_saliency_for_mask

        elif name == "Gc":
            gc_map, _ = eval_gc_lab(cache["lab_f"], fg_bool, omega=1.0, gamma=0.65, device=device)
            out["Gc"] = _minmax01(gc_map)

        elif name == "sp":
            sp_map, _ = eval_superpixel(cache["seg_slic"], fg_bool)
            out["sp"] = _minmax01(sp_map.to(device))

        elif name == "ctr":
            ctr_map, _ = eval_contrast(cache["lum"].to(device), cache["local_mean"].to(device), fg_bool,
                                       omega=1.0, gamma=0.65)
            out["ctr"] = _minmax01(ctr_map)

        elif name == "bri":
            bri_map, _ = eval_brightness(cache["lum"].to(device), fg_bool, omega=1.0, gamma=0.65)
            out["bri"] = _minmax01(bri_map)

        elif name == "tex":
            tex_map, _ = eval_texture(cache["gradmag"].to(device), fg_bool, omega=1.0, gamma=0.65)
            out["tex"] = _minmax01(tex_map)

        elif name == "frac" or name == "Sm":
            frac_map, Sm_map, _ = eval_sam_frac_sm(cache["seg_sam_full"], fg_bool)
            out.setdefault("frac", _minmax01(frac_map.to(device)))
            out.setdefault("Sm",   _minmax01(Sm_map.to(device)))

        elif name == "prompt":
            pm, _ = eval_sam_prompt(cache["sam_info"]["predictor"], resize_tf, fg_bool)
            out["prompt"] = _minmax01(pm.to(device))

    return out

@torch.no_grad()
def build_saliency_for_mask(
    raw_path: str,
    img_rgb_t: torch.Tensor,        # torch 3xHxW in [0..255]
    fg01: torch.Tensor,             # 0/1 mask (torch uint8)
    selected: Dict[str, float],
    resize_tf: ResizeLongestSide,
    mg: Optional[MarigoldRunner],
    device: torch.device,
    beta: float,
    point_map: str,
    depth_cfg: Dict[str, object],
    predictor: SamPredictor,
    mask_gen: SamAutomaticMaskGenerator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    maps = compute_selected_experts_maps(
        img_rgb=img_rgb_t, fg=fg01, selected=selected,
        predictor=predictor, resize_tf=resize_tf, mg=mg, device=device, mask_gen=mask_gen
    )
    # Depth sentinel handling (compute only if selected + Marigold provided)
    if "Gd" in selected and isinstance(maps.get("Gd"), torch.Tensor) and maps["Gd"].numel() == 1:
        if mg is None:
            print("Warning: Depth expert 'Gd' selected but Marigold not available. Skipping 'Gd'.")
            del maps["Gd"]
        else:
            d01 = mg.infer_depth01(
                raw_path,
                denoise_steps=depth_cfg.get("denoise"),
                processing_res=depth_cfg.get("proc_res"),
                ensemble_size=depth_cfg.get("ens", 1),
                seed=depth_cfg.get("seed")
            )
            d255 = (d01 * 255.0).clamp(0, 255).float()
            gd_map, _ = eval_gd(d255.to(device), fg01.bool().to(device), omega=1.0, gamma=0.65)
            maps["Gd"] = _minmax01(gd_map)

    H, W = fg01.shape
    # Only stack valid 2D maps that actually materialized
    valid = {k: v for k, v in selected.items() if k in maps and isinstance(maps[k], torch.Tensor) and maps[k].ndim == 2}
    if not valid:
        print(f"Warning: No valid expert maps computed. Returning zeros.")
        return torch.zeros((H, W), dtype=torch.float32), maps

    stack, weights = [], []
    for name, weight in valid.items():
        m = maps[name]
        if m.shape != (H, W):
            m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
        stack.append(m)
        weights.append(weight)

    exp_stack = torch.stack(stack, dim=0).to(device)
    w = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1, 1)

    blended = _minmax01((exp_stack * w).sum(dim=0))
    entropy = torch.zeros_like(blended) if exp_stack.shape[0] <= 1 else _minmax01(torch.std(exp_stack, dim=0))

    if point_map == "blended":
        sal = blended
    elif point_map == "entropy":
        sal = 1.0 - entropy
    else:  # "moe"
        sal = blended * torch.clamp(1.0 - float(beta) * entropy, 0.0, 1.0)
    return sal.detach().cpu(), maps


# ============================================================================
# 5) SAM PROMPTING & REFINEMENT
# ============================================================================
def gaussian_2d(shape, gamma_x=1, gamma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x / (2 * gamma_x * gamma_x) + y * y / (2 * gamma_y * gamma_y)))
    return h

def get_mask_embed(mask, img_embed):
    orig_H, orig_W = mask.shape[:2]
    embed_H, embed_W = img_embed.shape[-2:]
    if orig_H >= orig_W:
        resize_W = int(embed_H * orig_W / orig_H); resize_H = embed_H
    else:
        resize_H = int(embed_W * orig_H / orig_W); resize_W = embed_W
    mask_resize = F.interpolate(mask[None, None].float(), size=(resize_H, resize_W), mode='nearest')
    query_embed = (img_embed[:, :, :resize_H, :resize_W] * mask_resize).sum(dim=(-2, -1)) / mask_resize.sum()
    return query_embed, mask_resize

def extract_bboxes_expand(image_embeddings, mask, margin=0, img_path=None):
    ori_h, ori_w = mask.shape[-2:]
    if margin > 0 and ori_h > 0 and ori_w > 0:
        embed_H, embed_W = image_embeddings.shape[-2:]
        if ori_h >= ori_w:
            resize_W = int(embed_H * ori_w / ori_h); resize_H = embed_H
        else:
            resize_H = int(embed_W * ori_h / ori_w); resize_W = embed_W
        image_embeddings_resize = image_embeddings[:, :, :resize_H, :resize_W]
        image_embeddings_resize = F.interpolate(image_embeddings_resize, size=(ori_h, ori_w), mode='bilinear')
        image_embeddings_resize = image_embeddings_resize.permute(0, 2, 3, 1)
        image_embeddings_resize = image_embeddings_resize / image_embeddings_resize.norm(dim=-1, keepdim=True)

    boxes, box_masks, areas, expand_list = [], [], [], []
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        coord = torch.nonzero(m)
        try:
            y1, x1 = int(coord[:, 0].min()), int(coord[:, 1].min())
            y2, x2 = int(coord[:, 0].max()), int(coord[:, 1].max())
        except Exception:
            y1 = x1 = y2 = x2 = 0

        x1 = max(0, x1); y1 = max(0, y1)
        y2 = min(mask.shape[-2] - 1, y2); x2 = min(mask.shape[-1] - 1, x2)
        box_h = y2 - y1; box_w = x2 - x1
        final_x1, final_x2, final_y1, final_y2 = x1, x2, y1, y2
        changed = False

        if box_h > 0 and box_w > 0 and margin > 0 and ori_h > 0 and ori_w > 0:
            steph = min(box_h * 0.1, 10); stepw = min(box_w * 0.1, 10)
            query_embed, mask_resize = get_mask_embed(m, image_embeddings)
            query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
            sim = image_embeddings_resize @ query_embed.transpose(0, 1); sim = sim.squeeze(); sim = sim > 0.5
            temp_x1 = int(x1 - stepw)
            if temp_x1 > 0 and temp_x1 < x1:
                context_area = (y2 - y1) * (x1 - temp_x1)
                sim_context = sim[y1:y2, temp_x1:x1]; pos_area = sim_context.sum()
                if pos_area / max(1, context_area) > margin:
                    final_x1 = temp_x1; changed = True

            temp_x2 = int(x2 + stepw)
            if temp_x2 < ori_w and temp_x2 > x2:
                context_area = (y2 - y1) * (temp_x2 - x2)
                sim_context = sim[y1:y2, x2:temp_x2]; pos_area = sim_context.sum()
                if pos_area / max(1, context_area) > margin:
                    final_x2 = temp_x2; changed = True

            temp_y1 = int(y1 - steph)
            if temp_y1 > 0 and temp_y1 < y1:
                context_area = (y1 - temp_y1) * (x2 - x1)
                sim_context = sim[temp_y1:y1, x1:x2]; pos_area = sim_context.sum()
                if pos_area / max(1, context_area) > margin:
                    final_y1 = temp_y1; changed = True

            temp_y2 = int(y2 + steph)
            if temp_y2 < ori_h and temp_y2 > y2:
                context_area = (temp_y2 - y2) * (x2 - x1)
                sim_context = sim[y2:temp_y2, x1:x2]; pos_area = sim_context.sum()
                if pos_area / max(1, context_area) > margin:
                    final_y2 = temp_y2; changed = True

        expand_list.append(1 if changed else 0)
        x1, x2, y1, y2 = final_x1, final_x2, final_y1, final_y2
        boxes.append(torch.tensor([x1, y1, x2, y2]))
        box_mask = torch.zeros((m.shape[0], m.shape[1]), device=image_embeddings.device)
        box_mask[y1:y2, x1:x2] = 1
        box_masks.append(box_mask); areas.append(1.0 * (x2 - x1) * (y2 - y1))
    boxes = torch.stack(boxes, dim=0).reshape(-1, 4).to(image_embeddings.device)
    box_masks = torch.stack(box_masks, dim=0).to(image_embeddings.device)
    areas = torch.tensor(areas).reshape(-1).to(image_embeddings.device)
    expand_list = torch.tensor(expand_list).reshape(-1).to(image_embeddings.device)
    return boxes, box_masks, areas, expand_list

def strat_topk_saliency_suppressed(mask: np.ndarray, pm: np.ndarray, k: int = 3, suppression_frac: float = 0.03):
    avail = pm * (mask > 0).astype(pm.dtype)
    coords = np.column_stack(np.where(mask > 0))  # [N,2] as [y,x]
    N = coords.shape[0]
    if N == 0:
        return []
    suppression_n = max(1, int(np.ceil(N * suppression_frac)))
    H, W = mask.shape
    points: List[Tuple[int, int]] = []
    for _ in range(k):
        if avail.max() == 0:
            break
        idx_flat = np.argmax(avail); y0, x0 = np.unravel_index(idx_flat, avail.shape)
        points.append((x0, y0))
        # radial suppression using the k-th nearest distance inside the mask
        YY, XX = np.indices((H, W))
        d2 = (XX - x0) ** 2 + (YY - y0) ** 2
        in_mask = (mask > 0)
        d_vals = d2[in_mask]
        if d_vals.size == 0: break
        curr_n = min(suppression_n, d_vals.size)
        thresh2 = np.partition(d_vals, curr_n - 1)[curr_n - 1]
        avail[d2 <= thresh2] = 0
    return points

def extract_points(
    pred_masks: torch.ByteTensor,
    saliency_map: torch.Tensor,
    add_neg=True,
    use_mask=True,
    gamma=1.0,
    k_points: int = 3,
    suppression_frac: float = 0.03,
):
    device = pred_masks.device
    pc, pl, gdts = [], [], []
    for m in pred_masks:
        m_bool = m.bool(); m_np = m_bool.cpu().numpy().astype(np.uint8)
        pm_np = saliency_map.cpu().numpy()
        pts = strat_topk_saliency_suppressed(m_np, pm_np, k=k_points, suppression_frac=suppression_frac)
        for x0, y0 in pts:
            pc.append([x0, y0]); pl.append(1)

        # Negative-point selection (inside tight bbox only).
        if add_neg and m_bool.any():
            m_np = m_bool.cpu().numpy().astype(np.uint8)  # 0/1
            m_rev_np = (1 - m_np) * 255
            dist_rev_np = cv2.distanceTransform(m_rev_np, cv2.DIST_L2, 5)
            dt_r = torch.from_numpy(dist_rev_np.astype(np.float32)).to(device)
            ys, xs = torch.where(m_bool)
            y1 = int(ys.min().item()); y2 = int(ys.max().item())
            x1 = int(xs.min().item()); x2 = int(xs.max().item())
            dt_r_masked = dt_r.clone()
            mask_bbox = torch.zeros_like(dt_r_masked, dtype=torch.bool)
            mask_bbox[y1:y2+1, x1:x2+1] = True
            dt_r_masked[~mask_bbox] = 0
            if (dt_r_masked > 0).any():
                ys, xs = torch.where(dt_r_masked == dt_r_masked.max())
                pc.append([xs[0].item(), ys[0].item()]); pl.append(0)
            else:
                idx_bg = torch.nonzero(~m_bool, as_tuple=False)
                if idx_bg.numel() > 0:
                    by, bx = idx_bg[0].tolist()
                    pc.append([bx, by]); pl.append(0)

        if use_mask:
            m_np = (m_bool.cpu().numpy().astype(np.uint8)) * 255
            dist_np = cv2.distanceTransform(m_np, cv2.DIST_L2, 5); dist_np = dist_np / (dist_np.max() + 1e-8)
            dt = torch.from_numpy(dist_np.astype(np.float32)).to(device)
            area = max(m_bool.sum().float() / gamma, 1)
            g = torch.exp(-((dt - dt.max()) ** 2) / area)
            g[m_bool == 0] = 0; gdts.append(g)

    pc_t = torch.tensor(pc, device=device).view(len(pred_masks), -1, 2) if pc else torch.zeros((len(pred_masks), 0, 2), device=device, dtype=torch.long)
    pl_t = torch.tensor(pl, device=device).view(len(pred_masks), -1) if pl else torch.zeros((len(pred_masks), 0), device=device, dtype=torch.long)
    g_t  = torch.stack(gdts) if (use_mask and gdts) else torch.empty(0, device=device)
    return pc_t, pl_t, g_t

def extract_mask(pred_masks, gaus_dt, target_size, is01, strength=15, device=0, expand_list=0):
    pred_masks = pred_masks.float().unsqueeze(1)
    gaus_dt = gaus_dt.float().unsqueeze(1)
    if is01:
        pred_masks[pred_masks == 0] = -1; pred_masks[pred_masks == 1] = 1; padvalue = -1
    else:
        padvalue = -100
    pred_masks = F.interpolate(pred_masks, target_size, mode="bilinear", align_corners=False)
    gaus_dt = F.interpolate(gaus_dt, target_size, mode="bilinear", align_corners=False)
    h, w = pred_masks.shape[-2:]; padh = 1024 - h; padw = 1024 - w
    pred_masks = F.pad(pred_masks, (0, padw, 0, padh), 'constant', padvalue)
    pred_masks = F.interpolate(pred_masks, (256, 256), mode="bilinear", align_corners=False)
    gaus_dt = F.pad(gaus_dt, (0, padw, 0, padh), 'constant', 0)
    gaus_dt = F.interpolate(gaus_dt, (256, 256), mode="bilinear", align_corners=False)
    if is01:
        for i in range(len(pred_masks)):
            if expand_list[i] == 0:
                pred_masks[pred_masks <= 0] = -1 * strength; pred_masks[pred_masks > 0] = strength
            else:
                pred_masks[pred_masks <= 0] = -1; pred_masks[pred_masks > 0] = 1
        gaus_dt[gaus_dt <= 0] = 1; pred_masks = pred_masks * gaus_dt
    return pred_masks

def sam_input_prepare(image, pred_masks, img_emb, resize_tf, saliency_map,
                      use_point=True, use_box=True, use_mask=True,
                      add_neg=True, margin=0.0, gamma=1.0, strength=30,
                      k_points: int = 3, suppression_frac: float = 0.03):
    ori_sz = pred_masks.shape[-2:]
    inp = {'image': image, 'original_size': ori_sz}
    if use_box:
        bbs, bm, areas, ex = extract_bboxes_expand(img_emb, pred_masks, margin)
        inp['boxes'] = resize_tf.apply_boxes_torch(bbs, ori_sz)
    else:
        ex = torch.zeros((len(pred_masks)), device=image.device)
    pc, pl, gd = extract_points(pred_masks, saliency_map, add_neg, use_mask, gamma,
                                k_points=k_points, suppression_frac=suppression_frac)
    if use_point:
        inp['point_coords'] = resize_tf.apply_coords_torch(pc, ori_sz)
        inp['point_labels'] = pl
    # Keep raw (un-resized) for logging
    inp['raw_point_coords'] = pc
    inp['raw_point_labels'] = pl
    if use_mask:
        tgt_sz = image.shape[1:]
        inp['mask_inputs'] = extract_mask(pred_masks, gd, tgt_sz, True, strength, image.device, ex)
    return inp

# ---- minimal embed helpers (former cache removed; relies on predictor.set_image()) ----
@torch.no_grad()
def compute_sam_mask_features(mask_bin: np.ndarray, full_embed: torch.Tensor) -> np.ndarray:
    """
    Returns a (C,) mask-aligned SAM token (L2-normalized), where full_embed has shape (1, C, H', W').
    """
    m = (mask_bin > 127).astype(np.uint8)
    Hf, Wf = full_embed.shape[-2:]
    Hm, Wm = m.shape
    if Hm >= Wm:
        rW = int(Hf * Wm / max(1, Hm)); rH = Hf
    else:
        rH = int(Wf * Hm / max(1, Wm)); rW = Wf
    m_resized = cv2.resize(m, (rW, rH), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    m_res_t = torch.from_numpy(m_resized).to(full_embed.device)
    E_crop = full_embed[0, :, :rH, :rW]
    token = (E_crop * m_res_t).sum(dim=(1, 2)) / (m_res_t.sum() + 1e-6)
    token = torch.nn.functional.normalize(token, dim=0)  # (C,)
    return token.cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_sam_embeddings(predictor: SamPredictor, image_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    predictor.set_image(image_rgb)  # sets internal state
    full_embed = predictor.get_image_embedding()          # (1, C, H', W')
    pooled_embed = full_embed.mean(dim=(2, 3)).squeeze(0) # (C,)
    return full_embed, pooled_embed

def compute_cheap_mask_features(mask_bin: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    m = (mask_bin > 127).astype(np.uint8)
    H, W = m.shape
    area = m.mean()
    ys, xs = np.where(m)
    if xs.size == 0:
        bbox_fill, aspect = 0.0, 0.0
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
        bbox_fill = m.sum() / float(bw * bh)
        aspect = bw / float(bh)
    ph = np.abs(m[:, 1:] - m[:, :-1]).sum() if W > 1 else 0
    pv = np.abs(m[1:, :] - m[:-1, :]).sum() if H > 1 else 0
    perim = float(ph + pv)
    perim_norm = perim / max(H * W, 1)
    compactness = (4.0 * np.pi * m.sum()) / (perim**2 + 1e-6) if perim > 0 else 0.0
    num_labels, _ = cv2.connectedComponents(m, connectivity=4)
    num_components = float(max(0, num_labels - 1))
    return np.array([area, bbox_fill, aspect, perim_norm, compactness, num_components], dtype=np.float32)

def prepare_image_for_sam(image_np: np.ndarray, transform: ResizeLongestSide, device: torch.device) -> torch.Tensor:
    image_rs = transform.apply_image(image_np)
    image_t = torch.as_tensor(image_rs, device=device)
    return image_t.permute(2, 0, 1).contiguous()

def build_context_g(
    pooled_embed_np: np.ndarray,        # (256,)
    sam_mask_feats_np: np.ndarray,      # (256,)
    cheap_mask_feats_np: np.ndarray,    # (6,)
    router_cfg: dict,
    device: torch.device
) -> torch.Tensor:
    d_ctx = int(router_cfg.get("d_ctx", 256 + 6))
    g_combined = np.concatenate([
        pooled_embed_np,         # 256
        sam_mask_feats_np,       # 256
        cheap_mask_feats_np      # 6
    ]).astype(np.float32)        # total 518

    if g_combined.size > d_ctx:
        g_combined = g_combined[:d_ctx]
    elif g_combined.size < d_ctx:
        g_combined = np.pad(g_combined, (0, d_ctx - g_combined.size))

    return torch.from_numpy(g_combined).to(device)



@torch.no_grad()
def sam_refiner_router(
    image_path: str,
    image_rgb: np.ndarray,           # RGB HxWx3
    coarse_mask_255: np.ndarray,     # 0/255
    predictor: SamPredictor,         # single shared predictor
    full_img_embed: torch.Tensor,    # (1,C,H',W') for forward_with_image_embeddings
    selected_weights: Dict[str, float],
    sam_model,                       # SAM (only for .forward_with_image_embeddings)
    device: torch.device,
    resize_tf: ResizeLongestSide,
    mg: Optional[MarigoldRunner],
    beta: float,
    point_map: str,
    k_points: int,
    suppression_frac: float,
    use_point: bool, use_box: bool, use_mask: bool, add_neg: bool,
    margin: float, gamma: float, strength: int,
    depth_cfg: Dict[str, object],
    mask_gen: SamAutomaticMaskGenerator,
    log_points: bool = False,
    one_time_timer_dict: defaultdict = None,
    per_iter_timer_dict: defaultdict = None,
    timer_device: torch.device = None
) -> np.ndarray:
    
    one_time_timer = LatencyTimer(timer_device, one_time_timer_dict)
    per_iter_timer = LatencyTimer(timer_device, per_iter_timer_dict)
    """
    Refines a coarse mask using SAM; DOES NOT call the router (weights are provided).
    Saliency is computed once up-front and reused across iterations by default.
    """
    # Prepare tensors once
    img_t_cpu = load_image(image_path)                   # torch 3xHxW [0..255]
    image_t_for_sam = prepare_image_for_sam(image_rgb, resize_tf, device)
    masks_t = torch.as_tensor(coarse_mask_255[None, ...] > 0, dtype=torch.uint8, device=device)

    # One-time saliency for the initial coarse mask (no recompute across iters)
    one_time_timer.start("expert_saliency_comp")
    coarse01 = torch.from_numpy((coarse_mask_255 > 0).astype(np.uint8))
    sal_map, _ = build_saliency_for_mask(
        raw_path=image_path, img_rgb_t=img_t_cpu, fg01=coarse01,
        selected=selected_weights, resize_tf=resize_tf, mg=mg, device=device,
        beta=beta, point_map=point_map, depth_cfg=depth_cfg,
        predictor=predictor, mask_gen=mask_gen
    )
    one_time_timer.stop("expert_saliency_comp")

    sam_masks = None
    iters = int(depth_cfg.get("iters", 5))
    for it in range(iters):
        masks_cur = masks_t if it == 0 else (sam_masks > 0).to(torch.uint8)
        per_iter_timer.start("prompt_preparation")
        inp = sam_input_prepare(
            image_t_for_sam, masks_cur, full_img_embed, resize_tf, sal_map.to(device),
            use_point, use_box, use_mask, add_neg, margin, gamma, strength,
            k_points=k_points, suppression_frac=suppression_frac
        )
        per_iter_timer.stop("prompt_preparation")

        if log_points and 'raw_point_coords' in inp:
            rpc = inp['raw_point_coords'].detach().cpu().tolist()
            rpl = inp['raw_point_labels'].detach().cpu().tolist()
            for bi, (pts, labs) in enumerate(zip(rpc, rpl)):
                triplets = [(int(x), int(y), int(l)) for (x, y), l in zip(pts, labs)]
                print(f"[Iter {it}] mask {bi} points (x,y,label): {triplets}")

        per_iter_timer.start("sam_inference")
        out = sam_model.forward_with_image_embeddings(full_img_embed, [inp], multimask_output=True)[0]
        best = torch.argmax(out['iou_predictions'], dim=-1)
        sam_masks = torch.stack([m[idx] for m, idx in zip(out['masks'], best)], 0)
        per_iter_timer.stop("sam_inference")

    return (sam_masks > 0).detach().cpu().numpy().astype(np.uint8)


# ============================================================================
# 6) METRICS & CSV
# ============================================================================
def compute_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    b1 = m1 > 0; b2 = m2 > 0
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter) / union if union else 0.0

def compute_boundary_iou(m1: np.ndarray, m2: np.ndarray, dilation_ratio: float = 0.02) -> float:
    h, w = m1.shape[:2]
    d = max(1, int(np.hypot(h, w) * dilation_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*d+1, 2*d+1))
    def bdry(bw):
        u = (bw > 0).astype(np.uint8)
        return np.logical_xor(cv2.dilate(u, kernel), cv2.erode(u, kernel))
    b1 = bdry(m1); b2 = bdry(m2)
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter) / union if union else 0.0

def save_selected_experts_to_csv(selected_experts_per_image: Dict[str, Dict[str, float]], output_csv: str):
    rows = []
    for image_name, selected_experts in selected_experts_per_image.items():
        row = [image_name] + [selected_experts.get(expert, 0.0) for expert in EXPERT_NAMES]
        rows.append(row)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name'] + EXPERT_NAMES)
        writer.writerows(rows)
    print(f"→ Saved selected-experts CSV to {output_csv}")


# ============================================================================
# 7) I/O HELPERS
# ============================================================================
def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1)  # [0..255]
    return t

def parse_expert_weights_from_str(weights_str: str) -> Optional[Dict[str, float]]:
    if not weights_str: return None
    try:
        weights = {}
        for part in weights_str.split(','):
            key, value = part.strip().split('=')
            weights[key.strip()] = float(value.strip())
        s = sum(weights.values())
        if s > 0:
            for k in weights: weights[k] /= s
        return weights
    except Exception as e:
        print(f"Warning: Could not parse --expert_weights string '{weights_str}'. Error: {e}")
        return None

def load_sam(checkpoint_path: str, device: torch.device, model_type: str = "vit_h"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device); sam.eval()
    return sam


# ============================================================================
# 8) DRIVER: process_model + CLI
# ============================================================================
@torch.no_grad()
def process_model(
    input_root: str,
    pred_root: str,
    sam_model,
    predictor: SamPredictor,
    device: torch.device,
    output_csv: str,
    router: Optional[PairRouter],
    router_cfg: Optional[dict],
    mg: Optional[MarigoldRunner],
    beta: float,
    point_map: str,
    k_points: int,
    suppression_frac: float,
    iters: int,
    use_point: bool, use_box: bool, use_mask: bool, add_neg: bool,
    margin: float, gamma: float, strength: int,
    router_topk: int,
    depth_cfg: Dict[str, object],
    resize_tf: ResizeLongestSide,
    mask_gen: SamAutomaticMaskGenerator,
    manual_weights_str: Optional[str],
    log_points: bool = False,
    light_mode: bool = False,
    timers: Dict[str, defaultdict] = None,
):

    if timers is None:
        timers = {
            'one_time': defaultdict(float),
            'per_iter': defaultdict(float),
        }

    subdirs = sorted(d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)))
    rows = []
    per_image_selected: Dict[str, Dict[str, float]] = {}
    num_images_processed = 0

    one_time_timer = timers['one_time']
    per_iter_timer = timers['per_iter']
    timer = LatencyTimer(device, one_time_timer)

    model_name = os.path.basename(pred_root.rstrip('/'))
    for sub in tqdm(subdirs, desc=f"Processing {model_name}"):
        raw_p  = os.path.join(input_root, sub, "raw.png")
        gt_p   = os.path.join(input_root, sub, "gt.png")
        pred_p = os.path.join(pred_root, f"{sub}_pred.png")
        if not (os.path.isfile(raw_p) and os.path.isfile(gt_p) and os.path.isfile(pred_p)):
            continue

        pm = cv2.imread(pred_p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
        if pm is None or gt is None:
            continue

        pm8 = (pm > 0).astype(np.uint8) * 255
        gt8 = (gt > 0).astype(np.uint8) * 255

        # Unrefined metrics
        u_iou = compute_iou(pm8, gt8)
        u_biou = compute_boundary_iou(pm8, gt8)

        
        # --- SAM Image Setup (Category 1) ---
        timer.start("sam_image_setup")
        bgr = cv2.imread(raw_p, cv2.IMREAD_COLOR)
        if bgr is None: 
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Compute SAM embeddings ON DEMAND (no cache) — former cache removed
        img_emb_full, img_emb_pooled = compute_sam_embeddings(predictor, img_rgb)
        timer.stop("sam_image_setup")
        

        # Build router context g ONCE per image (pooled token + cheap mask feats)
        timer.start("des_router")
        if manual_weights_str:
            selected_weights = parse_expert_weights_from_str(manual_weights_str) or {}
        else:
            if not (router and router_cfg):
                raise ValueError("A valid --router_ckpt must be provided when not using --expert_weights.")
            cheap_feats = compute_cheap_mask_features(pm8, img_rgb)           # (6,)
            sam_feats   = compute_sam_mask_features(pm8, img_emb_full)        # (256,)
            pooled_np   = img_emb_pooled.cpu().numpy()                         # (256,)
            g = build_context_g(
                pooled_embed_np=pooled_np,
                sam_mask_feats_np=sam_feats,
                cheap_mask_feats_np=cheap_feats,
                router_cfg=router_cfg,
                device=device
            )
            selected_weights = get_weights_from_pair_router(router, g, topk=router_topk)
        timer.stop("des_router")
        per_image_selected[sub] = selected_weights

        # Refinement (router is NOT called again inside)
        depth_cfg_loc = dict(depth_cfg); depth_cfg_loc["iters"] = iters
        rf8 = sam_refiner_router(
            image_path=raw_p,
            image_rgb=img_rgb,
            coarse_mask_255=pm8,
            predictor=predictor,
            full_img_embed=img_emb_full,
            selected_weights=selected_weights,
            sam_model=sam_model,
            device=device,
            resize_tf=resize_tf,
            mg=mg,
            beta=beta, point_map=point_map, k_points=k_points, suppression_frac=suppression_frac,
            use_point=use_point, use_box=use_box, use_mask=use_mask, add_neg=add_neg,
            margin=margin, gamma=gamma, strength=strength,
            depth_cfg=depth_cfg_loc,
            mask_gen=mask_gen,
            log_points=log_points,
            one_time_timer_dict=one_time_timer,
            per_iter_timer_dict=per_iter_timer,
            timer_device=device,
        )[0] * 255

        r_iou = compute_iou(rf8, gt8)
        r_biou = compute_boundary_iou(rf8, gt8)
        rows.append((sub, u_iou, u_biou, r_iou, r_biou))
        num_images_processed += 1

    # Write primary metrics CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subdir', 'unrefined_iou', 'unrefined_boundary_iou', 'refined_iou', 'refined_boundary_iou'])
        w.writerows(rows)
    print(f"→ saved {output_csv}")

    # Write per-image expert weights CSV (post-topk)
    weights_csv = os.path.join(os.path.dirname(output_csv) or '.', f"{model_name}_weights.csv")
    save_selected_experts_to_csv(per_image_selected, weights_csv)

    # ---- LATENCY REPORT ----
    if timers is not None and num_images_processed > 0:
        one_time = timers['one_time']
        per_iter = timers['per_iter']

        print("\n" + "="*50)
        print(f"Latency Report (Avg. per Image over {num_images_processed} images)")
        print("="*50)

        # 1) one-time stuff
        total_one_time = 0.0
        print("--- One-Time Costs (ms) ---")
        for key, total_time in one_time.items():
            avg_ms = (total_time / num_images_processed) * 1000.0
            total_one_time += avg_ms
            print(f"- {key+':':<25} {avg_ms:.2f} ms")
        print(f"- {'TOTAL SETUP:':<25} {total_one_time:.2f} ms")

        # 2) per-iteration stuff
        print("\n--- Per-Iteration Costs (ms) ---")
        total_per_iter = 0.0

        # use the function arg first; fall back to depth_cfg
        iters_per_image = iters if iters else int(depth_cfg.get('iters', 1)) or 1

        for key, total_time in per_iter.items():
            avg_ms = (total_time / (num_images_processed * iters_per_image)) * 1000.0
            total_per_iter += avg_ms
            print(f"- {key+':':<25} {avg_ms:.2f} ms")
        print(f"- {'TOTAL PER ITERATION:':<25} {total_per_iter:.2f} ms")

    return num_images_processed


def parse_args():
    p = argparse.ArgumentParser(description="SAM refiner (single-file) with on-the-fly PairRouter + experts.")
    # Essential paths
    p.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint (vit_h.pth, etc.)')
    p.add_argument('--input_root', required=True, help='Root with <subdir>/{raw.png,gt.png}')
    p.add_argument('--pred_roots', nargs='+', required=True, help='Dirs with <subdir>_pred.png')
    p.add_argument('--output_dir', required=True, help='Where to dump each model CSV')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sam_model', choices=['vit_b','vit_l','vit_h'], default='vit_h')

    # Router
    p.add_argument('--router_ckpt', default=None, help='Path to PairRouter checkpoint.')
    p.add_argument('--router_topk', type=int, default=2, help="Top-k experts (≤ 2 enforced).")
    p.add_argument('--expert_weights', type=str, default=None,
                   help="Manual override: 'Dp=1.0,Gd=0.5,sp=0.8' (renormalized).")
    p.add_argument('--router_recompute_each_iter', action='store_true',
                   help="(Optional) If set, you may re-run the router each iter (not used by default).")

    # Saliency & refinement
    p.add_argument('--beta', type=float, default=1.4, help='Entropy damping beta for moe saliency.')
    p.add_argument('--point_map', choices=['moe','blended','entropy'], default='moe')
    p.add_argument('--k_points', type=int, default=3)
    p.add_argument('--suppression_frac', type=float, default=0.05, help="Suppression fraction for pos-point picking.")
    p.add_argument('--iters', type=int, default=8)

    # SAM prompts
    p.add_argument('--no_box', action='store_true')
    p.add_argument('--no_mask', action='store_true')
    p.add_argument('--no_neg_point', action='store_true')
    p.add_argument('--margin', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=4.0)
    p.add_argument('--strength', type=int, default=30)

    # Marigold (Depth 'Gd')
    p.add_argument('--marigold_checkpoint', default=None, help='HuggingFace id or local path')
    p.add_argument('--marigold_processing_res', type=int, default=None)
    p.add_argument('--marigold_denoise_steps', type=int, default=None)
    p.add_argument('--marigold_ensemble', type=int, default=1)
    p.add_argument('--marigold_half', action='store_true')
    p.add_argument('--marigold_seed', type=int, default=None)

    # Debug/logging
    p.add_argument('--log_points', action='store_true', help='Print placed points (x,y,label) each iter.')
    p.add_argument('--light_mode', action='store_true', help="Enable PromptMoE-Light (6 non-learned experts only).")

    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.router_topk > 2:
        raise ValueError("router_topk must be ≤ 2. Use 1 (utility) or 2 (pair-marginal).")


    # Load SAM ONCE
    sam_model = load_sam(args.checkpoint, device, model_type=args.sam_model)
    predictor = SamPredictor(sam_model)  # single shared predictor
    resize_tf = ResizeLongestSide(sam_model.image_encoder.img_size)  # reused
    mask_gen = SamAutomaticMaskGenerator(model=sam_model)            # reused

    # Router
    router, router_cfg = load_router(args.router_ckpt, device)
    if not args.expert_weights and not router:
        raise ValueError("Provide --router_ckpt or use --expert_weights to bypass the router.")
    if router:
        print("[Router] Ready.")

    # Marigold (optional)
    mg = None
    if args.marigold_checkpoint:
        if not _MARIGOLD_AVAILABLE:
            raise SystemExit("[Marigold] marigold package not installed but --marigold_checkpoint provided.")
        mg = MarigoldRunner(args.marigold_checkpoint, device, args.marigold_half)
        print("[Marigold] Depth pipeline initialized.")
    else:
        print("[Marigold] Not provided. If 'Gd' is selected it will be skipped with a warning.")

    depth_cfg = dict(
        denoise=args.marigold_denoise_steps,
        proc_res=args.marigold_processing_res,
        ens=args.marigold_ensemble,
        seed=args.marigold_seed,
        iters=args.iters,
    )

    timers = {
        'one_time': defaultdict(float),
        'per_iter': defaultdict(float)
    }
    total_iters = args.iters

    os.makedirs(args.output_dir, exist_ok=True)
    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip('/'))
        out_csv = os.path.join(args.output_dir, f"{model_name}.csv")
        print(f"--- Processing Model: {model_name} ---")
        # Reset timers for each model
        timers['one_time'].clear()
        timers['per_iter'].clear()
        num_images = process_model(
            input_root=args.input_root,
            pred_root=pred_root,
            sam_model=sam_model,
            predictor=predictor,
            device=device,
            output_csv=out_csv,
            router=router,
            router_cfg=router_cfg,
            mg=mg,
            beta=args.beta,
            point_map=args.point_map,
            k_points=args.k_points,
            suppression_frac=args.suppression_frac,
            iters=args.iters,
            use_point=True,
            use_box=(not args.no_box),
            use_mask=(not args.no_mask),
            add_neg=(not args.no_neg_point),
            margin=args.margin,
            gamma=args.gamma,
            strength=args.strength,
            router_topk=args.router_topk,
            depth_cfg=depth_cfg,
            resize_tf=resize_tf,
            mask_gen=mask_gen,
            manual_weights_str=args.expert_weights,
            log_points=args.log_points,
            light_mode=args.light_mode,
            timers=timers
        )
        

if __name__ == "__main__":
    main()
