#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM refiner with ROUTER + ON-THE-FLY experts (no precomputed expert PNGs).

What this does:
  - Loads SAM (vit_b/l/h) and runs your existing SAM refinement loop.
  - For each frame, a router selects a subset of experts to compute.
  - Only those experts (and their minimal prerequisites) are computed on-the-fly
    using functions imported from expertUtils.py.
  - A saliency map is built from the selected experts:
        blended = sum_i  w_i * expert_i
        entropy = std(expert_1, ..., expert_k)
        saliency = blended * clamp(1 - beta * entropy, 0, 1)
    (or 'blended' only, or 'entropy' inverted, via --point_map).
  - The saliency drives your point prompts exactly like your “perfect” code.
  - No expert PNGs are read/written; this is fully online.

Router:
  - If you pass --router_ckpt, a small CNN is loaded (expects 4ch: RGB+mask at 128x128)
    and outputs weights for the 10 experts. Top-K + thresholding select the experts.
  - If you do NOT pass a checkpoint, a deterministic rule-based router produces weights
    from cheap image stats and the coarse mask.

Optional:
  - --router_recompute_each_iter recomputes experts & saliency at each SAM iteration
    with the current mask (heavier, but sometimes better).

Outputs/CSV layout:
  - Exactly like your original evaluator: per-model CSV with
    subdir, unrefined_iou, unrefined_boundary_iou, refined_iou, refined_boundary_iou.
"""

from __future__ import annotations
import os, sys, csv, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm # ADD THIS IMPORT at the top of your script


from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator


# ====== import your expert code ======
# Assumes expertUtils.py is in the same directory or on PYTHONPATH.
from expertUtils import (
    # I/O + helpers
    load_image, MarigoldRunner, pre_lab, pre_luminance, pre_local_mean,
    pre_grad_mag, pre_slic, pre_sam,
    # EVAL (mask-relative) experts
    eval_dp, eval_gd, eval_gc_lab, eval_brightness, eval_contrast,
    eval_texture, eval_superpixel, eval_sam_frac_sm, eval_sam_prompt
)

from embedUtils import EmbeddingManager

# ------------------------------------------------------------------------------------
# Constants / Names
# ------------------------------------------------------------------------------------
EXPERT_NAMES = ["Dp","Gd","Gc","sp","ctr","bri","tex","frac","Sm","prompt"]

# ------------------------------------------------------------------------------------
# Small CNN router (optional learned router)
# ------------------------------------------------------------------------------------
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
                            enc_g.unsqueeze(1).unsqueeze(1).expand(B,E,E,H)], dim=-1)
        I = self.mlp_I(feats).squeeze(-1)
        I = 0.5*(I + I.transpose(1,2))
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
            w = w * mask; w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return w[0]
    
def build_context_g(
    embed_manager: EmbeddingManager,
    image_path: str,
    image_rgb: np.ndarray,
    coarse_mask: np.ndarray,
    router_cfg: dict
) -> torch.Tensor:
    """Builds the context vector 'g' using the on-the-fly EmbeddingManager."""
    d_ctx = router_cfg.get("d_ctx", 256 + 6) # Default to SAM token + cheap feats

    # Get pooled image embedding (g_img)
    _, pooled_embed = embed_manager.get_image_embeddings(image_path, image_rgb)
    g_img = pooled_embed.cpu().numpy()

    # Get mask features (g_mask)
    g_mask = embed_manager.get_mask_features(image_path, image_rgb, coarse_mask)

    # Combine, pad/trim to the dimension the router expects
    g_combined = np.concatenate([g_img, g_mask]).astype(np.float32)
    if g_combined.size < d_ctx:
        g_combined = np.pad(g_combined, (0, d_ctx - g_combined.size))
    elif g_combined.size > d_ctx:
        g_combined = g_combined[:d_ctx]

    return torch.from_numpy(g_combined).to(embed_manager.device)


def _resize_to_square_pad(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    x: [C,H,W] float in [0..255] (RGB) or [0..1] (mask)
    Output: [C,size,size]
    """
    C, H, W = x.shape
    scale = size / float(max(H, W))
    H2 = int(round(H * scale))
    W2 = int(round(W * scale))
    x2 = F.interpolate(x[None], size=(H2, W2), mode="bilinear", align_corners=False)[0]
    padh = size - H2
    padw = size - W2
    if padh > 0 or padw > 0:
        x2 = F.pad(x2, (0, padw, 0, padh))
    return x2

@torch.no_grad()
def get_weights_from_pair_router(
    router: PairRouter,
    context_g: torch.Tensor,
    topk: int
) -> Dict[str, float]:
    """
    Runs the PairRouter model on a context vector to get expert weights.
    """
    # Get the raw gating weights from the router model
    method = "utility" if topk == 1 else "pair_marginal"
    weights_tensor = router.gating_weights(context_g, method=method, topk=topk)


    # Create the dictionary of selected experts and their weights
    selected = {
        name: float(weights_tensor[i].item())
        for i, name in enumerate(EXPERT_NAMES)
        if weights_tensor[i].item() > 1e-6  # Filter out near-zero weights
    }

    # Optional but good practice: Re-normalize to ensure weights sum exactly to 1
    total_weight = sum(selected.values())
    if total_weight > 0:
        for key in selected:
            selected[key] /= total_weight
            
    return selected

def save_selected_experts_to_csv(selected_experts_dict, output_csv):
    """Saves the selected experts and their weights to a CSV file."""
    rows = []
    for image_name, selected_experts in selected_experts_dict.items():
        row = [image_name]
        for expert in EXPERT_NAMES:
            row.append(selected_experts.get(expert, 0.0))
        rows.append(row)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name'] + EXPERT_NAMES)
        writer.writerows(rows)
    print(f"→ Saved selected experts CSV to {output_csv}")


def load_router(
    router_ckpt: Optional[str],
    device: torch.device
) -> Tuple[Optional[PairRouter], Optional[dict]]:
    """
    Loads a PairRouter model and its configuration from a checkpoint.
    """
    if router_ckpt is None:
        return None, None
        
    if not os.path.exists(router_ckpt):
        print(f"[Router] Warning: Checkpoint not found at {router_ckpt}. Continuing without router.")
        return None, None

    payload = torch.load(router_ckpt, map_location=device)
    cfg = payload.get("cfg", {})
    
    model = PairRouter(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    
    print(f"[Router] Loaded PairRouter with context size d_ctx={cfg.get('d_ctx')}")
    return model, cfg


# ------------------------------------------------------------------------------------
# SAM prompt helpers (unchanged logic from your “perfect” code)
# ------------------------------------------------------------------------------------
def gaussian_2d(shape, gamma_x=1, gamma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x / (2 * gamma_x * gamma_x) + y * y / (2 * gamma_y * gamma_y)))
    return h

def get_mask_embed(mask, img_embed):
    orig_H, orig_W = mask.shape[:2]
    embed_H, embed_W = img_embed.shape[-2:]
    if orig_H >= orig_W:
        resize_W = int(embed_H * orig_W / orig_H)
        resize_H = embed_H
    else:
        resize_H = int(embed_W * orig_H / orig_W)
        resize_W = embed_W
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

def strat_topk_saliency_suppressed(
    mask: np.ndarray, pm: np.ndarray, k: int = 3, suppression_frac: float = 0.03
) -> List[Tuple[int, int]]:
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
        dists = np.hypot(coords[:, 1] - x0, coords[:, 0] - y0)
        if dists.size == 0:
            break
        curr_n = min(suppression_n, dists.size)
        thresh = np.partition(dists, curr_n - 1)[curr_n - 1]
        YY, XX = np.indices((H, W))
        suppress_mask = ((XX - x0) ** 2 + (YY - y0) ** 2) <= (thresh ** 2)
        avail[suppress_mask] = 0
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
        if add_neg and m_bool.any():
            m_np = m_bool.cpu().numpy().astype(np.uint8)  # 0/1
            m_rev_np = (1 - m_np) * 255                   # 0/255 background=0, outside=255
            dist_rev_np = cv2.distanceTransform(m_rev_np, cv2.DIST_L2, 5)
            dt_r = torch.from_numpy(dist_rev_np.astype(np.float32)).to(device)

            # --- bbox constraint (inclusive) ---
            ys, xs = torch.where(m_bool)
            y1 = int(ys.min().item()); y2 = int(ys.max().item())
            x1 = int(xs.min().item()); x2 = int(xs.max().item())
            dt_r_masked = dt_r.clone()
            mask_bbox = torch.zeros_like(dt_r_masked, dtype=torch.bool)
            mask_bbox[y1:y2+1, x1:x2+1] = True
            dt_r_masked[~mask_bbox] = 0
            # -----------------------------------

            if (dt_r_masked > 0).any():
                ys, xs = torch.where(dt_r_masked == dt_r_masked.max())
                pc.append([xs[0].item(), ys[0].item()]); pl.append(0)
            else:
                # Fallback: pick a background pixel if DT is degenerate
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
    # --- added: keep original (un-resized) points for printing/logging ---
    inp['raw_point_coords'] = pc  # [B, P, 2] tensor in original image coords (x, y)
    inp['raw_point_labels'] = pl  # [B, P] tensor (1=pos, 0=neg)
    # --------------------------------------------------------------------
    if use_mask:
        tgt_sz = image.shape[1:]
        inp['mask_inputs'] = extract_mask(pred_masks, gd, tgt_sz, True, strength, image.device, ex)
    return inp


# ------------------------------------------------------------------------------------
# On-the-fly experts & saliency assembly
# ------------------------------------------------------------------------------------
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
    predictor,                                   # SamPredictor
    resize_tf: ResizeLongestSide,
    mg: Optional[MarigoldRunner],
    device: torch.device,
    mask_gen: SamAutomaticMaskGenerator,         # NEW
) -> Dict[str, torch.Tensor]:


    """
    Compute only the selected expert maps, running minimal prerequisites.
    Returns dict: {name: map_tensor_on_device_in_[0,1]}
    """
    H, W = fg.shape
    fg_bool = fg.bool().to(device)

    # Lazy pre-feature cache
    cache = {}

    def need(*keys):
        return any(k in selected for k in keys)

    # Compute prereqs lazily
    if need("bri", "ctr", "tex", "Gc"):
        lum, _ = pre_luminance(img_rgb); cache["lum"] = lum  # CPU
    if need("ctr"):
        local_mean, _ = pre_local_mean(cache["lum"], k=7); cache["local_mean"] = local_mean # CPU
    if need("tex"):
        gradmag, _ = pre_grad_mag(cache["lum"]); cache["gradmag"] = gradmag # CPU
    if need("Gc"):
        lab_f, _ = pre_lab(img_rgb); cache["lab_f"] = lab_f  # np.float32 HxWx3
    if need("sp"):
        seg_slic, _ = pre_slic(img_rgb, n_segments=200, compactness=10.0, max_side=1024); cache["seg_slic"] = seg_slic # np.int32 HxW
    if need("prompt"):
        cache["sam_info"] = {"predictor": predictor}
    if need("frac", "Sm"):
        sam_info, _, seg_sam_full = pre_sam(img_rgb, mask_gen=mask_gen, resize_t=resize_tf)
        cache["seg_sam_full"] = seg_sam_full

    # Depth handled in build_saliency_for_mask to keep image-path access consolidated.

    # Now build maps (only for selected experts)
    out: Dict[str, torch.Tensor] = {}

    for name in selected.keys():
        if name == "Dp":
            dp_map, _ = eval_dp(fg, device)
            out["Dp"] = _minmax01(dp_map.to(device))

        elif name == "Gd":
            # Filled later in build_saliency_for_mask (needs image path).
            out["Gd"] = torch.tensor(-1.0, device=device)  # sentinel

        elif name == "Gc":
            gc_map, _ = eval_gc_lab(cache["lab_f"], fg_bool, omega=1.0, gamma=0.65, device=device)
            out["Gc"] = _minmax01(gc_map)

        elif name == "sp":
            sp_map, _ = eval_superpixel(cache["seg_slic"], fg_bool); out["sp"] = _minmax01(sp_map.to(device))

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
    img_rgb: torch.Tensor,
    fg01: torch.Tensor,
    selected: Dict[str, float],
    resize_tf,
    mg: Optional[MarigoldRunner],
    device: torch.device,
    beta: float,
    point_map: str,
    depth_cfg: Dict[str, object],
    predictor: SamPredictor,
    mask_gen: SamAutomaticMaskGenerator,       # NEW
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


    """
    Computes selected expert maps, assembles the saliency map, and returns both.
    This version correctly handles the on-demand computation of the depth expert.
    """
    # This function call correctly returns a placeholder for 'Gd' if it's selected
    maps = compute_selected_experts_maps(
        img_rgb, fg01, selected,
        predictor=predictor,
        resize_tf=resize_tf, mg=mg, device=device,
        mask_gen=mask_gen,                         # NEW ARG
    )




    # --- CORRECTED LOGIC FOR DEPTH EXPERT ---
    # Check if the 'Gd' expert was selected and is still a placeholder
    if "Gd" in selected and isinstance(maps.get("Gd"), torch.Tensor) and maps["Gd"].numel() == 1:
        if mg is None:
            print("Warning: Depth expert 'Gd' was selected, but Marigold is not available. Skipping.")
            # Remove the placeholder so it's not used later
            del maps["Gd"]
        else:
            # This block will now correctly execute
            # print("Computing Depth (Gd) expert...")
            d01 = mg.infer_depth01(
                raw_path,
                denoise_steps=depth_cfg.get("denoise"),
                processing_res=depth_cfg.get("proc_res"),
                ensemble_size=depth_cfg.get("ens", 1),
                seed=depth_cfg.get("seed")
            )
            d255 = (d01 * 255.0).clamp(0, 255).float()
            gd_map, _ = eval_gd(d255.to(device), fg01.bool().to(device), omega=1.0, gamma=0.65)
            maps["Gd"] = _minmax01(gd_map) # Overwrite the placeholder with the real map

    H, W = fg01.shape
    stack, weights = [], []
    
    # Filter for only successfully computed 2D maps before stacking
    valid_selected = {k: v for k, v in selected.items() if k in maps and isinstance(maps[k], torch.Tensor) and maps[k].ndim == 2}
    
    if not valid_selected:
        print(f"Warning: No valid expert maps were computed for the selection: {list(selected.keys())}")
        return torch.zeros((H, W), dtype=torch.float32), maps

    for name, weight in valid_selected.items():
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
    else: # "moe"
        sal = blended * torch.clamp(1.0 - float(beta) * entropy, 0.0, 1.0)

    sal = torch.clamp(sal, 0.0, 1.0).detach().cpu()
    return sal, maps


# ------------------------------------------------------------------------------------
# SAM refiner (unchanged iteration structure, but saliency is built on-the-fly)
# ------------------------------------------------------------------------------------
def prepare_image_for_sam(image_np: np.ndarray, transform: ResizeLongestSide, device: torch.device) -> torch.Tensor:
    image_rs = transform.apply_image(image_np)
    image_t = torch.as_tensor(image_rs, device=device)
    return image_t.permute(2, 0, 1).contiguous()

@torch.no_grad()
def sam_refiner_router(
    # --- Function Arguments ---
    image_path: str,
    coarse_masks: List[np.ndarray] | np.ndarray,
    sam,
    device: torch.device,
    resize_tf: Optional[ResizeLongestSide],
    embed_manager: EmbeddingManager,
    router: Optional[PairRouter],
    router_cfg: Optional[dict],
    mg: Optional[MarigoldRunner],
    beta: float,
    point_map: str,
    k_points: int,
    suppression_frac: float,
    use_point: bool, use_box: bool, use_mask: bool, add_neg: bool,
    margin: float, gamma: float, strength: int,
    router_topk: int,
    depth_cfg: Dict[str, object],
    args: argparse.Namespace,
    mask_gen: SamAutomaticMaskGenerator
) -> np.ndarray:
    """
    Refines a coarse mask using SAM, with expert selection guided by an on-the-fly
    PairRouter or manual weight overrides.
    """
    # --- Mask and Image Preparation ---
    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, 0)
    if coarse_masks.ndim == 2:
        coarse_masks = coarse_masks[None, ...]
    
    masks_t = torch.as_tensor(coarse_masks, dtype=torch.uint8, device=device)
    coarse_mask_np = (masks_t[0].cpu().numpy() > 0).astype(np.uint8) * 255

    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_t_cpu = load_image(image_path)
    resize_tf = resize_tf or ResizeLongestSide(sam.image_encoder.img_size)
    img_t = prepare_image_for_sam(img_rgb, resize_tf, device)

    # --- Intelligent Expert Routing with Manual Override ---
    selected: Optional[Dict[str, float]] = None

    # 1. Check for manual weight override from the command line first
    if args.expert_weights:
        print("[Router] Using manual weights from --expert_weights.")
        selected = parse_expert_weights_from_str(args.expert_weights)

    # 2. If no override is provided, use the PairRouter
    if selected is None:
        if not (router and router_cfg):
            raise ValueError("A valid --router_ckpt must be provided when not using --expert_weights.")
        
        context_g = build_context_g(
            embed_manager=embed_manager,
            image_path=image_path,
            image_rgb=img_rgb,
            coarse_mask=coarse_mask_np,
            router_cfg=router_cfg
        )
        selected = get_weights_from_pair_router(router, context_g, topk=router_topk)

    if not selected:
        raise RuntimeError("Expert selection failed. No experts were chosen or provided.")

    # --- Saliency Map and Refinement Loop ---
    img_emb, _ = embed_manager.get_image_embeddings(image_path, img_rgb)

    coarse01 = torch.from_numpy(coarse_mask_np // 255).to(torch.uint8)
    # unpack (saliency, maps) and use saliency tensor
    sal_map, _ = build_saliency_for_mask(
        image_path, img_t_cpu, coarse01, selected, resize_tf, mg, device,
        beta=beta, point_map=point_map, depth_cfg=depth_cfg,
        predictor=embed_manager.sam_predictor,
        mask_gen=mask_gen,                         # NEW ARG
    )




    # Iterative refinement using the generated saliency map
    sam_masks = None 
    for i in range(depth_cfg.get("iters", 5)):
        masks_cur = masks_t if i == 0 else (sam_masks > 0).to(torch.uint8)
        
        inp = sam_input_prepare(
            img_t, masks_cur, img_emb, resize_tf, sal_map.to(device),
            use_point, use_box, use_mask, add_neg, margin, gamma, strength,
            k_points=k_points, suppression_frac=suppression_frac
        )

        # --- added: print raw points placed this iteration (original image coordinates) ---
        if getattr(args, "log_points", False) and 'raw_point_coords' in inp and 'raw_point_labels' in inp:
            rpc = inp['raw_point_coords'].detach().cpu().tolist()  # [B, P, 2]
            rpl = inp['raw_point_labels'].detach().cpu().tolist()  # [B, P]
            for bi, (pts, labs) in enumerate(zip(rpc, rpl)):
                triplets = [(int(x), int(y), int(l)) for (x, y), l in zip(pts, labs)]
                print(f"[Iter {i}] mask {bi} points (x,y,label): {triplets}")

        # ---------------------------------------------------------------------------------

        with torch.no_grad():
            out = sam.forward_with_image_embeddings(img_emb, [inp], multimask_output=True)[0]
        
        best = torch.argmax(out['iou_predictions'], dim=-1)
        sam_masks = torch.stack([m[idx] for m, idx in zip(out['masks'], best)], 0)

    return (sam_masks > 0).detach().cpu().numpy().astype(np.uint8)



# ------------------------------------------------------------------------------------
# IoU utilities and driver (CSV output identical to your original)
# ------------------------------------------------------------------------------------
def compute_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    b1 = m1 > 0; b2 = m2 > 0
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter) / union if union else 0.0

def parse_expert_weights_from_str(weights_str: str) -> Optional[Dict[str, float]]:
    """Parses a comma-separated string of weights into a dictionary."""
    if not weights_str:
        return None
    try:
        weights = {}
        for part in weights_str.split(','):
            key, value = part.strip().split('=')
            weights[key.strip()] = float(value.strip())
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        return weights
    except Exception as e:
        print(f"Warning: Could not parse --expert_weights string '{weights_str}'. Error: {e}")
        return None

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

def load_sam(checkpoint_path: str, device: torch.device, model_type: str = "vit_h"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device); sam.eval()
    return sam

def process_model(
    # --- Function Arguments ---
    input_root: str,
    pred_root: str,
    sam,
    device: torch.device,
    output_csv: str,
    embed_manager: EmbeddingManager,
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
    args: argparse.Namespace,
    mask_gen: SamAutomaticMaskGenerator
):
    subdirs = sorted(d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d)))
    rows = []
    weights_rows = []
    weights_full_rows = []

    
    model_name = os.path.basename(pred_root.rstrip('/'))
    selected_experts_dict = {}
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

        # --- Router weights per image (subdir) ---  [ADDED]
        selected_weights_dict = None
        full_weights_vec = None
        if args.expert_weights:
            selected_weights_dict = parse_expert_weights_from_str(args.expert_weights) or {}
        elif router and router_cfg:
            img_bgr = cv2.imread(raw_p, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
            coarse_mask_for_ctx = (pm8 > 0).astype(np.uint8) * 255
            context_g = build_context_g(
                embed_manager=embed_manager,
                image_path=raw_p,
                image_rgb=img_rgb if img_rgb is not None else cv2.cvtColor(cv2.imread(raw_p), cv2.COLOR_BGR2RGB),
                coarse_mask=coarse_mask_for_ctx,
                router_cfg=router_cfg
            )
            # full vector (pre-topk)
            # method_full = "utility" if router_topk == 1 else "pair_marginal"
            # w_full = router.gating_weights(context_g, method=method_full, topk=None)
            # full_weights_vec = [float(w_full[i].item()) for i in range(len(EXPERT_NAMES))]
            # selected (post-topk normalized)
            selected_weights_dict = get_weights_from_pair_router(router, context_g, topk=router_topk)
            selected_experts_dict[sub] = selected_weights_dict
        else:
            selected_weights_dict = {}
        # align to fixed column order
        selected_row = [selected_weights_dict.get(name, 0.0) for name in EXPERT_NAMES]
        weights_rows.append((sub, *selected_row))
        if full_weights_vec is not None:
            weights_full_rows.append((sub, *full_weights_vec))
        # -------------------------------------------

        # Refine with router + on-the-fly experts
        depth_cfg_loc = dict(depth_cfg)
        depth_cfg_loc["iters"] = iters
        
        rf8 = sam_refiner_router(
            raw_p, [pm8.astype(np.float32) / 255.0], sam, device,
            ResizeLongestSide(sam.image_encoder.img_size),
            embed_manager,
            router,
            router_cfg,
            mg,
            beta, point_map, k_points, suppression_frac,
            use_point, use_box, use_mask, add_neg,
            margin, gamma, strength,
            router_topk,
            depth_cfg_loc,
            args,
            mask_gen
        )[0] * 255

        r_iou = compute_iou(rf8, gt8)
        r_biou = compute_boundary_iou(rf8, gt8)

        rows.append((sub, u_iou, u_biou, r_iou, r_biou))

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subdir', 'unrefined_iou', 'unrefined_boundary_iou', 'refined_iou', 'refined_boundary_iou'])
        w.writerows(rows)
    print(f"→ saved {output_csv}")

    # NEW: write per-image expert weights (post-topk) CSV
    # weights_csv = os.path.join(os.path.dirname(output_csv) or '.', f"{model_name}_weights.csv")
    # with open(weights_csv, 'w', newline='') as f:
    #     w = csv.writer(f)
    #     w.writerow(['subdir'] + EXPERT_NAMES)
    #     w.writerows(weights_rows)
    # print(f"→ saved {weights_csv}")

    # NEW: optional pre-topk (full) weights CSV
    # if weights_full_rows:
    #     weights_full_csv = os.path.join(os.path.dirname(output_csv) or '.', f"{model_name}_weights_full.csv")
    #     with open(weights_full_csv, 'w', newline='') as f:
    #         w = csv.writer(f)
    #         w.writerow(['subdir'] + [f"full_{n}" for n in EXPERT_NAMES])
    #         w.writerows(weights_full_rows)
    #     print(f"→ saved {weights_full_csv}")
    output_csv_path = os.path.join(os.path.dirname(output_csv) or '.', f"{model_name}_weights.csv")
    save_selected_experts_to_csv(selected_experts_dict, output_csv_path)




# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SAM refiner with on-the-fly PairRouter and experts.")
    
    # --- Essential Paths ---
    p.add_argument('--checkpoint', required=True, help='Path to SAM checkpoint (vit_h.pth, etc.)')
    p.add_argument('--input_root', required=True, help='Root with <subdir>/{raw.png,gt.png}')
    p.add_argument('--pred_roots', nargs='+', required=True, help='Dirs with <subdir>_pred.png')
    p.add_argument('--output_dir', required=True, help='Where to dump each model’s CSV')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sam_model', choices=['vit_b','vit_l','vit_h'], default='vit_h')

    # --- PairRouter Arguments ---
    p.add_argument('--router_ckpt', default=None, help='Path to PairRouter checkpoint.')
    p.add_argument('--router_topk', type=int, default=6, help="Select top-k experts from the router.")
    p.add_argument('--expert_weights', type=str, default=None,
                   help="Manual expert weights to override the router. Format: 'Dp=1.0,Gd=0.5,sp=0.8'")

    # --- Saliency & Refinement Hyperparameters (with tuned defaults) ---
    p.add_argument('--beta', type=float, default=1.4, help='Beta for entropy modulation in saliency blend.')
    p.add_argument('--point_map', choices=['moe','blended','entropy'], default='moe')
    p.add_argument('--k_points', type=int, default=3)
    p.add_argument('--suppression_frac', type=float, default=0.05, help="Suppression fraction for positive-point picking.")
    p.add_argument('--iters', type=int, default=8)

    # --- SAM Prompting Arguments ---
    p.add_argument('--no_box', action='store_true')
    p.add_argument('--no_mask', action='store_true')
    p.add_argument('--no_neg_point', action='store_true')
    p.add_argument('--margin', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=4.0)
    p.add_argument('--strength', type=int, default=30)

    # --- Marigold (Depth Expert) Arguments ---
    p.add_argument('--marigold_checkpoint', default=None, help='HuggingFace id or local path for Marigold depth')
    p.add_argument('--marigold_processing_res', type=int, default=None)
    p.add_argument('--marigold_denoise_steps', type=int, default=None)
    p.add_argument('--marigold_ensemble', type=int, default=1)
    p.add_argument('--marigold_half', action='store_true')
    p.add_argument('--marigold_seed', type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.router_topk > 2:
        raise ValueError("router_topk must be ≤ 2. Use 1 (utility) or 2 (pair-marginal).")

    # 1. Load the main SAM model for refinement
    sam = load_sam(args.checkpoint, device, model_type=args.sam_model)
    
    # 2. Initialize the EmbeddingManager for on-the-fly feature computation
    sam_predictor = SamPredictor(sam)  # one shared predictor bound to the loaded SAM
    embed_manager = EmbeddingManager.from_predictor(sam_predictor, device=args.device)
    mask_gen = SamAutomaticMaskGenerator(model=sam)



    # 3. Load the PairRouter and its config
    router, router_cfg = load_router(args.router_ckpt, device)
    if router:
        print("[Router] Loaded PairRouter checkpoint.")
    else:
        # This print statement is now removed because the fallback is gone.
        # The script will raise an error in sam_refiner_router if the checkpoint is missing.
        pass

    # 4. Initialize Marigold (if provided)
    mg = None
    if args.marigold_checkpoint:
        mg = MarigoldRunner(args.marigold_checkpoint, device, args.marigold_half)
        print("[Marigold] Depth pipeline initialized.")
    else:
        print("[Marigold] Not provided. If router selects 'Gd', an error will be raised.")

    depth_cfg = dict(
        denoise=args.marigold_denoise_steps,
        proc_res=args.marigold_processing_res,
        ens=args.marigold_ensemble,
        seed=args.marigold_seed,
        iters=args.iters
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip('/'))
        out_csv = os.path.join(args.output_dir, f"{model_name}.csv")
        print(f"--- Processing Model: {model_name} ---")
        
        # 5. Call process_model with all the correct arguments
        process_model(
            input_root=args.input_root,
            pred_root=pred_root,
            sam=sam,
            device=device,
            output_csv=out_csv,
            embed_manager=embed_manager,
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
            args=args,
            mask_gen=mask_gen,
        )


if __name__ == "__main__":
    main()
