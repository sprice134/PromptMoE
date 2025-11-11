#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
from typing import Optional, Iterator
import contextlib
import datetime

# Build once and pass through
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def parse_args():
    p = argparse.ArgumentParser(description="Run SAM refiner (one-file) with on-the-fly PairRouter and experts.")

    # --- Essential Paths ---
    p.add_argument("--samrefiner-path", required=True, help="Filesystem path to your main refiner script (the single-file).")
    p.add_argument("--checkpoint", required=True, help="Path to the SAM vit_*.pth checkpoint.")
    p.add_argument("--input_root", required=True, help="Root dir with raw/gt per subfolder.")
    p.add_argument("--pred_roots", required=True, nargs="+", help="One or more prediction roots with <subdir>_pred.png.")
    p.add_argument("--output_dir", required=True, help="Where to dump each model’s CSV.")
    p.add_argument("--device", default="cuda:0", help='Torch device, e.g., "cuda:0" or "cpu".')
    p.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h", "sam-vit_h"], default="vit_h", help="SAM model variant.")

    # --- PairRouter Arguments ---
    p.add_argument("--router_ckpt", default=None, help="Path to PairRouter checkpoint.")
    p.add_argument("--router_topk", type=int, default=2, help="Select top-k experts from the router (must be ≤ 2).")
    p.add_argument('--expert_weights', type=str, default=None,
                   help="Manual expert weights to override the router. Format: 'Dp=1.0,Gd=0.5,sp=0.8'")

    # --- Saliency & Refinement Hyperparameters (defaults kept from your runner) ---
    p.add_argument('--beta', type=float, default=1.397987726295555, help='Beta for entropy modulation in saliency blend.')
    p.add_argument('--point_map', choices=['moe','blended','entropy'], default='moe')
    p.add_argument('--k_points', type=int, default=3)
    p.add_argument('--suppression_frac', type=float, default=0.0960571445127933,
                   help="Suppression fraction for positive-point picking.")
    p.add_argument('--iters', type=int, default=8)

    # --- SAM Prompting Arguments ---
    p.add_argument("--no_box", action="store_true", help="Disable SAM box prompt.")
    p.add_argument("--no_mask", action="store_true", help="Disable SAM mask prior.")
    p.add_argument("--no_neg_point", action="store_true", help="Disable negative point.")
    p.add_argument("--margin", type=float, default=0.0, help="Box expansion margin.")
    p.add_argument("--gamma", type=float, default=4.0, help="Gaussian prior area divisor.")
    p.add_argument("--strength", type=int, default=30, help="Mask prior strength.")

    # --- Marigold (Depth Expert) Arguments ---
    p.add_argument('--marigold_checkpoint', default=None, help="HuggingFace id or local path for Marigold depth.")
    p.add_argument('--marigold_processing_res', type=int, default=None)
    p.add_argument('--marigold_denoise_steps', type=int, default=None)
    p.add_argument('--marigold_ensemble', type=int, default=1)
    p.add_argument('--marigold_half', action='store_true')
    p.add_argument('--marigold_seed', type=int, default=None)

    # --- Logging ---
    p.add_argument("--log_points", action="store_true", help="Capture the refiner's (x,y,label) prints into per-model logs.")
    p.add_argument("--log_dir", default=None, help="Directory for logs (defaults to <output_dir>/logs when --log_points is set).")
    p.add_argument('--light_mode', action='store_true', help="Enable PromptMoE-Light (6 non-learned experts only).")
    p.add_argument('--use_samhq', action='store_true', help="Tell the refiner to use SAM-HQ (image_encoder returns (emb, interm) and HQ decoder).")

    return p.parse_args()


class _Tee:
    """Write to multiple streams (e.g., console + file)."""
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def tee_stdout_stderr(stdout_file_path: Optional[str]) -> Iterator[None]:
    """Context that tees stdout/stderr to an optional file."""
    if stdout_file_path is None:
        yield
        return
    os.makedirs(os.path.dirname(stdout_file_path), exist_ok=True)
    with open(stdout_file_path, "a", buffering=1) as f:  # line-buffered
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n===== RUN START {stamp} =====\n")
        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            sys.stdout = _Tee(sys.stdout, f)
            sys.stderr = _Tee(sys.stderr, f)
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            sys.stdout, sys.stderr = orig_out, orig_err


def import_module_by_path(py_path: str):
    import importlib.util
    module_name = f"samrefiner_module_{os.path.basename(py_path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = parse_args()
    os.environ["PYTHONUNBUFFERED"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    if args.router_topk > 2:
        raise ValueError("router_topk must be ≤ 2 (1: utility, 2: pair-marginal).")

    if not args.expert_weights and not args.router_ckpt:
        raise ValueError("Provide --router_ckpt or use --expert_weights to bypass the router.")

    device_obj = torch.device(args.device)

    # Dynamically load the single-file refiner module
    sam_mod = import_module_by_path(args.samrefiner_path)

    # Load SAM ONCE and build shared utilities ONCE
    sam = sam_mod.load_sam(args.checkpoint, device_obj, model_type=args.sam_model)
    predictor = SamPredictor(sam)  # single shared predictor for the whole run
    resize_tf = ResizeLongestSide(sam.image_encoder.img_size)  # reused
    mask_gen = SamAutomaticMaskGenerator(model=sam)            # reused

    # Load PairRouter (optional)
    router, router_cfg = sam_mod.load_router(args.router_ckpt, device_obj)
    if router:
        print("[Router] Loaded PairRouter checkpoint.")

    # Initialize Marigold (optional)
    mg = None
    if args.marigold_checkpoint:
        mg = sam_mod.MarigoldRunner(args.marigold_checkpoint, device_obj, args.marigold_half)
        print("[Marigold] Depth pipeline initialized.")

    # Depth config block passed through to the refiner
    depth_cfg = dict(
        denoise=args.marigold_denoise_steps,
        proc_res=args.marigold_processing_res,
        ens=args.marigold_ensemble,
        seed=args.marigold_seed,
    )

    # Prepare logging directory
    log_dir = args.log_dir or (os.path.join(args.output_dir, "logs") if args.log_points else None)
    os.makedirs(args.output_dir, exist_ok=True)

    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip("/"))
        out_csv = os.path.join(args.output_dir, f"{model_name}.csv")
        log_file = None
        if args.log_points:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{model_name}_points.log")

        print(f"--- Processing Model: {model_name} ---")
        with tee_stdout_stderr(log_file):
            sam_mod.process_model(
                input_root=args.input_root,
                pred_root=pred_root,
                sam_model=sam,                   # matches single-file signature
                predictor=predictor,             # one predictor for the run
                device=device_obj,
                output_csv=out_csv,
                router=router,
                router_cfg=router_cfg,
                mg=mg,
                beta=float(args.beta),
                point_map=str(args.point_map),
                k_points=int(args.k_points),
                suppression_frac=float(args.suppression_frac),
                iters=int(args.iters),
                use_point=True,
                use_box=(not args.no_box),
                use_mask=(not args.no_mask),
                add_neg=(not args.no_neg_point),
                margin=float(args.margin),
                gamma=float(args.gamma),
                strength=int(args.strength),
                router_topk=int(args.router_topk),
                depth_cfg=depth_cfg,
                resize_tf=resize_tf,            # reuse the same ResizeLongestSide
                mask_gen=mask_gen,              # reuse the same mask generator
                manual_weights_str=args.expert_weights,
                log_points=args.log_points,
                light_mode=args.light_mode,
                use_samhq=args.use_samhq
            )


if __name__ == "__main__":
    main()
    '''
    python RunPromptMoE_HQ.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/paper_results/ablation/HQ/PromptMoE_HQ.py \
  --checkpoint /home/sprice/PromptMoE/checkpoints/sam_hq_vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/DAVIS585/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/STM \
  --output_dir /home/sprice/PromptMoE/paper_results/ablation/HQ/DAVIS585/ \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 --use_samhq


  python RunPromptMoE_HQ.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/paper_results/ablation/HQ/PromptMoE_HQ.py \
  --checkpoint /home/sprice/PromptMoE/checkpoints/sam_hq_vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
      /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
      /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
  --output_dir /home/sprice/PromptMoE/paper_results/ablation/HQ/BIG/HQ \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 --use_samhq


  python RunPromptMoE_HQ.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/paper_results/ablation/HQ/PromptMoE_HQ.py \
  --checkpoint /home/sprice/PromptMoE/checkpoints/sam_hq_vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/VOC/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/VOC/outputs/DeepLabV3 \
      /home/sprice/PromptMoE/DATASETS/VOC/outputs/FCN \
      /home/sprice/PromptMoE/DATASETS/VOC/outputs/LR-ASPP \
  --output_dir /home/sprice/PromptMoE/paper_results/ablation/HQ/VOC/HQ \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 --use_samhq


  python RunPromptMoE_HQ.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/paper_results/ablation/HQ/PromptMoE_HQ.py \
  --checkpoint /home/sprice/PromptMoE/checkpoints/sam_hq_vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/ECSSD/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/ECSSD/outputs/briaai_RMBG-1.4 \
      /home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZhengPeng7_BiRefNet-matting \
      /home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZQL9711_RMBG-2-Matting \
  --output_dir /home/sprice/PromptMoE/paper_results/ablation/HQ/ECSSD/HQ \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 --use_samhq

  python RunPromptMoE_HQ.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/paper_results/ablation/HQ/PromptMoE_HQ.py \
  --checkpoint /home/sprice/PromptMoE/checkpoints/sam_hq_vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/MSRA-B/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting \
  --output_dir /home/sprice/PromptMoE/paper_results/ablation/HQ/MSRA-B/HQ \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 --use_samhq

  python RunPromptMoE.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/condensed/PromptMoE.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/MSRA-B/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/MSRA-B/outputs/ZQL9711_RMBG-2-Matting \
  --output_dir /home/sprice/PromptMoE/SOTA/MSRA-B/PromptMoE_Reg_Condensed \
  --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2
    '''