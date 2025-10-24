import argparse
import os
import sys
import torch
from typing import Optional, Iterator
import contextlib
import datetime

# NEW: build once and pass through
from segment_anything import SamAutomaticMaskGenerator, SamPredictor


# In your RUNNER script: run_sam_pair_router.py

def parse_args():
    p = argparse.ArgumentParser(description="Run SAM refiner with on-the-fly PairRouter and experts.")
    
    # --- Essential Paths ---
    p.add_argument("--samrefiner-path", required=True, help="Filesystem path to your main refiner script.")
    p.add_argument("--checkpoint", required=True, help="Path to the SAM vit_*.pth checkpoint.")
    p.add_argument("--input_root", required=True, help="Root dir with raw/gt per subfolder.")
    p.add_argument("--pred_roots", required=True, nargs="+", help="One or more prediction roots with <subdir>_pred.png.")
    p.add_argument("--output_dir", required=True, help="Where to dump each model’s CSV.")
    p.add_argument("--device", default="cuda:0", help='Torch device, e.g., "cuda:0" or "cpu".')
    p.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_h", help="SAM model variant.")

    # --- PairRouter Arguments ---
    p.add_argument("--router_ckpt", default=None, help="Path to PairRouter checkpoint.")
    p.add_argument("--router_topk", type=int, default=6, help="Select top-k experts from the router.")
    p.add_argument('--expert_weights', type=str, default=None,
                   help="Manual expert weights to override the router. Format: 'Dp=1.0,Gd=0.5,sp=0.8'")

    # --- Saliency & Refinement Hyperparameters (with tuned defaults) ---
    p.add_argument('--beta', type=float, default=1.397987726295555, help='Beta for entropy modulation in saliency blend.')
    p.add_argument('--point_map', choices=['moe','blended','entropy'], default='moe')
    p.add_argument('--k_points', type=int, default=3)
    p.add_argument('--suppression_frac', type=float, default=0.0960571445127933, help="Suppression fraction for positive-point picking.")
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

    p.add_argument("--log_points", action="store_true", help="Capture the refiner's (x,y,label) prints into per-model logs.")
    p.add_argument("--log_dir", default=None, help="Directory for logs (defaults to <output_dir>/logs when --log_points is set).")

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
    # Create a unique module name to avoid conflicts
    module_name = f"samrefiner_module_{os.path.basename(py_path)}"
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

    device_obj = torch.device(args.device)

    # Dynamically load the main refiner script
    sam_mod = import_module_by_path(args.samrefiner_path)

    # Load the main SAM model for refinement
    sam = sam_mod.load_sam(args.checkpoint, device_obj, model_type=args.sam_model)

    # NEW: build a single predictor from the loaded SAM & reuse it in EmbeddingManager
    predictor = SamPredictor(sam)
    embed_manager = sam_mod.EmbeddingManager.from_predictor(predictor, device=args.device)

    # NEW: build a single shared mask generator (used by experts like frac/Sm/prompt)
    mask_gen = SamAutomaticMaskGenerator(model=sam)

    # Load the PairRouter and its config
    router, router_cfg = sam_mod.load_router(args.router_ckpt, device_obj)
    if router:
        print("[Router] Loaded PairRouter checkpoint.")

    # Initialize Marigold if provided
    mg = None
    if args.marigold_checkpoint:
        mg = sam_mod.MarigoldRunner(args.marigold_checkpoint, device_obj, args.marigold_half)
        print("[Marigold] Depth pipeline initialized.")

    # Prepare configuration dictionary for depth expert
    depth_cfg = dict(
        denoise=args.marigold_denoise_steps,
        proc_res=args.marigold_processing_res,
        ens=args.marigold_ensemble,
        seed=args.marigold_seed
    )

    # Loop through each prediction directory and process it
    # Prepare log directory
    log_dir = args.log_dir or (os.path.join(args.output_dir, "logs") if args.log_points else None)
    os.makedirs(args.output_dir, exist_ok=True)

    for pred_root in args.pred_roots:
        model_name = os.path.basename(pred_root.rstrip("/"))
        out_csv = os.path.join(args.output_dir, f"{model_name}.csv")
        log_file = None
        if args.log_points:
            # one log per model; includes the (x,y,label) prints from the refiner loop
            log_file = os.path.join(log_dir, f"{model_name}_points.log")

        print(f"--- Processing Model: {model_name} ---")

        with tee_stdout_stderr(log_file):
            sam_mod.process_model(
                input_root=args.input_root,
                pred_root=pred_root,
                sam=sam,
                device=device_obj,
                output_csv=out_csv,
                embed_manager=embed_manager,
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
                args=args,              # contains --expert_weights, etc.
                mask_gen=mask_gen,      # NEW: pass shared generator
            )


if __name__ == "__main__":
    main()

    '''
  
  python run_sam_pair_router.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/PromptMoE/sam_pair_router.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/PromptMoE/DATASETS/DAVIS585/images \
  --pred_roots \
      /home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/STM \
  --output_dir /home/sprice/PromptMoE/SOTA/DAVIS585/PromptMoE_Reg \
  --router_ckpt pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 
  
  python run_sam_pair_router.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/CVPR26_Dataset/final_results_Oct22/sam_pair_router.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/CVPR26_Dataset/datasets/CITYSCAPES/processedVal10_23 \
  --pred_roots \
      /home/sprice/CVPR26_Dataset/final_results_Oct22/outputs/Mask2Former-Instance \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Oct22/SOTA/CITYSCAPES/PromptMoE4 \
  --router_ckpt pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 


  python run_sam_pair_router.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/CVPR26_Dataset/final_results_Oct22/sam_pair_router.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/CVPR26_Dataset/datasets/CITYSCAPES/processedVal10_23 \
  --pred_roots \
      /home/sprice/CVPR26_Dataset/final_results_Oct22/outputs/Mask2Former-Instance-Swin-Large \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Oct22/SOTA/CITYSCAPES/PromptMoE \
  --router_ckpt pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 

  python run_sam_pair_router.py \
  --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
  --samrefiner-path /home/sprice/CVPR26_Dataset/final_results_Oct22/sam_pair_router.py \
  --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
  --input_root /home/sprice/CVPR26_Dataset/datasets/CITYSCAPES/processedVal10_23 \
  --pred_roots \
      /home/sprice/CVPR26_Dataset/final_results_Oct22/outputs/Mask2Former-Instance-Swin-Large \
      /home/sprice/CVPR26_Dataset/final_results_Oct22/outputs/Mask2Former-Instance-Swin-Small \
  --output_dir /home/sprice/CVPR26_Dataset/final_results_Oct22/SOTA/CITYSCAPES/PromptMoE6 \
  --router_ckpt pairRouter_10_13.pt \
  --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 


    '''