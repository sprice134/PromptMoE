#!/bin/bash
#SBATCH --job-name=BIG_Comp
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Total memory:   ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU}"

# Activate env
set +u
eval "$(conda shell.bash hook)"
conda activate PromptMoE
set -u

# Working dir
cd /home/sprice/PromptMoE

# ---------- CONSTANTS ----------
SAMREFINER_PATH="/home/sprice/PromptMoE/PromptMoE.py"
ROUTER_CKPT="/home/sprice/PromptMoE/pairRouter_10_13.pt"

CKPT_VIT_H="/home/sprice/SAMRefiner/vit_h.pth"
SAM_MODEL="vit_h"

# Depth (Marigold)
MARIGOLD="--marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half"

# Common knobs
BETA=1.4
ITERS=10
ROUTER_TOPK=2

# Output root (per dataset we will create Exp1..Exp7 under here)
OUT_ROOT="/home/sprice/PromptMoE/paper_results/ablation/components"

# DATASETS (names must match the paths below)
DATASETS=("VOC")

# ---------- PER-DATASET PATHS ----------
get_dataset_paths () {
  local dataset="$1"
  case "$dataset" in
    BIG)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/BIG/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3"
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN"
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP"
      )
      ;;
    ECSSD)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/ECSSD/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/briaai_RMBG-1.4"
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZhengPeng7_BiRefNet-matting"
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZQL9711_RMBG-2-Matting"
      )
      ;;
    DAVIS585)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/DAVIS585/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/SP"
        "/home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/STM"
      )
      ;;
    VOC)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/VOC/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/DeepLabV3"
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/FCN"
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/LR-ASPP"
      )
      ;;
    MSRA-B)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/MSRA-B/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/MSRA-B/outputs/briaai_RMBG-1.4"
        "/home/sprice/PromptMoE/DATASETS/MSRA-B/outputs/ZhengPeng7_BiRefNet-matting"
        "/home/sprice/PromptMoE/DATASETS/MSRA-B/outputs/ZQL9711_RMBG-2-Matting"
      )
      ;;
    CITY)
      INPUT_ROOT="/home/sprice/PromptMoE/DATASETS/CITYSCAPES/images"
      PRED_ROOTS=(
        "/home/sprice/PromptMoE/DATASETS/CITYSCAPES/outputs/Mask2Former-Instance-Swin-Large"
        "/home/sprice/PromptMoE/DATASETS/CITYSCAPES/outputs/Mask2Former-Instance-Swin-Small"
        "/home/sprice/PromptMoE/DATASETS/CITYSCAPES/outputs/Mask2Former-Instance-Swin-Tiny"
      )
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

# Helper to expand pred_roots into CLI array safely
build_pred_roots_args () {
  CLI_PRED_ROOTS=( )
  for pr in "${PRED_ROOTS[@]}"; do
    CLI_PRED_ROOTS+=( "$pr" )
  done
}

# ---------- MAIN ----------
for DS in "${DATASETS[@]}"; do
  echo ""
  echo "==================== DATASET: $DS ===================="
  get_dataset_paths "$DS"
  build_pred_roots_args

  # Experiment 7:
  # 3 positive, 1 negative, BOX, MASK, with a 0.05 suppression factor, AND ROUTER
  OUT_DIR="${OUT_ROOT}/${DS}/Exp7"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    ${MARIGOLD} \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --router_ckpt "${ROUTER_CKPT}" --router_topk ${ROUTER_TOPK} \
    --k_points 5 --suppression_frac 0.05 --iters ${ITERS} --beta ${BETA}
  set +x

  # Experiment 1:
  # 1 positive, NO box, NO mask, NO neg, NO router, ONLY dp
  OUT_DIR="${OUT_ROOT}/${DS}/Exp1"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 1 --suppression_frac 0.00 --iters ${ITERS} --beta ${BETA} \
    --no_box --no_mask --no_neg_point \
    --expert_weights "Dp=1.0"
  set +x

  # Experiment 2:
  # 1 positive, BOX, NO mask, NO neg, NO router, ONLY dp
  OUT_DIR="${OUT_ROOT}/${DS}/Exp2"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 1 --suppression_frac 0.00 --iters ${ITERS} --beta ${BETA} \
    --no_mask --no_neg_point \
    --expert_weights "Dp=1.0"
  set +x

  # Experiment 3:
  # 1 positive, BOX, MASK, NO neg, NO router, ONLY dp
  OUT_DIR="${OUT_ROOT}/${DS}/Exp3"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 1 --suppression_frac 0.00 --iters ${ITERS} --beta ${BETA} \
    --no_neg_point \
    --expert_weights "Dp=1.0"
  set +x

  # Experiment 4:
  # 1 positive, 1 negative, BOX, MASK, Only dp
  OUT_DIR="${OUT_ROOT}/${DS}/Exp4"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 1 --suppression_frac 0.00 --iters ${ITERS} --beta ${BETA} \
    --expert_weights "Dp=1.0"
  set +x

  # Experiment 5:
  # 3 positive, 1 negative, BOX, MASK, Only dp (no suppression factor)
  OUT_DIR="${OUT_ROOT}/${DS}/Exp5"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 5 --suppression_frac 0.0 --iters ${ITERS} --beta ${BETA} \
    --expert_weights "Dp=1.0"
  set +x

  # Experiment 6:
  # 3 positive, 1 negative, BOX, MASK, Only dp, with 0.05 suppression factor
  OUT_DIR="${OUT_ROOT}/${DS}/Exp6"
  mkdir -p "$OUT_DIR"
  set -x
  python RunPromptMoE.py \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points 5 --suppression_frac 0.05 --iters ${ITERS} --beta ${BETA} \
    --expert_weights "Dp=1.0"
  set +x

done

echo "All ablation runs completed."
