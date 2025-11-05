#!/bin/bash
#SBATCH --job-name=MSRA_Ex
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L40S
#SBATCH --partition=short
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Total memory:   ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU}"

# --- Conda init (disable nounset while sourcing Conda hooks) ---
set +u
eval "$(conda shell.bash hook)"
conda activate PromptMoE
set -u
# ---------------------------------------------------------------

# Working dir
cd /home/sprice/PromptMoE

# ---------- CONSTANTS ----------
SAMREFINER_PATH="/home/sprice/PromptMoE/PromptMoE.py"
ROUTER_CKPT="/home/sprice/PromptMoE/pairRouter_10_13.pt"

CKPT_VIT_H="/home/sprice/SAMRefiner/vit_h.pth"
SAM_MODEL="vit_h"

# Shared hyperparams (your defaults)
K_POINTS=5
SUPPRESSION_FRAC=0.05
ITERS=10
BETA=1.4
ROUTER_TOPK=2

# Depth (Marigold)
MARIGOLD="--marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half"

# Output root
OUT_ROOT="/home/sprice/PromptMoE/paper_results/ablation/experts"

# DATASETS
DATASETS=("VOC")

# ---------- EXPERIMENT DEFINITIONS ----------
# Expert names must match your code EXACTLY:
# EXPERT_NAMES = ["Dp","Gd","Gc","sp","ctr","bri","tex","frac","Sm","prompt"]
single_experts=("Dp" "Gd" "Gc" "sp" "ctr" "bri" "tex" "frac" "Sm" "prompt" )

# For AllExperts blended, we use equal weights (parser normalizes anyway).
ALL_WEIGHTS="Dp=1,Gd=1,Gc=1,sp=1,ctr=1,bri=1,tex=1,frac=1,Sm=1,prompt=1"

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
  idx=1
  for EXP in "${single_experts[@]}"; do
    printf -v TAG "E%02d_%s" "$idx" "$EXP"
    OUT_DIR="${OUT_ROOT}/${DS}/${TAG}"
    mkdir -p "$OUT_DIR"
    echo "---- $DS | ${TAG} (single expert: ${EXP}) -> $OUT_DIR ----"

    set -x
    python RunPromptMoE.py \
      ${MARIGOLD} \
      --samrefiner-path "${SAMREFINER_PATH}" \
      --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
      --input_root "${INPUT_ROOT}" \
      --pred_roots "${CLI_PRED_ROOTS[@]}" \
      --output_dir "${OUT_DIR}" \
      --k_points ${K_POINTS} \
      --suppression_frac ${SUPPRESSION_FRAC} \
      --iters ${ITERS} \
      --beta ${BETA} \
      --router_topk ${ROUTER_TOPK} \
      --expert_weights "${EXP}=1"
    set +x

    idx=$((idx+1))
  done

  # 2) All experts blended equally (manual weights)
  TAG="E11_AllExperts"
  OUT_DIR="${OUT_ROOT}/${DS}/${TAG}"
  mkdir -p "$OUT_DIR"
  echo "---- $DS | ${TAG} (all experts blended equally) -> $OUT_DIR ----"

  set -x
  python RunPromptMoE.py \
    ${MARIGOLD} \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --k_points ${K_POINTS} \
    --suppression_frac ${SUPPRESSION_FRAC} \
    --iters ${ITERS} \
    --beta ${BETA} \
    --router_topk ${ROUTER_TOPK} \
    --expert_weights "${ALL_WEIGHTS}"
  set +x

  # 3) Router-S (single-expert utility; topk=1)
  TAG="E12_Router-S"
  OUT_DIR="${OUT_ROOT}/${DS}/${TAG}"
  mkdir -p "$OUT_DIR"
  echo "---- $DS | ${TAG} (single-expert utility) -> $OUT_DIR ----"

  set -x
  python RunPromptMoE.py \
    ${MARIGOLD} \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --router_ckpt "${ROUTER_CKPT}" \
    --router_topk 1 \
    --k_points ${K_POINTS} \
    --suppression_frac ${SUPPRESSION_FRAC} \
    --iters ${ITERS} \
    --beta ${BETA}
  set +x

  # 4) Router-P (pair interaction maximum; topk=2)
  TAG="E13_Router-P"
  OUT_DIR="${OUT_ROOT}/${DS}/${TAG}"
  mkdir -p "$OUT_DIR"
  echo "---- $DS | ${TAG} (pair interaction maximum) -> $OUT_DIR ----"

  set -x
  python RunPromptMoE.py \
    ${MARIGOLD} \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
    --input_root "${INPUT_ROOT}" \
    --pred_roots "${CLI_PRED_ROOTS[@]}" \
    --output_dir "${OUT_DIR}" \
    --router_ckpt "${ROUTER_CKPT}" \
    --router_topk 2 \
    --k_points ${K_POINTS} \
    --suppression_frac ${SUPPRESSION_FRAC} \
    --iters ${ITERS} \
    --beta ${BETA}
  set +x

done

echo "All expert ablations completed."
