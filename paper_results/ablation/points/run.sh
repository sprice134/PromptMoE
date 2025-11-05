#!/bin/bash
#SBATCH --job-name=MSRA_B_Point
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

# Shared hyperparams
ITERS=10
BETA=1.4
ROUTER_TOPK=2

# Depth (Marigold)
MARIGOLD="--marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half"

# Output root
OUT_ROOT="/home/sprice/PromptMoE/paper_results/ablation/points"

# DATASETS
DATASETS=("DAVIS585")

# Grids
k_points_grid=(9)
supp_grid=(0.0 0.02 0.04 0.06 0.08 0.1)

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

  for K in "${k_points_grid[@]}"; do
    for SUPP in "${supp_grid[@]}"; do
      # format suppression for folder name: replace '.' with 'p'
      SUPP_STR="${SUPP/./p}"
      OUT_DIR="${OUT_ROOT}/${DS}/K${K}_S${SUPP_STR}"
      mkdir -p "$OUT_DIR"

      echo "---- $DS | k_points=$K | suppression=$SUPP -> $OUT_DIR ----"

      set -x
      python RunPromptMoE.py \
        ${MARIGOLD} \
        --samrefiner-path "${SAMREFINER_PATH}" \
        --checkpoint "${CKPT_VIT_H}" --sam_model "${SAM_MODEL}" \
        --input_root "${INPUT_ROOT}" \
        --pred_roots "${CLI_PRED_ROOTS[@]}" \
        --output_dir "${OUT_DIR}" \
        --router_ckpt "${ROUTER_CKPT}" --router_topk ${ROUTER_TOPK} \
        --k_points ${K} \
        --suppression_frac ${SUPP} \
        --iters ${ITERS} \
        --beta ${BETA}
      set +x
    done
  done
done

echo "All points/suppression ablations completed."
