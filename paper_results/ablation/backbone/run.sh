#!/bin/bash
#SBATCH --job-name=BACKBONE
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

CKPT_VIT_B="/home/sprice/SAMRefiner/vit_b.pth"
CKPT_VIT_L="/home/sprice/SAMRefiner/vit_l.pth"
CKPT_VIT_H="/home/sprice/SAMRefiner/vit_h.pth"

# Shared hyperparams
K_POINTS=5
SUPPRESSION_FRAC=0.05
ITERS=10
BETA=1.4
ROUTER_TOPK=2

# Depth (Marigold)
MARIGOLD="--marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half"

# Output root
OUT_ROOT="/home/sprice/PromptMoE/paper_results/ablation/backbone"

# ---------- FUNCTIONS ----------
run_job () {
  local dataset="$1"
  local sam_model="$2"     # vit_b | vit_l | vit_h
  local ckpt="$3"
  local input_root=""
  local -a pred_roots=()

  case "$dataset" in
    BIG)
      input_root="/home/sprice/PromptMoE/DATASETS/BIG/images"
      pred_roots=(
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3"
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN"
        "/home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP"
      )
      ;;
    ECSSD)
      input_root="/home/sprice/PromptMoE/DATASETS/ECSSD/images"
      pred_roots=(
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/briaai_RMBG-1.4"
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZhengPeng7_BiRefNet-matting"
        "/home/sprice/PromptMoE/DATASETS/ECSSD/outputs/ZQL9711_RMBG-2-Matting"
      )
      ;;
    DAVIS585)
      input_root="/home/sprice/PromptMoE/DATASETS/DAVIS585/images"
      pred_roots=(
        "/home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/SP"
        "/home/sprice/PromptMoE/DATASETS/DAVIS585/outputs/STM"
      )
      ;;
    VOC)
      input_root="/home/sprice/PromptMoE/DATASETS/VOC/images"
      pred_roots=(
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/DeepLabV3"
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/FCN"
        "/home/sprice/PromptMoE/DATASETS/VOC/outputs/LR-ASPP"
      )
      ;;
    MSRA-B)
      input_root="/home/sprice/PromptMoE/DATASETS/MSRA-B/images"
      pred_roots=(
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

  local out_dir="${OUT_ROOT}/${dataset}/${sam_model}"
  mkdir -p "$out_dir"

  echo ""
  echo "========== DATASET: ${dataset} | BACKBONE: ${sam_model} =========="
  echo "Output dir: $out_dir"

  # Build the pred_roots arguments
  CMD_PRED_ROOTS=( )
  for pr in "${pred_roots[@]}"; do
    CMD_PRED_ROOTS+=( "$pr" )
  done

  set -x
  python RunPromptMoE.py \
    ${MARIGOLD} \
    --samrefiner-path "${SAMREFINER_PATH}" \
    --checkpoint "${ckpt}" --sam_model "${sam_model}" \
    --input_root "${input_root}" \
    --pred_roots "${CMD_PRED_ROOTS[@]}" \
    --output_dir "${out_dir}" \
    --router_ckpt "${ROUTER_CKPT}" \
    --k_points ${K_POINTS} \
    --suppression_frac ${SUPPRESSION_FRAC} \
    --iters ${ITERS} \
    --beta ${BETA} \
    --router_topk ${ROUTER_TOPK}
  set +x
}

# ---------- MAIN LOOP ----------
DATASETS=("BIG")

# vit_h
for ds in "${DATASETS[@]}"; do
  run_job "$ds" "vit_h" "${CKPT_VIT_H}"
done

# vit_l
for ds in "${DATASETS[@]}"; do
  run_job "$ds" "vit_l" "${CKPT_VIT_L}"
done

# vit_b
for ds in "${DATASETS[@]}"; do
  run_job "$ds" "vit_b" "${CKPT_VIT_B}"
done





echo "All runs completed."
