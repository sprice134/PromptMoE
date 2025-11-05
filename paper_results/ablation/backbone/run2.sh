#!/bin/bash
#SBATCH --job-name=BACKBONE
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L40S:1
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

# ---- simple timing/logging helper (no Python edits needed) ----
LOGDIR=/home/sprice/PromptMoE/paper_results/ablation/backbone
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/timing-${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.txt"

run_and_log () {
  local label="$1"; shift          # human-friendly tag for the run
  local started_iso ended_iso
  local started_s ended_s duration exit_code
  local -a cmd=( "$@" )            # keep the full command intact

  started_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  started_s="$(date +%s)"

  {
    printf '=== START ===\n'
    printf 'label: %s\n' "$label"
    printf 'job_id: %s\n' "${SLURM_JOB_ID:-N/A}"
    printf 'host: %s\n' "$(hostname)"
    printf 'start_utc: %s\n' "$started_iso"
    printf 'cmd:'
    for arg in "${cmd[@]}"; do printf ' %q' "$arg"; done
    printf '\n'
  } >> "$LOGFILE"

  "${cmd[@]}"
  exit_code=$?

  ended_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  ended_s="$(date +%s)"
  duration=$(( ended_s - started_s ))

  {
    printf 'end_utc: %s\n' "$ended_iso"
    printf 'duration_s: %d\n' "$duration"
    printf 'exit_code: %d\n' "$exit_code"
    printf '=== END ===\n\n'
  } >> "$LOGFILE"

  return $exit_code
}

{
  echo "=== NODE INFO ==="
  echo "timestamp_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "job_id: ${SLURM_JOB_ID:-N/A}"
  echo "host: $(hostname)"
  echo "uname: $(uname -a)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo
  echo "--- GPU: nvidia-smi -L ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
    echo
    echo "--- GPU Topology: nvidia-smi topo -m ---"
    nvidia-smi topo -m || true
    echo
    echo "--- GPU Verbose: nvidia-smi -q ---"
    nvidia-smi -q || true
  else
    echo "nvidia-smi not found on PATH."
  fi
  echo
  echo "--- PCI (NVIDIA): lspci | grep -i nvidia ---"
  lspci | grep -i nvidia || true
  echo
  echo "--- CPU: lscpu ---"
  lscpu || true
  echo
  echo "--- Memory: free -h ---"
  free -h || true
  echo
} >> "$LOGFILE" 2>&1

{
  echo "JOB_START $(date -u '+%Y-%m-%dT%H:%M:%SZ')  job_id=${SLURM_JOB_ID:-N/A} host=$(hostname)"
} >> "$LOGFILE"
trap 'echo "JOB_END   $(date -u "+%Y-%m-%dT%H:%M:%SZ")  job_id=${SLURM_JOB_ID:-N/A}" >> "'"$LOGFILE"'"' EXIT
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
  run_and_log "dataset=${dataset} backbone=${sam_model}" \
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
DATASETS=("VOC")

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
