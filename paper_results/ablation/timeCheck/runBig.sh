#!/bin/bash
#SBATCH --job-name=TIMING_BIG
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=short
#SBATCH --output=slurm-%x-%j.out

# ---- simple timing/logging helper (no Python edits needed) ----
LOGDIR=/home/sprice/PromptMoE/paper_results/ablation/timeCheck
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

  # run the real command
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


# Optional: overall job start/end markers
{
  echo "JOB_START $(date -u '+%Y-%m-%dT%H:%M:%SZ')  job_id=${SLURM_JOB_ID:-N/A} host=$(hostname)"
} >> "$LOGFILE"
trap 'echo "JOB_END   $(date -u "+%Y-%m-%dT%H:%M:%SZ")  job_id=${SLURM_JOB_ID:-N/A}" >> "'"$LOGFILE"'"' EXIT
# ---------------------------------------------------------------

set +u
eval "$(conda shell.bash hook)"
conda activate PromptMoE
set -u
cd /home/sprice/PromptMoE

run_and_log "BIG" python RunPromptMoE.py \
    --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
    --samrefiner-path /home/sprice/PromptMoE/PromptMoE.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/PromptMoE-Lite \
    --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2 \
    --light_mode

run_and_log "BIG" python RunPromptMoE.py \
    --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
    --samrefiner-path /home/sprice/PromptMoE/PromptMoE.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/PromptMoE \
    --router_ckpt /home/sprice/PromptMoE/pairRouter_10_13.pt \
    --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2

run_and_log "BIG" python RunPromptMoE.py \
    --marigold_checkpoint prs-eth/marigold-depth-v1-1 --marigold_half \
    --samrefiner-path /home/sprice/PromptMoE/PromptMoE.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth --sam_model vit_h \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/PromptMoE-All \
    --expert_weights "Dp=1,Gd=1,Gc=1,sp=1,ctr=1,bri=1,tex=1,frac=1,Sm=1,prompt=1" \
    --k_points 5 --suppression_frac 0.05 --iters 10 --beta 1.4 --router_topk 2


set +u
eval "$(conda shell.bash hook)"
conda activate SAMRefiner
set -u
cd /home/sprice/PromptMoE/paper_results/sota_methods

run_and_log "BIG" python samRefiner.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/samRefiner

set +u
eval "$(conda shell.bash hook)"
conda activate CVPR_26
set -u
cd /home/sprice/PromptMoE/paper_results/sota_methods

run_and_log "BIG" python DualSight.py \
    --checkpoint /home/sprice/SAMRefiner/vit_h.pth \
    --model_type vit_h \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/DualSight


set +u
eval "$(conda shell.bash hook)"
conda activate segrefiner
set -u
cd /home/sprice/PromptMoE/paper_results/sota_methods

run_and_log "BIG" python segrefiner.py \
    --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
    --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
    --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/segrefiner_lr \
    --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
    --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_lr_latest.pth


run_and_log "BIG" python segrefiner.py \
      --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
      --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/segrefiner_hr \
      --config /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/refine_mask_config.py \
      --checkpoint /home/sprice/CVPR26_Dataset/SOTA/SegRefiner/segrefiner_hr_latest.pth

set +u
eval "$(conda shell.bash hook)"
conda activate segrefiner
set -u
cd /home/sprice/PromptMoE/paper_results/sota_methods


run_and_log "BIG" python cascadePSP.py \
      --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
      --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/cascadePSP_fast \
      --fast True

run_and_log "BIG" python cascadePSP.py \
      --input_root /home/sprice/PromptMoE/DATASETS/BIG/images \
      --pred_roots \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/DeepLabV3 \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/FCN \
        /home/sprice/PromptMoE/DATASETS/BIG/outputs/LR-ASPP \
      --output_dir /home/sprice/PromptMoE/paper_results/ablation/timeCheck/BIG/cascadePSP_slow \
      --fast False