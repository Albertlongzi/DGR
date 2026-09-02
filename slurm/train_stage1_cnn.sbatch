#!/bin/bash
#SBATCH --job-name=dgr_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# Stage 1: CNN front-end. Settings match configs/stage1_cnn.yaml.
set -euo pipefail

DGR_ROOT=${DGR_ROOT:?set DGR_ROOT to the repository root}
DGR_DATA=${DGR_DATA:?set DGR_DATA to the data root}
CONDA_ENV=${CONDA_ENV:-dgr}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${DGR_ROOT}:${PYTHONPATH:-}"
cd "${DGR_ROOT}"

NPZ_ROOT=${NPZ_ROOT:-"${DGR_DATA}/dwi_pair/pe_axis0"}
NPZ_ROOT2=${NPZ_ROOT2:-"${DGR_DATA}/dwi_pair/pe_axis1"}
OUT_DIR=${OUT_DIR:-"${DGR_ROOT}/runs/stage1_cnn"}

NPROC=${NPROC:-6}
MASTER_PORT=${MASTER_PORT:-29500}
RADIUS=${RADIUS:-2}
BATCH=${BATCH:-6}
EPOCHS=${EPOCHS:-25}
LR=${LR:-3e-4}
VAL_SPLIT=${VAL_SPLIT:-0.1}
WARMUP_STEPS=${WARMUP_STEPS:-3054}
BASE_CHANNELS=${BASE_CHANNELS:-64}
LATENT_DIM=${LATENT_DIM:-8}
PROMPT_K=${PROMPT_K:-8}
PROMPT_TEMP=${PROMPT_TEMP:-1.0}
SSIM_WEIGHT=${SSIM_WEIGHT:-0.25}
MS_W1=${MS_W1:-0.2}
MS_W2=${MS_W2:-0.05}

mkdir -p "${OUT_DIR}"

torchrun --nproc-per-node=${NPROC} --master_port ${MASTER_PORT} \
  scripts/restoration/train_stage1_cnn.py \
  --npz_root "${NPZ_ROOT}" \
  --npz_root2 "${NPZ_ROOT2}" \
  --out_dir "${OUT_DIR}" \
  --radius ${RADIUS} \
  --batch_size ${BATCH} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --val_split ${VAL_SPLIT} \
  --warmup_steps ${WARMUP_STEPS} \
  --base_channels ${BASE_CHANNELS} \
  --latent_dim ${LATENT_DIM} \
  --prompt_k ${PROMPT_K} \
  --prompt_temp ${PROMPT_TEMP} \
  --use_ssim_loss --ssim_weight ${SSIM_WEIGHT} \
  --ms_w1 ${MS_W1} --ms_w2 ${MS_W2} \
  --autoscale_lr 1 --autoscale_mode sqrt --autoscale_warmup 1
