#!/bin/bash
#SBATCH --job-name=dgr_stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=128:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

# Stage 2: conditional diffusion refinement, prediction_type="sample" (clean-image
# prediction rather than noise prediction). Settings match configs/stage2_diffusion.yaml.
set -euo pipefail

DGR_ROOT=${DGR_ROOT:?set DGR_ROOT to the repository root}
DGR_DATA=${DGR_DATA:?set DGR_DATA to the data root}
CONDA_ENV=${CONDA_ENV:-dgr}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${DGR_ROOT}:${PYTHONPATH:-}"
cd "${DGR_ROOT}"

NPZ_ROOT=${NPZ_ROOT:-"${DGR_DATA}/dwi_pair/pe_axis0"}
OUT_DIR=${OUT_DIR:-"${DGR_ROOT}/runs/stage2_diffusion"}
CNN_CKPT=${CNN_CKPT:?set CNN_CKPT to the frozen stage-1 checkpoint}

NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29502}
RADIUS=${RADIUS:-2}
BATCH=${BATCH:-6}
EPOCHS=${EPOCHS:-100}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
VAL_SPLIT=${VAL_SPLIT:-0.1}
VAL_INTERVAL=${VAL_INTERVAL:-2}
VAL_STEPS=${VAL_STEPS:-50}
VAL_STRENGTH=${VAL_STRENGTH:-0.3}
T2_COND_CHANNELS=${T2_COND_CHANNELS:-64}
T2_CONTRAST_MOD=${T2_CONTRAST_MOD:-none}
NUM_WORKERS=${NUM_WORKERS:-3}

mkdir -p "${OUT_DIR}"

torchrun --nproc-per-node=${NPROC} --master_port ${MASTER_PORT} \
  scripts/restoration/train_stage2_diffusion.py \
  --npz_root "${NPZ_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --cnn_ckpt "${CNN_CKPT}" \
  --batch_size ${BATCH} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --warmup_steps ${WARMUP_STEPS} \
  --radius ${RADIUS} \
  --val_split ${VAL_SPLIT} \
  --val_interval ${VAL_INTERVAL} \
  --val_steps ${VAL_STEPS} \
  --val_strength ${VAL_STRENGTH} \
  --use_dpm_solver_validation \
  --val_vis --val_vis_n 6 \
  --t2_cond_channels ${T2_COND_CHANNELS} \
  --t2_contrast_mod ${T2_CONTRAST_MOD} \
  --cnn_base_channels 64 --cnn_latent_dim 8 --cnn_prompt_k 8 --cnn_prompt_temp 1.0 \
  --cnn_memory_cache_size 10000 \
  --num_workers ${NUM_WORKERS} --num_workers_val 1 \
  --prefetch_factor 3 --prefetch_factor_val 2 \
  --gradient_accumulation_steps 1 --max_grad_norm 1.0
