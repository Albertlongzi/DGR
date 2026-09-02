#!/bin/bash
#SBATCH --job-name=dgr_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:6
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

# Full DGR inference: stage-1 CNN correction followed by diffusion refinement.
# Settings match configs/inference.yaml.
set -euo pipefail

DGR_ROOT=${DGR_ROOT:?set DGR_ROOT to the repository root}
CONDA_ENV=${CONDA_ENV:-dgr}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${DGR_ROOT}:${PYTHONPATH:-}"
cd "${DGR_ROOT}"

CKPT=${CKPT:?set CKPT to the stage-2 diffusion checkpoint}
CNN_CKPT=${CNN_CKPT:?set CNN_CKPT to the stage-1 CNN checkpoint}
TEST_ROOT=${TEST_ROOT:?set TEST_ROOT to the preprocessed test NPZ root}
OUT_DIR=${OUT_DIR:-"${DGR_ROOT}/outputs/dgr_infer"}

NUM_GPUS=${NUM_GPUS:-6}
STEPS=${STEPS:-100}
STRENGTH=${STRENGTH:-0.3}
ETA=${ETA:-0.0}
SAMPLER=${SAMPLER:-dpmsolver}
RADIUS=${RADIUS:-2}
SLICE_MODE=${SLICE_MODE:-all}
B_LOW=${B_LOW:-50}
B_HIGH=${B_HIGH:-1400}

mkdir -p "${OUT_DIR}"

python -u scripts/restoration/infer_dgr.py \
  --ckpt "${CKPT}" \
  --cnn_ckpt "${CNN_CKPT}" \
  --test_root ${TEST_ROOT} \
  --out_dir "${OUT_DIR}" \
  --radius ${RADIUS} \
  --steps ${STEPS} \
  --strength ${STRENGTH} \
  --eta ${ETA} \
  --sampler ${SAMPLER} \
  --slice_mode ${SLICE_MODE} \
  --b_low ${B_LOW} --b_high ${B_HIGH} \
  --t2_cond_channels 64 --t2_contrast_mod none \
  --cnn_base_channels 64 --cnn_latent_dim 8 --cnn_prompt_k 8 --cnn_prompt_temp 1.0 \
  --gamma_b14 0.4 --gamma_adc 0.7 --adc_plo 2.0 --adc_phi 99.0 \
  --num_gpus ${NUM_GPUS} \
  --save_npz --save_slices
