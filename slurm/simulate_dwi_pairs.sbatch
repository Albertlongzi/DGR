#!/bin/bash
#SBATCH --job-name=dgr_simulate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/simulate_%j.out
#SBATCH --error=logs/simulate_%j.err

# Forward distortion simulation: warp undistorted DWI through B0 variant fields
# to build the paired training set. CPU-bound (numpy + file I/O); no GPU required.
set -euo pipefail

DGR_ROOT=${DGR_ROOT:?set DGR_ROOT to the repository root}
DGR_DATA=${DGR_DATA:?set DGR_DATA to the data root}
CONDA_ENV=${CONDA_ENV:-dgr}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${DGR_ROOT}:${PYTHONPATH:-}"
cd "${DGR_ROOT}"

B0_ROOT=${B0_ROOT:-"${DGR_DATA}/B0_variants_poly/order_12"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${DGR_DATA}/dwi_pair"}
INPUT_ROOTS=${INPUT_ROOTS:-"${DGR_DATA}/preprocessed_local ${DGR_DATA}/preprocessed_fastmri ${DGR_DATA}/preprocessed_disease"}

MAX_B0_SUBJECTS_PER_DWI=${MAX_B0_SUBJECTS_PER_DWI:-11}
VARIANTS_LOCAL=${VARIANTS_LOCAL:-0}
VARIANTS_FASTMRI=${VARIANTS_FASTMRI:-1}
VARIANTS_DIEASE=${VARIANTS_DIEASE:-1}
SMOOTH_SIGMA=${SMOOTH_SIGMA:-1.5}
SEED=${SEED:-123}
NUM_WORKERS=${NUM_WORKERS:-6}

python -u scripts/simulation/generate_dwi_pairs.py \
  --input_roots ${INPUT_ROOTS} \
  --b0_root "${B0_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --max_b0_subjects_per_dwi ${MAX_B0_SUBJECTS_PER_DWI} \
  --variants_per_b0_subject ${VARIANTS_LOCAL} \
  --variants_per_b0_subject_fastmri ${VARIANTS_FASTMRI} \
  --variants_per_b0_subject_diease ${VARIANTS_DIEASE} \
  --smooth_sigma ${SMOOTH_SIGMA} \
  --seed ${SEED} \
  --num_workers ${NUM_WORKERS} \
  --overwrite
