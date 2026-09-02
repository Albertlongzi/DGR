# Let Distortion Guide Restoration (DGR)

<div align="center">

[![Paper](https://img.shields.io/badge/Radiology%20Advances-10.1093%2Fradadv%2Fumag031-1a7f37.svg)](https://doi.org/10.1093/radadv/umag031)
[![arXiv](https://img.shields.io/badge/arXiv-2601.00226-b31b1b.svg)](https://arxiv.org/abs/2601.00226)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-gated-yellow.svg)](https://huggingface.co/Zylong/DGR)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

**Physics-Informed Deep Learning for Geometric Distortion Correction in Prostate DWI**

</div>

> **Distortion-guided restoration: a physics-informed learning framework to correct prostate diffusion MRI artifacts**
> Ziyang Long, Nader Binesh, Lixia Wang, Archana Vadiraj Malaji, Chia-Chi Yang, Haoran Sun,
> Rola Saouaf, Timothy Daskivich, Hyung Kim, Yibin Xie, Debiao Li, Hsin-Jung Yang.
> *Radiology Advances* 3(4), 2026. [doi:10.1093/radadv/umag031](https://doi.org/10.1093/radadv/umag031)

---

## Overview

Prostate multiparametric MRI (mpMRI) is the clinical gold standard for prostate cancer detection and PI-RADS grading. However, the diffusion-weighted imaging (DWI) component — acquired using single-shot echo-planar imaging (ssEPI) — is highly vulnerable to B0 field inhomogeneities, causing severe geometric distortions in the form of spatial warping, pixel pile-up, and signal dropout. These artifacts are dramatically worse in patients with **hip prostheses** or **bowel distension**, precisely the demographic most at risk for prostate cancer.

**DGR** addresses this without any additional scan acquisition. By learning to invert a physics-based forward distortion simulator, DGR corrects severe geometric distortions using only routinely acquired DWI and T2-weighted images.

<div align="center">

```
Distorted ssEPI DWI  ──────────────────────────────►  Corrected DWI
(geometric warping,                                   (anatomically
 pixel pile-up,                                        faithful,
 signal dropout)                                       diagnostic quality)
```

</div>

This repository ships **both halves** of the method — you can regenerate the training data, not
just run the network:

| Half | What it does | Where |
|---|---|---|
| **Forward — simulation** | Takes a measured ΔB0 field, fits it, perturbs it into a family of physically plausible variants, converts each to a voxel displacement map from the EPI readout geometry, and warps *undistorted* DWI into *distorted* DWI. This manufactures the paired training data that does not exist clinically. | `dgr/physics/`, `dgr/utils/`, `scripts/simulation/` |
| **Reverse — restoration** | Inverts that forward model: a CNN front-end produces the geometric correction, then a conditional diffusion module refines it on the anatomical manifold under T2w guidance. | `dgr/models/`, `dgr/inference/`, `scripts/restoration/` |

---

## Key Features

- **No extra acquisitions required** — works with the standard clinical DWI + T2W protocol; no B0 field maps, no reverse phase-encoded scans
- **Physics-informed training** — forward ssEPI distortion simulator driven by real B0 field maps from hip-prosthesis patients, augmented via 12th-order field perturbation (>40,000 paired training samples)
- **Hybrid CNN–Diffusion architecture** — two-stage pipeline: coarse geometric correction via CNN, fine texture restoration via conditional diffusion refinement (SDEdit-style)
- **T2W anatomical conditioning** — uses the distortion-free T2W scan as an anatomical reference via deformable cross-attention
- **Clinically validated** — prospective cohort of 34 subjects with severe baseline distortion; blinded radiologist scoring shows significant improvement in geometric fidelity, image quality, and diagnostic confidence

---

## Method

```
                        ── FORWARD (simulation) ──
  DICOM / .mat  ─► b0_field_read ─► b0_registration ─► ΔB0 in T2 space
                                                          │
                        generate_b0_variants_{poly,sh}  ◄──┘
                                     │  (12th-order coefficient perturbation)
                                     ▼
                            B0 variant fields
                                     │
  undistorted DWI (b=50, b=1400) ────┼──► generate_dwi_pairs ──► paired (distorted, GT) NPZ
                                     │      (VDM → EPI forward splat)
                        ── REVERSE (restoration) ──
                                     ▼
        stage 1:  train_stage1_cnn      distorted DWI ──► geometry-corrected DWI
                                     ▼
        stage 2:  train_stage2_diffusion   + T2w guidance ──► refined DWI
                                     ▼
                  infer_dgr  ──►  evaluate_distortion_correction
```

### Forward simulator

`dgr/utils/warp.py :: compute_vdm_from_b0_2d_ESP` turns a ΔB0 field in Hz into a voxel
displacement map given the echo spacing, phase-encode line count, partial-Fourier factor and
in-plane acceleration. `dgr/utils/epi_warp.py :: forward_splat_with_fallback` then applies it as a
**conservative forward splat** along the PE axis, so signal pile-up and stretching are both
modelled rather than approximated by an interpolating pull-warp.

### Stage 1 — CNN Restoration Backbone

A 2.5D multi-scale encoder–decoder with residual blocks processes the distorted DWI alongside
co-registered T2W. A **contrast-aware deformable cross-attention** module treats distorted DWI as
queries and T2W as keys/values, computing adaptive spatial offsets to bridge the geometric
mismatch between the two modalities. Feature Pyramid Network (FPN)-style top-down aggregation
recovers fine spatial details.

### Stage 2 — Conditional Diffusion Refinement

A conditional diffusion UNet refines the coarse CNN output, conditioned on the T2W image and on
the frozen stage-1 result. It is trained with `prediction_type="sample"` — the network predicts
the clean image rather than the noise, so learning happens on the anatomical manifold. Inference
uses **SDEdit-style img2img initialization** with DPM-Solver for fast sampling.

```
 Distorted low-b DWI ─┐
 ADC map              ─┼─► [Stage 1: CNN Backbone] ──► Coarse-corrected DWI + ADC
 Co-reg. T2W          ─┘         (deformable                      │
                                  cross-attn)                      │
                                                                   ▼
                                                    [Stage 2: Diffusion Refinement]
                                                     (T2W + CNN output conditioning,
                                                      SDEdit img2img, DPM-Solver)
                                                                   │
                                                                   ▼
                                                     Final corrected DWI + ADC
                                                                   │
                                                                   ▼
                                                     High-b DWI (derived from ADC)
```

---

## Install

```bash
git clone https://github.com/Albertlongzi/DGR
cd DGR
conda create -n dgr python=3.8 -y && conda activate dgr
pip install -r requirements.txt
pip install -e .
```

The published results were produced with Python 3.8.20, PyTorch 2.4.1+cu121, diffusers 0.35.1,
NumPy 1.24.3. `monai` and `torchmetrics` are optional; the code falls back cleanly without them.

## Pretrained weights

Weights live on Hugging Face under a **gated** repository — access is reviewed and approved
manually, and the request form asks what you intend to use them for:

**https://huggingface.co/Zylong/DGR**

| Stage | File | Params | Size |
|---|---|---|---|
| 1 — CNN | `stage1_cnn/stage1_cnn.safetensors` | 32.1 M | 128 MB |
| 2 — diffusion | `stage2_diffusion/stage2_diffusion.safetensors` | 299.6 M | 1.20 GB |

```bash
hf auth login   # required: the repository is gated
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("Zylong/DGR", "stage1_cnn/stage1_cnn.safetensors")
hf_hub_download("Zylong/DGR", "stage2_diffusion/stage2_diffusion.safetensors")
PY
```

Each weight file ships with a `config.json` recording the architecture hyperparameters and the
noise-scheduler settings. `scripts/restoration/infer_dgr.py` accepts either a `.safetensors` or a
raw training `.pt`; given the former it reads the sibling `config.json`, which is where
`prediction_type="sample"` comes from — the diffusers default of `"epsilon"` would silently return
noise. To produce your own release files from a training checkpoint:

```bash
python tools/export_checkpoint.py --ckpt runs/stage2/diff_epoch_092.pt \
  --out_dir hf_export/stage2 --kind diffusion --name stage2_diffusion
```

---

## Usage

### 1. Forward simulation

```bash
# a) fit + perturb the measured B0 field into physically plausible variants
python scripts/simulation/generate_b0_variants_poly.py --help   # 2-D polynomial perturbation
python scripts/simulation/generate_b0_variants_sh.py   --help   # spherical-harmonic, orders >= 3 only

# b) warp undistorted DWI through the forward EPI model to build training pairs
python scripts/simulation/generate_dwi_pairs.py \
  --input_roots  /path/to/preprocessed_local /path/to/preprocessed_fastmri /path/to/preprocessed_disease \
  --b0_root      /path/to/B0_variants_poly/order_12 \
  --output_root  /path/to/dwi_pair \
  --max_b0_subjects_per_dwi 11 --smooth_sigma 1.5 --seed 123 --num_workers 6

# c) held-out test set, built the same way from a disjoint B0 pool
python scripts/simulation/generate_dwi_testset.py --help
```

The SH generator perturbs only orders ≥ 3, leaving orders 0–2 intact so the low-order
(shim-correctable) component stays physically consistent. The released training pairs were built
with the **12th-order polynomial** basis.

### 2. Stage 1 — CNN

```bash
torchrun --nproc-per-node=6 scripts/restoration/train_stage1_cnn.py \
  --npz_root /path/to/dwi_pair/pe_axis0 \
  --npz_root2 /path/to/dwi_pair/pe_axis1 \
  --out_dir  runs/stage1_cnn \
  --radius 2 --batch_size 6 --epochs 25 --lr 3e-4 --warmup_steps 3054 \
  --base_channels 64 --latent_dim 8 --prompt_k 8 --prompt_temp 1.0 \
  --use_ssim_loss --ssim_weight 0.25 --ms_w1 0.2 --ms_w2 0.05
```

### 3. Stage 2 — conditional diffusion

```bash
torchrun --nproc-per-node=4 scripts/restoration/train_stage2_diffusion.py \
  --npz_root /path/to/dwi_pair/pe_axis0 \
  --out_dir  runs/stage2_diffusion \
  --cnn_ckpt runs/stage1_cnn/mageultra_best.pt \
  --batch_size 6 --epochs 100 --lr 1e-4 --warmup_steps 1000 \
  --radius 2 --t2_cond_channels 64 --t2_contrast_mod none \
  --val_interval 2 --val_steps 50 --val_strength 0.3 --use_dpm_solver_validation
```

### 4. Inference

```bash
python scripts/restoration/infer_dgr.py \
  --cnn_ckpt  stage1_cnn/stage1_cnn.safetensors \
  --ckpt      stage2_diffusion/stage2_diffusion.safetensors \
  --test_root /path/to/preprocessed_test_npz \
  --out_dir   outputs/dgr_infer \
  --steps 100 --strength 0.3 --eta 0.0 --sampler dpmsolver \
  --radius 2 --t2_cond_channels 64 --b_low 50 --b_high 1400 \
  --slice_mode all --save_npz --save_slices
```

`--strength` is the SDEdit refinement strength. **0.3 is the value behind the released
checkpoints** — it refines. Raising it lets the diffusion prior invent structure, so do not
increase it without checking outputs against a reference.

### 5. Evaluation

```bash
python scripts/evaluation/evaluate_distortion_correction.py --help
```

Reports PSNR / SSIM / NMSE / MAE (whole-FOV and prostate-centred) plus wall-clock timing, against
FUGUE and TOPUP baselines.

SLURM job templates for all four steps are in `slurm/`; they take `DGR_ROOT` and `DGR_DATA` from
the environment and bake in no absolute paths. `configs/*.yaml` record the exact settings behind
the released checkpoints.

---

## Results

### Quantitative (Synthetic Benchmark, n=34)

| Method | low-b PSNR ↑ | low-b NMSE ↓ | ADC PSNR ↑ | ADC NMSE ↓ |
|:---|:---:|:---:|:---:|:---:|
| No correction | baseline | 0.364 | baseline | — |
| FUGUE (oracle field map) | — | — | — | — |
| TOPUP (oracle field map) | — | — | — | — |
| **DGR (ours)** | **23.88 ± 2.93 dB** | **0.089 ± 0.049** | **22.99 ± 1.97 dB** | **0.062 ± 0.028** |

DGR significantly outperforms FUGUE and TOPUP even when those baselines are given oracle (ground-truth) B0 field maps (paired Wilcoxon, p < 0.001).

### Clinical Study (Prospective Cohort, n=34, 5-point Likert scale)

| Criterion | Original ssEPI | DGR | p-value |
|:---|:---:|:---:|:---:|
| Geometric fidelity | 2.6 | **3.3** | < 0.001 |
| Overall image quality | 2.5 | **2.9** | < 0.001 |
| Diagnostic confidence | 2.5 | **3.0** | < 0.001 |

- Zero false negatives and zero false positives in lesion analysis (n=18 with histopathology)
- Inference time: **13–15 seconds** per subject on NVIDIA H100

---

## Repository Structure

```
DGR/
├── dgr/                   # library
│   ├── physics/           #   B0 field I/O, DICOM handling, SH & polynomial fitting, B0→T2 registration
│   ├── utils/             #   forward EPI model: VDM computation, splat / warp / resample kernels
│   ├── models/            #   phc_net → phc_e2e_mega_net → phc_e2e_mageultra_net (stage 1)
│   │                      #   diffusion_unet_diffusers (stage 2)
│   ├── data/              #   paired dual-b NPZ dataset with 2.5-D slice stacking
│   ├── inference/         #   DDIM / DDPM / DPM-Solver samplers with T2 + CNN conditioning
│   ├── losses/            #   SSIM, ADC consistency, relative intensity, TV / Jacobian penalties
│   └── conditioning/      #   T2W conditioning channel construction
├── scripts/
│   ├── simulation/        # forward-model entry points
│   ├── restoration/       # stage-1 / stage-2 training and DGR inference
│   └── evaluation/        # quantitative comparison against FUGUE / TOPUP
├── configs/               # YAML records of the settings behind the released checkpoints
├── slurm/                 # portable SLURM job templates
└── tools/                 # checkpoint export for Hugging Face
```

---

## Data

This work uses two datasets:

| Dataset | Subjects | Usage |
|:---|:---:|:---|
| [fastMRI Prostate](https://fastmri.med.nyu.edu/) | 314 exams | Training / Test |
| In-house (Cedars-Sinai Medical Center) | 130 exams | Training / Test |

B0 field maps were acquired from 11 patients with hip prostheses and augmented to 110 maps via 12th-order perturbation, driving the forward distortion simulator.

The clinical source data cannot be redistributed. The simulation half of this repository lets the
training pairs be regenerated from any DWI + T2w + ΔB0 source, including the public fastMRI
Prostate dataset. `configs/simulation.yaml` documents the expected NPZ keys.

---

## Citation

```bibtex
@article{long2026dgr,
  title   = {Distortion-guided restoration: a physics-informed learning framework
             to correct prostate diffusion MRI artifacts},
  author  = {Long, Ziyang and Binesh, Nader and Wang, Lixia and Malaji, Archana Vadiraj
             and Yang, Chia-Chi and Sun, Haoran and Saouaf, Rola and Daskivich, Timothy
             and Kim, Hyung and Xie, Yibin and Li, Debiao and Yang, Hsin-Jung},
  journal = {Radiology Advances},
  volume  = {3},
  number  = {4},
  year    = {2026},
  doi     = {10.1093/radadv/umag031}
}
```

---

## License

Code is released under the [MIT license](LICENSE). The pretrained weights are distributed
separately under a research-only license — see the
[Hugging Face repository](https://huggingface.co/Zylong/DGR).

**Not a medical device.** Not cleared or approved for clinical use, diagnosis, or treatment
planning by any regulatory body.

---

## Acknowledgements

This work was supported by NIH grants R01NS121544, R01HL156818, R01HL165211, R01HL181091, and R43NS120795. We thank the Research Imaging Core (RIC) at Cedars-Sinai Medical Center, MRI Technologist Mike Ngo, Irene Lee, and nurses Catherine Ubaldo-Prado and Lee Hyae for their support in data acquisition.
