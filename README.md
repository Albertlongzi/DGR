# DGR — Distortion-Guided Restoration for Prostate Diffusion MRI

Reference implementation of

> **Distortion-guided restoration: a physics-informed learning framework to correct prostate diffusion MRI artifacts**
> Ziyang Long, Nader Binesh, Lixia Wang, Archana Vadiraj Malaji, Chia-Chi Yang, Haoran Sun,
> Rola Saouaf, Timothy Daskivich, Hyung Kim, Yibin Xie, Debiao Li, Hsin-Jung Yang.
> *Radiology Advances* 3(4), 2026. [doi:10.1093/radadv/umag031](https://doi.org/10.1093/radadv/umag031)

DGR corrects susceptibility-induced geometric distortion in single-shot EPI prostate DWI
**without acquiring a field map or a reverse phase-encode scan**. It has two halves, and this
repository ships both:

| Half | What it does | Where |
|---|---|---|
| **Forward — simulation** | A physics forward model that takes a measured ΔB0 field, resamples it into a large family of physically plausible variants, and warps *undistorted* DWI into *distorted* DWI. This manufactures the paired training data that does not exist clinically. | `dgr/physics/`, `scripts/simulation/` |
| **Reverse — restoration** | A two-stage network that inverts that forward model: a CNN front-end produces the geometric correction, then a conditional diffusion module refines it on the anatomical manifold using the co-registered T2w image as guidance. | `dgr/models/`, `dgr/inference/`, `scripts/restoration/` |

---

## Install

```bash
conda create -n dgr python=3.8 -y
conda activate dgr
pip install -r requirements.txt
pip install -e .
```

The published results were produced with Python 3.8.20, PyTorch 2.4.1+cu121, diffusers 0.35.1,
NumPy 1.24.3. `monai` and `torchmetrics` are optional; the code falls back cleanly without them.

## Pretrained weights

Weights are hosted on Hugging Face under a **gated** repository — access is granted on request.

| Stage | File | Notes |
|---|---|---|
| Stage 1 — CNN | `stage1_cnn.safetensors` | MageUltra dual-b network, PE axis 0/1 |
| Stage 2 — diffusion | `stage2_diffusion.safetensors` | clean-image (`prediction_type="sample"`) UNet, T2 + CNN conditioned |

Training checkpoints carry optimizer state; use `tools/export_checkpoint.py` to strip it and emit
the inference-only `.safetensors` plus a `config.json` recording the architecture hyperparameters.
`scripts/restoration/infer_dgr.py` accepts either form — given a `.safetensors` it reads the
sibling `config.json` for the scheduler settings, which is where `prediction_type="sample"` comes
from (the diffusers default of `"epsilon"` would silently return noise).

```bash
python tools/export_checkpoint.py --ckpt /path/to/train_ckpt.pt --out_dir hf_export/ --kind diffusion
python tools/export_checkpoint.py --ckpt /path/to/train_ckpt.pt --out_dir hf_export/ --kind cnn
```

---

## Pipeline

```
                        ── FORWARD (simulation) ──
  DICOM / .mat  ─► b0_field_read ─► b0_registration ─► ΔB0 in T2 space
                                                          │
                        generate_b0_variants_{sh,poly}  ◄──┘
                                     │  (SH / polynomial coefficient perturbation)
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

### 1. Forward simulation

```bash
# a) fit + resample the measured B0 field into physically plausible variants
python scripts/simulation/generate_b0_variants_sh.py   --help   # spherical-harmonic perturbation (order >= 3)
python scripts/simulation/generate_b0_variants_poly.py --help   # 2-D polynomial perturbation

# b) warp undistorted DWI through the forward EPI model to build training pairs
python scripts/simulation/generate_dwi_pairs.py \
  --input_roots  /path/to/preprocessed_local /path/to/preprocessed_fastmri /path/to/preprocessed_disease \
  --b0_root      /path/to/B0_variants_poly/order_12 \
  --output_root  /path/to/dwi_pair \
  --max_b0_subjects_per_dwi 11 --smooth_sigma 1.5 --seed 123 --num_workers 6

# c) held-out test set built the same way, with a disjoint B0 pool
python scripts/simulation/generate_dwi_testset.py --help
```

The forward model itself lives in `dgr/utils/warp.py`
(`compute_vdm_from_b0_2d_ESP`: ΔB0 [Hz] → voxel displacement map, given echo spacing, PE
lines, partial Fourier and acceleration) and `dgr/utils/epi_warp.py`
(`forward_splat_with_fallback`: conservative forward splat along the PE axis, so signal pile-up
and stretching are both modelled rather than approximated by an interpolating pull-warp).

### 2. Stage 1 — CNN

```bash
torchrun --nproc-per-node=6 scripts/restoration/train_stage1_cnn.py \
  --npz_root /path/to/dwi_pair/pe_axis0 \
  --out_dir  runs/stage1_cnn \
  --radius 2 --batch_size 6 --epochs 25 --lr 1e-4 \
  --base_channels 64 --latent_dim 8 --prompt_k 8 --prompt_temp 1.0 \
  --b_low 50 --b_high 1400
```

### 3. Stage 2 — conditional diffusion

Trained with `prediction_type="sample"` (the network predicts the clean image, not the noise),
conditioned on the T2w image and on the frozen stage-1 CNN output.

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
  --ckpt      /path/to/stage2_diffusion.pt \
  --cnn_ckpt  /path/to/stage1_cnn.pt \
  --test_root /path/to/test_npz \
  --out_dir   outputs/dgr_infer \
  --steps 100 --strength 0.3 --eta 0.0 --sampler dpmsolver \
  --radius 2 --t2_cond_channels 64 --b_low 50 --b_high 1400 \
  --slice_mode all --save_npz --save_slices
```

`--strength` is the SDEdit refinement strength: 0.3 is the light refinement used in the paper.

### 5. Evaluation

```bash
python scripts/evaluation/evaluate_distortion_correction.py --help
```

Reports PSNR / SSIM / NMSE / MAE (whole-FOV and prostate-centred), plus wall-clock timing, against
FUGUE and TOPUP baselines.

---

## Repository layout

```
dgr/
  physics/      B0 field I/O, DICOM handling, SH & polynomial field fitting, B0→T2 registration
  utils/        forward EPI model: VDM computation, splat/warp/resample kernels
  models/       phc_net → phc_e2e_mega_net → phc_e2e_mageultra_net (stage 1);
                diffusion_unet_diffusers (stage 2)
  data/         paired dual-b NPZ dataset with 2.5-D slice stacking
  inference/    DDIM / DDPM / DPM-Solver samplers with T2 + CNN conditioning
  losses/       stage-1 losses (SSIM, ADC consistency, relative intensity, TV / Jacobian penalties)
  conditioning/ T2w conditioning channel construction
scripts/
  simulation/   forward-model entry points
  restoration/  stage-1 / stage-2 training and DGR inference
  evaluation/   quantitative comparison against FUGUE / TOPUP
slurm/          SLURM job templates
tools/          checkpoint export for Hugging Face
configs/        YAML records of the exact settings used for the published runs
```

## Data

The clinical prostate data used in the paper cannot be redistributed. The pipeline consumes
per-subject `.npz` volumes; `configs/simulation.yaml` documents the expected keys so the
simulation half can be reproduced on any DWI + T2w + ΔB0 source, including the public
[fastMRI Prostate](https://github.com/cai2r/fastMRI_prostate) dataset.

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

## License

See [LICENSE](LICENSE). Research use only; not a medical device and not cleared for clinical use.
