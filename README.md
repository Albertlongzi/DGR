# Let Distortion Guide Restoration (DGR)

<div align="center">

<!-- Replace with actual paper badge/link once published -->
![Status](https://img.shields.io/badge/status-code%20coming%20soon-orange)
![License](https://img.shields.io/badge/license-TBD-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

**Physics-Informed Deep Learning for Geometric Distortion Correction in Prostate DWI**

*Code and pre-trained weights will be released upon paper publication.*

</div>

---

## Overview

Prostate multiparametric MRI (mpMRI) is the clinical gold standard for prostate cancer detection and PI-RADS grading. However, the diffusion-weighted imaging (DWI) component — acquired using single-shot echo-planar imaging (ssEPI) — is highly vulnerable to B0 field inhomogeneities, causing severe geometric distortions in the form of spatial warping, pixel pile-up, and signal dropout. These artifacts are dramatically worse in patients with **hip prostheses** or **bowel distension**, precisely the demographic most at risk for prostate cancer.

**DGR** addresses this without any additional scan acquisition. By learning from a physics-based forward distortion simulator, DGR corrects severe geometric distortions using only routinely acquired DWI and T2-weighted images.

<div align="center">

```
Distorted ssEPI DWI  ──────────────────────────────►  Corrected DWI
(geometric warping,                                   (anatomically
 pixel pile-up,                                        faithful,
 signal dropout)                                       diagnostic quality)
```

</div>

---

## Key Features

- **No extra acquisitions required** — works with standard clinical DWI + T2W protocol; no B0 field maps, no reverse phase-encoded scans
- **Physics-informed training** — forward ssEPI distortion simulator using real B0 field maps from hip-prosthesis patients, augmented via spherical harmonic perturbation (>40,000 paired training samples)
- **Hybrid CNN–Diffusion architecture** — two-stage pipeline: coarse geometric correction via CNN, fine texture restoration via conditional diffusion refinement (SDEdit-style)
- **T2W anatomical conditioning** — uses the distortion-free T2W scan as an anatomical reference via deformable cross-attention
- **Clinically validated** — prospective cohort of 34 subjects with severe baseline distortion; blinded radiologist scoring shows significant improvement in geometric fidelity, image quality, and diagnostic confidence

---

## Method

DGR is a two-stage pipeline:

### Stage 1 — CNN Restoration Backbone

A 2.5D multi-scale encoder–decoder with residual blocks processes the distorted DWI alongside co-registered T2W. A **contrast-aware deformable cross-attention** module treats distorted DWI as queries and T2W as keys/values, computing adaptive spatial offsets to bridge the geometric mismatch between the two modalities. Feature Pyramid Network (FPN)-style top-down aggregation recovers fine spatial details.

### Stage 2 — Conditional Diffusion Refinement

A conditional diffusion model (SR3-style UNet, DDPM) refines the coarse CNN output. Using **SDEdit-style img2img initialization** (strength = 0.1), the diffusion model sharpens edges and restores realistic noise texture while preserving the coarse geometry from Stage 1. Inference uses DPM-Solver for fast sampling.

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
├── src/                   # Source code (coming soon)
│   ├── models/            #   CNN backbone & diffusion model
│   ├── data/              #   Dataset loaders & preprocessing
│   ├── simulator/         #   Physics-based forward distortion simulator
│   └── utils/             #   Training utilities & metrics
├── configs/               # Training & inference configuration files
├── data/                  # Data directory (see Data section below)
├── models/                # Pre-trained model weights (coming soon)
├── results/               # Evaluation outputs & visualizations
└── README.md
```

---

## Data

This work uses two datasets:

| Dataset | Subjects | Usage |
|:---|:---:|:---|
| [fastMRI Prostate](https://fastmri.med.nyu.edu/) | 314 exams | Training / Test |
| In-house (Cedars-Sinai Medical Center) | 130 exams | Training / Test |

B0 field maps were acquired from 11 patients with hip prostheses and augmented to 110 maps via 12th-order spherical harmonic perturbation, driving the forward distortion simulator.

---

## Citation

> **Code and pre-trained weights will be made publicly available upon paper publication.**

If you find this work useful, please cite our paper (BibTeX will be provided upon publication):

```bibtex
@article{DGR2025,
  title   = {Let Distortion Guide Restoration: Physics-Informed Deep Learning
             for Geometric Distortion Correction in Prostate Diffusion-Weighted MRI},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

---

## Acknowledgements

This work was supported by NIH grants R01NS121544, R01HL156818, R01HL165211, R01HL181091, and R43NS120795. We thank the Research Imaging Core (RIC) at Cedars-Sinai Medical Center, MRI Technologist Mike Ngo, Irene Lee, and nurses Catherine Ubaldo-Prado and Lee Hyae for their support in data acquisition.

---

<div align="center">
<sub>⭐ Star this repo to get notified when code is released!</sub>
</div>
