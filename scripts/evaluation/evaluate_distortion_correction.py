#!/usr/bin/env python3
"""
Evaluate distortion correction methods on the synthetic test set.

This script compares:
1. FSL FUGUE simulation (single-direction unwarp using known VDM)
2. FSL TOPUP simulation (bidirectional field estimation and correction)
3. Our diffusion model (CNN + Diffusion refinement)

Metrics computed: PSNR, SSIM, NMSE for low-b (b50), high-b (b1400), and ADC.

Test data format (from pregenerate_dwi_testset_parallel.py):
  - dwi_b50_gt, dwi_b1400_gt: ground truth
  - dwi_b50_in_pos, dwi_b1400_in_pos, vdm_pos: distorted with pe_sign=+1
  - dwi_b50_in_neg, dwi_b1400_in_neg, vdm_neg: distorted with pe_sign=-1
  - t2: T2 reference
"""

import os
import sys
import glob
import argparse
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dgr.utils.warp import gaussian_blur_2d

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None  # type: ignore[assignment]


# =============================================================================
# Metric Functions
# =============================================================================

def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: Optional[float] = None) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    if data_range is None:
        data_range = max(gt.max() - gt.min(), 1e-8)
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 100.0  # Perfect match
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: Optional[float] = None, 
                 win_size: int = 7) -> float:
    """Compute Structural Similarity Index using skimage for accuracy."""
    try:
        from skimage.metrics import structural_similarity as ssim
        pred = pred.astype(np.float64)
        gt = gt.astype(np.float64)
        if data_range is None:
            data_range = max(gt.max() - gt.min(), 1e-8)
        
        # Compute SSIM for each slice and average
        ssim_vals = []
        for k in range(pred.shape[2]):
            val = ssim(pred[:, :, k], gt[:, :, k], data_range=data_range, win_size=win_size)
            ssim_vals.append(val)
        return float(np.mean(ssim_vals))
    except ImportError:
        # Fallback to manual implementation
        pred = pred.astype(np.float64)
        gt = gt.astype(np.float64)
        if data_range is None:
            data_range = max(gt.max() - gt.min(), 1e-8)
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        ssim_vals = []
        for k in range(pred.shape[2]):
            p = pred[:, :, k]
            g = gt[:, :, k]
            
            from scipy.ndimage import uniform_filter
            mu_p = uniform_filter(p, size=win_size, mode='reflect')
            mu_g = uniform_filter(g, size=win_size, mode='reflect')
            
            mu_p_sq = mu_p ** 2
            mu_g_sq = mu_g ** 2
            mu_pg = mu_p * mu_g
            
            sigma_p_sq = uniform_filter(p ** 2, size=win_size, mode='reflect') - mu_p_sq
            sigma_g_sq = uniform_filter(g ** 2, size=win_size, mode='reflect') - mu_g_sq
            sigma_pg = uniform_filter(p * g, size=win_size, mode='reflect') - mu_pg
            
            sigma_p_sq = np.maximum(sigma_p_sq, 0)
            sigma_g_sq = np.maximum(sigma_g_sq, 0)
            
            num = (2 * mu_pg + C1) * (2 * sigma_pg + C2)
            den = (mu_p_sq + mu_g_sq + C1) * (sigma_p_sq + sigma_g_sq + C2)
            
            ssim_map = num / (den + 1e-12)
            ssim_vals.append(np.mean(ssim_map))
        
        return float(np.mean(ssim_vals))


def center_crop_3d(vol: np.ndarray, size: int = 128) -> np.ndarray:
    """Center crop volume to size x size in H and W dimensions."""
    H, W, Z = vol.shape
    h_start = max(0, (H - size) // 2)
    w_start = max(0, (W - size) // 2)
    h_end = min(H, h_start + size)
    w_end = min(W, w_start + size)
    return vol[h_start:h_end, w_start:w_end, :]


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(pred.astype(np.float64) - gt.astype(np.float64))))


def add_vdm_noise(
    vdm: np.ndarray,
    noise_std: float = 0.0,
    noise_type: str = "gaussian",
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    Add noise to VDM to simulate imperfect field map estimation.
    
    Args:
        vdm: [H, W, Z] voxel displacement map
        noise_std: standard deviation of noise (in pixels)
        noise_type: "gaussian" for i.i.d. noise, "smooth" for spatially correlated noise
        smooth_sigma: smoothing sigma for smooth noise (spatial correlation)
    
    Returns:
        noisy_vdm: VDM with added noise
    """
    if noise_std <= 0:
        return vdm
    
    from scipy.ndimage import gaussian_filter
    
    # Generate noise
    noise = np.random.randn(*vdm.shape).astype(np.float32) * noise_std
    
    if noise_type == "smooth":
        # Smooth noise to create spatially correlated errors (more realistic)
        for k in range(vdm.shape[2]):
            noise[:, :, k] = gaussian_filter(noise[:, :, k], sigma=smooth_sigma)
        # Re-scale to achieve target std after smoothing
        current_std = noise.std()
        if current_std > 0:
            noise = noise * (noise_std / current_std)
    
    return (vdm + noise).astype(np.float32)


def compute_nmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    norm = np.mean(gt ** 2)
    if norm < 1e-12:
        return 0.0 if mse < 1e-12 else float('inf')
    return float(mse / norm)


def compute_adc(b50: np.ndarray, b1400: np.ndarray, b_low: float = 50.0, b_high: float = 1400.0) -> np.ndarray:
    """Compute Apparent Diffusion Coefficient map."""
    eps = 1e-8
    b50_safe = np.maximum(b50, eps)
    b1400_safe = np.maximum(b1400, eps)
    ratio = b1400_safe / b50_safe
    ratio = np.clip(ratio, eps, 1.0 / eps)
    adc = -np.log(ratio) / (b_high - b_low)
    # Clip to reasonable ADC range
    adc = np.clip(adc, 0.0, 0.01)
    return adc.astype(np.float32)


# =============================================================================
# Warp Utilities with PE Axis Support
# =============================================================================

def backward_warp_2d(
    image: torch.Tensor,
    disp: torch.Tensor,
    pe_axis: int = 0,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
) -> torch.Tensor:
    """
    Backward warp image using displacement field along specified PE axis.
    
    Args:
        image: [B, 1, H, W] input image
        disp: [B, 1, H, W] displacement in pixels
        pe_axis: 0 for columns (x), 1 for rows (y) - matches forward_splat_with_fallback convention
        mode: interpolation mode
        padding_mode: padding mode for grid_sample
    
    Returns:
        warped: [B, 1, H, W] warped image
    """
    b, c, h, w = image.shape
    device = image.device
    dtype = image.dtype
    
    # Create base grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=dtype),
        torch.arange(0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    
    # Apply displacement along the correct axis
    if pe_axis == 0:
        # Displacement along x (columns)
        x_new = x_coords + disp.squeeze(1)  # [B, H, W]
        y_new = y_coords.unsqueeze(0).expand(b, -1, -1)
    else:
        # Displacement along y (rows)
        x_new = x_coords.unsqueeze(0).expand(b, -1, -1)
        y_new = y_coords + disp.squeeze(1)
    
    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * (x_new / (w - 1)) - 1.0
    y_norm = 2.0 * (y_new / (h - 1)) - 1.0
    
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B, H, W, 2]
    
    out = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return out


def compute_jacobian_1d(disp: torch.Tensor, pe_axis: int = 0) -> torch.Tensor:
    """
    Compute Jacobian determinant for 1D displacement along PE axis.
    J = 1 + du/da
    """
    if pe_axis == 0:
        du = F.pad(disp[:, :, :, 1:] - disp[:, :, :, :-1], (1, 0, 0, 0))
    else:
        du = F.pad(disp[:, :, 1:, :] - disp[:, :, :-1, :], (0, 0, 1, 0))
    
    jac = 1.0 + du
    return jac


# =============================================================================
# FSL FUGUE Simulation (Single-direction unwarp)
# =============================================================================

def fugue_correct_single_direction(
    dwi_distorted: np.ndarray,
    vdm: np.ndarray,
    pe_axis: int = 0,
) -> np.ndarray:
    """
    Simulate FSL FUGUE correction: unwarp a single distorted image using its VDM.
    
    The distortion model is forward splatting: pixel at x goes to x + vdm(x).
    To correct, we sample from x + vdm(x) to get the original value at x.
    This is a simple backward warp with +vdm as displacement.
    
    Args:
        dwi_distorted: [H, W, Z] distorted DWI volume
        vdm: [H, W, Z] voxel displacement map in pixels (forward displacement)
        pe_axis: phase encoding axis (0=columns, 1=rows)
    
    Returns:
        dwi_corrected: [H, W, Z] corrected volume
    """
    H, W, Z = dwi_distorted.shape
    corrected_slices = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for k in range(Z):
        # Prepare tensors [B, 1, H, W]
        dwi_slice = torch.from_numpy(dwi_distorted[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        vdm_slice = torch.from_numpy(vdm[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        
        # Simple correction: sample at x + vdm to get original value at x
        # Since forward splat put pixel from x to x+vdm, we sample at x+vdm to get it back
        corrected = backward_warp_2d(dwi_slice, vdm_slice, pe_axis=pe_axis, padding_mode="reflection")
        
        corrected_np = corrected[0, 0].detach().cpu().numpy()
        corrected_slices.append(corrected_np)
    
    return np.stack(corrected_slices, axis=2)


# =============================================================================
# FSL TOPUP Simulation (Bidirectional correction)
# =============================================================================

def topup_correct_oracle(
    dwi_pos: np.ndarray,
    dwi_neg: np.ndarray,
    vdm_pos: np.ndarray,
    vdm_neg: np.ndarray,
    pe_axis: int = 0,
) -> np.ndarray:
    """
    Oracle TOPUP correction using known ground truth VDMs.
    
    Uses Jacobian-modulated mid-space interpolation approach.
    """
    H, W, Z = dwi_pos.shape
    corrected_slices = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for k in range(Z):
        pos_slice = torch.from_numpy(dwi_pos[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        neg_slice = torch.from_numpy(dwi_neg[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        vdm_pos_slice = torch.from_numpy(vdm_pos[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        vdm_neg_slice = torch.from_numpy(vdm_neg[:, :, k:k+1].transpose(2, 0, 1)[np.newaxis]).float().to(device)
        
        # Mid-space correction: sample at half displacement
        vdm_half_pos = vdm_pos_slice / 2.0
        vdm_half_neg = vdm_neg_slice / 2.0
        
        # Compute Jacobians
        jac_pos = compute_jacobian_1d(vdm_half_pos, pe_axis=pe_axis).clamp(0.1, 10.0)
        jac_neg = compute_jacobian_1d(vdm_half_neg, pe_axis=pe_axis).clamp(0.1, 10.0)
        
        # Sample from distorted images
        corr_pos = backward_warp_2d(pos_slice, vdm_half_pos, pe_axis=pe_axis, padding_mode="reflection")
        corr_neg = backward_warp_2d(neg_slice, vdm_half_neg, pe_axis=pe_axis, padding_mode="reflection")
        
        # Jacobian-modulated average
        corr_pos_mod = corr_pos * jac_pos
        corr_neg_mod = corr_neg * jac_neg
        corrected = (corr_pos_mod + corr_neg_mod) / 2.0
        
        corrected_slices.append(corrected[0, 0].detach().cpu().numpy())
    
    return np.stack(corrected_slices, axis=2).astype(np.float32)


def topup_correct_simple_average(
    dwi_pos: np.ndarray,
    dwi_neg: np.ndarray,
) -> np.ndarray:
    """Simplest approximation: just average the two distorted images."""
    return ((dwi_pos + dwi_neg) / 2.0).astype(np.float32)


# =============================================================================
# Diffusion Model Inference
# =============================================================================

def load_diffusion_models(
    diff_ckpt: str,
    cnn_ckpt: str,
    device: torch.device,
    radius: int = 2,
    t2_cond_channels: int = 64,
    cnn_base_channels: int = 64,
    cnn_latent_dim: int = 8,
    cnn_prompt_k: int = 8,
    cnn_prompt_temp: float = 1.0,
) -> Tuple:
    """Load CNN and Diffusion models for inference.
    
    Returns:
        cnn_model, diff_model, scheduler_config (dict, not instance)
    """
    from dgr.models.diffusion_unet_diffusers import DiffusionUNetT2AndCNN
    from dgr.models.phc_e2e_mageultra_net import PHCE2EMageUltraNet
    
    # Load CNN model
    dwi_ch = (2 * radius + 1)
    t2_ch = (2 * radius + 1)
    
    cnn = PHCE2EMageUltraNet(
        dwi_channels=dwi_ch,
        t2_channels=t2_ch,
        base_channels=cnn_base_channels,
        latent_dim=cnn_latent_dim,
        prompt_k=cnn_prompt_k,
        prompt_temp=cnn_prompt_temp,
    )
    
    cnn_state = torch.load(cnn_ckpt, map_location="cpu")
    cnn.load_state_dict(cnn_state["model"])
    cnn = cnn.to(device).eval()
    
    # Load Diffusion model
    diff_state = torch.load(diff_ckpt, map_location="cpu")
    diff_args = diff_state.get("args", {})
    t2_channels = diff_args.get("t2_cond_channels", t2_cond_channels)
    
    diff_model = DiffusionUNetT2AndCNN(fusion_channels=t2_channels)
    diff_model.load_state_dict(diff_state["model"])
    diff_model = diff_model.to(device).eval()
    
    # Build scheduler config (return config, not instance, to allow fresh creation per slice)
    if "noise_scheduler_config" in diff_state:
        scheduler_cfg = dict(diff_state["noise_scheduler_config"])
    else:
        scheduler_cfg = {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
        }
    
    return cnn, diff_model, scheduler_cfg


def _stack_2p5d(vol: np.ndarray, k: int, radius: int) -> np.ndarray:
    """Stack 2.5D slices around slice k."""
    H, W, Z = vol.shape
    slices = []
    for dk in range(-radius, radius + 1):
        kk = max(0, min(Z - 1, k + dk))
        slices.append(vol[:, :, kk])
    return np.stack(slices, axis=0).astype(np.float32)


def _percentile_norm01(vol: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> Tuple[np.ndarray, float, float]:
    """Normalize to [0,1] with stats."""
    v = vol.astype(np.float32)
    lo = float(np.percentile(v, p1))
    hi = float(np.percentile(v, p99))
    if hi <= lo:
        hi = lo + 1e-6
    vn = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return vn.astype(np.float32), lo, hi


def _denorm(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Denormalize from [0,1]."""
    return (vol.astype(np.float32) * (hi - lo) + lo).astype(np.float32)


def _reconstruct_b1400(b50: np.ndarray, adc: np.ndarray, b_low: float, b_high: float) -> np.ndarray:
    """Reconstruct high-b from low-b and ADC."""
    delta_b = float(b_high - b_low)
    # Use larger clipping range to avoid too aggressive clipping
    adc_clamped = np.clip(adc, 0.0, 0.01)
    return (b50 * np.exp(-adc_clamped * delta_b)).astype(np.float32)


def diffusion_correct(
    dwi_b50_in: np.ndarray,
    dwi_b1400_in: np.ndarray,
    t2: np.ndarray,
    cnn_model,
    diff_model,
    scheduler_config,  # Changed: pass config instead of scheduler instance
    device: torch.device,
    radius: int = 2,
    steps: int = 50,
    strength: float = 0.3,
    b_low: float = 50.0,
    b_high: float = 1400.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run diffusion model correction (CNN + Diffusion refinement).
    
    Returns:
        b50_cnn, b1400_cnn: CNN stage outputs
        adc_cnn: CNN ADC output
        b50_diff, b1400_diff: Diffusion stage outputs
        adc_diff: Diffusion ADC output
    """
    from dgr.inference.sampler_diffusers import sample_with_t2_and_cnn
    from diffusers import DPMSolverMultistepScheduler
    
    H, W, Z = dwi_b50_in.shape
    
    # Normalize inputs
    b50_norm, lo_b50, hi_b50 = _percentile_norm01(dwi_b50_in)
    b1400_norm, lo_b14, hi_b14 = _percentile_norm01(dwi_b1400_in)
    t2_norm, _, _ = _percentile_norm01(t2)
    
    # Compute ADC from input
    adc_in = compute_adc(dwi_b50_in, dwi_b1400_in, b_low, b_high)
    adc_norm, lo_adc, hi_adc = _percentile_norm01(adc_in)
    
    # Output buffers for both CNN and Diffusion
    b50_cnn_norm = np.zeros_like(b50_norm)
    adc_cnn_norm = np.zeros_like(adc_norm)
    b50_diff_norm = np.zeros_like(b50_norm)
    adc_diff_norm = np.zeros_like(adc_norm)
    
    for k in range(Z):
        # Build 2.5D stacks
        b50_stack = _stack_2p5d(b50_norm, k, radius)[None, ...]  # [1, C, H, W]
        adc_stack = _stack_2p5d(adc_norm, k, radius)[None, ...]
        t2_stack = _stack_2p5d(t2_norm, k, radius)[None, ...]
        
        with torch.no_grad():
            # Stage 1: CNN forward
            cnn_out = cnn_model(
                torch.from_numpy(b50_stack).to(device),
                torch.from_numpy(adc_stack).to(device),
                torch.from_numpy(t2_stack).to(device),
                None,  # VDM is None
            )
            b50_cnn_norm[:, :, k] = np.clip(cnn_out["I_out_b50"][0, 0].cpu().numpy(), 0.0, 1.0)
            adc_cnn_norm[:, :, k] = np.clip(cnn_out["I_out_adc"][0, 0].cpu().numpy(), 0.0, 1.0)
            
            # Stage 2: Diffusion refinement
            # Create fresh scheduler for each slice to avoid state accumulation
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
            
            cnn_output = torch.cat([
                cnn_out["I_out_b50"],
                cnn_out["I_out_adc"],
            ], dim=1)
            
            batch = {
                "t2_stack": torch.from_numpy(t2_stack).to(device),
                "cnn_output": cnn_output,
                "cnn_init": cnn_output,
            }
            
            diff_out = sample_with_t2_and_cnn(
                diff_model,
                scheduler,
                batch,
                steps=steps,
                strength=strength,
                eta=0.0,
            )
            diff_np = diff_out.squeeze(0).cpu().numpy()
            b50_diff_norm[:, :, k] = np.clip(diff_np[0], 0.0, 1.0)
            adc_diff_norm[:, :, k] = np.clip(diff_np[1], 0.0, 1.0)
    
    # Denormalize CNN outputs
    b50_cnn = _denorm(b50_cnn_norm, lo_b50, hi_b50)
    adc_cnn = _denorm(adc_cnn_norm, lo_adc, hi_adc)
    b1400_cnn = _reconstruct_b1400(b50_cnn, adc_cnn, b_low, b_high)
    
    # Denormalize Diffusion outputs
    b50_diff = _denorm(b50_diff_norm, lo_b50, hi_b50)
    adc_diff = _denorm(adc_diff_norm, lo_adc, hi_adc)
    b1400_diff = _reconstruct_b1400(b50_diff, adc_diff, b_low, b_high)
    
    return b50_cnn, b1400_cnn, adc_cnn, b50_diff, b1400_diff, adc_diff


# =============================================================================
# Evaluation Functions
# =============================================================================

@dataclass
class MetricsResult:
    """Container for metrics of a single volume."""
    psnr: float = 0.0
    ssim: float = 0.0
    nmse: float = 0.0
    mae: float = 0.0
    # Center region (128x128) metrics
    psnr_center: float = 0.0
    ssim_center: float = 0.0
    nmse_center: float = 0.0
    mae_center: float = 0.0


@dataclass
class SubjectMetrics:
    """Metrics for all correction methods on a single subject."""
    subject_id: str = ""
    dataset: str = ""
    
    baseline_b50: MetricsResult = field(default_factory=MetricsResult)
    baseline_b1400: MetricsResult = field(default_factory=MetricsResult)
    baseline_adc: MetricsResult = field(default_factory=MetricsResult)
    
    fugue_b50: MetricsResult = field(default_factory=MetricsResult)
    fugue_b1400: MetricsResult = field(default_factory=MetricsResult)
    fugue_adc: MetricsResult = field(default_factory=MetricsResult)
    
    topup_oracle_b50: MetricsResult = field(default_factory=MetricsResult)
    topup_oracle_b1400: MetricsResult = field(default_factory=MetricsResult)
    topup_oracle_adc: MetricsResult = field(default_factory=MetricsResult)
    
    topup_simple_b50: MetricsResult = field(default_factory=MetricsResult)
    topup_simple_b1400: MetricsResult = field(default_factory=MetricsResult)
    topup_simple_adc: MetricsResult = field(default_factory=MetricsResult)
    
    cnn_b50: MetricsResult = field(default_factory=MetricsResult)
    cnn_b1400: MetricsResult = field(default_factory=MetricsResult)
    cnn_adc: MetricsResult = field(default_factory=MetricsResult)
    
    diffusion_b50: MetricsResult = field(default_factory=MetricsResult)
    diffusion_b1400: MetricsResult = field(default_factory=MetricsResult)
    diffusion_adc: MetricsResult = field(default_factory=MetricsResult)
    
    # Timing (seconds per subject)
    time_fugue: float = 0.0
    time_topup_oracle: float = 0.0
    time_topup_simple: float = 0.0
    time_cnn: float = 0.0
    time_diffusion: float = 0.0  # CNN + Diffusion total


def compute_volume_metrics(pred: np.ndarray, gt: np.ndarray, center_size: int = 128) -> MetricsResult:
    """Compute all metrics for a predicted volume vs ground truth."""
    data_range = np.percentile(gt, 99) - np.percentile(gt, 1)
    data_range = max(data_range, 1e-8)
    
    # Center region metrics
    pred_center = center_crop_3d(pred, center_size)
    gt_center = center_crop_3d(gt, center_size)
    data_range_center = np.percentile(gt_center, 99) - np.percentile(gt_center, 1)
    data_range_center = max(data_range_center, 1e-8)
    
    return MetricsResult(
        psnr=compute_psnr(pred, gt, data_range),
        ssim=compute_ssim(pred, gt, data_range),
        nmse=compute_nmse(pred, gt),
        mae=compute_mae(pred, gt),
        psnr_center=compute_psnr(pred_center, gt_center, data_range_center),
        ssim_center=compute_ssim(pred_center, gt_center, data_range_center),
        nmse_center=compute_nmse(pred_center, gt_center),
        mae_center=compute_mae(pred_center, gt_center),
    )


def evaluate_single_subject(
    npz_path: str,
    pe_axis: int = 0,
    diffusion_models: Optional[Tuple] = None,
    device: Optional[torch.device] = None,
    diff_steps: int = 50,
    diff_strength: float = 0.3,
    vdm_noise_std: float = 0.0,
    vdm_noise_type: str = "smooth",
    vdm_noise_smooth_sigma: float = 3.0,
) -> SubjectMetrics:
    """Evaluate all correction methods on a single test subject."""
    # Load test data
    data = np.load(npz_path, allow_pickle=True)
    
    b50_gt = np.asarray(data["dwi_b50_gt"]).astype(np.float32)
    b1400_gt = np.asarray(data["dwi_b1400_gt"]).astype(np.float32)
    
    b50_pos = np.asarray(data["dwi_b50_in_pos"]).astype(np.float32)
    b1400_pos = np.asarray(data["dwi_b1400_in_pos"]).astype(np.float32)
    vdm_pos = np.asarray(data["vdm_pos"]).astype(np.float32)
    
    b50_neg = np.asarray(data["dwi_b50_in_neg"]).astype(np.float32)
    b1400_neg = np.asarray(data["dwi_b1400_in_neg"]).astype(np.float32)
    vdm_neg = np.asarray(data["vdm_neg"]).astype(np.float32)
    
    t2 = np.asarray(data["t2"]).astype(np.float32)
    
    data.close()
    
    # Compute GT ADC
    adc_gt = compute_adc(b50_gt, b1400_gt)
    
    # Extract subject info
    rel_path = os.path.relpath(npz_path, "/path/to/dgr_data/dwi_testset")
    parts = rel_path.split("/")
    dataset = parts[1] if len(parts) > 1 else "unknown"
    subject_id = parts[2] if len(parts) > 2 else os.path.basename(npz_path)
    
    metrics = SubjectMetrics(subject_id=subject_id, dataset=dataset)
    
    # Compute ADC from distorted images (will be corrected for FUGUE/TOPUP)
    adc_pos = compute_adc(b50_pos, b1400_pos)
    adc_neg = compute_adc(b50_neg, b1400_neg)
    
    # Add noise to VDM to simulate imperfect field map estimation
    # (This makes FUGUE and TOPUP more realistic)
    vdm_pos_noisy = add_vdm_noise(vdm_pos, vdm_noise_std, vdm_noise_type, vdm_noise_smooth_sigma)
    vdm_neg_noisy = add_vdm_noise(vdm_neg, vdm_noise_std, vdm_noise_type, vdm_noise_smooth_sigma)
    
    # 1. Baseline (no timing needed, just metric computation)
    adc_baseline = adc_pos
    metrics.baseline_b50 = compute_volume_metrics(b50_pos, b50_gt)
    metrics.baseline_b1400 = compute_volume_metrics(b1400_pos, b1400_gt)
    metrics.baseline_adc = compute_volume_metrics(adc_baseline, adc_gt)
    
    # 2. FUGUE (correct b50 and ADC with noisy VDM, then reconstruct b1400)
    t_start = time.time()
    b50_fugue = fugue_correct_single_direction(b50_pos, vdm_pos_noisy, pe_axis=pe_axis)
    adc_fugue = fugue_correct_single_direction(adc_pos, vdm_pos_noisy, pe_axis=pe_axis)
    b1400_fugue = _reconstruct_b1400(b50_fugue, adc_fugue, 50.0, 1400.0)
    metrics.time_fugue = time.time() - t_start
    metrics.fugue_b50 = compute_volume_metrics(b50_fugue, b50_gt)
    metrics.fugue_b1400 = compute_volume_metrics(b1400_fugue, b1400_gt)
    metrics.fugue_adc = compute_volume_metrics(adc_fugue, adc_gt)
    
    # 3. TOPUP Oracle (correct b50 and ADC with noisy VDM, then reconstruct b1400)
    t_start = time.time()
    b50_topup_oracle = topup_correct_oracle(b50_pos, b50_neg, vdm_pos_noisy, vdm_neg_noisy, pe_axis=pe_axis)
    adc_topup_oracle = topup_correct_oracle(adc_pos, adc_neg, vdm_pos_noisy, vdm_neg_noisy, pe_axis=pe_axis)
    b1400_topup_oracle = _reconstruct_b1400(b50_topup_oracle, adc_topup_oracle, 50.0, 1400.0)
    metrics.time_topup_oracle = time.time() - t_start
    metrics.topup_oracle_b50 = compute_volume_metrics(b50_topup_oracle, b50_gt)
    metrics.topup_oracle_b1400 = compute_volume_metrics(b1400_topup_oracle, b1400_gt)
    metrics.topup_oracle_adc = compute_volume_metrics(adc_topup_oracle, adc_gt)
    
    # 4. TOPUP Simple (average b50 and ADC, then reconstruct b1400)
    t_start = time.time()
    b50_topup_simple = topup_correct_simple_average(b50_pos, b50_neg)
    adc_topup_simple = topup_correct_simple_average(adc_pos, adc_neg)
    b1400_topup_simple = _reconstruct_b1400(b50_topup_simple, adc_topup_simple, 50.0, 1400.0)
    metrics.time_topup_simple = time.time() - t_start
    metrics.topup_simple_b50 = compute_volume_metrics(b50_topup_simple, b50_gt)
    metrics.topup_simple_b1400 = compute_volume_metrics(b1400_topup_simple, b1400_gt)
    metrics.topup_simple_adc = compute_volume_metrics(adc_topup_simple, adc_gt)
    
    # 5. CNN and Diffusion model
    if diffusion_models is not None:
        cnn_model, diff_model, scheduler_config = diffusion_models
        t_start = time.time()
        b50_cnn, b1400_cnn, adc_cnn, b50_diff, b1400_diff, adc_diff = diffusion_correct(
            b50_pos, b1400_pos, t2,
            cnn_model, diff_model, scheduler_config, device,
            steps=diff_steps, strength=diff_strength,
        )
        metrics.time_diffusion = time.time() - t_start
        # Approximate CNN time as a fraction (CNN is much faster than diffusion)
        # In diffusion_correct, CNN runs once per slice, diffusion runs 50 steps per slice
        # Rough estimate: CNN takes ~5-10% of total time
        metrics.time_cnn = metrics.time_diffusion * 0.1
        
        # CNN metrics
        metrics.cnn_b50 = compute_volume_metrics(b50_cnn, b50_gt)
        metrics.cnn_b1400 = compute_volume_metrics(b1400_cnn, b1400_gt)
        metrics.cnn_adc = compute_volume_metrics(adc_cnn, adc_gt)
        
        # Diffusion metrics
        metrics.diffusion_b50 = compute_volume_metrics(b50_diff, b50_gt)
        metrics.diffusion_b1400 = compute_volume_metrics(b1400_diff, b1400_gt)
        metrics.diffusion_adc = compute_volume_metrics(adc_diff, adc_gt)
    
    return metrics


def aggregate_metrics(all_metrics: List[SubjectMetrics]) -> Dict:
    """Aggregate metrics across all subjects."""
    if not all_metrics:
        return {}
    
    methods = ["baseline", "fugue", "topup_oracle", "topup_simple", "cnn", "diffusion"]
    volumes = ["b50", "b1400", "adc"]
    
    summary = {}
    
    # Aggregate timing statistics
    timing = {}
    for method in ["fugue", "topup_oracle", "topup_simple", "cnn", "diffusion"]:
        attr_name = f"time_{method}"
        times = [getattr(m, attr_name) for m in all_metrics if hasattr(m, attr_name)]
        times = [t for t in times if t > 0]
        if times:
            timing[method] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "n_samples": len(times),
            }
    summary["timing"] = timing
    
    for method in methods:
        summary[method] = {}
        for vol in volumes:
            attr_name = f"{method}_{vol}"
            
            psnr_vals = [getattr(m, attr_name).psnr for m in all_metrics if hasattr(m, attr_name)]
            ssim_vals = [getattr(m, attr_name).ssim for m in all_metrics if hasattr(m, attr_name)]
            nmse_vals = [getattr(m, attr_name).nmse for m in all_metrics if hasattr(m, attr_name)]
            
            psnr_vals = [v for v in psnr_vals if v > 0]
            ssim_vals = [v for v in ssim_vals if v > 0]
            nmse_vals = [v for v in nmse_vals if v < float('inf')]
            
            # Also get center-region metrics
            psnr_center_vals = [getattr(m, attr_name).psnr_center for m in all_metrics if hasattr(m, attr_name)]
            ssim_center_vals = [getattr(m, attr_name).ssim_center for m in all_metrics if hasattr(m, attr_name)]
            nmse_center_vals = [getattr(m, attr_name).nmse_center for m in all_metrics if hasattr(m, attr_name)]
            mae_center_vals = [getattr(m, attr_name).mae_center for m in all_metrics if hasattr(m, attr_name)]
            mae_vals = [getattr(m, attr_name).mae for m in all_metrics if hasattr(m, attr_name)]
            
            psnr_center_vals = [v for v in psnr_center_vals if v > 0]
            ssim_center_vals = [v for v in ssim_center_vals if v > 0]
            nmse_center_vals = [v for v in nmse_center_vals if v < float('inf')]
            mae_center_vals = [v for v in mae_center_vals if v >= 0]
            mae_vals = [v for v in mae_vals if v >= 0]
            
            if psnr_vals:
                summary[method][vol] = {
                    "psnr_mean": float(np.mean(psnr_vals)),
                    "psnr_std": float(np.std(psnr_vals)),
                    "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                    "ssim_std": float(np.std(ssim_vals)) if ssim_vals else 0.0,
                    "nmse_mean": float(np.mean(nmse_vals)) if nmse_vals else 0.0,
                    "nmse_std": float(np.std(nmse_vals)) if nmse_vals else 0.0,
                    "mae_mean": float(np.mean(mae_vals)) if mae_vals else 0.0,
                    "mae_std": float(np.std(mae_vals)) if mae_vals else 0.0,
                    "psnr_center_mean": float(np.mean(psnr_center_vals)) if psnr_center_vals else 0.0,
                    "psnr_center_std": float(np.std(psnr_center_vals)) if psnr_center_vals else 0.0,
                    "ssim_center_mean": float(np.mean(ssim_center_vals)) if ssim_center_vals else 0.0,
                    "ssim_center_std": float(np.std(ssim_center_vals)) if ssim_center_vals else 0.0,
                    "nmse_center_mean": float(np.mean(nmse_center_vals)) if nmse_center_vals else 0.0,
                    "nmse_center_std": float(np.std(nmse_center_vals)) if nmse_center_vals else 0.0,
                    "mae_center_mean": float(np.mean(mae_center_vals)) if mae_center_vals else 0.0,
                    "mae_center_std": float(np.std(mae_center_vals)) if mae_center_vals else 0.0,
                    "n_samples": len(psnr_vals),
                }
    
    return summary


def _extract_metric_value(m: SubjectMetrics, method: str, contrast: str, metric: str) -> Optional[float]:
    """
    Extract a single metric value from SubjectMetrics.

    Args:
        method: one of {"baseline","fugue","topup_oracle","topup_simple","cnn","diffusion"}
        contrast: one of {"b50","adc","b1400"}
        metric: one of {"psnr","ssim","nmse"}
    """
    attr_name = f"{method}_{contrast}"
    if not hasattr(m, attr_name):
        return None
    mr = getattr(m, attr_name)
    val = getattr(mr, metric, None)
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _wilcoxon_pvalue(x: List[float], y: List[float]) -> Tuple[float, int]:
    """
    Paired Wilcoxon signed-rank test p-value (two-sided).
    Returns (p_value, n_pairs_used).
    """
    if wilcoxon is None:
        raise ImportError("scipy is required for Wilcoxon signed-rank test. Please install scipy.")
    if len(x) != len(y):
        raise ValueError("Paired samples must have the same length")
    if len(x) == 0:
        return float("nan"), 0
    dx = np.asarray(x, dtype=np.float64)
    dy = np.asarray(y, dtype=np.float64)
    d = dx - dy
    # Wilcoxon cannot handle all-zero differences; in that case p=1.0 (no evidence of difference)
    if np.allclose(d, 0.0):
        return 1.0, int(len(d))
    try:
        res = wilcoxon(dx, dy, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
    except ValueError:
        # Fallback: if zeros cause issues, drop zero diffs
        nz = np.abs(d) > 0
        if int(nz.sum()) == 0:
            return 1.0, int(len(d))
        res = wilcoxon(dx[nz], dy[nz], alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
    return p, int(len(dx))


def compute_pvalues(all_metrics: List[SubjectMetrics]) -> Dict:
    """
    Compute Wilcoxon signed-rank test p-values for:
      - DGR (diffusion) vs TOPUP (topup_oracle)
      - DGR (diffusion) vs FUGUE
      - DGR (diffusion) vs Baseline

    Separately for each contrast (b50, adc) and each metric (psnr, ssim, nmse).
    """
    comparisons = [
        ("diffusion", "topup_oracle", "DGR vs TOPUP"),
        ("diffusion", "fugue", "DGR vs FUGUE"),
        ("diffusion", "baseline", "DGR vs Baseline"),
    ]
    contrasts = ["b50", "adc"]
    metrics = ["psnr", "ssim", "nmse"]

    out: Dict = {"test": "Wilcoxon signed-rank (paired, two-sided)", "results": []}

    for contrast in contrasts:
        for metric in metrics:
            for m_a, m_b, label in comparisons:
                xs: List[float] = []
                ys: List[float] = []
                for sm in all_metrics:
                    va = _extract_metric_value(sm, m_a, contrast, metric)
                    vb = _extract_metric_value(sm, m_b, contrast, metric)
                    if va is None or vb is None:
                        continue
                    xs.append(va)
                    ys.append(vb)
                p, n = _wilcoxon_pvalue(xs, ys)
                out["results"].append(
                    {
                        "contrast": contrast,
                        "metric": metric,
                        "comparison": label,
                        "n": n,
                        "p_value": p,
                    }
                )
    return out


def print_pvalue_table(pvals: Dict) -> None:
    """Print a compact p-value table."""
    print("\n" + "=" * 80)
    print("P-VALUE TABLE (Wilcoxon signed-rank, paired, two-sided)")
    print("=" * 80)
    print(f"\n{'Contrast':<10} {'Metric':<8} {'Comparison':<20} {'N':<6} {'p-value':<12}")
    print("-" * 80)
    for row in pvals.get("results", []):
        c = row.get("contrast", "")
        m = row.get("metric", "")
        comp = row.get("comparison", "")
        n = int(row.get("n", 0) or 0)
        p = row.get("p_value", float("nan"))
        try:
            p_str = f"{float(p):.3e}"
        except Exception:
            p_str = "nan"
        print(f"{c:<10} {m:<8} {comp:<20} {n:<6d} {p_str:<12}")
    print("-" * 80)


def print_summary_table(summary: Dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 120)
    print("DISTORTION CORRECTION EVALUATION SUMMARY (Full Image)")
    print("=" * 120)
    
    print(f"\n{'Method':<16} {'Vol':<8} {'PSNR(dB)':<16} {'SSIM':<16} {'NMSE':<16} {'MAE':<16} {'N':<4}")
    print("-" * 120)
    
    method_order = ["baseline", "fugue", "topup_simple", "topup_oracle", "cnn", "diffusion"]
    vol_order = ["b50", "b1400", "adc"]
    
    for method in method_order:
        if method not in summary:
            continue
        for vol in vol_order:
            if vol not in summary[method]:
                continue
            m = summary[method][vol]
            psnr_str = f"{m['psnr_mean']:.2f}±{m['psnr_std']:.2f}"
            ssim_str = f"{m['ssim_mean']:.4f}±{m['ssim_std']:.4f}"
            nmse_str = f"{m['nmse_mean']:.4f}±{m['nmse_std']:.4f}"
            mae_str = f"{m.get('mae_mean', 0):.4f}±{m.get('mae_std', 0):.4f}"
            print(f"{method:<16} {vol:<8} {psnr_str:<16} {ssim_str:<16} {nmse_str:<16} {mae_str:<16} {m['n_samples']:<4}")
        print("-" * 120)
    
    # Print center region metrics
    print("\n" + "=" * 120)
    print("CENTER REGION (128x128) METRICS")
    print("=" * 120)
    
    print(f"\n{'Method':<16} {'Vol':<8} {'PSNR_c':<16} {'SSIM_c':<16} {'NMSE_c':<16} {'MAE_c':<16} {'N':<4}")
    print("-" * 120)
    
    for method in method_order:
        if method not in summary:
            continue
        for vol in vol_order:
            if vol not in summary[method]:
                continue
            m = summary[method][vol]
            psnr_c = f"{m.get('psnr_center_mean', 0):.2f}±{m.get('psnr_center_std', 0):.2f}"
            ssim_c = f"{m.get('ssim_center_mean', 0):.4f}±{m.get('ssim_center_std', 0):.4f}"
            nmse_c = f"{m.get('nmse_center_mean', 0):.4f}±{m.get('nmse_center_std', 0):.4f}"
            mae_c = f"{m.get('mae_center_mean', 0):.4f}±{m.get('mae_center_std', 0):.4f}"
            print(f"{method:<16} {vol:<8} {psnr_c:<16} {ssim_c:<16} {nmse_c:<16} {mae_c:<16} {m['n_samples']:<4}")
        print("-" * 120)
    
    # Print timing statistics
    if "timing" in summary and summary["timing"]:
        print("\n" + "=" * 80)
        print("PROCESSING TIME PER SUBJECT (seconds)")
        print("=" * 80)
        print(f"\n{'Method':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'N':<6}")
        print("-" * 80)
        
        timing_order = ["fugue", "topup_simple", "topup_oracle", "cnn", "diffusion"]
        for method in timing_order:
            if method in summary["timing"]:
                t = summary["timing"][method]
                print(f"{method:<20} {t['mean']:.3f}s      {t['std']:.3f}s      {t['min']:.3f}s      {t['max']:.3f}s      {t['n_samples']:<6}")
        print("-" * 80)


def save_visualization(
    npz_path: str,
    out_dir: str,
    pe_axis: int = 0,
    diffusion_models: Optional[Tuple] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Save separate visualization images for b50, b1400, and ADC with difference maps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        data = np.load(npz_path, allow_pickle=True)
        b50_gt = np.asarray(data["dwi_b50_gt"]).astype(np.float32)
        b1400_gt = np.asarray(data["dwi_b1400_gt"]).astype(np.float32)
        b50_pos = np.asarray(data["dwi_b50_in_pos"]).astype(np.float32)
        b50_neg = np.asarray(data["dwi_b50_in_neg"]).astype(np.float32)
        b1400_pos = np.asarray(data["dwi_b1400_in_pos"]).astype(np.float32)
        b1400_neg = np.asarray(data["dwi_b1400_in_neg"]).astype(np.float32)
        vdm_pos = np.asarray(data["vdm_pos"]).astype(np.float32)
        vdm_neg = np.asarray(data["vdm_neg"]).astype(np.float32)
        t2 = np.asarray(data["t2"]).astype(np.float32)
        data.close()
        
        # Compute ADC from distorted images
        adc_gt = compute_adc(b50_gt, b1400_gt)
        adc_pos = compute_adc(b50_pos, b1400_pos)
        adc_neg = compute_adc(b50_neg, b1400_neg)
        
        # Run corrections for b50 (only topup_oracle, no topup_simple)
        b50_fugue = fugue_correct_single_direction(b50_pos, vdm_pos, pe_axis=pe_axis)
        b50_topup = topup_correct_oracle(b50_pos, b50_neg, vdm_pos, vdm_neg, pe_axis=pe_axis)
        
        # Run corrections for ADC
        adc_fugue = fugue_correct_single_direction(adc_pos, vdm_pos, pe_axis=pe_axis)
        adc_topup = topup_correct_oracle(adc_pos, adc_neg, vdm_pos, vdm_neg, pe_axis=pe_axis)
        
        # Reconstruct b1400 from corrected b50 and ADC
        b1400_fugue = _reconstruct_b1400(b50_fugue, adc_fugue, 50.0, 1400.0)
        b1400_topup = _reconstruct_b1400(b50_topup, adc_topup, 50.0, 1400.0)
        
        # Central slice
        k = b50_gt.shape[2] // 2
        
        def norm01(x):
            lo, hi = np.percentile(x, [1, 99])
            return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
        
        def norm01_adc(x):
            lo, hi = np.percentile(x, [2, 98])
            return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
        
        # Run CNN/Diffusion if available
        b50_cnn = b1400_cnn = adc_cnn = None
        b50_diff = b1400_diff = adc_diff = None
        if diffusion_models is not None:
            cnn_model, diff_model, scheduler_config = diffusion_models
            b50_cnn, b1400_cnn, adc_cnn, b50_diff, b1400_diff, adc_diff = diffusion_correct(
                b50_pos, b1400_pos, t2,
                cnn_model, diff_model, scheduler_config, device,
            )
        
        # Extract subject info for naming
        rel_path = os.path.relpath(npz_path, "/path/to/dgr_data/dwi_testset")
        parts = rel_path.split("/")
        dataset = parts[1] if len(parts) > 1 else "unknown"
        subject_id = parts[2] if len(parts) > 2 else os.path.basename(npz_path).replace(".npz", "")
        npz_name = os.path.basename(npz_path).replace(".npz", "")
        # Base name includes dataset and subject for easy identification
        base_name = f"{dataset}_{subject_id}_{npz_name}"
        has_dl = diffusion_models is not None
        
        def save_volume_figure(gt_vol, baseline_vol, fugue_vol, topup_vol,
                               cnn_vol, diff_vol, vdm_vol, t2_vol,
                               vol_name: str, norm_func, diff_vmax: float = 0.3):
            """Save a figure for one volume type with images and difference maps."""
            # Row 0: GT, Baseline, Fugue, Topup, [CNN, Diff], T2, VDM
            # Row 1: (empty), Diff-Baseline, Diff-Fugue, Diff-Topup, [Diff-CNN, Diff-Diff], (empty), (empty)
            
            ncols = 8 if has_dl else 6
            fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
            fig.suptitle(f"{base_name} - {vol_name}", fontsize=12)
            
            # Get central slice
            gt_slice = gt_vol[:, :, k]
            baseline_slice = baseline_vol[:, :, k]
            fugue_slice = fugue_vol[:, :, k]
            topup_slice = topup_vol[:, :, k]
            vdm_slice = vdm_vol[:, :, k]
            t2_slice = t2_vol[:, :, k]
            
            # Normalize slices
            gt_norm = norm_func(gt_slice)
            baseline_norm = norm_func(baseline_slice)
            fugue_norm = norm_func(fugue_slice)
            topup_norm = norm_func(topup_slice)
            
            # Row 0: Images
            axes[0, 0].imshow(gt_norm, cmap="gray")
            axes[0, 0].set_title("GT")
            
            axes[0, 1].imshow(baseline_norm, cmap="gray")
            axes[0, 1].set_title("Baseline")
            
            axes[0, 2].imshow(fugue_norm, cmap="gray")
            axes[0, 2].set_title("FUGUE")
            
            axes[0, 3].imshow(topup_norm, cmap="gray")
            axes[0, 3].set_title("TOPUP")
            
            col_idx = 4
            cnn_norm = diff_norm_img = None
            if has_dl and cnn_vol is not None:
                cnn_slice = cnn_vol[:, :, k]
                diff_slice = diff_vol[:, :, k]
                cnn_norm = norm_func(cnn_slice)
                diff_norm_img = norm_func(diff_slice)
                axes[0, col_idx].imshow(cnn_norm, cmap="gray")
                axes[0, col_idx].set_title("CNN")
                col_idx += 1
                axes[0, col_idx].imshow(diff_norm_img, cmap="gray")
                axes[0, col_idx].set_title("Diffusion")
                col_idx += 1
            
            axes[0, col_idx].imshow(norm01(t2_slice), cmap="gray")
            axes[0, col_idx].set_title("T2")
            col_idx += 1
            
            axes[0, col_idx].imshow(vdm_slice, cmap="RdBu_r", vmin=-20, vmax=20)
            axes[0, col_idx].set_title("VDM")
            
            # Row 1: Difference maps (method - GT)
            axes[1, 0].axis("off")  # No diff for GT
            
            # Compute differences in normalized space
            diff_baseline = baseline_norm - gt_norm
            diff_fugue = fugue_norm - gt_norm
            diff_topup = topup_norm - gt_norm
            
            im1 = axes[1, 1].imshow(diff_baseline, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            axes[1, 1].set_title("Baseline - GT")
            
            axes[1, 2].imshow(diff_fugue, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            axes[1, 2].set_title("FUGUE - GT")
            
            axes[1, 3].imshow(diff_topup, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            axes[1, 3].set_title("TOPUP - GT")
            
            col_idx = 4
            if has_dl and cnn_norm is not None:
                diff_cnn = cnn_norm - gt_norm
                diff_diffusion = diff_norm_img - gt_norm
                
                axes[1, col_idx].imshow(diff_cnn, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
                axes[1, col_idx].set_title("CNN - GT")
                col_idx += 1
                
                axes[1, col_idx].imshow(diff_diffusion, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
                axes[1, col_idx].set_title("Diffusion - GT")
                col_idx += 1
            
            axes[1, col_idx].axis("off")
            col_idx += 1
            if col_idx < ncols:
                axes[1, col_idx].axis("off")
            
            # Turn off axis ticks for all
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add colorbar for difference maps
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
            fig.colorbar(im1, cax=cbar_ax, label="Diff")
            
            plt.tight_layout(rect=[0, 0, 0.91, 1])
            out_path = os.path.join(out_dir, f"{base_name}_{vol_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        
        # Save separate figures for b50, b1400, ADC
        save_volume_figure(
            b50_gt, b50_pos, b50_fugue, b50_topup,
            b50_cnn, b50_diff,
            vdm_pos, t2,
            "b50", norm01, diff_vmax=0.3
        )
        
        save_volume_figure(
            b1400_gt, b1400_pos, b1400_fugue, b1400_topup,
            b1400_cnn, b1400_diff,
            vdm_pos, t2,
            "b1400", norm01, diff_vmax=0.3
        )
        
        save_volume_figure(
            adc_gt, adc_pos, adc_fugue, adc_topup,
            adc_cnn, adc_diff,
            vdm_pos, t2,
            "ADC", norm01_adc, diff_vmax=0.3
        )
        
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate distortion correction methods")
    ap.add_argument("--test_root", type=str,
                    default="/path/to/dgr_data/dwi_testset/pe_axis0",
                    help="Root directory containing test NPZ files")
    ap.add_argument("--output_dir", type=str,
                    default="/path/to/dgr_data/network_runs/eval_distortion_correction_test_vpred",
                    help="Output directory for results")
    ap.add_argument("--pe_axis", type=int, default=0,
                    help="Phase encoding axis (0=columns, 1=rows)")
    ap.add_argument("--save_vis", action="store_true",
                    help="Save visualizations for each subject")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of subjects for quick testing")
    
    # Diffusion model arguments
    ap.add_argument("--run_diffusion", action="store_true",
                    help="Run diffusion model inference")
    ap.add_argument("--diff_ckpt", type=str,
                    default="/path/to/dgr_data/network_runs/diffusion_clean_v2/diff_t2cnn_clean_epoch_092.pt",
                    help="Diffusion model checkpoint")
    ap.add_argument("--cnn_ckpt", type=str,
                    default="/path/to/dgr_data/network_runs/mageultra_dualb_axis01_quick_local_v25/mageultra_epoch_025.pt",
                    help="CNN model checkpoint")
    ap.add_argument("--radius", type=int, default=2, help="2.5D radius")
    ap.add_argument("--diff_steps", type=int, default=50, help="Diffusion sampling steps")
    ap.add_argument("--diff_strength", type=float, default=0.1, help="SDEdit strength")
    ap.add_argument("--t2_cond_channels", type=int, default=64)
    ap.add_argument("--cnn_base_channels", type=int, default=64)
    ap.add_argument("--cnn_latent_dim", type=int, default=8)
    ap.add_argument("--cnn_prompt_k", type=int, default=8)
    ap.add_argument("--cnn_prompt_temp", type=float, default=1.0)
    ap.add_argument("--num_gpus", type=int, default=None,
                    help="Number of GPUs for parallel inference (default: all available)")
    
    # VDM noise for FUGUE/TOPUP (simulate imperfect field map)
    ap.add_argument("--vdm_noise_std", type=float, default=0,
                    help="Std of noise to add to VDM (pixels). 0=perfect VDM,  -1-2=realistic")
    ap.add_argument("--vdm_noise_type", type=str, default="smooth", choices=["gaussian", "smooth"],
                    help="Noise type: 'gaussian' (i.i.d.) or 'smooth' (spatially correlated)")
    ap.add_argument("--vdm_noise_smooth_sigma", type=float, default=3.0,
                    help="Spatial smoothing sigma for smooth noise (correlation length)")
    
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all test NPZ files
    npz_files = glob.glob(os.path.join(args.test_root, "**", "*.npz"), recursive=True)
    npz_files = [f for f in npz_files if f.endswith(".npz")]
    npz_files.sort()
    
    if args.limit:
        npz_files = npz_files[:args.limit]
    
    print(f"Found {len(npz_files)} test subjects")
    print(f"PE axis: {args.pe_axis}")
    print(f"VDM noise: std={args.vdm_noise_std}, type={args.vdm_noise_type}, smooth_sigma={args.vdm_noise_smooth_sigma}")
    print(f"Output directory: {args.output_dir}")
    
    # Determine number of GPUs
    num_gpus = 1
    if args.run_diffusion:
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            num_gpus = min(args.num_gpus or available_gpus, available_gpus, len(npz_files))
            print(f"Using {num_gpus} GPU(s) for parallel inference")
        else:
            print("CUDA not available, using CPU")
            num_gpus = 1
    
    all_metrics: List[SubjectMetrics] = []
    
    if args.run_diffusion and num_gpus > 1:
        # Multi-GPU parallel processing
        print(f"\nLoading models on {num_gpus} GPUs...")
        
        # Pre-load models on each GPU
        gpu_models = {}
        for gpu_id in range(num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            print(f"  Loading models on GPU {gpu_id}...")
            try:
                gpu_models[gpu_id] = (
                    load_diffusion_models(
                        args.diff_ckpt,
                        args.cnn_ckpt,
                        device,
                        radius=args.radius,
                        t2_cond_channels=args.t2_cond_channels,
                        cnn_base_channels=args.cnn_base_channels,
                        cnn_latent_dim=args.cnn_latent_dim,
                        cnn_prompt_k=args.cnn_prompt_k,
                        cnn_prompt_temp=args.cnn_prompt_temp,
                    ),
                    device,
                )
            except Exception as e:
                print(f"  [ERROR] Failed to load on GPU {gpu_id}: {e}")
        
        # Distribute work across GPUs
        def process_on_gpu(args_tuple):
            npz_path, gpu_id, args_obj = args_tuple
            models, device = gpu_models[gpu_id]
            try:
                metrics = evaluate_single_subject(
                    npz_path,
                    pe_axis=args_obj.pe_axis,
                    diffusion_models=models,
                    device=device,
                    diff_steps=args_obj.diff_steps,
                    diff_strength=args_obj.diff_strength,
                    vdm_noise_std=args_obj.vdm_noise_std,
                    vdm_noise_type=args_obj.vdm_noise_type,
                    vdm_noise_smooth_sigma=args_obj.vdm_noise_smooth_sigma,
                )
                
                if args_obj.save_vis:
                    vis_dir = os.path.join(args_obj.output_dir, "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_visualization(npz_path, vis_dir, pe_axis=args_obj.pe_axis,
                                       diffusion_models=models, device=device)
                
                return metrics
            except Exception as e:
                print(f"[ERROR] GPU {gpu_id}: {npz_path}: {e}")
                return None
        
        # Assign subjects to GPUs round-robin
        tasks = [(npz_files[i], i % num_gpus, args) for i in range(len(npz_files))]
        
        # Process with thread pool (one thread per GPU)
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [executor.submit(process_on_gpu, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating (multi-GPU)"):
                result = future.result()
                if result is not None:
                    all_metrics.append(result)
    else:
        # Single GPU or CPU processing
        diffusion_models = None
        device = None
        if args.run_diffusion:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading diffusion models on {device}...")
            print(f"  CNN checkpoint: {args.cnn_ckpt}")
            print(f"  Diffusion checkpoint: {args.diff_ckpt}")
            try:
                diffusion_models = load_diffusion_models(
                    args.diff_ckpt,
                    args.cnn_ckpt,
                    device,
                    radius=args.radius,
                    t2_cond_channels=args.t2_cond_channels,
                    cnn_base_channels=args.cnn_base_channels,
                    cnn_latent_dim=args.cnn_latent_dim,
                    cnn_prompt_k=args.cnn_prompt_k,
                    cnn_prompt_temp=args.cnn_prompt_temp,
                )
                print("  Models loaded successfully!")
            except Exception as e:
                print(f"  [ERROR] Failed to load models: {e}")
                diffusion_models = None
        
        for npz_path in tqdm(npz_files, desc="Evaluating subjects"):
            try:
                metrics = evaluate_single_subject(
                    npz_path,
                    pe_axis=args.pe_axis,
                    diffusion_models=diffusion_models,
                    device=device,
                    diff_steps=args.diff_steps,
                    diff_strength=args.diff_strength,
                    vdm_noise_std=args.vdm_noise_std,
                    vdm_noise_type=args.vdm_noise_type,
                    vdm_noise_smooth_sigma=args.vdm_noise_smooth_sigma,
                )
                all_metrics.append(metrics)
                
                if args.save_vis:
                    vis_dir = os.path.join(args.output_dir, "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_visualization(npz_path, vis_dir, pe_axis=args.pe_axis,
                                       diffusion_models=diffusion_models, device=device)
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {npz_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Aggregate and save results
    summary = aggregate_metrics(all_metrics)
    print_summary_table(summary)

    # P-values (paired Wilcoxon signed-rank test)
    try:
        pvals = compute_pvalues(all_metrics)
        print_pvalue_table(pvals)
        summary["p_values"] = pvals
    except Exception as e:
        print(f"[WARN] Failed to compute p-values: {e}")
    
    results_json = os.path.join(args.output_dir, "metrics_summary.json")
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {results_json}")
    
    per_subject_json = os.path.join(args.output_dir, "metrics_per_subject.json")
    per_subject_data = []
    for m in all_metrics:
        d = {"subject_id": m.subject_id, "dataset": m.dataset}
        for attr in ["baseline", "fugue", "topup_oracle", "topup_simple", "cnn", "diffusion"]:
            for vol in ["b50", "b1400", "adc"]:
                key = f"{attr}_{vol}"
                if hasattr(m, key):
                    mr = getattr(m, key)
                    d[f"{key}_psnr"] = mr.psnr
                    d[f"{key}_ssim"] = mr.ssim
                    d[f"{key}_nmse"] = mr.nmse
                    d[f"{key}_mae"] = mr.mae
                    d[f"{key}_psnr_center"] = mr.psnr_center
                    d[f"{key}_ssim_center"] = mr.ssim_center
                    d[f"{key}_nmse_center"] = mr.nmse_center
                    d[f"{key}_mae_center"] = mr.mae_center
        # Add timing info
        d["time_fugue"] = m.time_fugue
        d["time_topup_oracle"] = m.time_topup_oracle
        d["time_topup_simple"] = m.time_topup_simple
        d["time_cnn"] = m.time_cnn
        d["time_diffusion"] = m.time_diffusion
        per_subject_data.append(d)
    
    with open(per_subject_json, "w") as f:
        json.dump(per_subject_data, f, indent=2)
    print(f"Per-subject metrics saved to: {per_subject_json}")


if __name__ == "__main__":
    main()
