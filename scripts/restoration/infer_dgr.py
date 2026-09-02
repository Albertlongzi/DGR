"""
Inference script for T2+CNN conditional diffusion model.

Pipeline:
  1. Load CNN (MageUltra) for Stage 1 distortion correction
  2. Load Diffusion model (DiffusionUNetT2AndCNN) for Stage 2 refinement
  3. For each slice:
     a. Run CNN to get initial correction (b50_cnn, adc_cnn)
     b. Add noise to CNN output (SDEdit style)
     c. Denoise with (T2, CNN_output) conditioning
  4. Reconstruct high-b from low-b and ADC
  5. Save results and visualization panels

Supports multi-GPU inference via --num_gpus flag (default: all available).
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Optional, Tuple, List
import multiprocessing as mp

import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dgr.models.diffusion_unet_diffusers import DiffusionUNetT2AndCNN
from dgr.models.phc_e2e_mageultra_net import PHCE2EMageUltraNet
from dgr.inference.sampler_diffusers import sample_with_t2_and_cnn
from dgr.data.npz_variant_dualb_dataset import _stack_2p5d
from dgr.conditioning.t2_conditioning import process_t2_conditioning


def _load_np_payload(path: str) -> Dict[str, np.ndarray]:
    if path.endswith(".npy"):
        obj = np.load(path, allow_pickle=True)
        payload = obj.item() if isinstance(obj, np.ndarray) else obj
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected payload in {path}: {type(payload)}")
        return payload
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _compute_adc_from_two_b(
    s_low: np.ndarray,
    b_low: float,
    s_high: np.ndarray,
    b_high: float,
    eps: float = 1e-6,
) -> np.ndarray:
    assert s_low.shape == s_high.shape, f"ADC shape mismatch: {s_low.shape} vs {s_high.shape}"
    den = float(b_high - b_low) if float(b_high - b_low) != 0 else 1.0
    sl = np.clip(s_low.astype(np.float32), eps, None)
    sh = np.clip(s_high.astype(np.float32), eps, None)
    adc = (np.log(sl) - np.log(sh)) / den
    return np.clip(adc, 0.0, 0.003).astype(np.float32)


def _percentile_norm01(vol: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> np.ndarray:
    v = vol.astype(np.float32)
    lo = float(np.percentile(v, p1))
    hi = float(np.percentile(v, p99))
    if hi <= lo:
        hi = lo + 1e-6
    out = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _percentile_norm01_with_stats(
    vol: np.ndarray,
    p1: float = 1.0,
    p99: float = 99.0,
) -> Tuple[np.ndarray, float, float]:
    v = vol.astype(np.float32)
    lo = float(np.percentile(v, p1))
    hi = float(np.percentile(v, p99))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1e-6
    if hi <= lo:
        hi = lo + 1e-6
    vn = (v - lo) / (hi - lo)
    return np.clip(vn, 0.0, 1.0).astype(np.float32), lo, hi


def _denorm_from_stats(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (vol.astype(np.float32) * (hi - lo) + lo).astype(np.float32)


def _adc_vmin_vmax(vol: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> Tuple[float, float]:
    v = vol.astype(np.float32)
    lo = float(np.percentile(v, p_lo))
    hi = float(np.percentile(v, p_hi))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1e-6
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _extract_scalar(payload: Dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in payload:
        return default
    arr = np.array(payload[key]).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return default
    return float(arr[0])


def _maybe_use_ms(vol: Optional[np.ndarray], ref_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if vol is None:
        return None
    arr = np.asarray(vol).astype(np.float32)
    if arr.shape != ref_shape:
        return None
    return arr


def _prepare_scheduler(cfg: Dict, sampler: str) -> object:
    if sampler == "dpmsolver":
        return DPMSolverMultistepScheduler.from_config(dict(cfg))
    if sampler == "ddpm":
        return DDPMScheduler(**cfg)
    return DDIMScheduler(**cfg)


def _save_panel_png_with_stages(
    out_path: str,
    b50_in: np.ndarray,
    b50_cnn: np.ndarray,  # kept for API compatibility but not plotted
    b50_diff: np.ndarray,
    b14_in: np.ndarray,
    b14_cnn: np.ndarray,  # kept for API compatibility but not plotted
    b14_diff: np.ndarray,
    t2: np.ndarray,
    adc_in: np.ndarray,
    adc_cnn: np.ndarray,  # kept for API compatibility but not plotted
    adc_diff: np.ndarray,
    adc_gt: Optional[np.ndarray],
    b50_gt: Optional[np.ndarray],
    b14_gt: Optional[np.ndarray],
    k: int,
    gamma_b14: float,
    gamma_adc: float,
    adc_plo: float = 2.0,
    adc_phi: float = 98.0,
) -> None:
    """
    Save visualization panel for radiologist blind review:
    - Row 0: b50 (A, B, T2) - no model info in labels
    - Row 1: b1400 (A, B, T2)
    - Row 2: ADC (A, B, T2)
    Note: CNN results are not plotted (kept in API for compatibility).
    Each image now includes a colorbar showing the gray level range.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except Exception as exc:
        print(f"[WARN] Matplotlib unavailable, skip panel: {exc}")
        return

    def _disp_b14(img: np.ndarray) -> np.ndarray:
        img01 = np.clip(img, 0.0, 1.0)
        g = gamma_b14 if gamma_b14 > 0 else 1.0
        return np.power(img01, g)

    def _disp_adc(img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        vmin, vmax = _adc_vmin_vmax(img, p_lo=adc_plo, p_hi=adc_phi)
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
        g = gamma_adc if gamma_adc > 0 else 1.0
        img_gamma = np.power(img_norm, g)
        return img_gamma * (vmax - vmin) + vmin, vmin, vmax

    def _add_colorbar(ax, im, label: str = ""):
        """Add a colorbar to the right of the given axis."""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        if label:
            cbar.set_label(label, fontsize=8)
        return cbar

    # Columns: A (input), B (diff output), T2  (no CNN column)
    ncols = 3
    fig, axes = plt.subplots(3, ncols, figsize=(5 * ncols, 12))
    t2_slice = t2[:, :, k]

    # Row 0: b50 (normalized 0-1)
    im00 = axes[0, 0].imshow(b50_in[:, :, k], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("b50 A")
    axes[0, 0].axis("off")
    _add_colorbar(axes[0, 0], im00, "Intensity")

    im01 = axes[0, 1].imshow(b50_diff[:, :, k], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("b50 B")
    axes[0, 1].axis("off")
    _add_colorbar(axes[0, 1], im01, "Intensity")

    im02 = axes[0, 2].imshow(t2_slice, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("T2")
    axes[0, 2].axis("off")
    _add_colorbar(axes[0, 2], im02, "Intensity")

    # Row 1: b1400 (gamma-corrected, displayed 0-1)
    im10 = axes[1, 0].imshow(_disp_b14(b14_in[:, :, k]), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("b1400 A")
    axes[1, 0].axis("off")
    _add_colorbar(axes[1, 0], im10, "Intensity")

    im11 = axes[1, 1].imshow(_disp_b14(b14_diff[:, :, k]), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("b1400 B")
    axes[1, 1].axis("off")
    _add_colorbar(axes[1, 1], im11, "Intensity")

    im12 = axes[1, 2].imshow(t2_slice, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].axis("off")
    _add_colorbar(axes[1, 2], im12, "Intensity")

    # Row 2: ADC (with percentile-based vmin/vmax)
    adc_in_disp, vmin_in, vmax_in = _disp_adc(adc_in[:, :, k])
    im20 = axes[2, 0].imshow(adc_in_disp, cmap="gray", vmin=vmin_in, vmax=vmax_in)
    axes[2, 0].set_title("ADC A")
    axes[2, 0].axis("off")
    _add_colorbar(axes[2, 0], im20, "ADC")

    adc_diff_disp, vmin_diff, vmax_diff = _disp_adc(adc_diff[:, :, k])
    im21 = axes[2, 1].imshow(adc_diff_disp, cmap="gray", vmin=vmin_diff, vmax=vmax_diff)
    axes[2, 1].set_title("ADC B")
    axes[2, 1].axis("off")
    _add_colorbar(axes[2, 1], im21, "ADC")

    im22 = axes[2, 2].imshow(t2_slice, cmap="gray", vmin=0, vmax=1)
    axes[2, 2].axis("off")
    _add_colorbar(axes[2, 2], im22, "Intensity")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _prepare_case_payload(
    path: str,
    b_low: float,
    b_high: float,
) -> Dict[str, np.ndarray]:
    payload = _load_np_payload(path)

    def _get_first(keys: List[str]) -> np.ndarray:
        for key in keys:
            if key in payload:
                return np.asarray(payload[key]).astype(np.float32)
        raise KeyError(f"{path} missing keys {keys}")

    b50_in = _get_first(["dwi_b50_in", "dwi_b50", "dwi_in"])
    b14_in = _get_first(["dwi_b1400_in", "dwi_b1400"])
    t2 = None
    if "t2_n" in payload:
        t2 = np.asarray(payload["t2_n"]).astype(np.float32)
    elif "t2" in payload:
        t2 = _percentile_norm01(np.asarray(payload["t2"]).astype(np.float32))
    else:
        t2 = np.zeros_like(b50_in, dtype=np.float32)

    lo_in = _extract_scalar(payload, "lo_in", float(np.percentile(b50_in, 1.0)))
    hi_in = _extract_scalar(payload, "hi_in", float(np.percentile(b50_in, 99.0)))
    if hi_in <= lo_in:
        hi_in = lo_in + 1e-6

    adc_in_phy = _compute_adc_from_two_b(b50_in, b_low, b14_in, b_high)
    lo_ai = _extract_scalar(payload, "lo_ai", float(np.percentile(adc_in_phy, 1.0)))
    hi_ai = _extract_scalar(payload, "hi_ai", float(np.percentile(adc_in_phy, 99.0)))
    if hi_ai <= lo_ai:
        hi_ai = lo_ai + 1e-6

    # Try to load ground truth from various keys
    b50_gt = payload.get("dwi_b50_gt")
    b14_gt = payload.get("dwi_b1400_gt")
    gt_source = "dwi_*_gt" if b50_gt is not None else None
    # Also try MS (multi-shot) volumes as pseudo-GT
    if b50_gt is None and "dwi_ms_b50" in payload:
        b50_gt = payload["dwi_ms_b50"]
        gt_source = "dwi_ms_*"
    if b14_gt is None and "dwi_ms_b1400" in payload:
        b14_gt = payload["dwi_ms_b1400"]
        if gt_source is None:
            gt_source = "dwi_ms_*"
    b50_gt = _maybe_use_ms(b50_gt, b50_in.shape)
    b14_gt = _maybe_use_ms(b14_gt, b14_in.shape)

    if b50_gt is not None:
        lo_gt = _extract_scalar(payload, "lo_gt", float(np.percentile(b50_gt, 1.0)))
        hi_gt = _extract_scalar(payload, "hi_gt", float(np.percentile(b50_gt, 99.0)))
        if hi_gt <= lo_gt:
            hi_gt = lo_gt + 1e-6
        adc_gt_phy = _compute_adc_from_two_b(b50_gt, b_low, b14_gt, b_high) if b14_gt is not None else None
        if adc_gt_phy is not None:
            lo_ag = _extract_scalar(payload, "lo_ag", float(np.percentile(adc_gt_phy, 1.0)))
            hi_ag = _extract_scalar(payload, "hi_ag", float(np.percentile(adc_gt_phy, 99.0)))
            if hi_ag <= lo_ag:
                hi_ag = lo_ag + 1e-6
        else:
            lo_ag = hi_ag = None
    else:
        lo_gt = hi_gt = lo_ag = hi_ag = None
        adc_gt_phy = None

    return {
        "b50_in_raw": b50_in.astype(np.float32),
        "b14_in_raw": b14_in.astype(np.float32),
        "b50_gt_raw": b50_gt.astype(np.float32) if b50_gt is not None else None,
        "b14_gt_raw": b14_gt.astype(np.float32) if b14_gt is not None else None,
        "adc_in_phy": adc_in_phy.astype(np.float32),
        "adc_gt_phy": adc_gt_phy.astype(np.float32) if adc_gt_phy is not None else None,
        "t2": t2.astype(np.float32),
        "lo_in": float(lo_in),
        "hi_in": float(hi_in),
        "lo_ai": float(lo_ai),
        "hi_ai": float(hi_ai),
        "lo_gt": float(lo_gt) if lo_gt is not None else None,
        "hi_gt": float(hi_gt) if hi_gt is not None else None,
        "lo_ag": float(lo_ag) if lo_ag is not None else None,
        "hi_ag": float(hi_ag) if hi_ag is not None else None,
        "gt_source": gt_source,
    }


def _load_checkpoint(path: str) -> Tuple[Dict, Dict]:
    """Load either a training checkpoint (.pt) or a released .safetensors file.

    Returns ``(state_dict, meta)``. For a ``.pt`` the metadata is the checkpoint dict
    itself; for a ``.safetensors`` it is the sibling ``config.json`` written by
    ``tools/export_checkpoint.py``, which carries the same architecture and scheduler
    information with the optimizer state stripped.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path)
        meta = {}
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            meta = dict(cfg.get("arch") or {})
            if cfg.get("noise_scheduler"):
                meta["noise_scheduler_config"] = cfg["noise_scheduler"]
        return state, meta

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    meta = dict(ckpt.get("args") or {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and "noise_scheduler_config" in ckpt:
        meta["noise_scheduler_config"] = ckpt["noise_scheduler_config"]
    return state, meta


def _build_scheduler_config(meta: Dict) -> Dict:
    if "noise_scheduler_config" in meta:
        return dict(meta["noise_scheduler_config"])
    return {
        "num_train_timesteps": 1000,
        "beta_schedule": "linear",
        "prediction_type": "epsilon",
    }


def _load_cnn_model(args, device: torch.device) -> PHCE2EMageUltraNet:
    """Load MageUltra CNN for Stage 1 distortion correction."""
    dwi_ch = (2 * args.radius + 1)
    t2_ch = (2 * args.radius + 1)
    
    cnn = PHCE2EMageUltraNet(
        dwi_channels=dwi_ch,
        t2_channels=t2_ch,
        base_channels=args.cnn_base_channels,
        latent_dim=args.cnn_latent_dim,
        prompt_k=args.cnn_prompt_k,
        prompt_temp=args.cnn_prompt_temp,
    )
    
    state, _ = _load_checkpoint(args.cnn_ckpt)
    cnn.load_state_dict(state)
    cnn = cnn.to(device).eval()
    
    return cnn


def _load_diffusion_model(args, device: torch.device) -> Tuple[DiffusionUNetT2AndCNN, Dict]:
    """Load T2+CNN conditional diffusion model for Stage 2 refinement."""
    model_state, meta = _load_checkpoint(args.ckpt)
    t2_cond_channels = meta.get("t2_cond_channels", args.t2_cond_channels)
    
    model = DiffusionUNetT2AndCNN(
        fusion_channels=t2_cond_channels,
    )
    model.load_state_dict(model_state)
    model = model.to(device).eval()
    
    scheduler_cfg = _build_scheduler_config(meta)
    
    return model, scheduler_cfg


def _reconstruct_b1400(b50: np.ndarray, adc: np.ndarray, b_low: float, b_high: float) -> np.ndarray:
    """Reconstruct high-b from low-b and ADC: S_high = S_low * exp(-ADC * delta_b)"""
    delta_b = float(b_high - b_low)
    adc_clamped = np.clip(adc, 0.0, 0.005)  # Clamp to avoid overflow
    return (b50 * np.exp(-adc_clamped * delta_b)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for T2+CNN conditional diffusion model")
    
    # Checkpoints
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to diffusion model checkpoint")
    parser.add_argument("--cnn_ckpt", type=str, 
                        default="/path/to/dgr_data/network_runs/mageultra_dualb_axis01_quick_local_v23/mageultra_best.pt",
                        help="Path to CNN (MageUltra) checkpoint for Stage 1")
    
    # Data
    parser.add_argument("--test_root", type=str, nargs="+",
                        default=["/path/to/dgr_data/validation_data2_TSE",
                                 "/path/to/dgr_data/preprocessed_test_npz2"],
                        help="Directory(s) containing NPZ/NPY files for inference")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--max_cases", type=int, default=None)
    
    # Diffusion sampling
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.3,
                        help="SDEdit strength: 0.0=no refinement, 1.0=full reconstruction")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm", "dpmsolver"], default="dpmsolver")
    
    # Physics
    parser.add_argument("--b_low", type=float, default=50.0)
    parser.add_argument("--b_high", type=float, default=1400.0)
    
    # Output
    parser.add_argument("--slice_mode", type=str, choices=["central", "all"], default="all")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_slices", action="store_true", help="Save per-slice PNG panels")
    
    # Visualization
    parser.add_argument("--gamma_b14", type=float, default=0.4)
    parser.add_argument("--gamma_adc", type=float, default=0.5,
                        help="ADC gamma: <1 brightens darks, >1 darkens darks")
    parser.add_argument("--adc_plo", type=float, default=2.0,
                        help="Lower percentile for ADC window (default: 2)")
    parser.add_argument("--adc_phi", type=float, default=98.0,
                        help="Upper percentile for ADC window (default: 98)")
    
    # Model architecture
    parser.add_argument("--t2_cond_channels", type=int, default=64,
                        help="Channel width for T2 conditioning module")
    parser.add_argument("--t2_contrast_mod", type=str, default="none",
                        choices=["none", "lcn", "canny_binary", "canny_grayscale"])
    parser.add_argument("--t2_canny_low", type=int, default=50)
    parser.add_argument("--t2_canny_high", type=int, default=150)
    
    # CNN architecture (must match training)
    parser.add_argument("--cnn_base_channels", type=int, default=64)
    parser.add_argument("--cnn_latent_dim", type=int, default=8)
    parser.add_argument("--cnn_prompt_k", type=int, default=8)
    parser.add_argument("--cnn_prompt_temp", type=float, default=1.0)
    
    # Multi-GPU
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use. Default: all available.")
    
    return parser.parse_args()


def _process_single_case(
    path: str,
    test_roots: List[str],
    args: argparse.Namespace,
    cnn_model,
    diff_model,
    scheduler_cfg: Dict,
    device: torch.device,
    gpu_id: int = 0,
) -> None:
    """Process a single case (called by worker process or main process)."""
    # Determine which test_root this file belongs to
    test_root_for_file = test_roots[0]
    for tr in test_roots:
        if path.startswith(tr):
            test_root_for_file = tr
            break
    
    rel = os.path.relpath(path, test_root_for_file)
    name = os.path.splitext(rel.replace(os.sep, "_"))[0]
    print(f"\n[GPU {gpu_id}] {'='*50}")
    print(f"[GPU {gpu_id}] Processing: {name}")
    print(f"[GPU {gpu_id}]   Source: {path}")

    try:
        case = _prepare_case_payload(path, args.b_low, args.b_high)
    except Exception as e:
        print(f"[GPU {gpu_id}]   [SKIP] Failed to load: {e}")
        return

    b50_raw = case["b50_in_raw"]
    b14_raw = case["b14_in_raw"]
    t2 = case["t2"]
    adc_in_phy = case["adc_in_phy"]
    lo_in, hi_in = case["lo_in"], case["hi_in"]
    lo_ai, hi_ai = case["lo_ai"], case["hi_ai"]

    scale_in = max(1e-6, hi_in - lo_in)
    scale_ai = max(1e-6, hi_ai - lo_ai)
    b50_in_norm = np.clip((b50_raw - lo_in) / scale_in, 0.0, 1.0).astype(np.float32)
    b14_in_disp = np.clip((b14_raw - lo_in) / scale_in, 0.0, 1.0).astype(np.float32)
    adc_in_norm = np.clip((adc_in_phy - lo_ai) / scale_ai, 0.0, 1.0).astype(np.float32)

    # Prepare GT for visualization
    b50_gt = case["b50_gt_raw"]
    b14_gt = case["b14_gt_raw"]
    adc_gt_phy = case["adc_gt_phy"]
    if b50_gt is not None:
        lo_gt = case["lo_gt"] if case["lo_gt"] is not None else float(np.percentile(b50_gt, 1.0))
        hi_gt = case["hi_gt"] if case["hi_gt"] is not None else float(np.percentile(b50_gt, 99.0))
        scale_gt = max(1e-6, hi_gt - lo_gt)
        b50_gt_disp = np.clip((b50_gt - lo_gt) / scale_gt, 0.0, 1.0).astype(np.float32)
        b14_gt_disp = np.clip((b14_gt - lo_gt) / scale_gt, 0.0, 1.0).astype(np.float32) if b14_gt is not None else None
        if adc_gt_phy is not None:
            lo_ag = case["lo_ag"] if case["lo_ag"] is not None else float(np.percentile(adc_gt_phy, 1.0))
            hi_ag = case["hi_ag"] if case["hi_ag"] is not None else float(np.percentile(adc_gt_phy, 99.0))
            scale_ag = max(1e-6, hi_ag - lo_ag)
            adc_gt_norm = np.clip((adc_gt_phy - lo_ag) / scale_ag, 0.0, 1.0).astype(np.float32)
        else:
            adc_gt_norm = None
    else:
        b50_gt_disp = None
        b14_gt_disp = None
        adc_gt_norm = None

    H, W, Z = b50_in_norm.shape
    print(f"[GPU {gpu_id}]   Shape: {H}x{W}x{Z}")
    gt_source = case.get("gt_source")
    if b50_gt is not None:
        print(f"[GPU {gpu_id}]   Has GT: Yes (source: {gt_source})")
    else:
        print(f"[GPU {gpu_id}]   Has GT: No")

    # Initialize output volumes
    b50_cnn_norm = np.zeros_like(b50_in_norm, dtype=np.float32)
    adc_cnn_norm = np.zeros_like(adc_in_norm, dtype=np.float32)
    b50_diff_norm = np.zeros_like(b50_in_norm, dtype=np.float32)
    adc_diff_norm = np.zeros_like(adc_in_norm, dtype=np.float32)

    scheduler = _prepare_scheduler(scheduler_cfg, args.sampler)

    print(f"[GPU {gpu_id}]   Running inference on {Z} slices...")
    for k in range(Z):
        # Build 2.5D stacks
        b50_stack = _stack_2p5d(b50_in_norm, k, args.radius)[None, ...]  # [1, C, H, W]
        adc_stack = _stack_2p5d(adc_in_norm, k, args.radius)[None, ...]
        t2_stack = _stack_2p5d(t2, k, args.radius)[None, ...]
        
        with torch.no_grad():
            # Stage 1: CNN forward
            cnn_out = cnn_model(
                torch.from_numpy(b50_stack).to(device),
                torch.from_numpy(adc_stack).to(device),
                torch.from_numpy(t2_stack).to(device),
                None,  # VDM is None for TSE
            )
            b50_cnn_k = cnn_out["I_out_b50"][0, 0].cpu().numpy()
            adc_cnn_k = cnn_out["I_out_adc"][0, 0].cpu().numpy()
            b50_cnn_norm[:, :, k] = np.clip(b50_cnn_k, 0.0, 1.0)
            adc_cnn_norm[:, :, k] = np.clip(adc_cnn_k, 0.0, 1.0)
            
            # Stage 2: Diffusion refinement
            # Prepare CNN output as init_latent and conditioning
            cnn_output = torch.cat([
                cnn_out["I_out_b50"],  # [1, 1, H, W]
                cnn_out["I_out_adc"],  # [1, 1, H, W]
            ], dim=1)  # [1, 2, H, W]
            
            # Prepare batch for sampler
            batch: Dict[str, torch.Tensor] = {
                "t2_stack": torch.from_numpy(t2_stack).to(device),
                "cnn_output": cnn_output,  # For conditioning
                "cnn_init": cnn_output,    # As init_latent for SDEdit
            }
            
            # Run diffusion sampling
            diff_out = sample_with_t2_and_cnn(
                diff_model,
                scheduler,
                batch,
                steps=args.steps,
                strength=args.strength,
                eta=args.eta,
            )
            diff_np = diff_out.squeeze(0).cpu().numpy()
            b50_diff_norm[:, :, k] = np.clip(diff_np[0], 0.0, 1.0)
            adc_diff_norm[:, :, k] = np.clip(diff_np[1], 0.0, 1.0)

    # De-normalize to physical scale
    b50_cnn_phy = _denorm_from_stats(b50_cnn_norm, lo_in, hi_in)
    adc_cnn_phy = _denorm_from_stats(adc_cnn_norm, lo_ai, hi_ai)
    b50_diff_phy = _denorm_from_stats(b50_diff_norm, lo_in, hi_in)
    adc_diff_phy = _denorm_from_stats(adc_diff_norm, lo_ai, hi_ai)

    # Reconstruct b1400 from b50 and ADC
    b14_cnn_phy = _reconstruct_b1400(b50_cnn_phy, adc_cnn_phy, args.b_low, args.b_high)
    b14_diff_phy = _reconstruct_b1400(b50_diff_phy, adc_diff_phy, args.b_low, args.b_high)
    
    # Normalize for display
    b14_cnn_disp = np.clip((b14_cnn_phy - lo_in) / scale_in, 0.0, 1.0).astype(np.float32)
    b14_diff_disp = np.clip((b14_diff_phy - lo_in) / scale_in, 0.0, 1.0).astype(np.float32)

    # Compute metrics if GT available
    if b50_gt is not None:
        mae_b50_cnn = float(np.mean(np.abs(b50_cnn_phy - b50_gt)))
        mae_b50_diff = float(np.mean(np.abs(b50_diff_phy - b50_gt)))
        print(f"[GPU {gpu_id}]   MAE(b50): CNN={mae_b50_cnn:.6f}, DIFF={mae_b50_diff:.6f}")
        if adc_gt_phy is not None:
            mae_adc_cnn = float(np.mean(np.abs(adc_cnn_phy - adc_gt_phy)))
            mae_adc_diff = float(np.mean(np.abs(adc_diff_phy - adc_gt_phy)))
            print(f"[GPU {gpu_id}]   MAE(ADC): CNN={mae_adc_cnn:.6f}, DIFF={mae_adc_diff:.6f}")

    # Save NPZ
    if args.save_npz:
        out_npz = os.path.join(args.out_dir, f"{name}_pred.npz")
        save_dict = {
            # Stage 1 (CNN)
            "dwi_b50_cnn": b50_cnn_phy.astype(np.float32),
            "dwi_b1400_cnn": b14_cnn_phy.astype(np.float32),
            "adc_cnn": adc_cnn_phy.astype(np.float32),
            # Stage 2 (Diffusion)
            "dwi_b50_diff": b50_diff_phy.astype(np.float32),
            "dwi_b1400_diff": b14_diff_phy.astype(np.float32),
            "adc_diff": adc_diff_phy.astype(np.float32),
            # Input
            "dwi_b50_in": b50_raw.astype(np.float32),
            "dwi_b1400_in": b14_raw.astype(np.float32),
            "adc_in": adc_in_phy.astype(np.float32),
            # Stats
            "lo_in": np.array([lo_in], dtype=np.float32),
            "hi_in": np.array([hi_in], dtype=np.float32),
            "lo_ai": np.array([lo_ai], dtype=np.float32),
            "hi_ai": np.array([hi_ai], dtype=np.float32),
        }
        if b50_gt is not None:
            save_dict["dwi_b50_gt"] = b50_gt.astype(np.float32)
        if b14_gt is not None:
            save_dict["dwi_b1400_gt"] = b14_gt.astype(np.float32)
        if adc_gt_phy is not None:
            save_dict["adc_gt"] = adc_gt_phy.astype(np.float32)
        np.savez_compressed(out_npz, **save_dict)
        print(f"[GPU {gpu_id}]   Saved NPZ: {out_npz}")

    # Save visualization panel (central slice)
    k_vis = Z // 2
    panel_png = os.path.join(args.out_dir, f"{name}_panel.png")
    _save_panel_png_with_stages(
        panel_png,
        b50_in_norm, b50_cnn_norm, b50_diff_norm,
        b14_in_disp, b14_cnn_disp, b14_diff_disp,
        t2,
        adc_in_norm, adc_cnn_norm, adc_diff_norm,
        adc_gt_norm,
        b50_gt_disp, b14_gt_disp,
        k_vis,
        args.gamma_b14, args.gamma_adc,
        adc_plo=args.adc_plo, adc_phi=args.adc_phi,
    )
    print(f"[GPU {gpu_id}]   Saved panel: {panel_png}")

    # Save per-slice panels
    if args.save_slices:
        vis_dir = os.path.join(args.out_dir, f"{name}_slices")
        os.makedirs(vis_dir, exist_ok=True)
        for k in range(Z):
            slice_png = os.path.join(vis_dir, f"slice_{k:03d}.png")
            _save_panel_png_with_stages(
                slice_png,
                b50_in_norm, b50_cnn_norm, b50_diff_norm,
                b14_in_disp, b14_cnn_disp, b14_diff_disp,
                t2,
                adc_in_norm, adc_cnn_norm, adc_diff_norm,
                adc_gt_norm,
                b50_gt_disp, b14_gt_disp,
                k,
                args.gamma_b14, args.gamma_adc,
                adc_plo=args.adc_plo, adc_phi=args.adc_phi,
            )
        print(f"[GPU {gpu_id}]   Saved slices: {vis_dir}")


def _gpu_worker(
    gpu_id: int,
    paths: List[str],
    test_roots: List[str],
    args: argparse.Namespace,
) -> None:
    """Worker function for multi-GPU inference. Each worker handles a subset of cases on one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load models on this GPU
    cnn_model = _load_cnn_model(args, device)
    diff_model, scheduler_cfg = _load_diffusion_model(args, device)
    
    print(f"[GPU {gpu_id}] Loaded models, processing {len(paths)} cases...")
    
    for path in paths:
        _process_single_case(
            path, test_roots, args, cnn_model, diff_model, scheduler_cfg, device, gpu_id
        )
    
    print(f"[GPU {gpu_id}] Finished all {len(paths)} cases.")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine number of GPUs
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus is None:
        num_gpus = max(1, num_gpus_available)
    else:
        num_gpus = min(args.num_gpus, num_gpus_available) if num_gpus_available > 0 else 1
    
    print(f"Available GPUs: {num_gpus_available}, Using: {num_gpus}")

    # Collect test files from all test_root directories
    npz_paths = []
    test_roots = args.test_root if isinstance(args.test_root, list) else [args.test_root]
    for test_root in test_roots:
        paths = sorted(glob.glob(os.path.join(test_root, "**", "*.npz"), recursive=True))
        paths += sorted(glob.glob(os.path.join(test_root, "**", "*.npy"), recursive=True))
        npz_paths.extend(paths)
        print(f"  Found {len(paths)} files in {test_root}")
    
    if not npz_paths:
        raise RuntimeError(f"No NPZ/NPY files found under {test_roots}")
    if args.max_cases is not None and args.max_cases > 0:
        npz_paths = npz_paths[:args.max_cases]
    print(f"\nTotal: {len(npz_paths)} cases to process")

    if num_gpus <= 1 or len(npz_paths) == 1:
        # Single GPU mode
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using single device: {device}")
        
        print(f"\n[Stage 1] Loading CNN (MageUltra) from: {args.cnn_ckpt}")
        cnn_model = _load_cnn_model(args, device)
        print(f"  CNN config: base_channels={args.cnn_base_channels}, latent_dim={args.cnn_latent_dim}, "
              f"prompt_k={args.cnn_prompt_k}, prompt_temp={args.cnn_prompt_temp}")
        
        print(f"\n[Stage 2] Loading Diffusion model from: {args.ckpt}")
        diff_model, scheduler_cfg = _load_diffusion_model(args, device)
        print(f"  Diffusion config: t2_cond_channels={args.t2_cond_channels}, "
              f"steps={args.steps}, strength={args.strength}, sampler={args.sampler}")
        
        for path in npz_paths:
            _process_single_case(
                path, test_roots, args, cnn_model, diff_model, scheduler_cfg, device, gpu_id=0
            )
    else:
        # Multi-GPU mode using multiprocessing with spawn (required for CUDA)
        print(f"\n=== Multi-GPU Mode: {num_gpus} GPUs ===")
        
        # Split paths among GPUs
        paths_per_gpu = [[] for _ in range(num_gpus)]
        for i, path in enumerate(npz_paths):
            paths_per_gpu[i % num_gpus].append(path)
        
        for gpu_id in range(num_gpus):
            print(f"  GPU {gpu_id}: {len(paths_per_gpu[gpu_id])} cases")
        
        # Use 'spawn' start method for CUDA compatibility
        ctx = mp.get_context('spawn')
        
        # Launch worker processes
        processes = []
        for gpu_id in range(num_gpus):
            if len(paths_per_gpu[gpu_id]) == 0:
                continue
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, paths_per_gpu[gpu_id], test_roots, args),
            )
            p.start()
            processes.append(p)
        
        # Wait for all workers to finish
        for p in processes:
            p.join()

    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
