"""
Clean-Image-Prediction Diffusion Training Script

Key difference from standard noise-prediction diffusion:
- prediction_type="sample": Network directly predicts clean image, not noise
- Training loss: MSE(predicted_clean, target_clean) instead of MSE(predicted_noise, noise)
- The model learns on the anatomical manifold, not in noise space

Reference: arXiv:2511.13720 - Learning on the anatomical manifold for medical imaging
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler, ConcatDataset
from typing import Dict, Optional, Tuple
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dgr.data.npz_variant_dualb_dataset import NPZVariantDualBDataset
from dgr.models.diffusion_unet_diffusers import DiffusionUNetT2AndCNN
from dgr.inference.sampler_diffusers import sample_with_t2_and_cnn_clean
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from dgr.conditioning.t2_conditioning import process_t2_conditioning
from pathlib import Path
import hashlib
import threading
import queue


def _compute_adc_from_two_b_torch(s_low: torch.Tensor,
                                  b_low: float,
                                  s_high: torch.Tensor,
                                  b_high: float,
                                  eps: float = 1e-6) -> torch.Tensor:
    den = max(1e-6, float(b_high - b_low))
    dtype = s_low.dtype
    sl = torch.clamp(s_low.float(), min=eps)
    sh = torch.clamp(s_high.float(), min=eps)
    adc = (torch.log(sl) - torch.log(sh)) / den
    return torch.clamp(adc, 0.0, 0.003).to(dtype)


def _expand_stat_to_ndim(stat: torch.Tensor, target_ndim: int) -> torch.Tensor:
    if stat.ndim == 0:
        stat = stat.view(1, 1)
    if stat.ndim == 1:
        stat = stat.unsqueeze(1)
    if stat.ndim > 2:
        stat = stat.view(stat.shape[0], -1)
    stat = stat[:, :1]
    while stat.ndim < target_ndim:
        stat = stat.unsqueeze(-1)
    return stat


def _apply_norm_with_stats(tensor: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor - lo) / torch.clamp(hi - lo, min=1e-6), 0.0, 1.0)


def _maybe_squeeze_channel_dim(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.ndim == 5 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def _ensure_gpu_preproc(batch: Dict[str, torch.Tensor], b_low: float, b_high: float) -> None:
    """
    Normalize raw payloads emitted by NPZVariantDualBDataset directly on the GPU.
    """
    if 'adc_stack' in batch:
        return
    squeeze_keys = [
        'dwi_b50_stack', 'dwi_b1400_stack', 't2_stack',
        'dwi_b50_in', 'dwi_b1400_in',
        'dwi_b50_gt', 'dwi_b1400_gt',
        'adc_in', 'adc_gt'
    ]
    for key in squeeze_keys:
        if key in batch and isinstance(batch[key], torch.Tensor):
            squeezed = _maybe_squeeze_channel_dim(batch[key])
            if squeezed is not None:
                batch[key] = squeezed
    required_keys = [
        'dwi_b50_stack', 'dwi_b1400_stack',
        'dwi_b50_in', 'dwi_b1400_in',
        'dwi_b50_gt', 'dwi_b1400_gt',
        'norm_lo_b50_in', 'norm_hi_b50_in',
        'norm_lo_adc_in', 'norm_hi_adc_in',
        'norm_lo_b50_gt', 'norm_hi_b50_gt',
        'norm_lo_adc_gt', 'norm_hi_adc_gt'
    ]
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise RuntimeError(f"Missing keys for GPU preprocessing: {missing}")

    b50_stack_raw = batch['dwi_b50_stack']
    b14_stack_raw = batch['dwi_b1400_stack']
    lo_in = _expand_stat_to_ndim(batch['norm_lo_b50_in'], b50_stack_raw.ndim)
    hi_in = _expand_stat_to_ndim(batch['norm_hi_b50_in'], b50_stack_raw.ndim)

    adc_stack_phy = _compute_adc_from_two_b_torch(b50_stack_raw, b_low, b14_stack_raw, b_high)

    batch['dwi_b50_stack'] = _apply_norm_with_stats(b50_stack_raw, lo_in, hi_in)
    batch['dwi_b1400_stack'] = _apply_norm_with_stats(b14_stack_raw, lo_in, hi_in)

    lo_ai = _expand_stat_to_ndim(batch['norm_lo_adc_in'], adc_stack_phy.ndim)
    hi_ai = _expand_stat_to_ndim(batch['norm_hi_adc_in'], adc_stack_phy.ndim)
    batch['adc_stack'] = _apply_norm_with_stats(adc_stack_phy, lo_ai, hi_ai)

    b50_in_raw = batch['dwi_b50_in']
    b14_in_raw = batch['dwi_b1400_in']
    lo_in_slice = _expand_stat_to_ndim(batch['norm_lo_b50_in'], b50_in_raw.ndim)
    hi_in_slice = _expand_stat_to_ndim(batch['norm_hi_b50_in'], b50_in_raw.ndim)
    batch['dwi_b50_in'] = _apply_norm_with_stats(b50_in_raw, lo_in_slice, hi_in_slice)
    batch['dwi_b1400_in'] = _apply_norm_with_stats(b14_in_raw, lo_in_slice, hi_in_slice)
    adc_in_phy = _compute_adc_from_two_b_torch(b50_in_raw, b_low, b14_in_raw, b_high)
    lo_ai_slice = _expand_stat_to_ndim(batch['norm_lo_adc_in'], adc_in_phy.ndim)
    hi_ai_slice = _expand_stat_to_ndim(batch['norm_hi_adc_in'], adc_in_phy.ndim)
    batch['adc_in'] = _apply_norm_with_stats(adc_in_phy, lo_ai_slice, hi_ai_slice)

    b50_gt_raw = batch['dwi_b50_gt']
    b14_gt_raw = batch['dwi_b1400_gt']
    lo_gt = _expand_stat_to_ndim(batch['norm_lo_b50_gt'], b50_gt_raw.ndim)
    hi_gt = _expand_stat_to_ndim(batch['norm_hi_b50_gt'], b50_gt_raw.ndim)
    batch['dwi_b50_gt'] = _apply_norm_with_stats(b50_gt_raw, lo_gt, hi_gt)
    batch['dwi_b1400_gt'] = _apply_norm_with_stats(b14_gt_raw, lo_gt, hi_gt)
    adc_gt_phy = _compute_adc_from_two_b_torch(b50_gt_raw, b_low, b14_gt_raw, b_high)
    lo_ag = _expand_stat_to_ndim(batch['norm_lo_adc_gt'], adc_gt_phy.ndim)
    hi_ag = _expand_stat_to_ndim(batch['norm_hi_adc_gt'], adc_gt_phy.ndim)
    batch['adc_gt'] = _apply_norm_with_stats(adc_gt_phy, lo_ag, hi_ag)


def _prepare_t2_conditioning(batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    """
    Prepare T2 conditioning image. T2 is required for this model.
    T2 is already normalized to [0,1] by the dataset (t2_n key in NPY).
    """
    t2_stack = batch.get('t2_stack')
    if t2_stack is None:
        raise RuntimeError("T2 stack is required for T2-only diffusion model but not found in batch")
    
    t2 = t2_stack
    if t2.ndim == 5:
        t2 = t2.squeeze(1)
    mid = t2.shape[1] // 2
    t2_slice = t2[:, mid:mid + 1]  # [B, 1, H, W]
    
    # T2 is already normalized to [0,1] from dataset preprocessing
    # Optionally apply contrast modification (canny edge, etc.)
    if args.t2_contrast_mod == "none":
        return t2_slice
    return process_t2_conditioning(
        t2_slice,
        method=args.t2_contrast_mod,
        low_thresh=args.t2_canny_low,
        high_thresh=args.t2_canny_high,
    )


def _prepare_diffusion_targets(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare diffusion targets (GT) and distorted inputs.
    For T2-only model: target is (dwi_b50_gt, adc_gt), distorted input stored for validation only.
    """
    distorted_low = batch['dwi_b50_in']
    distorted_adc = batch.get('adc_in')
    if distorted_adc is None:
        distorted_adc = torch.zeros_like(distorted_low)
    distorted_pair = torch.cat([distorted_low, distorted_adc], dim=1)
    batch['dwi_in'] = distorted_pair  # Keep for validation with CNN
    
    target_clean = torch.cat([batch['dwi_b50_gt'], batch['adc_gt']], dim=1)
    batch['dwi_gt'] = target_clean
    return distorted_pair, target_clean


def _load_mageultra_cnn(ckpt_path: str, device: torch.device, args: argparse.Namespace):
    """
    Load MageUltra CNN for validation preprocessing.
    """
    from dgr.models.phc_e2e_mageultra_net import PHCE2EMageUltraNet
    
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
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cnn.load_state_dict(ckpt["model"])
    cnn = cnn.to(device).eval()
    
    return cnn


def _run_cnn_preprocessing(
    cnn_model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Run MageUltra CNN to get initial distortion correction.
    Returns: [B, 2, H, W] tensor of (b50_out, adc_out)
    """
    with torch.no_grad():
        # Get stacks for CNN input
        dwi_b50_stack = batch['dwi_b50_stack']
        adc_stack = batch['adc_stack']
        t2_stack = batch['t2_stack']
        
        # CNN forward
        out = cnn_model(dwi_b50_stack, adc_stack, t2_stack, None)
        
        b50_out = out["I_out_b50"]  # [B, 1, H, W]
        adc_out = out["I_out_adc"]  # [B, 1, H, W]
        
        # Concatenate
        cnn_output = torch.cat([b50_out, adc_out], dim=1)  # [B, 2, H, W]
        
    return cnn_output


# ==============================================================================
# CNN Cache Manager for Lazy Caching
# ==============================================================================
class CNNCacheManager:
    """
    Manages lazy caching of CNN outputs to avoid recomputing every epoch.
    
    - First epoch: compute CNN outputs on-the-fly and cache to disk/memory
    - Subsequent epochs: load from cache
    - Thread-safe async saving to not block training
    """
    
    def __init__(self, cache_dir: str, use_memory_cache: bool = True, max_memory_items: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_memory_cache = use_memory_cache
        self.max_memory_items = max_memory_items
        
        # In-memory cache (faster than disk)
        self._memory_cache: Dict[str, torch.Tensor] = {}
        self._memory_lock = threading.Lock()
        
        # Async save queue
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_save_thread = False
        
    def _start_save_thread(self):
        """Start background thread for async disk saves."""
        if self._save_thread is not None and self._save_thread.is_alive():
            return
        self._stop_save_thread = False
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
    
    def _save_worker(self):
        """Background worker that saves tensors to disk."""
        while not self._stop_save_thread:
            try:
                item = self._save_queue.get(timeout=1.0)
                if item is None:
                    break
                cache_key, tensor = item
                cache_path = self.cache_dir / f"{cache_key}.pt"
                if not cache_path.exists():
                    torch.save(tensor.cpu(), cache_path)
                self._save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CNN_CACHE] Save error: {e}")
    
    def stop(self):
        """Stop the background save thread."""
        self._stop_save_thread = True
        self._save_queue.put(None)
        if self._save_thread is not None:
            self._save_thread.join(timeout=5.0)
    
    def _generate_cache_key(self, batch: Dict[str, torch.Tensor], batch_idx: int, sample_idx: int) -> str:
        """Generate a unique cache key for a sample."""
        # Use npz_path if available, otherwise use a hash of the input
        npz_paths = batch.get('npz_path', [])
        slice_indices = batch.get('slice_idx', [])
        
        if npz_paths and len(npz_paths) > sample_idx:
            npz_path = npz_paths[sample_idx] if isinstance(npz_paths, list) else str(npz_paths)
            slice_idx = slice_indices[sample_idx] if isinstance(slice_indices, (list, torch.Tensor)) else 0
            if isinstance(slice_idx, torch.Tensor):
                slice_idx = slice_idx.item()
            # Create a short hash of the path
            path_hash = hashlib.md5(str(npz_path).encode()).hexdigest()[:12]
            return f"{path_hash}_s{slice_idx}"
        else:
            # Fallback: hash the input tensor
            dwi_hash = hashlib.md5(batch['dwi_b50_stack'][sample_idx].cpu().numpy().tobytes()).hexdigest()[:12]
            return f"batch{batch_idx}_sample{sample_idx}_{dwi_hash}"
    
    def get_or_compute(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        cnn_model,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get CNN outputs for a batch, either from cache or by computing.
        
        Returns: [B, 2, H, W] tensor of CNN outputs
        """
        batch_size = batch['dwi_b50_stack'].shape[0]
        cnn_outputs = []
        needs_compute_indices = []
        cache_keys = []
        
        # Check cache for each sample
        for i in range(batch_size):
            cache_key = self._generate_cache_key(batch, batch_idx, i)
            cache_keys.append(cache_key)
            
            # Try memory cache first
            with self._memory_lock:
                if cache_key in self._memory_cache:
                    cnn_outputs.append(self._memory_cache[cache_key].to(device))
                    continue
            
            # Try disk cache
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                try:
                    cached = torch.load(cache_path, map_location=device)
                    cnn_outputs.append(cached)
                    # Also store in memory cache
                    if self.use_memory_cache:
                        with self._memory_lock:
                            if len(self._memory_cache) < self.max_memory_items:
                                self._memory_cache[cache_key] = cached.cpu()
                    continue
                except Exception:
                    pass
            
            # Need to compute this sample
            cnn_outputs.append(None)
            needs_compute_indices.append(i)
        
        # Compute missing samples
        if needs_compute_indices:
            # Create a mini-batch of samples that need computation
            mini_batch = {}
            for key in ['dwi_b50_stack', 'adc_stack', 't2_stack']:
                if key in batch:
                    mini_batch[key] = batch[key][needs_compute_indices]
            
            # Run CNN
            with torch.no_grad():
                out = cnn_model(
                    mini_batch['dwi_b50_stack'],
                    mini_batch['adc_stack'],
                    mini_batch['t2_stack'],
                    None
                )
                computed = torch.cat([out["I_out_b50"], out["I_out_adc"]], dim=1)
            
            # Fill in computed results and schedule caching
            self._start_save_thread()
            for j, orig_idx in enumerate(needs_compute_indices):
                cnn_out_j = computed[j:j+1]
                cnn_outputs[orig_idx] = cnn_out_j.squeeze(0)
                
                # Cache in memory
                cache_key = cache_keys[orig_idx]
                if self.use_memory_cache:
                    with self._memory_lock:
                        if len(self._memory_cache) < self.max_memory_items:
                            self._memory_cache[cache_key] = cnn_out_j.squeeze(0).cpu()
                
                # Schedule async disk save
                self._save_queue.put((cache_key, cnn_out_j.squeeze(0)))
        
        # Stack all outputs
        return torch.stack(cnn_outputs, dim=0)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        disk_count = len(list(self.cache_dir.glob("*.pt")))
        with self._memory_lock:
            memory_count = len(self._memory_cache)
        return {
            "disk_cached": disk_count,
            "memory_cached": memory_count,
            "pending_saves": self._save_queue.qsize(),
        }


def _save_val_visuals(batch: Dict[str, torch.Tensor], preds: torch.Tensor, 
                      cnn_preds: Optional[torch.Tensor], epoch: int, 
                      args: argparse.Namespace, max_samples: int) -> int:
    if max_samples <= 0:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[VAL][WARN] Matplotlib unavailable for vis: {exc}", flush=True)
        return 0
    save_root = os.path.join(args.out_dir, "val_vis")
    os.makedirs(save_root, exist_ok=True)
    limit = min(max_samples, preds.shape[0])
    t2_stack = batch.get("t2_stack")
    if t2_stack is not None:
        if t2_stack.ndim == 5:
            t2_stack = t2_stack.squeeze(1)
        t2_mid = t2_stack[:, t2_stack.shape[1] // 2]
    else:
        t2_mid = None

    def _adc_range(arr: np.ndarray) -> Tuple[float, float]:
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    for idx in range(limit):
        low_in = batch['dwi_in'][idx, 0].detach().cpu().numpy()
        adc_in = batch['dwi_in'][idx, 1].detach().cpu().numpy()
        low_out = preds[idx, 0].detach().cpu().numpy()
        adc_out = preds[idx, 1].detach().cpu().numpy()
        low_gt = batch['dwi_gt'][idx, 0].detach().cpu().numpy()
        adc_gt = batch['dwi_gt'][idx, 1].detach().cpu().numpy()
        t2_np = None
        if t2_mid is not None:
            t2_np = t2_mid[idx].detach().cpu().numpy()
        
        # Check if we have CNN predictions
        has_cnn = cnn_preds is not None
        if has_cnn:
            low_cnn = cnn_preds[idx, 0].detach().cpu().numpy()
            adc_cnn = cnn_preds[idx, 1].detach().cpu().numpy()
        
        ncols = 5 if has_cnn else 4
        if t2_np is not None:
            ncols += 1
        
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6))
        col = 0
        
        # Low-b row
        axes[0, col].imshow(low_in, cmap='gray', vmin=0, vmax=1); axes[0, col].set_title("low-b IN"); axes[0, col].axis('off'); col += 1
        if has_cnn:
            axes[0, col].imshow(low_cnn, cmap='gray', vmin=0, vmax=1); axes[0, col].set_title("low-b CNN"); axes[0, col].axis('off'); col += 1
        axes[0, col].imshow(low_out, cmap='gray', vmin=0, vmax=1); axes[0, col].set_title("low-b DIFF"); axes[0, col].axis('off'); col += 1
        axes[0, col].imshow(low_gt, cmap='gray', vmin=0, vmax=1); axes[0, col].set_title("low-b GT"); axes[0, col].axis('off'); col += 1
        if t2_np is not None:
            axes[0, col].imshow(t2_np, cmap='gray', vmin=0, vmax=1); axes[0, col].set_title("T2"); axes[0, col].axis('off')
        
        col = 0
        # ADC row
        vmin_in, vmax_in = _adc_range(adc_in)
        axes[1, col].imshow(adc_in, cmap='viridis', vmin=vmin_in, vmax=vmax_in); axes[1, col].set_title("ADC IN"); axes[1, col].axis('off'); col += 1
        if has_cnn:
            vmin_cnn, vmax_cnn = _adc_range(adc_cnn)
            axes[1, col].imshow(adc_cnn, cmap='viridis', vmin=vmin_cnn, vmax=vmax_cnn); axes[1, col].set_title("ADC CNN"); axes[1, col].axis('off'); col += 1
        vmin_out, vmax_out = _adc_range(adc_out)
        axes[1, col].imshow(adc_out, cmap='viridis', vmin=vmin_out, vmax=vmax_out); axes[1, col].set_title("ADC DIFF"); axes[1, col].axis('off'); col += 1
        vmin_gt, vmax_gt = _adc_range(adc_gt)
        axes[1, col].imshow(adc_gt, cmap='viridis', vmin=vmin_gt, vmax=vmax_gt); axes[1, col].set_title("ADC GT"); axes[1, col].axis('off'); col += 1
        if ncols > col:
            axes[1, col].axis('off')
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_root, f"epoch{epoch:03d}_sample{idx:02d}.png"), dpi=140)
        plt.close(fig)
    return limit


def parse_args():
    p = argparse.ArgumentParser(description="Train T2-only conditional diffusion model on MageUltra NPZ dataset")
    p.add_argument("--npz_root", type=str, required=True, help="Primary NPZ dataset root")
    p.add_argument("--npz_root2", type=str, default="", help="Optional secondary NPZ root to concatenate")
    p.add_argument("--numeric_only", action="store_true", help="Restrict dataset traversal to numeric-named folders")
    p.add_argument("--numeric_only2", action="store_true", help="Numeric-only filter for secondary dataset")
    p.add_argument("--axis_filter", type=int, choices=[0, 1], default=None, help="Filter to a single phase-encode axis")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--val_steps", type=int, default=25)
    p.add_argument("--warmup_steps", type=int, default=1000,
                   help="Linear warmup steps fed into the LR scheduler")
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   help="Scheduler name passed to diffusers.get_scheduler (default: cosine)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--use_dpm_solver_validation", action="store_true",
                   help="Use DPMSolver++ multistep scheduler for validation sampling")
    p.add_argument("--val_strength", type=float, default=0.5, help="Img2img strength during validation sampling")
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", 4)))
    p.add_argument("--num_workers_val", type=int, default=int(os.environ.get("NUM_WORKERS_VAL", max(1, int(os.environ.get("NUM_WORKERS", 4)) // 2))))
    p.add_argument("--prefetch_factor", type=int, default=int(os.environ.get("PREFETCH_FACTOR", 2)))
    p.add_argument("--prefetch_factor_val", type=int, default=int(os.environ.get("PREFETCH_FACTOR_VAL", os.environ.get("PREFETCH_FACTOR", 2))))
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    
    # T2 conditioning module parameters
    p.add_argument("--t2_cond_channels", type=int, default=64, help="Channel width for T2 conditioning module")
    p.add_argument("--t2_contrast_mod", type=str, default="none",
                   choices=["none", "lcn", "canny_binary", "canny_grayscale"],
                   help="T2 conditioning transform")
    p.add_argument("--t2_canny_low", type=int, default=50)
    p.add_argument("--t2_canny_high", type=int, default=150)
    
    p.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from.")
    p.add_argument("--b_low", type=float, default=50.0)
    p.add_argument("--b_high", type=float, default=1400.0)
    p.add_argument("--val_vis", action="store_true", help="Enable validation visualizations")
    p.add_argument("--val_vis_n", type=int, default=4, help="Max number of validation panels to save per epoch")
    
    # CNN (MageUltra) for conditioning and validation preprocessing
    p.add_argument("--cnn_ckpt", type=str, 
                   default="/path/to/dgr_data/network_runs/mageultra_dualb_axis01_quick_local_v23/mageultra_best.pt",
                   help="Path to MageUltra CNN checkpoint for conditioning")
    p.add_argument("--cnn_base_channels", type=int, default=64, help="CNN base channels (must match CNN training)")
    p.add_argument("--cnn_latent_dim", type=int, default=8, help="CNN latent dim (must match CNN training)")
    p.add_argument("--cnn_prompt_k", type=int, default=8, help="CNN prompt_k (must match CNN training)")
    p.add_argument("--cnn_prompt_temp", type=float, default=1.0, help="CNN prompt_temp (must match CNN training)")
    
    # CNN cache settings
    p.add_argument("--cnn_cache_dir", type=str, default="",
                   help="Directory to cache CNN outputs (default: <out_dir>/cnn_cache)")
    p.add_argument("--cnn_memory_cache_size", type=int, default=10000,
                   help="Max number of CNN outputs to keep in memory cache")
    
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if has_cuda:
            torch.cuda.manual_seed_all(args.seed)

    use_ddp = dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    if has_cuda:
        torch.cuda.set_device(device)

    # Build NPZ dataset(s)
    datasets = [
        NPZVariantDualBDataset(
            args.npz_root,
            radius=args.radius,
            numeric_only_subfolders=args.numeric_only,
            axis_filter=args.axis_filter,
            defer_norm_to_gpu=True,
        )
    ]
    if args.npz_root2:
        datasets.append(
            NPZVariantDualBDataset(
                args.npz_root2,
                radius=args.radius,
                numeric_only_subfolders=args.numeric_only2,
                axis_filter=args.axis_filter,
                defer_norm_to_gpu=True,
            )
        )
    full_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    total_len = len(full_dataset)
    val_len = max(1, int(total_len * args.val_split))
    train_len = max(1, total_len - val_len)
    generator = torch.Generator().manual_seed(args.seed)
    tr_set, va_set = random_split(full_dataset, [train_len, val_len], generator=generator)

    sampler_tr = DistributedSampler(tr_set, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    sampler_va = DistributedSampler(va_set, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    def _build_loader(dataset, sampler, shuffle, workers, prefetch):
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers and workers > 0,
        )
        if workers > 0 and prefetch > 0:
            loader_kwargs["prefetch_factor"] = prefetch
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            drop_last=False,
            **loader_kwargs,
        )

    ld_tr = _build_loader(tr_set, sampler_tr, shuffle=True, workers=args.num_workers, prefetch=args.prefetch_factor)
    ld_va = _build_loader(va_set, sampler_va, shuffle=False, workers=args.num_workers_val, prefetch=args.prefetch_factor_val) if val_len > 0 else None

    # Create T2+CNN diffusion model (refines CNN outputs with T2 guidance)
    model = DiffusionUNetT2AndCNN(
        fusion_channels=args.t2_cond_channels,
    ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    # Load CNN for conditioning (required for this model)
    cnn_model = None
    if args.cnn_ckpt and os.path.exists(args.cnn_ckpt):
        try:
            cnn_model = _load_mageultra_cnn(args.cnn_ckpt, device, args)
            if rank == 0:
                print(f"[INFO] Loaded MageUltra CNN from {args.cnn_ckpt}")
        except Exception as e:
            if rank == 0:
                print(f"[ERROR] Failed to load CNN checkpoint: {e}")
            raise RuntimeError(f"CNN checkpoint is required for T2+CNN model but failed to load: {e}")
    else:
        raise RuntimeError(f"CNN checkpoint is required for T2+CNN model but not found: {args.cnn_ckpt}")
    
    # Initialize CNN cache manager
    cnn_cache_dir = args.cnn_cache_dir if args.cnn_cache_dir else os.path.join(args.out_dir, "cnn_cache")
    cnn_cache = CNNCacheManager(
        cache_dir=cnn_cache_dir,
        use_memory_cache=True,
        max_memory_items=args.cnn_memory_cache_size,
    )
    if rank == 0:
        print(f"[INFO] CNN cache directory: {cnn_cache_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ==========================================================================
    # KEY CHANGE: prediction_type="sample" instead of "epsilon"
    # The network directly predicts clean images, not noise.
    # This makes the model learn on the anatomical manifold.
    # ==========================================================================
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="linear",
        prediction_type="sample",  # ← CLEAN IMAGE PREDICTION
    )
    if args.use_dpm_solver_validation:
        val_scheduler = DPMSolverMultistepScheduler.from_config(dict(noise_scheduler.config))
    else:
        val_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="linear",
            prediction_type="sample",  # ← Must match training scheduler
        )

    total_train_steps = max(1, len(ld_tr)) // max(1, args.gradient_accumulation_steps) * args.epochs
    try:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_train_steps,
        )
    except Exception:
        lr_scheduler = None

    start_epoch = 1
    global_step = 0
    model_to_save = model.module if isinstance(model, DDP) else model
    if args.resume_from:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        model_to_save.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)

    keys_to_gpu = {
        'dwi_b50_stack', 'dwi_b1400_stack', 't2_stack',
        'dwi_b50_gt', 'dwi_b1400_gt', 'adc_gt',
        'dwi_b50_in', 'dwi_b1400_in', 'dwi_in', 'adc_in',
        'norm_lo_b50_in', 'norm_hi_b50_in',
        'norm_lo_adc_in', 'norm_hi_adc_in',
        'norm_lo_b50_gt', 'norm_hi_b50_gt',
        'norm_lo_adc_gt', 'norm_hi_adc_gt',
    }

    # ==========================================================================
    # CHECKPOINT LOGIC: Best 3 + every 10th epoch (milestone)
    # ==========================================================================
    best_checkpoints = []
    max_best_checkpoints = 3
    milestone_epochs = set(range(10, args.epochs + 1, 10))  # 10, 20, 30, ...
    milestone_checkpoints = set()  # Track which milestones have been saved

    def save_checkpoint(epoch: int, val_loss: float, is_milestone: bool = False) -> None:
        nonlocal best_checkpoints, milestone_checkpoints
        if rank != 0:
            return
        
        scheduler_info = {
            "num_train_timesteps": args.num_train_timesteps,
            "beta_schedule": "linear",
            "prediction_type": "sample",  # ← Clean-image prediction
        }
        val_scheduler_info = dict(val_scheduler.config) if args.use_dpm_solver_validation else None
        ckpt_path = os.path.join(args.out_dir, f"diff_t2cnn_clean_epoch_{epoch:03d}.pt")
        
        torch.save(
            {
                "epoch": epoch,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "noise_scheduler_config": scheduler_info,
                "val_scheduler_config": val_scheduler_info,
                "args": vars(args),
                "global_step": global_step,
                "val_loss": val_loss,
                "model_type": "DiffusionUNetT2AndCNN",
                "prediction_type": "sample",  # ← Mark as clean-image prediction
                "is_milestone": is_milestone,
            },
            ckpt_path,
        )
        
        # Handle best checkpoints (excluding milestones from removal)
        if not is_milestone:
            best_checkpoints.append((val_loss, epoch, ckpt_path))
            best_checkpoints.sort(key=lambda x: x[0])
            
            # Remove worst checkpoint if exceeding max, but protect milestones
            while len(best_checkpoints) > max_best_checkpoints:
                _, worst_epoch, worst_path = best_checkpoints.pop()
                # Don't remove milestone checkpoints
                if worst_epoch not in milestone_checkpoints:
                    if os.path.exists(worst_path):
                        os.remove(worst_path)
                        print(f"[CHECKPOINT] Removed epoch {worst_epoch:03d} (higher val loss)")
        else:
            # This is a milestone checkpoint - track it
            milestone_checkpoints.add(epoch)
            print(f"[CHECKPOINT] Saved milestone epoch {epoch:03d}")
        
        print("[CHECKPOINT] Best 3 leaderboard:")
        for idx, (loss, ep, path) in enumerate(best_checkpoints[:max_best_checkpoints], start=1):
            marker = " (milestone)" if ep in milestone_checkpoints else ""
            print(f"  #{idx}: epoch {ep:03d} | val_loss={loss:.6f}{marker}")
        
        if milestone_checkpoints:
            print(f"[CHECKPOINT] Milestone epochs saved: {sorted(milestone_checkpoints)}")

    if rank == 0:
        print("=" * 70)
        print("CLEAN-IMAGE-PREDICTION Diffusion Training")
        print("(Learning on the Anatomical Manifold)")
        print("=" * 70)
        print(f"  Model: DiffusionUNetT2AndCNN")
        print(f"  Prediction Type: SAMPLE (clean image, NOT noise)")
        print(f"  Conditioning: T2 anatomy + CNN output")
        print(f"  T2 cond channels: {args.t2_cond_channels}")
        print(f"  Target: (dwi_b50_gt, adc_gt)")
        print(f"  Training: Add noise to GT, predict clean GT directly")
        print(f"  Loss: MSE(predicted_clean, target_clean)")
        print(f"  CNN checkpoint: {args.cnn_ckpt}")
        print(f"  CNN cache: {cnn_cache_dir}")
        print(f"  Checkpoint strategy: Best 3 by val_loss + every 10th epoch")
        print("=" * 70)

    batch_idx_global = 0
    for ep in range(start_epoch, args.epochs + 1):
        if sampler_tr is not None:
            sampler_tr.set_epoch(ep)
        model.train()
        tot = 0.0
        it = 0
        for batch in ld_tr:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and (k in keys_to_gpu or k == 't2_stack'):
                    batch[k] = v.to(device, non_blocking=has_cuda)
            _ensure_gpu_preproc(batch, args.b_low, args.b_high)

            # Prepare targets: (dwi_b50_gt, adc_gt)
            _, target_clean = _prepare_diffusion_targets(batch)

            # T2 conditioning (required, already normalized [0,1])
            t2_image = _prepare_t2_conditioning(batch, args)
            t2_image = t2_image.to(device, non_blocking=has_cuda)
            
            # Get CNN output (from cache or compute)
            # CNN output tells the model "what the CNN already predicted"
            cnn_output = cnn_cache.get_or_compute(batch, batch_idx_global, cnn_model, device)
            batch_idx_global += 1

            # Add noise to GT (same as before - this is the forward diffusion process)
            noise = torch.randn_like(target_clean)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps,
                (target_clean.size(0),), device=device
            ).long()
            target_noisy = noise_scheduler.add_noise(target_clean, noise, timesteps)

            # ==========================================================================
            # KEY CHANGE: Clean-image prediction instead of noise prediction
            # The model outputs the clean sample directly, not the noise.
            # Loss is computed against the original clean target.
            # This makes the model learn on the anatomical manifold.
            # ==========================================================================
            sample_pred = model(target_noisy, t2_image, cnn_output, timesteps).sample
            loss = F.mse_loss(sample_pred, target_clean)  # ← Predict clean, not noise
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (it + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            tot += float(loss.item() * args.gradient_accumulation_steps)
            it += 1

            if rank == 0 and (it % 100 == 0 or it == 1):
                cache_stats = cnn_cache.get_cache_stats()
                print(f"[TRAIN] epoch={ep} step={it} | loss={tot / max(1, it):.6f} | global_step={global_step} | cache: mem={cache_stats['memory_cached']}, disk={cache_stats['disk_cached']}")

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            if rank == 0:
                print(f"[EARLY-STOP] max_steps reached at epoch {ep}")
            break

        val_loss = None
        if ld_va is not None and ep % args.val_interval == 0:
            if sampler_va is not None:
                sampler_va.set_epoch(ep)
            model.eval()
            tot_v = 0.0
            count_v = 0
            vis_quota = args.val_vis_n if (rank == 0 and args.val_vis) else 0
            with torch.no_grad():
                for batch in ld_va:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor) and (k in keys_to_gpu or k == 't2_stack'):
                            batch[k] = v.to(device, non_blocking=has_cuda)
                    _ensure_gpu_preproc(batch, args.b_low, args.b_high)
                    _prepare_diffusion_targets(batch)
                    
                    # Step 1: Run CNN to get initial correction
                    cnn_output = _run_cnn_preprocessing(cnn_model, batch, device)
                    
                    # Step 2: Prepare batch for sampler
                    batch['cnn_output'] = cnn_output  # [B, 2, H, W] - for conditioning
                    batch['cnn_init'] = cnn_output    # [B, 2, H, W] - as init_latent for SDEdit
                    
                    # Run diffusion sampling with T2 + CNN conditioning
                    # Using clean-image prediction sampler (prediction_type="sample")
                    pred_batch = sample_with_t2_and_cnn_clean(
                        model,
                        val_scheduler,
                        batch,
                        steps=args.val_steps,
                        strength=args.val_strength,
                    )
                    
                    val_loss_step = F.l1_loss(pred_batch, batch["dwi_gt"]).item()
                    tot_v += val_loss_step
                    count_v += 1
                    if rank == 0 and vis_quota > 0 and args.val_vis:
                        used = _save_val_visuals(batch, pred_batch, cnn_output, ep, args, vis_quota)
                        vis_quota = max(0, vis_quota - used)
            if use_ddp:
                tensor = torch.tensor([tot_v, count_v], device=device, dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tot_v = float(tensor[0].item())
                count_v = max(1, int(tensor[1].item()))
            val_loss = tot_v / max(1, count_v)
            if rank == 0:
                print(f"[VAL] epoch={ep} | loss={val_loss:.6f}")
                
                # Check if this is a milestone epoch
                is_milestone = (ep in milestone_epochs)
                
                # Save checkpoint (either as best or milestone)
                save_checkpoint(ep, val_loss, is_milestone=is_milestone)
                
        if rank == 0:
            train_loss = tot / max(1, it)
            if val_loss is not None:
                print(f"Epoch {ep:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
            else:
                print(f"Epoch {ep:03d} | train_loss={train_loss:.6f} | val_skipped")

    # Stop CNN cache background thread
    cnn_cache.stop()
    
    if rank == 0:
        cache_stats = cnn_cache.get_cache_stats()
        print("\n" + "=" * 70)
        print("CLEAN-IMAGE-PREDICTION TRAINING COMPLETED")
        print("=" * 70)
        print(f"  Prediction Type: SAMPLE (clean image prediction)")
        print(f"  CNN Cache Stats: disk={cache_stats['disk_cached']}, memory={cache_stats['memory_cached']}")
        if best_checkpoints:
            print("  Final best checkpoints:")
            for idx, (loss, ep, path) in enumerate(best_checkpoints[:max_best_checkpoints], start=1):
                marker = " (milestone)" if ep in milestone_checkpoints else ""
                print(f"    #{idx}: epoch {ep:03d} | val_loss={loss:.6f}{marker} -> {os.path.basename(path)}")
        if milestone_checkpoints:
            print(f"  Milestone checkpoints: {sorted(milestone_checkpoints)}")
        print("=" * 70)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
