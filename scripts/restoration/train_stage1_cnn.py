import os
import sys
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import shutil
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dgr.models.phc_e2e_mageultra_net import PHCE2EMageUltraNet
from dgr.data.npz_variant_dualb_dataset import NPZVariantDualBDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PHC E2E (MageUltra) on dual-b NPZ variants")
    p.add_argument("--npz_root", type=str, required=True)
    p.add_argument("--npz_root2", type=str, default="", help="Optional second dataset root for mixed training")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt) to load model weights from")
    p.add_argument("--numeric_only", action="store_true", help="Use only numeric-named subfolders")
    p.add_argument("--numeric_only2", action="store_true", help="Use only numeric-named subfolders for second dataset")
    p.add_argument("--axis_filter", type=int, default=None, choices=[0, 1], help="Filter to a single pe_axis (0 or 1); default None uses both")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=64)
    # Prompt/domain conditioning
    p.add_argument("--latent_dim", type=int, default=0, help="T2 prompt code dimension (0 disables)")
    p.add_argument("--prompt_k", type=int, default=8, help="Number of prompt codes in the codebook")
    p.add_argument("--prompt_temp", type=float, default=1.0, help="Softmax temperature for prompt gating")
    # Multi-scale and SSIM (align with single-stream)
    p.add_argument("--use_ssim_loss", action="store_true")
    p.add_argument("--ssim_weight", type=float, default=0.1)
    p.add_argument("--ms_w1", type=float, default=0.2)
    p.add_argument("--ms_w2", type=float, default=0.05)
    p.add_argument("--val_vis", action="store_true")
    p.add_argument("--val_vis_n", type=int, default=8)
    p.add_argument(
        "--zero_t2_guidance",
        action="store_true",
        help="Ablation: replace T2 guidance input with zeros during training and validation.",
    )
    p.add_argument("--amp", type=int, default=1, choices=[0, 1], help="Enable CUDA AMP autocast/GradScaler")
    # Optional scheduler
    p.add_argument("--warmup_steps", type=int, default=3054, help="Warmup steps (per-rank) for cosine schedule; 0 disables scheduler")
    # ADC supervision / b-values
    p.add_argument("--adc_loss_weight", type=float, default=1.0, help="Deprecated (no longer used when ADC is a primary output)")
    p.add_argument("--b_low", type=float, default=50.0, help="Low b-value used to compute ADC")
    p.add_argument("--b_high", type=float, default=1400.0, help="High b-value used to compute ADC")
    # Curriculum flags kept for compatibility but unused in the new ADC-as-input/output setup
    p.add_argument("--relint_loss_weight", type=float, default=0.0, help="Unused when ADC is a primary supervised output")
    p.add_argument("--adc_curriculum_steps", type=int, default=0, help="Unused when ADC is a primary supervised output")
    p.add_argument("--relint_curriculum_steps", type=int, default=0, help="Unused when ADC is a primary supervised output")
    p.add_argument("--curriculum_mode", type=str, default="linear", choices=["linear", "cosine"], help="Unused when ADC is a primary supervised output")
    # LR/Warmup autoscaling controls
    p.add_argument("--autoscale_lr", type=int, default=1, choices=[0, 1], help="Enable LR autoscaling by global batch")
    p.add_argument("--autoscale_mode", type=str, default="linear", choices=["linear", "sqrt"], help="LR autoscaling rule")
    p.add_argument("--autoscale_warmup", type=int, default=1, choices=[0, 1], help="Enable warmup autoscaling by global batch")
    return p.parse_args()


def _percentile_norm01_t(t: torch.Tensor) -> torch.Tensor:
    # normalize per-slice image to [0,1]
    b, c, h, w = t.shape
    x = t.view(b, c, -1)
    lo = x.kthvalue(max(1, int(0.01 * x.shape[-1]))).values.view(b, c, 1)
    hi = x.kthvalue(max(1, int(0.99 * x.shape[-1]))).values.view(b, c, 1)
    y = (x - lo) / torch.clamp(hi - lo, min=1e-6)
    return y.view(b, c, h, w).clamp(0.0, 1.0)


def _compute_adc_from_two_b_torch(s_low: torch.Tensor,
                                  b_low: float,
                                  s_high: torch.Tensor,
                                  b_high: float,
                                  eps: float = 1e-6) -> torch.Tensor:
    """
    Compute ADC map (physical units) from two b-value signals using torch ops.
    Operates in float32 for stability and casts back to the original dtype.
    """
    den = max(1e-6, float(b_high - b_low))
    dtype = s_low.dtype
    sl = torch.clamp(s_low.float(), min=eps)
    sh = torch.clamp(s_high.float(), min=eps)
    adc = (torch.log(sl) - torch.log(sh)) / den
    return torch.clamp(adc, 0.0, 0.003).to(dtype)


def _expand_stat_to_ndim(stat: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """
    Expand a per-sample scalar tensor to match target tensor dimensionality for broadcasting.
    stat shape can be [B], [B,1], or [B,1,...]; output shape becomes [B,1,1,...].
    """
    if stat.ndim == 0:
        stat = stat.view(1, 1)
    if stat.ndim == 1:
        stat = stat.unsqueeze(1)
    if stat.ndim > 2:
        stat = stat.view(stat.shape[0], -1)
    stat = stat[:, :1]  # ensure single channel
    while stat.ndim < target_ndim:
        stat = stat.unsqueeze(-1)
    return stat


def _apply_norm_with_stats(tensor: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor - lo) / torch.clamp(hi - lo, min=1e-6), 0.0, 1.0)


def _ensure_gpu_preproc(batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    """
    If the batch was emitted by the dataset in raw (unnormalized) form, build normalized
    low-b, ADC inputs, and GT tensors directly on GPU to reduce CPU overhead.
    """
    if 'adc_stack' in batch:
        # Already normalized payload (legacy mode)
        return
    required_keys = ['dwi_b50_stack', 'dwi_b1400_stack', 'dwi_b50_in', 'dwi_b1400_in',
                     'dwi_b50_gt', 'dwi_b1400_gt',
                     'norm_lo_b50_in', 'norm_hi_b50_in',
                     'norm_lo_adc_in', 'norm_hi_adc_in',
                     'norm_lo_b50_gt', 'norm_hi_b50_gt',
                     'norm_lo_adc_gt', 'norm_hi_adc_gt']
    missing = [k for k in required_keys if k not in batch]
    if missing:
        raise RuntimeError(f"Missing keys for GPU preprocessing: {missing}")

    b50_stack_raw = batch['dwi_b50_stack']
    b14_stack_raw = batch['dwi_b1400_stack']
    lo_in = _expand_stat_to_ndim(batch['norm_lo_b50_in'], b50_stack_raw.ndim)
    hi_in = _expand_stat_to_ndim(batch['norm_hi_b50_in'], b50_stack_raw.ndim)

    # ADC stack computed from raw (physical) inputs prior to normalization
    adc_stack_phy = _compute_adc_from_two_b_torch(b50_stack_raw, args.b_low, b14_stack_raw, args.b_high)

    batch['dwi_b50_stack'] = _apply_norm_with_stats(b50_stack_raw, lo_in, hi_in)
    batch['dwi_b1400_stack'] = _apply_norm_with_stats(b14_stack_raw, lo_in, hi_in)

    lo_ai = _expand_stat_to_ndim(batch['norm_lo_adc_in'], adc_stack_phy.ndim)
    hi_ai = _expand_stat_to_ndim(batch['norm_hi_adc_in'], adc_stack_phy.ndim)
    batch['adc_stack'] = _apply_norm_with_stats(adc_stack_phy, lo_ai, hi_ai)

    # Single-slice inputs
    b50_in_raw = batch['dwi_b50_in']
    b14_in_raw = batch['dwi_b1400_in']
    lo_in_slice = _expand_stat_to_ndim(batch['norm_lo_b50_in'], b50_in_raw.ndim)
    hi_in_slice = _expand_stat_to_ndim(batch['norm_hi_b50_in'], b50_in_raw.ndim)
    batch['dwi_b50_in'] = _apply_norm_with_stats(b50_in_raw, lo_in_slice, hi_in_slice)
    batch['dwi_b1400_in'] = _apply_norm_with_stats(b14_in_raw, lo_in_slice, hi_in_slice)
    adc_in_phy = _compute_adc_from_two_b_torch(b50_in_raw, args.b_low, b14_in_raw, args.b_high)
    lo_ai_slice = _expand_stat_to_ndim(batch['norm_lo_adc_in'], adc_in_phy.ndim)
    hi_ai_slice = _expand_stat_to_ndim(batch['norm_hi_adc_in'], adc_in_phy.ndim)
    batch['adc_in'] = _apply_norm_with_stats(adc_in_phy, lo_ai_slice, hi_ai_slice)

    # Ground truth normalization
    b50_gt_raw = batch['dwi_b50_gt']
    b14_gt_raw = batch['dwi_b1400_gt']
    lo_gt = _expand_stat_to_ndim(batch['norm_lo_b50_gt'], b50_gt_raw.ndim)
    hi_gt = _expand_stat_to_ndim(batch['norm_hi_b50_gt'], b50_gt_raw.ndim)
    batch['dwi_b50_gt'] = _apply_norm_with_stats(b50_gt_raw, lo_gt, hi_gt)
    batch['dwi_b1400_gt'] = _apply_norm_with_stats(b14_gt_raw, lo_gt, hi_gt)
    adc_gt_phy = _compute_adc_from_two_b_torch(b50_gt_raw, args.b_low, b14_gt_raw, args.b_high)
    lo_ag = _expand_stat_to_ndim(batch['norm_lo_adc_gt'], adc_gt_phy.ndim)
    hi_ag = _expand_stat_to_ndim(batch['norm_hi_adc_gt'], adc_gt_phy.ndim)
    batch['adc_gt'] = _apply_norm_with_stats(adc_gt_phy, lo_ag, hi_ag)

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Distributed init (torchrun)
    use_ddp = dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Auto-scale LR and/or warmup relative to single-stream single-GPU baseline if enabled
    # Baseline: batch_size=4, lr=3e-4, warmup_steps=9000 (single-stream sbatch common config)
    baseline_bs = 4
    baseline_lr = 3e-4
    baseline_warmup = 9000
    global_batch = max(1, args.batch_size * world_size)
    if int(args.autoscale_lr) == 1:
        if args.autoscale_mode == "linear":
            scaled_lr = baseline_lr * (global_batch / float(baseline_bs))
        else:
            # sqrt scaling (often better with Adam-like optimizers)
            scaled_lr = baseline_lr * math.sqrt(global_batch / float(baseline_bs))
        args.lr = float(scaled_lr)
    # warmup scaling: keep sample-equivalence by default; can disable to keep a larger warmup
    if int(args.autoscale_warmup) == 1:
        scaled_warmup = int(max(1, round(baseline_warmup * (baseline_bs / float(global_batch)))))
        args.warmup_steps = int(scaled_warmup)
    if rank == 0:
        print(
            "[AUTO-SCALE] "
            f"global_batch={global_batch} (bs={args.batch_size} x world={world_size}) "
            f"| autoscale_lr={args.autoscale_lr} mode={args.autoscale_mode} -> lr={args.lr:.3e} "
            f"| autoscale_warmup={args.autoscale_warmup} -> warmup_steps={args.warmup_steps}",
            flush=True
        )

    # Seeding (different per-rank)
    base_seed = 42 if args.seed is None else int(args.seed)
    seed_this = base_seed + rank
    torch.manual_seed(seed_this)
    torch.cuda.manual_seed_all(seed_this)

    # Datasets
    ds_a = NPZVariantDualBDataset(
        root=args.npz_root,
        radius=args.radius,
        normalize_in=True,
        normalize_gt=True,
        normalize_t2=True,
        numeric_only_subfolders=bool(args.numeric_only),
        seed=args.seed,
        axis_filter=args.axis_filter,
        b_low=float(args.b_low),
        b_high=float(args.b_high),
        defer_norm_to_gpu=True,
    )
    use_mixed = bool(args.npz_root2)
    if use_mixed and len(args.npz_root2) > 0:
        ds_b = NPZVariantDualBDataset(
            root=args.npz_root2,
            radius=args.radius,
            normalize_in=True,
            normalize_gt=True,
            normalize_t2=True,
            numeric_only_subfolders=bool(args.numeric_only2),
            seed=args.seed,
            axis_filter=args.axis_filter,
            b_low=float(args.b_low),
            b_high=float(args.b_high),
            defer_norm_to_gpu=True,
        )
        # If root2 produced no items, disable mixed mode gracefully
        try:
            if len(ds_b) <= 0:
                use_mixed = False
                ds_b = None  # type: ignore[assignment]
        except Exception:
            use_mixed = False
            ds_b = None  # type: ignore[assignment]
    else:
        ds_b = None  # type: ignore[assignment]

    # Splits
    val_len_a = int(len(ds_a) * args.val_split)
    tr_len_a = len(ds_a) - val_len_a
    tr_set_a, va_set_a = random_split(ds_a, [tr_len_a, val_len_a])
    # Preserve dataset-internal order per split (subject-grouped) for hierarchical batching
    try:
        tr_set_a.indices = sorted(tr_set_a.indices)  # type: ignore[attr-defined]
        va_set_a.indices = sorted(va_set_a.indices)  # type: ignore[attr-defined]
    except Exception:
        pass
    if use_mixed and ds_b is not None:
        val_len_b = int(len(ds_b) * args.val_split)
        tr_len_b = len(ds_b) - val_len_b
        tr_set_b, va_set_b = random_split(ds_b, [tr_len_b, val_len_b])
        try:
            tr_set_b.indices = sorted(tr_set_b.indices)  # type: ignore[attr-defined]
            va_set_b.indices = sorted(va_set_b.indices)  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        tr_set_b = None  # type: ignore[assignment]
        va_set_b = None  # type: ignore[assignment]

    def collate(batch):
        keys = batch[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
        return out

    # DataLoaders & Samplers: disable sampler shuffling to keep subject-grouped order from dataset/subset
    tr_sampler_a = torch.utils.data.distributed.DistributedSampler(tr_set_a, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    va_sampler_a = torch.utils.data.distributed.DistributedSampler(va_set_a, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    if use_mixed and ds_b is not None:
        tr_sampler_b = torch.utils.data.distributed.DistributedSampler(tr_set_b, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
        va_sampler_b = torch.utils.data.distributed.DistributedSampler(va_set_b, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    else:
        tr_sampler_b = None
        va_sampler_b = None

    # DataLoaders: increase workers, enable pin_memory and persistent workers
    # Allow env override to throttle I/O pressure
    try:
        num_workers_tr = int(os.environ.get("NUM_WORKERS_TRAIN", os.environ.get("NUM_WORKERS", "4")))
    except Exception:
        num_workers_tr = 4
    try:
        num_workers_va = int(os.environ.get("NUM_WORKERS_VAL", str(max(1, num_workers_tr // 2))))
    except Exception:
        num_workers_va = max(1, num_workers_tr // 2)
    prefetch_factor_tr = int(os.environ.get("PREFETCH_FACTOR_TRAIN", os.environ.get("PREFETCH_FACTOR", "2")))
    prefetch_factor_va = int(os.environ.get("PREFETCH_FACTOR_VAL", os.environ.get("PREFETCH_FACTOR", "2")))
    ld_tr_a = DataLoader(
        tr_set_a,
        batch_size=args.batch_size,
        shuffle=(tr_sampler_a is None),
        sampler=tr_sampler_a,
        num_workers=num_workers_tr,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor_tr,
    )
    ld_va_a = DataLoader(
        va_set_a,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=va_sampler_a,
        num_workers=num_workers_va,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor_va,
    )
    if use_mixed and ds_b is not None:
        ld_tr_b = DataLoader(
            tr_set_b,
            batch_size=args.batch_size,
            shuffle=(tr_sampler_b is None),
            sampler=tr_sampler_b,
            num_workers=num_workers_tr,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor_tr,
        )
        ld_va_b = DataLoader(
            va_set_b,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=va_sampler_b,
            num_workers=num_workers_va,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor_va,
        )
    else:
        ld_tr_b = None  # type: ignore[assignment]
        ld_va_b = None  # type: ignore[assignment]

    # Model
    dwi_ch = (2 * args.radius + 1)  # per DWI stream
    t2_ch = (2 * args.radius + 1)
    model = PHCE2EMageUltraNet(
        dwi_channels=dwi_ch,
        t2_channels=t2_ch,
        base_channels=max(32, args.base_channels),
        latent_dim=max(0, int(args.latent_dim)),
        prompt_k=max(1, int(args.prompt_k)) if args.latent_dim > 0 else 0,
        prompt_temp=float(args.prompt_temp),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    # Resume weights if provided (load before wrapping with DDP)
    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=False)
            if rank == 0:
                print(f"[RESUME] Loaded weights from {args.resume}", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[RESUME] Failed to load checkpoint '{args.resume}': {e}", flush=True)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = bool(device.type == 'cuda' and int(args.amp) == 1)
    scaler = GradScaler(enabled=amp_enabled)
    # Optional warmup + cosine scheduler
    if args.warmup_steps > 0:
        total_steps_est = None  # determined per-epoch later
        def _lr_lambda(step: int) -> float:
            nonlocal total_steps_est
            if total_steps_est is None or total_steps_est <= 0:
                return 1.0
            if step < args.warmup_steps:
                return float(step) / max(1, args.warmup_steps)
            progress = float(step - args.warmup_steps) / max(1, total_steps_est - args.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    else:
        scheduler = None
    # Enable cuDNN benchmark for speed
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # Training (no VDM augmentation in TSE variant)
    # With ADC as primary supervised output we do not need ADC/relative losses helpers here
    def _compute_adc_from_two_b_np(s_low: np.ndarray, b_low: float, s_high: np.ndarray, b_high: float, eps: float = 1e-6) -> np.ndarray:
        den = float(b_high - b_low) if float(b_high - b_low) != 0 else 1.0
        sl = np.clip(s_low.astype(np.float32), eps, 1.0)
        sh = np.clip(s_high.astype(np.float32), eps, 1.0)
        adc = (np.log(sl) - np.log(sh)) / den
        adc = np.clip(adc, 0.0, 0.003).astype(np.float32)
        return adc
    def _adc_vmin_vmax_np(img: np.ndarray, p1: float = 5.0, p99: float = 95.0) -> Tuple[float, float]:
        lo = float(np.percentile(img, p1))
        hi = float(np.percentile(img, p99))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 0.003
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def get_progressive_weights(epoch: int, ssim_weight: float) -> Tuple[float, float]:
        if epoch < 8:
            return 1.0, ssim_weight * 0.1
        if epoch < 15:
            return 0.8, ssim_weight * 0.2
        if epoch < 22:
            return 0.7, ssim_weight * 0.3
        # if epoch < 20:
        #     return 0.6, ssim_weight * 0.4
        return 0.5, ssim_weight * 0.5

    def get_bvalue_weights(epoch: int) -> Tuple[float, float]:
        # Warm-start: favor b50 early, converge to 0.5/0.5
        if epoch < 10:
            return 0.7, 0.3
        if epoch < 20:
            return 0.6, 0.4
        return 0.5, 0.5

    def _train_one_epoch(epoch: int) -> float:
        model.train()
        tot = 0.0
        it = 0
        t0 = time.time()
        # Prepare SSIM metric once (training epoch scope)
        ssim_metric = None
        if args.use_ssim_loss:
            try:
                import torchmetrics  # type: ignore
                ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # type: ignore[attr-defined]
            except Exception:
                ssim_metric = None
        # Note: Removed signal-quality weighted L1 and curriculum losses
        # per-epoch subject-level reshuffle in dataset to implement hierarchical shuffling
        try:
            # reshuffle underlying datasets (affects Subsets) with epoch-dependent seed
            if hasattr(tr_set_a, "dataset") and hasattr(tr_set_a.dataset, "reshuffle_subjects"):
                tr_set_a.dataset.reshuffle_subjects(seed=(args.seed or 42) + int(epoch))  # type: ignore[attr-defined]
            if use_mixed and tr_set_b is not None and hasattr(tr_set_b, "dataset") and hasattr(tr_set_b.dataset, "reshuffle_subjects"):
                tr_set_b.dataset.reshuffle_subjects(seed=(args.seed or 42) + int(epoch) + 12345)  # type: ignore[attr-defined]
        except Exception:
            pass
        # set epoch on samplers for proper sharding (no internal shuffle)
        if tr_sampler_a is not None:
            tr_sampler_a.set_epoch(epoch)
        if tr_sampler_b is not None:
            tr_sampler_b.set_epoch(epoch)
        use_dual = (use_mixed and ld_tr_b is not None and len(ld_tr_b) > 0)
        if use_dual:
            steps_per_epoch = max(len(ld_tr_a), len(ld_tr_b))  # type: ignore[arg-type]
            ita = iter(ld_tr_a)
            itb = iter(ld_tr_b)  # type: ignore[arg-type]
        else:
            steps_per_epoch = len(ld_tr_a)
            ita = iter(ld_tr_a)
        # set total steps for scheduler ETA
        if scheduler is not None:
            nonlocal total_steps_est
            total_steps_est = steps_per_epoch * args.epochs
        # Epoch header (rank 0): align with single-stream logs
        if rank == 0:
            total_steps_hdr = steps_per_epoch * args.epochs
            try:
                ws = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                ws = 1
            print(f"[EPOCH] {epoch}/{args.epochs} | steps_per_epoch={steps_per_epoch} | total_steps={total_steps_hdr} | "
                  f"warmup_steps={args.warmup_steps} (per-rank, world_size={ws}) | amp={int(amp_enabled)}",
                  flush=True)
        data_time_acc = 0.0
        comp_time_acc = 0.0
        for step_idx in range(steps_per_epoch):
            wall_t0 = time.time()
            batch = None
            if use_dual:
                try:
                    ba = next(ita)
                except StopIteration:
                    ita = iter(ld_tr_a)
                    ba = next(ita)
                try:
                    bb = next(itb)  # type: ignore[assignment]
                except StopIteration:
                    itb = iter(ld_tr_b)  # type: ignore[assignment]
                    try:
                        bb = next(itb)  # type: ignore[assignment]
                    except StopIteration:
                        use_dual = False
                        batch = ba
                    else:
                        batch = {k: torch.cat([ba[k], bb[k]], dim=0) for k in ba.keys()}
                else:
                    batch = {k: torch.cat([ba[k], bb[k]], dim=0) for k in ba.keys()}
            if batch is None:
                try:
                    batch = next(ita)
                except StopIteration:
                    ita = iter(ld_tr_a); batch = next(ita)

            # H2D transfer (whitelist to reduce PCIe/H2D overhead)
            keys_to_gpu = {
                'dwi_b50_stack', 'dwi_b1400_stack', 't2_stack',
                'dwi_b50_gt', 'dwi_b1400_gt',
                # single-slice inputs needed for GPU preproc
                'dwi_b50_in', 'dwi_b1400_in',
                'norm_lo_b50_in', 'norm_hi_b50_in',
                'norm_lo_adc_in', 'norm_hi_adc_in',
                'norm_lo_b50_gt', 'norm_hi_b50_gt',
                'norm_lo_adc_gt', 'norm_hi_adc_gt',
            }
            for k in list(batch.keys()):
                if k in keys_to_gpu and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)
            data_time = time.time() - wall_t0

            # Build normalized inputs/targets on device if needed
            _ensure_gpu_preproc(batch, args)
            if args.zero_t2_guidance:
                batch['t2_stack'] = torch.zeros_like(batch['t2_stack'])

            # Forward
            comp_t0 = time.time()
            with autocast(enabled=amp_enabled):
                out = model(
                    batch['dwi_b50_stack'][:, 0: dwi_ch],
                    batch['adc_stack'][:, 0: dwi_ch],
                    batch['t2_stack'][:, 0: t2_ch],
                    None,
                )
            # Multi-scale L1 per b-value
            ms_w1 = float(args.ms_w1)
            ms_w2 = float(args.ms_w2)
            # Per-stream multi-scale L1 (unweighted)
            with autocast(enabled=amp_enabled):
                l1_b50_main = F.l1_loss(out['I_out_b50'], batch['dwi_b50_gt'])
                l1_b50_s1 = F.l1_loss(out['I_s1_b50'], batch['dwi_b50_gt'])
                l1_b50_s2 = F.l1_loss(out['I_s2_b50'], batch['dwi_b50_gt'])
                l1_b50_total = l1_b50_main + ms_w1 * l1_b50_s1 + ms_w2 * l1_b50_s2
                l1_adc_main = F.l1_loss(out['I_out_adc'], batch['adc_gt'])
                l1_adc_s1 = F.l1_loss(out['I_s1_adc'], batch['adc_gt'])
                l1_adc_s2 = F.l1_loss(out['I_s2_adc'], batch['adc_gt'])
                l1_adc_total = l1_adc_main + ms_w1 * l1_adc_s1 + ms_w2 * l1_adc_s2
            # Optional SSIM: compute per-stream and sum (no cross-b weighting)
            with autocast(enabled=amp_enabled):
                if args.use_ssim_loss:
                    # SSIM on b50 only; ADC SSIM disabled
                    if ssim_metric is not None:
                        try:
                            ssim_b50 = 1.0 - ssim_metric(out['I_out_b50'].clamp(0, 1), batch['dwi_b50_gt'].clamp(0, 1))
                        except Exception:
                            ssim_b50 = torch.tensor(0.0, device=device, dtype=l1_b50_total.dtype)
                    else:
                        ssim_b50 = torch.tensor(0.0, device=device, dtype=l1_b50_total.dtype)
                    ssim_adc = torch.tensor(0.0, device=device, dtype=l1_adc_total.dtype)
                    alpha, beta = get_progressive_weights(epoch, float(args.ssim_weight))
                    loss_b50 = alpha * l1_b50_total + beta * ssim_b50
                    loss_adc = alpha * l1_adc_total + beta * ssim_adc
                    loss = loss_b50 + loss_adc
                else:
                    loss = l1_b50_total + l1_adc_total

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            tot += float(loss.item())
            it += 1
            comp_time = time.time() - comp_t0
            data_time_acc += data_time
            comp_time_acc += comp_time
            if (it % 100 == 0 or it <= 5) and (rank == 0):
                elapsed = time.time() - t0
                steps_per_sec = it / max(1e-6, elapsed)
                imgs_per_sec = steps_per_sec * args.batch_size
                eta_epoch = (steps_per_epoch - it) / max(1e-6, steps_per_sec)
                total_done = (epoch - 1) * steps_per_epoch + it
                total_steps = steps_per_epoch * args.epochs
                eta_total = (total_steps - total_done) / max(1e-6, steps_per_sec)
                lr_curr = opt.param_groups[0]["lr"]
                steps_per_sec = it / max(1e-6, (time.time() - t0))
                imgs_per_sec = (args.batch_size * (world_size if use_ddp else 1)) * steps_per_sec
                avg_data = data_time_acc / max(1, it)
                avg_comp = comp_time_acc / max(1, it)
                print(f"[TRAIN] ep={epoch} {it}/{steps_per_epoch} | loss={float(loss.item()):.6f} "
                      f"| L1b50={float(l1_b50_main.item()):.6f} L1adc={float(l1_adc_main.item()):.6f} "
                      f"| lr={lr_curr:.2e} | {steps_per_sec:.2f} steps/s ({imgs_per_sec:.1f} imgs/s) "
                      f"| ETA ep={eta_epoch/60:.1f}m total={eta_total/3600:.1f}h | data={avg_data*1000:.1f} ms/step compute={avg_comp*1000:.1f} ms/step", flush=True)
            if args.max_steps and it >= args.max_steps:
                break
        return tot / max(1, it)

    def _validate(epoch: int) -> float:
        model.eval()
        tot_v = 0.0
        itv = 0
        # Optional SSIM
        ssim_module: Optional[object] = None
        try:
            import torchmetrics  # type: ignore
            ssim_module = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # type: ignore[attr-defined]
        except Exception:
            ssim_module = None
        # Helper to read scalar normalization stats from batch (supports [B] or [B,1])
        def _get_scalar_from_batch(bdict: Dict[str, torch.Tensor], key: str, i: int, default: float = 0.0) -> float:
            if key not in bdict:
                return default
            t = bdict[key]
            try:
                if t.ndim == 1:
                    return float(t[i].detach().cpu().item())
                return float(t[i, 0].detach().cpu().item())
            except Exception:
                try:
                    return float(t.flatten()[i].detach().cpu().item())
                except Exception:
                    return default
        # Accumulators
        ssim_b50_sum = 0.0
        ssim_b14_sum = 0.0
        nmse_b50_num = 0.0
        nmse_b50_den = 0.0
        nmse_b14_num = 0.0
        nmse_b14_den = 0.0
        # Domain-wise (dataset label) stats
        per_label_stats: Dict[int, Dict[str, float]] = {}
        # Visualization setup
        save_vis_left = int(max(1, args.val_vis_n))
        with torch.no_grad():
            for batch in ld_va_a:
                for k in batch:
                    batch[k] = batch[k].to(device)
                _ensure_gpu_preproc(batch, args)
                if args.zero_t2_guidance:
                    batch['t2_stack'] = torch.zeros_like(batch['t2_stack'])
                dom_ids = batch.get('domain_id', None)
                if dom_ids is not None:
                    dom_ids = dom_ids.view(dom_ids.shape[0], -1).squeeze(1).long()
                out = model(
                    batch['dwi_b50_stack'][:, 0: dwi_ch],
                    batch['adc_stack'][:, 0: dwi_ch],
                    batch['t2_stack'][:, 0: t2_ch],
                    None,
                )
                # Validation: report main L1 average (b50 and ADC)
                l1_b50 = F.l1_loss(out['I_out_b50'], batch['dwi_b50_gt'])
                l1_adc = F.l1_loss(out['I_out_adc'], batch['adc_gt'])
                loss = 0.5 * (l1_b50 + l1_adc)
                tot_v += float(loss.item())
                itv += 1
                # Per-label L1 aggregation on main heads
                if 'dataset_label_id' in batch:
                    # Per-sample mean absolute error
                    l1_b50_vec = torch.mean(torch.abs(out['I_out_b50'] - batch['dwi_b50_gt']), dim=(1, 2, 3))
                    l1_adc_vec = torch.mean(torch.abs(out['I_out_adc'] - batch['adc_gt']), dim=(1, 2, 3))
                    lids = batch['dataset_label_id'].view(-1)
                    uniq = torch.unique(lids)
                    for lid_t in uniq:
                        lid = int(lid_t.item())
                        mask = (lids == lid)
                        cnt = int(mask.sum().item())
                        if cnt <= 0:
                            continue
                        e = per_label_stats.get(lid, {"cnt": 0.0, "l1b50": 0.0, "l1adc": 0.0})
                        e["cnt"] += cnt
                        e["l1b50"] += float(l1_b50_vec[mask].sum().item())
                        e["l1adc"] += float(l1_adc_vec[mask].sum().item())
                        per_label_stats[lid] = e
                # SSIM
                if ssim_module is not None:
                    try:
                        ssim_b50 = 1.0 - float(ssim_module(out['I_out_b50'].clamp(0, 1), batch['dwi_b50_gt'].clamp(0, 1)).item())
                        ssim_b50_sum += ssim_b50
                    except Exception:
                        pass
                # NMSE accumulators
                def _acc_nmse(pred: torch.Tensor, gt: torch.Tensor):
                    nonlocal nmse_b50_num, nmse_b50_den, nmse_b14_num, nmse_b14_den
                    num = torch.sum((pred - gt) ** 2).double().item()
                    den = torch.sum(gt ** 2).double().item() + 1e-12
                    return num, den
                n_b50, d_b50 = _acc_nmse(out['I_out_b50'], batch['dwi_b50_gt'])
                n_adc, d_adc = _acc_nmse(out['I_out_adc'], batch['adc_gt'])
                nmse_b50_num += n_b50; nmse_b50_den += d_b50
                nmse_b14_num += n_adc; nmse_b14_den += d_adc
                # Visualization (rank 0 only) - remove VDM panel for TSE variant
                if rank == 0 and args.val_vis and save_vis_left > 0:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')  # type: ignore
                        import matplotlib.pyplot as plt  # type: ignore
                        vis_dir = os.path.join(args.out_dir, f"val_vis_epoch_{epoch:03d}")
                        os.makedirs(vis_dir, exist_ok=True)
                        B = batch['dwi_b50_in'].shape[0] if 'dwi_b50_in' in batch else batch['dwi_b50_stack'].shape[0]
                        # Balanced selection across dataset labels: prefer 1 per label first, then up to 2
                        # Trackers initialized outside loop
                        try:
                            selected_per_label
                        except NameError:
                            from collections import defaultdict
                            selected_per_label = defaultdict(set)  # type: ignore[name-defined]
                            labels_seen_once = set()  # type: ignore[name-defined]
                            max_per_label = 2  # type: ignore[name-defined]
                        for i in range(B):
                            if save_vis_left <= 0:
                                break
                            label_id = int(batch.get('dataset_label_id', torch.tensor([-1], device=device))[i].item()) if 'dataset_label_id' in batch else -1
                            subject_id = int(batch.get('subject_id', torch.tensor([i], device=device))[i].item()) if 'subject_id' in batch else i
                            if subject_id in selected_per_label[label_id]:
                                continue
                            if label_id not in labels_seen_once:
                                labels_seen_once.add(label_id)
                            else:
                                if len(selected_per_label[label_id]) >= max_per_label:
                                    continue
                            # pick central slice from stacks for visualization alignment is already slice-level
                            b50_in_np = (batch['dwi_b50_in'][i, 0] if 'dwi_b50_in' in batch else batch['dwi_b50_stack'][i, args.radius]).detach().cpu().numpy()
                            # High-b IN: prefer provided; otherwise reconstruct from low-b + ADC (physical)
                            try:
                                b14_in_np = (batch['dwi_b1400_in'][i, 0] if 'dwi_b1400_in' in batch else batch['dwi_b1400_stack'][i, args.radius]).detach().cpu().numpy()
                            except Exception:
                                # Fallback reconstruction for IN using physical scales
                                lo_b50_in = _get_scalar_from_batch(batch, 'norm_lo_b50_in', i, 0.0)
                                hi_b50_in = _get_scalar_from_batch(batch, 'norm_hi_b50_in', i, 1.0)
                                lo_adc_in = _get_scalar_from_batch(batch, 'norm_lo_adc_in', i, 0.0)
                                hi_adc_in = _get_scalar_from_batch(batch, 'norm_hi_adc_in', i, 1.0)
                                scale_b50_in = max(1e-6, (hi_b50_in - lo_b50_in))
                                scale_adc_in = max(1e-6, (hi_adc_in - lo_adc_in))
                                adc_in_np_tmp = batch['adc_in'][i, 0].detach().cpu().numpy()
                                adc_in_phys = adc_in_np_tmp * scale_adc_in + lo_adc_in
                                b50_in_phys = b50_in_np * scale_b50_in + lo_b50_in
                                delta_b_in = float(args.b_high) - float(args.b_low)
                                b14_in_phys = b50_in_phys * np.exp(-delta_b_in * adc_in_phys)
                                # Re-normalize for display using b50 IN scale for consistency
                                b14_in_np = np.clip((b14_in_phys - lo_b50_in) / scale_b50_in, 0.0, 1.0)
                            b50_gt_np = batch['dwi_b50_gt'][i, 0].detach().cpu().numpy()
                            b14_gt_np = batch['dwi_b1400_gt'][i, 0].detach().cpu().numpy()
                            b50_out_np = out['I_out_b50'][i, 0].detach().cpu().numpy()
                            # Denormalize predicted b50/ADC using GT stats, compute physical b1400, then re-normalize for display
                            lo_b50_gt = _get_scalar_from_batch(batch, 'norm_lo_b50_gt', i, 0.0)
                            hi_b50_gt = _get_scalar_from_batch(batch, 'norm_hi_b50_gt', i, 1.0)
                            lo_adc_gt = _get_scalar_from_batch(batch, 'norm_lo_adc_gt', i, 0.0)
                            hi_adc_gt = _get_scalar_from_batch(batch, 'norm_hi_adc_gt', i, 1.0)
                            scale_b50_gt = max(1e-6, (hi_b50_gt - lo_b50_gt))
                            scale_adc_gt = max(1e-6, (hi_adc_gt - lo_adc_gt))
                            b50_out_phys = b50_out_np * scale_b50_gt + lo_b50_gt
                            adc_out_np = out['I_out_adc'][i, 0].detach().cpu().numpy()
                            adc_out_phys = adc_out_np * scale_adc_gt + lo_adc_gt
                            delta_b = float(args.b_high) - float(args.b_low)
                            b14_out_phys = b50_out_phys * np.exp(-delta_b * adc_out_phys)
                            # Re-normalize b1400 with the same (b50_gt) scale for consistent panel display
                            b14_out_np = np.clip((b14_out_phys - lo_b50_gt) / scale_b50_gt, 0.0, 1.0)
                            t2_mid_np = batch['t2_stack'][i, args.radius].detach().cpu().numpy()
                            # no VDM in this variant
                            # Build ADC maps (grayscale with per-image dynamic range)
                            # Denormalize ADC to physical scale for visualization consistency
                            lo_adc_in = _get_scalar_from_batch(batch, 'norm_lo_adc_in', i, 0.0)
                            hi_adc_in = _get_scalar_from_batch(batch, 'norm_hi_adc_in', i, 1.0)
                            lo_adc_in = _get_scalar_from_batch(batch, 'norm_lo_adc_in', i, 0.0)
                            hi_adc_in = _get_scalar_from_batch(batch, 'norm_hi_adc_in', i, 1.0)
                            scale_adc_in = max(1e-6, (hi_adc_in - lo_adc_in))
                            adc_in_np_n = batch['adc_in'][i, 0].detach().cpu().numpy()
                            adc_in_phys = adc_in_np_n * scale_adc_in + lo_adc_in
                            adc_out_np_n = out['I_out_adc'][i, 0].detach().cpu().numpy()
                            adc_out_phys = adc_out_np_n * scale_adc_gt + lo_adc_gt
                            adc_gt_np_n = batch['adc_gt'][i, 0].detach().cpu().numpy()
                            adc_gt_phys = adc_gt_np_n * scale_adc_gt + lo_adc_gt
                            vmin_in, vmax_in = _adc_vmin_vmax_np(adc_in_phys)
                            vmin_out, vmax_out = _adc_vmin_vmax_np(adc_out_phys)
                            vmin_gt, vmax_gt = _adc_vmin_vmax_np(adc_gt_phys)
                            # Panel: 3 rows x 4 cols (b50 row, b1400 row, ADC row, without VDM)
                            import matplotlib.pyplot as plt  # type: ignore
                            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                            # Use gamma compression to improve visibility of low-to-mid ADC regions
                            try:
                                from matplotlib.colors import PowerNorm  # type: ignore
                                norm_in = PowerNorm(gamma=0.6, vmin=vmin_in, vmax=vmax_in)
                                norm_out = PowerNorm(gamma=0.6, vmin=vmin_out, vmax=vmax_out)
                                norm_gt = PowerNorm(gamma=0.6, vmin=vmin_gt, vmax=vmax_gt)
                            except Exception:
                                norm_in = None
                                norm_out = None
                                norm_gt = None
                            # Row 0: b50
                            axes[0, 0].imshow(b50_in_np, cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('b50 IN'); axes[0,0].axis('off')
                            axes[0, 1].imshow(b50_out_np, cmap='gray', vmin=0, vmax=1); axes[0,1].set_title('b50 OUT'); axes[0,1].axis('off')
                            axes[0, 2].imshow(b50_gt_np, cmap='gray', vmin=0, vmax=1); axes[0,2].set_title('b50 GT'); axes[0,2].axis('off')
                            axes[0, 3].imshow(t2_mid_np, cmap='gray', vmin=0, vmax=1); axes[0,3].set_title('T2 mid'); axes[0,3].axis('off')
                            # Row 1: b1400
                            axes[1, 0].imshow(b14_in_np, cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('b1400 IN'); axes[1,0].axis('off')
                            axes[1, 1].imshow(b14_out_np, cmap='gray', vmin=0, vmax=1); axes[1,1].set_title('b1400 OUT'); axes[1,1].axis('off')
                            axes[1, 2].imshow(b14_gt_np, cmap='gray', vmin=0, vmax=1); axes[1,2].set_title('b1400 GT'); axes[1,2].axis('off')
                            axes[1, 3].imshow(t2_mid_np, cmap='gray', vmin=0, vmax=1); axes[1,3].set_title('T2 mid'); axes[1,3].axis('off')
                            # Row 2: ADC (grayscale per-image range)
                            axes[2, 0].imshow(adc_in_phys, cmap='gray', vmin=None if norm_in is not None else vmin_in, vmax=None if norm_in is not None else vmax_in, norm=norm_in); axes[2,0].set_title('ADC IN'); axes[2,0].axis('off')
                            axes[2, 1].imshow(adc_out_phys, cmap='gray', vmin=None if norm_out is not None else vmin_out, vmax=None if norm_out is not None else vmax_out, norm=norm_out); axes[2,1].set_title('ADC OUT'); axes[2,1].axis('off')
                            axes[2, 2].imshow(adc_gt_phys, cmap='gray', vmin=None if norm_gt is not None else vmin_gt, vmax=None if norm_gt is not None else vmax_gt, norm=norm_gt); axes[2,2].set_title('ADC GT'); axes[2,2].axis('off')
                            axes[2, 3].axis('off')
                            plt.tight_layout()
                            fig.savefig(os.path.join(vis_dir, f"case_ep{epoch:03d}_{save_vis_left:02d}_label{label_id}_sub{subject_id}.png"), dpi=140)
                            plt.close(fig)
                            selected_per_label[label_id].add(subject_id)
                            save_vis_left -= 1
                    except Exception as e:
                        # Make the failure visible in logs instead of silently skipping
                        print(f"[VAL][WARN] Failed to save visualization (epoch={epoch}): {e}", flush=True)
        # Aggregate metrics
        avg_loss = tot_v / max(1, itv)
        if rank == 0:
            try:
                nmse_b50 = nmse_b50_num / max(1e-12, nmse_b50_den)
                nmse_adc = nmse_b14_num / max(1e-12, nmse_b14_den)
                if ssim_module is not None and itv > 0:
                    ssim_b50_avg = ssim_b50_sum / itv
                    print(f"[VAL] ep={epoch} | L1={avg_loss:.6f} | NMSE(b50)={nmse_b50:.6f} NMSE(ADC)={nmse_adc:.6f} "
                          f"| (1-SSIM) b50={ssim_b50_avg:.6f}", flush=True)
                else:
                    print(f"[VAL] ep={epoch} | L1={avg_loss:.6f} | NMSE(b50)={nmse_b50:.6f} NMSE(ADC)={nmse_adc:.6f}", flush=True)
                # Print per-label L1 stats (main heads)
                for lid in sorted(per_label_stats.keys()):
                    s = per_label_stats[lid]
                    cnt = max(1.0, s["cnt"])
                    l1b50_mean = s["l1b50"] / cnt
                    l1adc_mean = s["l1adc"] / cnt
                    label_name = ("Local_train_data2_preprocessed" if lid == 0 else
                                  "preprocessed_diease_prostate_outputs" if lid == 1 else
                                  "preprocessed_fastmri_prostate" if lid == 2 else
                                  f"label{lid}")
                    print(f"[VAL][domain] ep={epoch} | {label_name} | L1_b50={l1b50_mean:.6f} L1_adc={l1adc_mean:.6f} main={0.5*(l1b50_mean+l1adc_mean):.6f}",
                          flush=True)
            except Exception:
                pass
        return avg_loss

    best = float('inf')
    top_records: list = []  # (val, epoch, ckpt_path)
    for ep in range(1, args.epochs + 1):
        tr = _train_one_epoch(ep)
        va = _validate(ep)
        if rank == 0:
            print(f"[EPOCH] {ep:03d} | train {tr:.6f} | val {va:.6f}", flush=True)
            ckpt = os.path.join(args.out_dir, f"mageultra_epoch_{ep:03d}.pt")
            # unwrap if DDP
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({'epoch': ep, 'model': state_dict, 'opt': opt.state_dict(), 'tr': tr, 'va': va}, ckpt)
            if va < best:
                best = va
                best_path = os.path.join(args.out_dir, f"mageultra_best.pt")
                torch.save({'epoch': ep, 'model': state_dict, 'opt': opt.state_dict(), 'tr': tr, 'va': va}, best_path)
                print(f"[CHECKPOINT] Updated best to epoch {ep:03d} (val={va:.6f}) -> {os.path.basename(best_path)}")
            # Maintain and report top-3 checkpoints
            try:
                top_records.append((float(va), int(ep), ckpt))
                top_records = sorted(top_records, key=lambda x: x[0])[:3]
                print("[CHECKPOINT] Top checkpoints:", flush=True)
                for rank_i, (v, epp, pathp) in enumerate(top_records, start=1):
                    print(f"  #{rank_i}: epoch {epp:03d} (val={v:.6f}) -> {os.path.basename(pathp)}", flush=True)
                # Write summary file and convenience copies
                try:
                    with open(os.path.join(args.out_dir, "top_checkpoints.txt"), "w", encoding="utf-8") as ftop:
                        ftop.write("Top checkpoints (lowest val first):\n")
                        for rank_i, (v, epp, pathp) in enumerate(top_records, start=1):
                            ftop.write(f"#{rank_i}: epoch {epp:03d} (val={v:.6f}) -> {os.path.basename(pathp)}\n")
                    # Copy to mageultra_top{i}.pt
                    for i, (_, _, pathp) in enumerate(top_records, start=1):
                        dst = os.path.join(args.out_dir, f"mageultra_top{i}.pt")
                        shutil.copyfile(pathp, dst)
                except Exception:
                    pass
            except Exception:
                pass

    # Clean up DDP
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
