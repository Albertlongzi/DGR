import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler


def _normalize_tensor(
    t: torch.Tensor,
    target_batch_size: int,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
    keep_channels: bool = False,
) -> torch.Tensor:
    """
    Squeeze redundant dimensions, ensure BCHW layout, and match batch/spatial sizes.
    """
    while t.ndim > 4:
        squeezed = False
        for dim in range(2, t.ndim):
            if t.shape[dim] == 1:
                t = t.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            mid = t.shape[-1] // 2
            t = t.select(-1, mid)
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    if not keep_channels and t.shape[1] != 1:
        t = t[:, :1]
    if t.shape[0] != target_batch_size:
        if t.shape[0] == 1:
            t = t.expand(target_batch_size, -1, -1, -1)
        else:
            t = t[:target_batch_size]
    if t.shape[-2:] != target_hw:
        t = F.interpolate(t, size=target_hw, mode=mode, align_corners=(mode == "bilinear"))
    return t

def _maybe_crop(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    if x.shape[-2:] == (orig_h, orig_w):
        return x
    H, W = x.shape[-2], x.shape[-1]
    sh = max(0, (H - orig_h) // 2)
    sw = max(0, (W - orig_w) // 2)
    return x[:, :, sh:sh + orig_h, sw:sw + orig_w]


# ==============================================================================
# Legacy sampler for DiffusionUNetWithFusion (DWI+T2 conditioning)
# ==============================================================================
def _run_img2img_sampling(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int,
    strength: float,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """Legacy sampler for models that use DWI+T2 conditioning."""
    device = next(model.parameters()).device
    dwi_cond = batch["dwi_in"].to(device)
    t2_stack = batch.get("t2_stack")
    if t2_stack is not None:
        t2_stack = t2_stack.to(device)

    target_batch = dwi_cond.shape[0] if dwi_cond.ndim == 4 else 1
    target_hw = dwi_cond.shape[-2:] if dwi_cond.ndim >= 3 else (256, 256)

    dwi_cond = _normalize_tensor(dwi_cond, target_batch, target_hw, mode="bilinear", keep_channels=True)

    if t2_stack is not None:
        if t2_stack.ndim == 5:
            t2_stack = t2_stack.squeeze(1)
        mid = t2_stack.shape[1] // 2
        t2_image = t2_stack[:, mid:mid + 1]
        t2_image = _normalize_tensor(t2_image, target_batch, target_hw, mode="bilinear")
    else:
        t2_image = None

    if t2_image is not None:
        init_latent = t2_image.repeat(1, dwi_cond.shape[1], 1, 1).clone()
    else:
        init_latent = dwi_cond.clone()

    try:
        backbone = getattr(model, "module", model)
        if hasattr(backbone, "unet"):
            backbone = backbone.unet
        down_levels = len(getattr(backbone, "down_blocks"))
        req_mult = 2 ** down_levels
    except Exception:
        req_mult = 16
    orig_h, orig_w = init_latent.shape[-2], init_latent.shape[-1]
    pad_h = (req_mult - (orig_h % req_mult)) % req_mult
    pad_w = (req_mult - (orig_w % req_mult)) % req_mult
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        init_latent = F.pad(init_latent, (pad_left, pad_right, pad_top, pad_bottom))
        dwi_cond = F.pad(dwi_cond, (pad_left, pad_right, pad_top, pad_bottom))
        if t2_image is not None:
            t2_image = F.pad(t2_image, (pad_left, pad_right, pad_top, pad_bottom))
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    scheduler.set_timesteps(steps, device=device)
    t_start = int(max(0.0, min(1.0, strength)) * steps)
    start_idx = max(0, steps - t_start)
    timesteps = scheduler.timesteps[start_idx:]

    if t_start == 0:
        if pad_h or pad_w:
            init_latent = _maybe_crop(init_latent, orig_h, orig_w)
        return init_latent
    elif t_start >= steps:
        x = torch.randn_like(init_latent)
    else:
        init_noise = torch.randn_like(init_latent)
        t0 = timesteps[0]
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            if isinstance(t0, torch.Tensor):
                t_noise = t0.unsqueeze(0)
            else:
                t_noise = torch.tensor([t0], device=init_latent.device, dtype=torch.long)
        else:
            t_noise = t0
        x = scheduler.add_noise(init_latent, init_noise, t_noise)

    for t in timesteps:
        noise_pred = model(x, dwi_cond, t2_image, t).sample
        step_kwargs = {}
        # eta is only supported by DDIM scheduler, not DPMSolver
        if eta is not None and isinstance(scheduler, DDIMScheduler):
            step_kwargs["eta"] = eta
        x = scheduler.step(noise_pred, t, x, **step_kwargs).prev_sample

    if pad_h or pad_w:
        x = _maybe_crop(x, orig_h, orig_w)
    return x


# ==============================================================================
# NEW: T2-Only sampler for DiffusionUNetT2Only
# ==============================================================================
def _run_t2only_sampling(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int,
    strength: float,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    Sampler for T2-only conditioning model (DiffusionUNetT2Only).
    
    Workflow:
      1. If 'cnn_init' is in batch, use it as the starting point (CNN preprocessed)
      2. Otherwise, use T2 repeated as init_latent (SDEdit-style)
      3. Add noise to init_latent based on strength
      4. Denoise with T2 conditioning only
    """
    device = next(model.parameters()).device
    
    # Get T2 conditioning
    t2_stack = batch.get("t2_stack")
    if t2_stack is not None:
        t2_stack = t2_stack.to(device)
        if t2_stack.ndim == 5:
            t2_stack = t2_stack.squeeze(1)
        mid = t2_stack.shape[1] // 2
        t2_image = t2_stack[:, mid:mid + 1]
    else:
        t2_image = None
    
    # Determine init_latent source
    cnn_init = batch.get("cnn_init")
    if cnn_init is not None:
        # Use CNN output as starting point
        cnn_init = cnn_init.to(device)
        target_batch = cnn_init.shape[0]
        target_hw = cnn_init.shape[-2:]
        init_latent = cnn_init.clone()
    elif t2_image is not None:
        # SDEdit style: use T2 repeated as init
        target_batch = t2_image.shape[0]
        target_hw = t2_image.shape[-2:]
        init_latent = t2_image.repeat(1, 2, 1, 1).clone()  # [B, 2, H, W]
    else:
        # Fallback: use dwi_gt if available, otherwise random
        dwi_gt = batch.get("dwi_gt")
        if dwi_gt is not None:
            dwi_gt = dwi_gt.to(device)
            target_batch = dwi_gt.shape[0]
            target_hw = dwi_gt.shape[-2:]
            init_latent = dwi_gt.clone()
        else:
            raise ValueError("No valid init_latent source: need cnn_init, t2_stack, or dwi_gt")
    
    # Normalize T2 image
    if t2_image is not None:
        t2_image = _normalize_tensor(t2_image, target_batch, target_hw, mode="bilinear")
    
    # Compute padding
    try:
        backbone = getattr(model, "module", model)
        if hasattr(backbone, "unet"):
            backbone = backbone.unet
        down_levels = len(getattr(backbone, "down_blocks"))
        req_mult = 2 ** down_levels
    except Exception:
        req_mult = 16
    
    orig_h, orig_w = init_latent.shape[-2], init_latent.shape[-1]
    pad_h = (req_mult - (orig_h % req_mult)) % req_mult
    pad_w = (req_mult - (orig_w % req_mult)) % req_mult
    
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        init_latent = F.pad(init_latent, (pad_left, pad_right, pad_top, pad_bottom))
        if t2_image is not None:
            t2_image = F.pad(t2_image, (pad_left, pad_right, pad_top, pad_bottom))
    
    # Setup scheduler
    scheduler.set_timesteps(steps, device=device)
    t_start = int(max(0.0, min(1.0, strength)) * steps)
    start_idx = max(0, steps - t_start)
    timesteps = scheduler.timesteps[start_idx:]
    
    if t_start == 0:
        # No denoising needed
        if pad_h or pad_w:
            init_latent = _maybe_crop(init_latent, orig_h, orig_w)
        return init_latent
    elif t_start >= steps:
        # Full noise
        x = torch.randn_like(init_latent)
    else:
        # Partial noise (SDEdit style)
        init_noise = torch.randn_like(init_latent)
        t0 = timesteps[0]
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            if isinstance(t0, torch.Tensor):
                t_noise = t0.unsqueeze(0)
            else:
                t_noise = torch.tensor([t0], device=init_latent.device, dtype=torch.long)
        else:
            t_noise = t0
        x = scheduler.add_noise(init_latent, init_noise, t_noise)
    
    # Denoising loop with T2-only conditioning
    for t in timesteps:
        # T2-only model: forward(noisy_latent, t2_image, timestep)
        noise_pred = model(x, t2_image, t).sample
        step_kwargs = {}
        # eta is only supported by DDIM scheduler, not DPMSolver
        if eta is not None and isinstance(scheduler, DDIMScheduler):
            step_kwargs["eta"] = eta
        x = scheduler.step(noise_pred, t, x, **step_kwargs).prev_sample
    
    if pad_h or pad_w:
        x = _maybe_crop(x, orig_h, orig_w)
    return x


# ==============================================================================
# Public API
# ==============================================================================
@torch.no_grad()
def sample_with_ddim(
    model,
    scheduler: DDIMScheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Deterministic DDIM sampling wrapper (legacy, for DWI+T2 conditioning).
    """
    return _run_img2img_sampling(model, scheduler, batch, steps=steps, strength=strength, eta=eta)


@torch.no_grad()
def sample_with_ddpm(
    model,
    scheduler: DDPMScheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
) -> torch.Tensor:
    """
    Stochastic DDPM sampling wrapper (legacy, for DWI+T2 conditioning).
    """
    return _run_img2img_sampling(model, scheduler, batch, steps=steps, strength=strength)


@torch.no_grad()
def sample_with_dpmsolver(
    model,
    scheduler: DPMSolverMultistepScheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
) -> torch.Tensor:
    """
    DPMSolver++ multistep sampling wrapper (legacy, for DWI+T2 conditioning).
    """
    return _run_img2img_sampling(model, scheduler, batch, steps=steps, strength=strength)


@torch.no_grad()
def sample_with_t2only(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    T2-only conditioning sampler for DiffusionUNetT2Only.
    
    Args:
        model: DiffusionUNetT2Only model
        scheduler: Any diffusers scheduler (DDPM, DDIM, DPMSolver)
        batch: Dict with keys:
            - 't2_stack': T2 volume [B, C, H, W] or [B, 1, C, H, W]
            - 'cnn_init' (optional): CNN preprocessed output [B, 2, H, W]
            - 'dwi_gt' (optional): Ground truth for fallback
        steps: Number of denoising steps
        strength: Noise strength (0.0 = no noise, 1.0 = full noise)
        eta: DDIM eta parameter (only used for DDIM scheduler)
    
    Returns:
        Denoised output [B, 2, H, W]
    """
    return _run_t2only_sampling(model, scheduler, batch, steps=steps, strength=strength, eta=eta)


# ==============================================================================
# NEW: T2 + CNN Conditioning Sampler
# ==============================================================================
def _run_t2_and_cnn_sampling(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int,
    strength: float,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    Sampler for T2 + CNN conditioning model (DiffusionUNetT2AndCNN).
    
    Workflow:
      1. Use 'cnn_init' as the starting point (CNN preprocessed output)
      2. Add noise to cnn_init based on strength (SDEdit style)
      3. Denoise with (T2, cnn_output) conditioning
    
    The model refines CNN output using T2 anatomical guidance.
    """
    device = next(model.parameters()).device
    
    # Get T2 conditioning
    t2_stack = batch.get("t2_stack")
    if t2_stack is not None:
        t2_stack = t2_stack.to(device)
        if t2_stack.ndim == 5:
            t2_stack = t2_stack.squeeze(1)
        mid = t2_stack.shape[1] // 2
        t2_image = t2_stack[:, mid:mid + 1]
    else:
        t2_image = None
    
    # Get CNN output for conditioning (what CNN already predicted)
    cnn_output = batch.get("cnn_output")
    if cnn_output is not None:
        cnn_output = cnn_output.to(device)
    
    # Get init_latent (CNN output to refine)
    cnn_init = batch.get("cnn_init")
    if cnn_init is not None:
        cnn_init = cnn_init.to(device)
        target_batch = cnn_init.shape[0]
        target_hw = cnn_init.shape[-2:]
        init_latent = cnn_init.clone()
    else:
        raise ValueError("cnn_init is required for T2+CNN model sampling")
    
    # If cnn_output not provided separately, use cnn_init
    if cnn_output is None:
        cnn_output = init_latent.clone()
    
    # Normalize T2 image
    if t2_image is not None:
        t2_image = _normalize_tensor(t2_image, target_batch, target_hw, mode="bilinear")
    
    # Compute padding
    try:
        backbone = getattr(model, "module", model)
        if hasattr(backbone, "unet"):
            backbone = backbone.unet
        down_levels = len(getattr(backbone, "down_blocks"))
        req_mult = 2 ** down_levels
    except Exception:
        req_mult = 16
    
    orig_h, orig_w = init_latent.shape[-2], init_latent.shape[-1]
    pad_h = (req_mult - (orig_h % req_mult)) % req_mult
    pad_w = (req_mult - (orig_w % req_mult)) % req_mult
    
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        init_latent = F.pad(init_latent, (pad_left, pad_right, pad_top, pad_bottom))
        cnn_output = F.pad(cnn_output, (pad_left, pad_right, pad_top, pad_bottom))
        if t2_image is not None:
            t2_image = F.pad(t2_image, (pad_left, pad_right, pad_top, pad_bottom))
    
    # Setup scheduler
    scheduler.set_timesteps(steps, device=device)
    t_start = int(max(0.0, min(1.0, strength)) * steps)
    start_idx = max(0, steps - t_start)
    timesteps = scheduler.timesteps[start_idx:]
    
    if t_start == 0:
        # No denoising needed - return CNN output as-is
        if pad_h or pad_w:
            init_latent = _maybe_crop(init_latent, orig_h, orig_w)
        return init_latent
    elif t_start >= steps:
        # Full noise
        x = torch.randn_like(init_latent)
    else:
        # Partial noise (SDEdit style) - add noise to CNN output
        init_noise = torch.randn_like(init_latent)
        t0 = timesteps[0]
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            if isinstance(t0, torch.Tensor):
                t_noise = t0.unsqueeze(0)
            else:
                t_noise = torch.tensor([t0], device=init_latent.device, dtype=torch.long)
        else:
            t_noise = t0
        x = scheduler.add_noise(init_latent, init_noise, t_noise)
    
    # Denoising loop with T2 + CNN conditioning
    for t in timesteps:
        # T2+CNN model: forward(noisy_latent, t2_image, cnn_output, timestep)
        noise_pred = model(x, t2_image, cnn_output, t).sample
        step_kwargs = {}
        # eta is only supported by DDIM scheduler, not DPMSolver
        if eta is not None and isinstance(scheduler, DDIMScheduler):
            step_kwargs["eta"] = eta
        x = scheduler.step(noise_pred, t, x, **step_kwargs).prev_sample
    
    if pad_h or pad_w:
        x = _maybe_crop(x, orig_h, orig_w)
    return x


@torch.no_grad()
def sample_with_t2_and_cnn(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    T2 + CNN conditioning sampler for DiffusionUNetT2AndCNN (noise prediction).
    
    This sampler REFINES CNN output using T2 anatomical guidance.
    The diffusion model learns "what the CNN got wrong" and corrects it.
    
    Args:
        model: DiffusionUNetT2AndCNN model
        scheduler: Any diffusers scheduler (DDPM, DDIM, DPMSolver)
        batch: Dict with keys:
            - 't2_stack': T2 volume [B, C, H, W] or [B, 1, C, H, W]
            - 'cnn_output': CNN output for conditioning [B, 2, H, W]
            - 'cnn_init': CNN output as init_latent [B, 2, H, W]
        steps: Number of denoising steps
        strength: Noise strength (0.0 = no change, 1.0 = full reconstruction)
            - Low strength (0.1-0.3): Minor refinement of CNN output
            - Medium strength (0.3-0.5): Moderate correction
            - High strength (0.5-0.8): Major reconstruction with CNN guidance
        eta: DDIM eta parameter (only used for DDIM scheduler)
    
    Returns:
        Refined output [B, 2, H, W]
    """
    return _run_t2_and_cnn_sampling(model, scheduler, batch, steps=steps, strength=strength, eta=eta)


# ==============================================================================
# NEW: Clean-Image-Prediction Sampler (prediction_type="sample")
# ==============================================================================
def _run_t2_and_cnn_sampling_clean(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int,
    strength: float,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    Sampler for T2 + CNN conditioning model with CLEAN-IMAGE PREDICTION.
    
    Key difference from _run_t2_and_cnn_sampling:
    - The model outputs predicted clean sample, not noise
    - Scheduler must have prediction_type="sample"
    - The scheduler.step() handles the math correctly for sample prediction
    
    Reference: arXiv:2511.13720 - Learning on the anatomical manifold
    """
    device = next(model.parameters()).device
    
    # Get T2 conditioning
    t2_stack = batch.get("t2_stack")
    if t2_stack is not None:
        t2_stack = t2_stack.to(device)
        if t2_stack.ndim == 5:
            t2_stack = t2_stack.squeeze(1)
        mid = t2_stack.shape[1] // 2
        t2_image = t2_stack[:, mid:mid + 1]
    else:
        t2_image = None
    
    # Get CNN output for conditioning
    cnn_output = batch.get("cnn_output")
    if cnn_output is not None:
        cnn_output = cnn_output.to(device)
    
    # Get init_latent (CNN output to refine)
    cnn_init = batch.get("cnn_init")
    if cnn_init is not None:
        cnn_init = cnn_init.to(device)
        target_batch = cnn_init.shape[0]
        target_hw = cnn_init.shape[-2:]
        init_latent = cnn_init.clone()
    else:
        raise ValueError("cnn_init is required for T2+CNN model sampling")
    
    if cnn_output is None:
        cnn_output = init_latent.clone()
    
    # Normalize T2 image
    if t2_image is not None:
        t2_image = _normalize_tensor(t2_image, target_batch, target_hw, mode="bilinear")
    
    # Compute padding
    try:
        backbone = getattr(model, "module", model)
        if hasattr(backbone, "unet"):
            backbone = backbone.unet
        down_levels = len(getattr(backbone, "down_blocks"))
        req_mult = 2 ** down_levels
    except Exception:
        req_mult = 16
    
    orig_h, orig_w = init_latent.shape[-2], init_latent.shape[-1]
    pad_h = (req_mult - (orig_h % req_mult)) % req_mult
    pad_w = (req_mult - (orig_w % req_mult)) % req_mult
    
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        init_latent = F.pad(init_latent, (pad_left, pad_right, pad_top, pad_bottom))
        cnn_output = F.pad(cnn_output, (pad_left, pad_right, pad_top, pad_bottom))
        if t2_image is not None:
            t2_image = F.pad(t2_image, (pad_left, pad_right, pad_top, pad_bottom))
    
    # Setup scheduler
    scheduler.set_timesteps(steps, device=device)
    t_start = int(max(0.0, min(1.0, strength)) * steps)
    start_idx = max(0, steps - t_start)
    timesteps = scheduler.timesteps[start_idx:]
    
    if t_start == 0:
        if pad_h or pad_w:
            init_latent = _maybe_crop(init_latent, orig_h, orig_w)
        return init_latent
    elif t_start >= steps:
        x = torch.randn_like(init_latent)
    else:
        init_noise = torch.randn_like(init_latent)
        t0 = timesteps[0]
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            if isinstance(t0, torch.Tensor):
                t_noise = t0.unsqueeze(0)
            else:
                t_noise = torch.tensor([t0], device=init_latent.device, dtype=torch.long)
        else:
            t_noise = t0
        x = scheduler.add_noise(init_latent, init_noise, t_noise)
    
    # Denoising loop - model outputs clean sample prediction
    # The scheduler.step() handles the math for prediction_type="sample"
    for t in timesteps:
        # Model predicts the CLEAN SAMPLE directly, not noise
        sample_pred = model(x, t2_image, cnn_output, t).sample
        
        step_kwargs = {}
        if eta is not None and isinstance(scheduler, DDIMScheduler):
            step_kwargs["eta"] = eta
        
        # Scheduler handles sample prediction internally
        # It computes: x_{t-1} using the predicted x_0 (sample_pred)
        x = scheduler.step(sample_pred, t, x, **step_kwargs).prev_sample
    
    if pad_h or pad_w:
        x = _maybe_crop(x, orig_h, orig_w)
    return x


@torch.no_grad()
def sample_with_t2_and_cnn_clean(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 50,
    strength: float = 0.8,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    T2 + CNN conditioning sampler for CLEAN-IMAGE PREDICTION diffusion.
    
    This is for models trained with prediction_type="sample".
    The model directly predicts clean images instead of noise.
    
    Key benefits (per arXiv:2511.13720):
    - Learning on the anatomical manifold
    - More stable training for medical images
    - Better preservation of anatomical structures
    
    Args:
        model: DiffusionUNetT2AndCNN model (trained with prediction_type="sample")
        scheduler: Scheduler with prediction_type="sample" (DDPM, DDIM, DPMSolver)
        batch: Dict with keys:
            - 't2_stack': T2 volume
            - 'cnn_output': CNN output for conditioning
            - 'cnn_init': CNN output as init_latent
        steps: Number of denoising steps
        strength: Noise strength (0.0 = no change, 1.0 = full reconstruction)
        eta: DDIM eta parameter
    
    Returns:
        Refined output [B, 2, H, W]
    """
    return _run_t2_and_cnn_sampling_clean(model, scheduler, batch, steps=steps, strength=strength, eta=eta)


@torch.no_grad()
def fast_validation(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps: int = 10,
    strength: float = 0.8,
    use_ddim: bool = True,
) -> float:
    """
    Fast validation using reduced steps for training monitoring.
    
    Args:
        model: DiffusionUNetWithFusion module
        scheduler: DDIM or DDPM scheduler
        batch: validation batch
        steps: number of sampling steps (reduced for speed)
        strength: strength of noise (0.0 = no noise, 1.0 = full noise)
        use_ddim: whether to use DDIM (faster) or DDPM
    
    Returns:
        l1_loss: L1 loss between predicted and ground truth
    """
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        x_pred = sample_with_dpmsolver(model, scheduler, batch, steps=steps, strength=strength)
    elif use_ddim and isinstance(scheduler, DDIMScheduler):
        x_pred = sample_with_ddim(model, scheduler, batch, steps=steps, strength=strength)
    else:
        x_pred = sample_with_ddpm(model, scheduler, batch, steps=steps, strength=strength)
    
    return F.l1_loss(x_pred, batch["dwi_gt"]).item()


@torch.no_grad()
def comprehensive_validation(
    model,
    scheduler,
    batch: Dict[str, torch.Tensor],
    steps_list: list = [5, 10, 20, 50],
    strength: float = 0.8,
    use_ddim: bool = True,
) -> Dict[str, float]:
    """
    Comprehensive validation with multiple step counts.
    
    Args:
        model: DiffusionUNetWithFusion module
        scheduler: DDIM or DDPM scheduler
        batch: validation batch
        steps_list: list of step counts to test
        strength: strength of noise (0.0 = no noise, 1.0 = full noise)
        use_ddim: whether to use DDIM or DDPM
    
    Returns:
        dict: validation metrics for different step counts
    """
    results = {}
    
    for steps in steps_list:
        if isinstance(scheduler, DPMSolverMultistepScheduler):
            x_pred = sample_with_dpmsolver(model, scheduler, batch, steps=steps, strength=strength)
        elif use_ddim and isinstance(scheduler, DDIMScheduler):
            x_pred = sample_with_ddim(model, scheduler, batch, steps=steps, strength=strength)
        else:
            x_pred = sample_with_ddpm(model, scheduler, batch, steps=steps, strength=strength)
        
        l1_loss = F.l1_loss(x_pred, batch["dwi_gt"]).item()
        mse_loss = F.mse_loss(x_pred, batch["dwi_gt"]).item()
        
        results[f'val_l1_{steps}steps'] = l1_loss
        results[f'val_mse_{steps}steps'] = mse_loss
    
    return results
