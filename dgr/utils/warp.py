import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_vdm_from_b0(field_map_hz: np.ndarray,
                         dwell_time: float = 2.2266e-4,
                         tramp: float = 200.0,
                         gamma: float = 42.56,
                         scaling_factor: float = 1873800.0) -> np.ndarray:
    """
    Compute a voxel displacement map (VDM) along image x-axis from B0 (Hz).

    Args:
        field_map_hz: [H,W,Z] frequency map in Hz registered to target space.
        dwell_time: dwell time per pixel (s).
        tramp: ramp time constant.
        gamma: gyromagnetic ratio constant (as used in user's script).
        scaling_factor: overall scaling (empirical compatibility).

    Returns:
        vdm: [H,W,Z] displacement in pixels (float32), along +x direction.
    """
    fmap = np.asarray(field_map_hz, dtype=np.float64)
    H, W, Z = fmap.shape
    n = (np.arange(W, dtype=np.float64) + 1.0).reshape(1, W, 1)
    factor = (2.0 * float(tramp) + n * float(dwell_time))
    vdm = -1.0 * float(gamma) * (fmap * 2.0 * math.pi) * factor / float(scaling_factor)
    return vdm.astype(np.float32)


def compute_vdm_from_b0_2d(b0_slice_hz: np.ndarray,
                           dwell_time: float = 2.2266e-4,
                           tramp: float = 200.0,
                           gamma: float = 42.56,
                           scaling_factor: float = 1873800.0) -> np.ndarray:
    """VDM for a single 2D slice along x-direction."""
    fmap = np.asarray(b0_slice_hz, dtype=np.float64)
    H, W = fmap.shape
    n = (np.arange(W, dtype=np.float64) + 1.0).reshape(1, W)
    factor = (2.0 * float(tramp) + n * float(dwell_time)).reshape(1, W) 
    vdm = -1.0 * float(gamma) * (fmap * 2.0 * math.pi) * factor / float(scaling_factor)
    return vdm.astype(np.float32)

def compute_vdm_from_b0_2d_ESP(b0_slice_hz: np.ndarray,
                           esp_s: float = 0.00068,
                           npe: int = 100,
                           pf: float = 1.0,
                           r: float = 2.0,
                           pe_axis: int = 1,          # 对 2D 图，通常 HxW，PE 多为 axis=1 (列/纵向)
                           pe_sign: int = +1,
                           return_mm: bool = False,
                           pe_pixel_size_mm: float = 2.0
                           ) -> np.ndarray:
    """
    单张 2D 切片的 VDM。参数与上面一致。
    """
    fmap = np.asarray(b0_slice_hz, dtype=np.float64)

    neff = (npe * pf) / r
    trt = max(neff - 1, 0) * esp_s

    vdm_px = pe_sign * fmap * trt
    if not return_mm:
        return vdm_px.astype(np.float32)

    if pe_pixel_size_mm is None:
        raise ValueError("return_mm=True 时需要提供 pe_pixel_size_mm")
    vdm_mm = vdm_px * float(pe_pixel_size_mm)
    return vdm_mm.astype(np.float32)


def grid_warp_x(
    image: torch.Tensor,
    disp_x: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
    align_corners: bool = True,
    jacobian_compensation: bool = False,
) -> torch.Tensor:
    """
    Differentiable warp along x using disp field in pixels.

    Args:
        image: [B,1,H,W] float tensor.
        disp_x: [B,1,H,W] displacement in pixels along x (cols).
        mode: interpolation mode.

    Returns:
        warped: [B,1,H,W]
    """
    assert image.ndim == 4 and disp_x.ndim == 4
    b, c, h, w = image.shape
    device = image.device
    dtype = image.dtype
    # auto-align displacement spatial size to the image (debug-safety)
    if disp_x.shape[-2] != h or disp_x.shape[-1] != w:
        try:
            disp_x = F.interpolate(disp_x, size=(h, w), mode="bilinear", align_corners=True)
        except Exception:
            # fallback: center crop/pad
            Hd, Wd = disp_x.shape[-2], disp_x.shape[-1]
            top = max(0, (h - Hd) // 2)
            left = max(0, (w - Wd) // 2)
            pad = (left, max(0, w - Wd - left), top, max(0, h - Hd - top))
            if any(pad):
                disp_x = F.pad(disp_x, pad, mode="constant", value=0.0)
            Hd2, Wd2 = disp_x.shape[-2], disp_x.shape[-1]
            off_h = max(0, (Hd2 - h) // 2)
            off_w = max(0, (Wd2 - w) // 2)
            disp_x = disp_x[:, :, off_h:off_h + h, off_w:off_w + w]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=dtype),
        torch.arange(0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_new = x_coords + disp_x.squeeze(1)
    # normalize to [-1, 1]
    x_norm = 2.0 * (x_new / (w - 1)) - 1.0  # [B,H,W]
    y_norm = 2.0 * (y_coords / (h - 1)) - 1.0  # [H,W]
    y_norm = y_norm.unsqueeze(0).expand(b, -1, -1)  # [B,H,W]
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]
    out = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if jacobian_compensation:
        # intensity (pile-up) compensation via Jacobian of inverse mapping: detJ = 1 + dw/dx
        detj = 1.0 + finite_diff_x(disp_x)
        detj = detj.clamp_min(0.1)
        out = out * detj
    return out


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create 1D Gaussian kernel normalized to sum=1."""
    if sigma <= 0:
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k
    radius = int(truncate * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def gaussian_blur_2d(field: torch.Tensor, sigma: float = 1.0, truncate: float = 3.0) -> torch.Tensor:
    """Apply separable Gaussian blur to [B,1,H,W] tensor using reflect padding."""
    assert field.ndim == 4 and field.size(1) == 1
    if sigma <= 0:
        return field
    device = field.device
    dtype = field.dtype
    k1d = _gaussian_kernel1d(sigma, truncate, device=device, dtype=dtype)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    pad_r = (kx.shape[-1] - 1) // 2
    pad_t = (ky.shape[-2] - 1) // 2
    # pad horizontally for x-pass only, keep height unchanged
    out = F.pad(field, (pad_r, pad_r, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, padding=0, groups=1)
    # then pad vertically for y-pass only
    out = F.pad(out, (0, 0, pad_t, pad_t), mode="reflect")
    out = F.conv2d(out, ky, padding=0, groups=1)
    return out


def grid_warp_2d(image: torch.Tensor, flow_xy: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Backwarp using 2D flow (u_x,u_y) in pixels.

    Args:
        image: [B,1,H,W]
        flow_xy: [B,2,H,W]
    Returns:
        warped: [B,1,H,W]
    """
    b, c, h, w = image.shape
    device = image.device
    dtype = image.dtype
    # auto-align flow spatial size to the image
    if flow_xy.shape[-2] != h or flow_xy.shape[-1] != w:
        try:
            flow_xy = F.interpolate(flow_xy, size=(h, w), mode="bilinear", align_corners=True)
        except Exception:
            Hd, Wd = flow_xy.shape[-2], flow_xy.shape[-1]
            top = max(0, (h - Hd) // 2)
            left = max(0, (w - Wd) // 2)
            pad = (left, max(0, w - Wd - left), top, max(0, h - Hd - top))
            if any(pad):
                flow_xy = F.pad(flow_xy, pad, mode="constant", value=0.0)
            Hd2, Wd2 = flow_xy.shape[-2], flow_xy.shape[-1]
            off_h = max(0, (Hd2 - h) // 2)
            off_w = max(0, (Wd2 - w) // 2)
            flow_xy = flow_xy[:, :, off_h:off_h + h, off_w:off_w + w]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=dtype),
        torch.arange(0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_new = x_coords + flow_xy[:, 0:1].squeeze(1)
    y_new = y_coords + flow_xy[:, 1:2].squeeze(1)
    x_norm = 2.0 * (x_new / (w - 1)) - 1.0
    y_norm = 2.0 * (y_new / (h - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(image, grid, mode=mode, padding_mode="zeros", align_corners=True)


def finite_diff_x(t: torch.Tensor) -> torch.Tensor:
    """Forward difference along x for [B,1,H,W]."""
    return F.pad(t[:, :, :, 1:] - t[:, :, :, :-1], (1, 0, 0, 0))


def tv_loss(t: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Total variation L1 for [B,1,H,W]."""
    dy = F.pad(t[:, :, 1:, :] - t[:, :, :-1, :], (0, 0, 1, 0))
    dx = F.pad(t[:, :, :, 1:] - t[:, :, :, :-1], (1, 0, 0, 0))
    return weight * (dx.abs().mean() + dy.abs().mean())


def jacobian_det_penalty_1d_x(disp_x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    Penalize negative Jacobian determinant for mapping [x + u(x,y), y].
    detJ = 1 + du/dx. We penalize ReLU(-(1 + du/dx)).
    Args:
        disp_x: [B,1,H,W] pixel displacement along x.
    """
    du_dx = finite_diff_x(disp_x)
    detj = 1.0 + du_dx
    penalty = F.relu(-detj).mean()
    return weight * penalty



def screened_poisson_smooth_2d(field: torch.Tensor, lam: float = 0.2, iters: int = 40) -> torch.Tensor:
    """Solve (I - lam*Δ) v = field for v using Jacobi iterations with Neumann-like boundaries.

    Args:
        field: [B,1,H,W] tensor (target signal to be smoothed)
        lam: nonnegative regularization parameter (larger -> smoother)
        iters: number of Jacobi iterations

    Returns:
        v: [B,1,H,W] smoothed result
    """
    assert field.ndim == 4 and field.size(1) == 1
    if iters <= 0 or lam <= 0.0:
        return field
    v = field.clone()
    denom = 1.0 + 4.0 * lam
    for _ in range(int(iters)):
        vp = F.pad(v, (1, 1, 1, 1), mode="replicate")
        v_new = (field + lam * (vp[:, :, 1:-1, 0:-2] + vp[:, :, 1:-1, 2:] + vp[:, :, 0:-2, 1:-1] + vp[:, :, 2:, 1:-1])) / denom
        v = v_new
    return v



def _sample_along_x(field: torch.Tensor, offset_x: torch.Tensor) -> torch.Tensor:
    """Sample scalar field at coordinates (x + offset_x, y).

    Args:
        field: [B,1,H,W]
        offset_x: [B,1,H,W]
    Returns:
        sampled: [B,1,H,W]
    """
    assert field.ndim == 4 and offset_x.ndim == 4
    b, _, h, w = field.shape
    device = field.device
    dtype = field.dtype
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=dtype),
        torch.arange(0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_new = x_coords.unsqueeze(0) + offset_x.squeeze(1)
    x_norm = 2.0 * (x_new / (w - 1)) - 1.0
    y_norm = 2.0 * (y_coords / (h - 1)) - 1.0
    y_norm = y_norm.unsqueeze(0).expand(b, -1, -1)
    grid = torch.stack([x_norm, y_norm], dim=-1)
    out = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out


def invert_disp_field_x(disp_x: torch.Tensor, lam: float = 0.2, iters_fp: int = 5, smooth_iters: int = 20, gaussian_sigma: float = 0.0) -> torch.Tensor:
    """Approximate inverse 1D-x displacement via fixed-point iterations with screened-Poisson smoothing.

    Solves for w such that x = x' + w(x'), approximating w_{n+1} = -u(x' + w_n(x')).
    After each update, apply screened Poisson smoothing to stabilize.

    Args:
        disp_x: [B,1,H,W] forward displacement u(x) in pixels
        lam: smoothing strength for screened Poisson
        iters_fp: number of fixed-point iterations
        smooth_iters: Jacobi iterations within smoother

    Returns:
        inv_disp_x: [B,1,H,W] approximate inverse displacement field
    """
    assert disp_x.ndim == 4 and disp_x.size(1) == 1
    w = torch.zeros_like(disp_x)
    if iters_fp <= 0:
        return -disp_x
    for _ in range(int(iters_fp)):
        # sample u at x' using current inverse guess w: u(x' + w)
        u_at = _sample_along_x(disp_x, w)
        w_new = -u_at
        if gaussian_sigma > 0.0:
            w_new = gaussian_blur_2d(w_new, sigma=float(gaussian_sigma))
        elif lam > 0 and smooth_iters > 0:
            w_new = screened_poisson_smooth_2d(w_new, lam=lam, iters=smooth_iters)
        w = w_new
    return w

