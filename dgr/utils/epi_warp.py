"""
EPI distortion simulation methods for DWI warping.

This module contains three different approaches to simulate EPI distortion:
1. PSF-aware convolution method
2. k-space forward model method  
3. Splat-based displacement method
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, convolve1d
from scipy.signal import resample
from typing import Optional, Tuple, Dict, Any
from dgr.utils.splat import bilinear_splat_2d


def make_window(L: int, kind: str = "hann", tukey_alpha: float = 0.3) -> np.ndarray:
    """Create window function for apodization."""
    n = np.arange(L)
    if kind == "hann":
        w = 0.5 - 0.5*np.cos(2*np.pi*n/(L-1))
    elif kind == "hamming":
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(L-1))
    elif kind == "tukey":
        # scipy.signal.tukey 的简化版近似
        a = tukey_alpha
        w = np.ones(L)
        m = int(np.floor(a*(L-1)/2.0))
        ramp = 0.5*(1 - np.cos(np.pi*np.arange(m+1)/m)) if m>0 else np.array([1.0])
        w[:m+1] = ramp
        w[-(m+1):] = ramp[::-1]
    else:
        w = np.ones(L)
    # 保 DC/均值不变
    w = w / w.mean()
    return w


def _compute_1d_psf_kernel(delta_f_hz: float,
                          esp_s: float = 0.00068,
                          npe: int = 100,
                          pf: float = 1.0,
                          r: float = 2.0,
                          original_pixel_size_mm: float = 2.0,
                          target_pixel_size_mm: float = 0.56,
                          apply_hann: bool = True,
                          t2star_s: float = 0.05,
                          window: str = 'hann',
                          kaiser_beta: float = 8.0) -> np.ndarray:
    """
    基于真实序列时间基构造1D复数PSF核，并从DWI像素单位复数重采样到T2像素单位。
    - 时间基仅由序列参数给出：t_k = (k - (N_eff-1)/2) * ESP
    - 频域线权重：H[k] = exp(i 2π Δf t_k) * Hann(k) * exp(-|t_k| / T2*)
    - PSF核：h_df = IFFT_k{ H[k] }
    - 重采样：按比例 s = ps_DWI / ps_T2 将 h_df 从 DWI 像素单位复数插值到 T2 像素单位
    """
    # 有效相位编码步数（整数）
    neff = int(np.round((npe * pf) / max(1e-6, r)))
    neff = max(3, neff)

    # 逐线时间基（以回波中心为0）
    k_idx = np.arange(neff)
    t_k = (k_idx - (neff - 1) / 2.0) * esp_s

    # 频域线权重 H[k]
    H = np.exp(1j * 2.0 * np.pi * float(delta_f_hz) * t_k)
    if apply_hann and neff > 1:
        if window == 'kaiser':
            # Kaiser window
            x = (k_idx - (neff - 1) / 2.0) / ((neff - 1) / 2.0)
            x = np.clip(x, -1.0, 1.0)
            w = np.i0(kaiser_beta * np.sqrt(1 - x * x)) / np.i0(kaiser_beta)
        else:
            # Hann window
            w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (k_idx / (neff - 1)))
        H = H * w
    if (t2star_s is not None) and (t2star_s > 0):
        H = H * np.exp(-np.abs(t_k) / t2star_s)

    # 用带限重采样（FFT-based）将 h_dwi 拉伸到T2像素单位
    s = float(original_pixel_size_mm) / float(target_pixel_size_mm)
    new_len = int(np.round(neff * s))
    new_len = max(31, new_len)
    if new_len % 2 == 0:
        new_len += 1
    # 先得到 DWI 像素单位下的核并居中
    h_dwi = np.fft.ifft(H)
    h_dwi = np.fft.fftshift(h_dwi)
    # 再用FFT-based重采样到 T2 像素单位
    h_t2 = resample(h_dwi, new_len)

    # L1 归一化（防止数值问题）
    norm = np.sum(np.abs(h_t2))
    if norm > 1e-12:
        h_t2 = h_t2 / norm
    else:
        # 回退：单位脉冲
        h_t2 = np.zeros_like(h_t2)
        h_t2[len(h_t2) // 2] = 1.0 + 0j

    # 零相位中心化：使中心抽头相位为0（实正）
    mid = int(len(h_t2) // 2)
    center = h_t2[mid]
    ang = np.angle(center)
    if np.isfinite(ang):
        h_t2 = h_t2 * np.exp(-1j * ang)

    return h_t2.astype(np.complex64)


def psf_aware_convolution(img_2d_np: np.ndarray,
                         delta_f_map: np.ndarray,
                         pe_axis: int,
                         delta_f_range: tuple = (-600, 600),
                         delta_f_step: float = 5.0,
                         esp_s: float = 0.00068,
                         npe: int = 100,
                         pf: float = 1.0,
                         r: float = 2.0,
                         original_pixel_size_mm: float = 2.0,
                         target_pixel_size_mm: float = 0.56,
                         delta_f_sign: float = 1.0,
                         apply_hann: bool = False,
                         t2star_s: float = None) -> np.ndarray:
    """
    PSF-aware convolution method for EPI distortion simulation.
    
    Args:
        img_2d_np: Input magnitude image
        delta_f_map: B0 off-resonance field in Hz
        pe_axis: Phase encoding axis (0=rows, 1=cols)
        delta_f_range: Range of frequency offsets to consider
        delta_f_step: Step size for frequency discretization
        esp_s: Echo spacing in seconds
        npe: Number of phase encoding steps
        pf: Partial Fourier factor
        r: Acceleration factor
        original_pixel_size_mm: Original DWI pixel size
        target_pixel_size_mm: Target T2 pixel size
        delta_f_sign: Sign convention for frequency map
        apply_hann: Whether to apply Hann window
        t2star_s: T2* decay time
        
    Returns:
        Distorted image
    """
    from scipy import ndimage

    H, W = img_2d_np.shape

    # 预计算PSF核库（T2像素单位下的复数核）
    df_min, df_max = delta_f_range
    df_vals = np.arange(df_min, df_max + delta_f_step, delta_f_step, dtype=np.float32)
    psf_bank = {}
    for df in df_vals:
        psf_bank[df] = _compute_1d_psf_kernel(
            df, esp_s, npe, pf, r,
            original_pixel_size_mm, target_pixel_size_mm,
            apply_hann=apply_hann, t2star_s=t2star_s
        )

    # 输出幅值结果（最终为 abs(y/(c+eps))）
    out_img = np.zeros_like(img_2d_np, dtype=np.float32)

    # 对 Δf 做轻度 1D 高斯（仅沿 PE 轴），仅用于权重/索引，不改变真实位移
    if pe_axis == 0:
        df_smooth = ndimage.gaussian_filter1d(delta_f_sign * delta_f_map, sigma=1.2, axis=1, mode='reflect')
    else:
        df_smooth = ndimage.gaussian_filter1d(delta_f_sign * delta_f_map, sigma=1.2, axis=0, mode='reflect')

    def convolve_line(line_data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return ndimage.convolve1d(
            line_data.astype(np.complex128),
            kernel.astype(np.complex128),
            mode='reflect',
            origin=0
        )

    # 根据相位编码方向按行或按列处理
    if pe_axis == 0:
        # PE=L↔R：沿列方向作用（逐行处理）
        for i in range(H):
            row = img_2d_np[i, :]
            row_df = np.clip(df_smooth[i, :], df_min, df_max)

            # 非相干功率叠加：仅在权重域决定需要的 bin，逐 bin 做一次整行卷积
            pos = (row_df - df_min) / max(1e-6, delta_f_step)
            idx = np.floor(pos).astype(np.int32)
            idx = np.clip(idx, 0, len(df_vals) - 1)
            lo = max(0, int(idx.min()) - 1)
            hi = min(len(df_vals) - 1, int(idx.max()) + 1)

            y2 = np.zeros(W, dtype=np.float64)

            bins_used = 0
            for bi in range(lo, hi + 1):
                df_b = float(df_vals[bi])
                # 三角权重（连续 overlap-add），仅用于权重/能量分配
                w_b = 1.0 - np.abs(row_df - df_b) / max(1e-6, delta_f_step)
                if w_b.max() <= 0.0:
                    continue
                w_b = np.clip(w_b, 0.0, 1.0)
                # 幅值加权源信号
                src_b = row.astype(np.complex128) * w_b.astype(np.complex128)
                # 复核卷积 → 功率累加（非相干）
                y_b = convolve_line(src_b, psf_bank[df_b])
                y2 += np.abs(y_b) ** 2
                bins_used += 1

            if bins_used <= 0:
                out_row = np.abs(row).astype(np.float32)
            else:
                out_row = np.sqrt(y2).astype(np.float32)
            out_img[i, :] = out_row

    else:
        # PE=A↔P：沿行方向作用（逐列处理）
        for j in range(W):
            col = img_2d_np[:, j]
            col_df = np.clip(df_smooth[:, j], df_min, df_max)

            pos = (col_df - df_min) / max(1e-6, delta_f_step)
            idx = np.floor(pos).astype(np.int32)
            idx = np.clip(idx, 0, len(df_vals) - 1)
            lo = max(0, int(idx.min()) - 1)
            hi = min(len(df_vals) - 1, int(idx.max()) + 1)

            y2 = np.zeros(H, dtype=np.float64)

            bins_used = 0
            for bi in range(lo, hi + 1):
                df_b = float(df_vals[bi])
                w_b = 1.0 - np.abs(col_df - df_b) / max(1e-6, delta_f_step)
                if w_b.max() <= 0.0:
                    continue
                w_b = np.clip(w_b, 0.0, 1.0)
                src_b = col.astype(np.complex128) * w_b.astype(np.complex128)
                y_b = convolve_line(src_b, psf_bank[df_b])
                y2 += np.abs(y_b) ** 2
                bins_used += 1

            if bins_used <= 0:
                out_col = np.abs(col).astype(np.float32)
            else:
                out_col = np.sqrt(y2).astype(np.float32)
            out_img[:, j] = out_col

    # 数值健壮性
    if np.any(~np.isfinite(out_img)):
        out_img = np.nan_to_num(out_img, nan=0.0, posinf=0.0, neginf=0.0)

    return out_img.astype(np.float32)


def forward_epi_offresonance(img_2d_np: np.ndarray,
                             delta_f_map: np.ndarray,
                             esp_s: float = 0.00021,
                             npe: int = 320,
                             pf: float = 1.0,
                             r: float = 1.0,
                             pe_axis: int = 1,
                             delta_f_sign: float = +1.0,
                             phase0: float = 0,
                             zigzag: bool = False,
                             df_clip_hz: float = 400.0,
                             apodize: bool = True) -> np.ndarray:
    """
    k-space forward model for EPI distortion simulation.
    
    Args:
        img_2d_np: Input magnitude image
        delta_f_map: B0 off-resonance field in Hz
        esp_s: Echo spacing in seconds
        npe: Number of phase encoding lines
        pf: Partial Fourier fraction
        r: Parallel imaging acceleration
        pe_axis: Phase encoding axis (0=rows, 1=cols)
        delta_f_sign: Sign convention for frequency map
        phase0: Constant phase offset
        zigzag: Whether to emulate zigzag readout
        df_clip_hz: Frequency clipping range
        apodize: Whether to apply apodization
        
    Returns:
        Distorted image
    """
    assert img_2d_np.shape == delta_f_map.shape
    H_in, W_in = img_2d_np.shape

    # Clip and smooth frequency map
    df = np.clip(delta_f_map, -df_clip_hz, +df_clip_hz)
    df = gaussian_filter(df, sigma=3)
    mag = img_2d_np
    
    # Build complex image
    S0 = mag * np.exp(1j * float(phase0))

    # k-space container
    K = np.zeros((H_in, W_in), dtype=np.complex128)
    t_pe = np.arange(H_in) * esp_s
    N_eff = H_in
    
    # Line-by-line acquisition simulation
    for m in range(N_eff):
        # Get line signal
        S_line = S0[m, :]
        
        # Apply phase evolution for this line
        phase_line = 2 * np.pi * df[m, :] * t_pe[m]
        phase_wrapped = np.mod(phase_line + np.pi, 2*np.pi) - np.pi
        S_line_phased = S_line * np.exp(1j * phase_wrapped)
        
        # 1D FFT along readout direction
        K_line = np.fft.fft(S_line_phased, norm='ortho')
        
        # Place line in k-space
        K[m, :] = K_line

    # Apply apodization
    if apodize:
        if pe_axis == 1:
            w = make_window(H_in, kind="tukey", tukey_alpha=0.3)
            w = (1-0.4) + 0.4*w
            K *= w[:, None]
        else:
            w = make_window(W_in, kind="tukey", tukey_alpha=0.3)
            w = (1-0.4) + 0.4*w
            K *= w[None, :]

    # Reconstruct image
    S_dist = np.fft.ifft2(K, norm='ortho')
    dwi_in = np.abs(S_dist).astype(np.float32)
    
    return dwi_in


def forward_splat_with_fallback(img_2d_np: np.ndarray,
                               disp_2d_np: np.ndarray,
                               pe_axis: int,
                               valid_mask_np: Optional[np.ndarray] = None,
                               tau: float = 0.0,
                               alpha: float = 0.0,
                               hole_thresh: float = 5e-3) -> np.ndarray:
    """
    Splat-based displacement method for EPI distortion simulation.
    
    Args:
        img_2d_np: Input magnitude image
        disp_2d_np: Displacement field in pixels
        pe_axis: Phase encoding axis (0=rows, 1=cols)
        valid_mask_np: Valid region mask
        tau: Temperature parameter for splat
        alpha: Coverage normalization parameter
        hole_thresh: Threshold for hole filling
        
    Returns:
        Distorted image
    """
    img = torch.from_numpy(img_2d_np).float()[None, None]  # [1,1,H,W]
    disp = torch.from_numpy(disp_2d_np).float()[None, None]  # [1,1,H,W]
    H, W = img.shape[-2:]
    max_u = float(np.nanmax(np.abs(disp_2d_np)))
    
    # Clamp displacement to prevent overflow
    # if max_u > 0.45 * min(H, W):
    #     print(f"[WARN] huge displacement: {max_u:.1f}px on {H}x{W}. "
    #           "check px_scale/voxel size; clamping in fallback.", flush=True)
    clip_u = 0.45 * min(H, W)
    disp_2d_np = np.clip(disp_2d_np, -clip_u, clip_u)

    # Binary foreground as "quality"
    if valid_mask_np is None:
        Mbin_np = np.ones_like(img_2d_np, dtype=np.float32)
    else:
        Mbin_np = (valid_mask_np > 0.5).astype(np.float32)
    Mbin = torch.from_numpy(Mbin_np).float()[None, None]

    # Reflective padding
    pad_pix = int(np.ceil(np.percentile(np.abs(disp_2d_np), 95))) + 2
    pad_tuple = (pad_pix, pad_pix, pad_pix, pad_pix)
    img_p = F.pad(img, pad_tuple, mode='reflect')
    Mbin_p = F.pad(Mbin, pad_tuple, mode='constant', value=0.0)
    disp_p = F.pad(disp, pad_tuple, mode='reflect')

    # Construct flow field
    if pe_axis == 0:
        flow = torch.cat([disp_p, torch.zeros_like(disp_p)], dim=1)
    else:
        flow = torch.cat([torch.zeros_like(disp_p), disp_p], dim=1)

    # Splat intensity and coverage
    num, _ = bilinear_splat_2d(img_p * Mbin_p, flow)
    _, cov = bilinear_splat_2d(Mbin_p, flow)
    cov_np = cov.squeeze().cpu().numpy()
    # print(f"[COV] holes<0.05: {(cov_np<0.05).mean():.3%}  "
    #       f"stretch(cov<0.5): {(cov_np<0.5).mean():.3%}  "
    #       f"pile(cov>1.5): {(cov_np>1.5).mean():.3%}")

    # Power normalization
    alpha_eff = 0.5 if alpha is None else float(alpha)
    
        # 计算 cov 后加上：
    cov_safe = cov.clamp_min(1e-6)

    # 距离“未变形”的程度：d=|log cov|
    d = (cov_safe.log().abs())

    # 自适应 alpha: d 小→ alpha≈1；d 大→ alpha≈0
    # s 控制过渡的平滑区宽度，推荐 s≈0.15~0.25；lambda≤1 控制最大“回退”幅度
    s = 0.5
    lam = 0.9
    alpha_map = 1.0 - lam * torch.tanh(d / s)          # ∈(0,1]
    # print(f"max alpha_map: {alpha_map.max():.3f}, min alpha_map: {alpha_map.min():.3f}")
    # 可选：限制下界，避免极端处过亮
    # alpha_map = torch.clamp(alpha_map, 0.1, 1.0)

    Yp = num / cov_safe.pow(alpha_map)
    # Yp = num / cov.clamp_min(1e-6).pow(alpha_eff)

    # Hole filling with Gaussian smoothing
    holes = (cov < hole_thresh)
    if holes.any():
        sigma = 1.0
        kernel_size = 5
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=Yp.dtype, device=Yp.device) - kernel_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d[None, None, :, :]
        
        non_hole = (~holes).to(dtype=Yp.dtype)
        # Reflective padding
        pad_size = kernel_size // 2
        Yn = Yp * non_hole
        Yn_p = F.pad(Yn, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        Mh_p = F.pad(non_hole, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        sum_nb = F.conv2d(Yn_p, kernel, padding=0)
        cnt_nb = F.conv2d(Mh_p, kernel, padding=0)
        
        # Fill holes with weighted neighborhood
        has_nb = (cnt_nb > 0.1)
        fill_vals = torch.where(has_nb, sum_nb / cnt_nb.clamp_min(1.0), Yp)
        Yp = torch.where(holes, fill_vals, Yp)

    # Crop back to original size
    Y = Yp[..., pad_pix:-pad_pix, pad_pix:-pad_pix]
    return Y.squeeze().cpu().numpy()


def warp_dwi_image(img_2d_np: np.ndarray,
                   delta_f_map: np.ndarray,
                   method: str = "kspace",
                   pe_axis: int = 1,
                   **kwargs) -> np.ndarray:
    """
    Unified interface for EPI distortion simulation methods.
    
    Args:
        img_2d_np: Input magnitude image
        delta_f_map: B0 off-resonance field in Hz
        method: Method to use ("psf", "kspace", "splat")
        pe_axis: Phase encoding axis (0=rows, 1=cols)
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Distorted image
    """
    if method == "psf":
        return psf_aware_convolution(
            img_2d_np=img_2d_np,
            delta_f_map=delta_f_map,
            pe_axis=pe_axis,
            **kwargs
        )
    elif method == "kspace":
        return forward_epi_offresonance(
            img_2d_np=img_2d_np,
            delta_f_map=delta_f_map,
            pe_axis=pe_axis,
            **kwargs
        )
    elif method == "splat":
        # For splat method, delta_f_map should be displacement field
        return forward_splat_with_fallback(
            img_2d_np=img_2d_np,
            disp_2d_np=delta_f_map,  # In splat method, this is displacement
            pe_axis=pe_axis,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'psf', 'kspace', 'splat'")
