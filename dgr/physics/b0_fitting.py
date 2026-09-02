from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

try:
    from scipy.special import lpmv
except Exception as _e:  # pragma: no cover
    lpmv = None

try:
    from scipy.ndimage import gaussian_filter
except Exception as _e:  # pragma: no cover
    gaussian_filter = None


@dataclass
class FitInputs:
    field_map_hz: np.ndarray
    mask: np.ndarray
    voxel_size_mm: Tuple[float, float, float]
    order: int = 12
    fov_size_mm: Optional[Tuple[float, float, float]] = None


@dataclass
class FitOutputs:
    coeffs: np.ndarray
    bestfit_field_hz: np.ndarray
    residual_field_hz: np.ndarray
    stats: Dict[str, float]
    num_terms: int


def build_coordinate_1d(n: int, voxel_mm: float, fov_mm: Optional[float]) -> np.ndarray:
    if fov_mm is not None and fov_mm > 0:
        return np.linspace(-fov_mm / 2.0 + voxel_mm / 2.0, fov_mm / 2.0 - voxel_mm / 2.0, n, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    return idx * voxel_mm


def _theta_phi_from_xyz(x_cm: np.ndarray, y_cm: np.ndarray, z_cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(x_cm * x_cm + y_cm * y_cm + z_cm * z_cm)
    theta = np.arctan2(np.sqrt(x_cm * x_cm + y_cm * y_cm), z_cm)
    phi = np.arctan2(y_cm, x_cm)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    return r, theta, phi


def _design_matrix_real_sh(x_cm: np.ndarray, y_cm: np.ndarray, z_cm: np.ndarray, order: int) -> np.ndarray:
    if lpmv is None:
        raise RuntimeError("scipy is required for SH fitting (scipy.special.lpmv not found)")

    r, theta, phi = _theta_phi_from_xyz(x_cm, y_cm, z_cm)
    cos_theta = np.cos(theta)

    cols = []
    for l in range(order + 1):
        r_pow_l = np.power(r, l)
        P_l0 = lpmv(0, l, cos_theta)
        cols.append(r_pow_l * P_l0)
        for m in range(1, l + 1):
            P_lm = lpmv(m, l, cos_theta)
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi)
            cols.append(r_pow_l * P_lm * cos_mphi)
            cols.append(r_pow_l * P_lm * sin_mphi)

    X = np.stack(cols, axis=1).astype(np.float64, copy=False)
    return X


def gaussian_smooth_masked(field_map_hz: np.ndarray, mask: np.ndarray, sigma_vox: float) -> np.ndarray:
    """Gaussian smoothing restricted to mask via normalized convolution.
    - Smooth field*mask and mask separately, then divide; outside mask keep original.
    """
    if gaussian_filter is None:
        raise RuntimeError("scipy.ndimage is required for Gaussian smoothing")
    field = np.asarray(field_map_hz, dtype=np.float64)
    m = (np.asarray(mask) > 0).astype(np.float64)
    if field.shape != m.shape:
        raise ValueError("field_map_hz and mask must have the same shape")
    # normalized convolution
    num = gaussian_filter(field * m, sigma=sigma_vox, mode="nearest")
    den = gaussian_filter(m, sigma=sigma_vox, mode="nearest")
    eps = 1e-8
    sm = np.where(den > eps, num / (den + eps), field)
    # outside mask: keep original
    out = np.where(m > 0, sm, field)
    return out.astype(np.float64, copy=False)


def spherical_harmonics_fit(inputs: FitInputs, ridge_lambda: float = 0.0) -> FitOutputs:
    field_map = np.asarray(inputs.field_map_hz, dtype=np.float64)
    mask = np.asarray(inputs.mask) > 0
    if field_map.shape != mask.shape:
        raise ValueError("field_map_hz and mask must have the same shape")
    if inputs.order < 0:
        raise ValueError("order must be >= 0")

    nx, ny, nz = field_map.shape
    dx, dy, dz = [float(v) for v in inputs.voxel_size_mm]
    fx = inputs.fov_size_mm[0] if inputs.fov_size_mm else None
    fy = inputs.fov_size_mm[1] if inputs.fov_size_mm else None
    fz = inputs.fov_size_mm[2] if inputs.fov_size_mm else None

    x1d_mm = build_coordinate_1d(nx, dx, fx)
    y1d_mm = build_coordinate_1d(ny, dy, fy)
    z1d_mm = build_coordinate_1d(nz, dz, fz)

    Xcm, Ycm, Zcm = np.meshgrid(x1d_mm / 10.0, y1d_mm / 10.0, z1d_mm / 10.0, indexing="ij")

    roi_idx = mask
    y_vec = field_map[roi_idx].reshape(-1)
    x_cm_vec = Xcm[roi_idx].reshape(-1)
    y_cm_vec = Ycm[roi_idx].reshape(-1)
    z_cm_vec = Zcm[roi_idx].reshape(-1)

    X = _design_matrix_real_sh(x_cm_vec, y_cm_vec, z_cm_vec, inputs.order)
    num_terms = X.shape[1]

    if ridge_lambda and ridge_lambda > 0:
        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge_lambda
        Xty = X.T @ y_vec
        coeffs = np.linalg.solve(XtX, Xty)
    else:
        coeffs, *_ = np.linalg.lstsq(X, y_vec, rcond=None)

    y_hat = X @ coeffs
    bestfit = np.zeros_like(field_map, dtype=np.float64)
    bestfit[roi_idx] = y_hat
    resid = field_map - bestfit

    values = resid[roi_idx]
    stats = {
        "min": float(np.min(values)) if values.size else math.nan,
        "max": float(np.max(values)) if values.size else math.nan,
        "mean": float(np.mean(values)) if values.size else math.nan,
        "median": float(np.median(values)) if values.size else math.nan,
        "std": float(np.std(values, ddof=0)) if values.size else math.nan,
        "num_voxels": int(values.size),
    }

    return FitOutputs(
        coeffs=coeffs.astype(np.float64, copy=False),
        bestfit_field_hz=bestfit.astype(np.float64, copy=False),
        residual_field_hz=resid.astype(np.float64, copy=False),
        stats=stats,
        num_terms=num_terms,
    )


def spherical_harmonics_fit_full(
    inputs: FitInputs,
    ridge_lambda: float = 0.0,
    standardize_design: bool = False,
) -> Tuple[FitOutputs, np.ndarray]:
    """Fit on ROI (inputs.mask) but also predict SH field on the entire grid.
    Returns FitOutputs (same as spherical_harmonics_fit) plus predicted_all (3D).
    If standardize_design, columns of X (ROI) and X_all use same mean/std.
    """
    field_map = np.asarray(inputs.field_map_hz, dtype=np.float64)
    mask = np.asarray(inputs.mask) > 0
    if field_map.shape != mask.shape:
        raise ValueError("field_map_hz and mask must have the same shape")
    if inputs.order < 0:
        raise ValueError("order must be >= 0")

    nx, ny, nz = field_map.shape
    dx, dy, dz = [float(v) for v in inputs.voxel_size_mm]
    fx = inputs.fov_size_mm[0] if inputs.fov_size_mm else None
    fy = inputs.fov_size_mm[1] if inputs.fov_size_mm else None
    fz = inputs.fov_size_mm[2] if inputs.fov_size_mm else None

    x1d_mm = build_coordinate_1d(nx, dx, fx)
    y1d_mm = build_coordinate_1d(ny, dy, fy)
    z1d_mm = build_coordinate_1d(nz, dz, fz)

    Xcm, Ycm, Zcm = np.meshgrid(x1d_mm / 10.0, y1d_mm / 10.0, z1d_mm / 10.0, indexing="ij")

    # ROI vectors
    roi_idx = mask
    y_vec = field_map[roi_idx].reshape(-1)
    x_cm_vec = Xcm[roi_idx].reshape(-1)
    y_cm_vec = Ycm[roi_idx].reshape(-1)
    z_cm_vec = Zcm[roi_idx].reshape(-1)

    X = _design_matrix_real_sh(x_cm_vec, y_cm_vec, z_cm_vec, inputs.order)
    num_terms = X.shape[1]

    # Optionally standardize columns for stability
    if standardize_design:
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0) + 1e-12
        Xs = (X - col_mean) / col_std
    else:
        Xs = X
        col_mean = None
        col_std = None

    if ridge_lambda and ridge_lambda > 0:
        XtX = Xs.T @ Xs
        XtX.flat[:: XtX.shape[0] + 1] += ridge_lambda
        Xty = Xs.T @ y_vec
        coeffs = np.linalg.solve(XtX, Xty)
    else:
        coeffs, *_ = np.linalg.lstsq(Xs, y_vec, rcond=None)

    # Predict on ROI for stats
    y_hat = Xs @ coeffs
    bestfit = np.zeros_like(field_map, dtype=np.float64)
    bestfit[roi_idx] = y_hat
    resid = field_map - bestfit
    values = resid[roi_idx]
    stats = {
        "min": float(np.min(values)) if values.size else math.nan,
        "max": float(np.max(values)) if values.size else math.nan,
        "mean": float(np.mean(values)) if values.size else math.nan,
        "median": float(np.median(values)) if values.size else math.nan,
        "std": float(np.std(values, ddof=0)) if values.size else math.nan,
        "num_voxels": int(values.size),
    }

    # Predict on ALL voxels
    x_all = Xcm.reshape(-1)
    y_all = Ycm.reshape(-1)
    z_all = Zcm.reshape(-1)
    X_all = _design_matrix_real_sh(x_all, y_all, z_all, inputs.order)
    if standardize_design:
        X_all = (X_all - col_mean) / col_std
    y_all_hat = X_all @ coeffs
    pred_all = y_all_hat.reshape(field_map.shape)

    fit_outputs = FitOutputs(
        coeffs=coeffs.astype(np.float64, copy=False),
        bestfit_field_hz=bestfit.astype(np.float64, copy=False),
        residual_field_hz=resid.astype(np.float64, copy=False),
        stats=stats,
        num_terms=num_terms,
    )
    return fit_outputs, pred_all.astype(np.float64, copy=False)


def design_matrix_poly2d(x_mm: np.ndarray, y_mm: np.ndarray, order: int) -> np.ndarray:
    """Build 2D polynomial design matrix up to total order.

    Columns are monomials x^i y^j for i>=0, j>=0, i+j<=order, in lexicographic order.
    x_mm, y_mm are 1D vectors of the same length.
    """
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((i, j))
    cols = []
    for i, j in terms:
        cols.append((x_mm ** i) * (y_mm ** j))
    X = np.stack(cols, axis=1).astype(np.float64, copy=False)
    return X


def polynomial_fit_2d_slice(field_slice: np.ndarray, mask_slice: np.ndarray,
                            dx_mm: float, dy_mm: float, order: int,
                            ridge_lambda: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 2D polynomial on one slice within mask and predict on full slice.

    Returns: (pred_full_slice, coeffs)
    """
    field = np.asarray(field_slice, dtype=np.float64)
    m = (np.asarray(mask_slice) > 0)
    nx, ny = field.shape
    # coordinates centered at FOV center
    x1d = build_coordinate_1d(nx, dx_mm, None)
    y1d = build_coordinate_1d(ny, dy_mm, None)
    Xmm, Ymm = np.meshgrid(x1d, y1d, indexing="ij")
    roi = m
    if not np.any(roi):
        return np.zeros_like(field, dtype=np.float64), np.zeros(((order + 1) * (order + 2)) // 2, dtype=np.float64)
    y_vec = field[roi].reshape(-1)
    x_vec = Xmm[roi].reshape(-1)
    y2_vec = Ymm[roi].reshape(-1)
    X = design_matrix_poly2d(x_vec, y2_vec, order)
    if ridge_lambda and ridge_lambda > 0:
        XtX = X.T @ X
        XtX.flat[:: XtX.shape[0] + 1] += ridge_lambda
        Xty = X.T @ y_vec
        coeffs = np.linalg.solve(XtX, Xty)
    else:
        coeffs, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
    # predict on full grid
    X_full = design_matrix_poly2d(Xmm.reshape(-1), Ymm.reshape(-1), order)
    pred = (X_full @ coeffs).reshape(nx, ny)
    return pred.astype(np.float64, copy=False), coeffs.astype(np.float64, copy=False)


def polynomial_fit_2d_volume(field_map_hz: np.ndarray, mask: np.ndarray,
                             voxel_size_mm: Tuple[float, float, float], order: int,
                             ridge_lambda: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slice 2D polynomial fitting over a 3D volume.

    Returns: (pred_volume, coeffs_per_slice) where coeffs_per_slice has shape (num_coeffs, nz)
    """
    field = np.asarray(field_map_hz, dtype=np.float64)
    m = (np.asarray(mask) > 0)
    if field.shape != m.shape:
        raise ValueError("field_map_hz and mask must have the same shape")
    nx, ny, nz = field.shape
    dx, dy, _ = [float(v) for v in voxel_size_mm]
    num_coeffs = ((order + 1) * (order + 2)) // 2
    coeffs_all = np.zeros((num_coeffs, nz), dtype=np.float64)
    pred = np.zeros_like(field, dtype=np.float64)
    for k in range(nz):
        pred_k, coeffs_k = polynomial_fit_2d_slice(field[:, :, k], m[:, :, k], dx, dy, order, ridge_lambda)
        pred[:, :, k] = pred_k
        coeffs_all[:, k] = coeffs_k
    return pred.astype(np.float64, copy=False), coeffs_all.astype(np.float64, copy=False)


__all__ = [
    "FitInputs",
    "FitOutputs",
    "build_coordinate_1d",
    "spherical_harmonics_fit",
    "spherical_harmonics_fit_full",
    "gaussian_smooth_masked",
    "design_matrix_poly2d",
    "polynomial_fit_2d_slice",
    "polynomial_fit_2d_volume",
]



