from typing import Tuple

import numpy as np


def resample_b0_with_nib(
    src_vol: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Resample a volume (assumed shape [H,W,Z]) from src_affine onto a target grid
    defined by tgt_shape and tgt_affine using nibabel.processing.resample_from_to.

    This path is robust to orientation and voxel size differences and avoids
    edge replication artifacts by controlling interpolation mode/cval.
    """
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    src_img = nib.Nifti1Image(src_vol.astype(np.float32), affine=src_affine.astype(np.float64))
    # dummy data for target grid
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))

    out_img = resample_from_to(src_img, tgt_img, order=order, mode=mode, cval=cval)
    out = out_img.get_fdata(dtype=np.float32)
    return out.astype(np.float32)


def resample_mask_with_nib(
    src_mask: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbor resample of mask using exact affines (no XY center shift)."""
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    src_img = nib.Nifti1Image(src_mask.astype(np.float32), affine=src_affine.astype(np.float64))
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))
    out_img = resample_from_to(src_img, tgt_img, order=0, mode=mode, cval=cval)
    out = out_img.get_fdata(dtype=np.float32)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def resample_b0_center_xy(
    src_vol: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Resample src onto target grid but enforce XY-center alignment:
    - Adjust src affine translation so that XY world-center matches target world-center
    - Leave Z translation unchanged, preserving Z handling
    - Then resample via nibabel resample_from_to
    """
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    src_shape = src_vol.shape
    # centers in voxel coordinates (axis order matches array order)
    src_c_vox = np.array([(src_shape[0] - 1) / 2.0, (src_shape[1] - 1) / 2.0, (src_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c_vox = np.array([(tgt_shape[0] - 1) / 2.0, (tgt_shape[1] - 1) / 2.0, (tgt_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)

    src_c_world = src_affine @ src_c_vox
    tgt_c_world = tgt_affine @ tgt_c_vox

    delta = (tgt_c_world - src_c_world)
    # zero out Z shift to keep Z handling unchanged
    delta[2] = 0.0

    src_aff_adj = src_affine.copy()
    src_aff_adj[:3, 3] += delta[:3]

    src_img = nib.Nifti1Image(src_vol.astype(np.float32), affine=src_aff_adj.astype(np.float64))
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))

    out_img = resample_from_to(src_img, tgt_img, order=order, mode=mode, cval=cval)
    out = out_img.get_fdata(dtype=np.float32)
    return out.astype(np.float32)


def resample_mask_center_xy(
    src_mask: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Resample a binary/float mask using nearest-neighbor while aligning XY centers.
    Returns float32 mask in [0,1].
    """
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    src_shape = src_mask.shape
    src_c_vox = np.array([(src_shape[0] - 1) / 2.0, (src_shape[1] - 1) / 2.0, (src_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c_vox = np.array([(tgt_shape[0] - 1) / 2.0, (tgt_shape[1] - 1) / 2.0, (tgt_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)

    src_c_world = src_affine @ src_c_vox
    tgt_c_world = tgt_affine @ tgt_c_vox
    delta = (tgt_c_world - src_c_world)
    delta[2] = 0.0
    src_aff_adj = src_affine.copy()
    src_aff_adj[:3, 3] += delta[:3]

    src_img = nib.Nifti1Image(src_mask.astype(np.float32), affine=src_aff_adj.astype(np.float64))
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))
    # order=0 nearest
    out_img = resample_from_to(src_img, tgt_img, order=0, mode=mode, cval=cval)
    out = out_img.get_fdata(dtype=np.float32)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def resample_b0_align_xy_to_target(
    src_vol: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Build a synthetic source affine that copies the target XY orientation (unit directions),
    keeps source in-plane pixel sizes, and forces Z-span mapping similar to registration.
    Then resample using nibabel world-space resampling.
    """
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    # derive src in-plane pixel sizes from its affine
    dr_src = _norm(src_affine[:3, 0])
    dc_src = _norm(src_affine[:3, 1])
    # target orientation unit vectors
    row_dir = tgt_affine[:3, 0]
    col_dir = tgt_affine[:3, 1]
    row_dir_u = row_dir / (np.linalg.norm(row_dir) + 1e-8)
    col_dir_u = col_dir / (np.linalg.norm(col_dir) + 1e-8)
    # forced Z span mapping
    n_src = max(1, src_vol.shape[2] - 1)
    n_tgt = max(1, tgt_shape[2] - 1)
    z_vec_forced = tgt_affine[:3, 2] * (n_tgt / float(n_src))

    A = np.eye(4, dtype=np.float64)
    A[:3, 0] = row_dir_u * dr_src
    A[:3, 1] = col_dir_u * dc_src
    A[:3, 2] = z_vec_forced

    # center alignment
    src_c = np.array([(src_vol.shape[0] - 1) / 2.0, (src_vol.shape[1] - 1) / 2.0, (src_vol.shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c = np.array([(tgt_shape[0] - 1) / 2.0, (tgt_shape[1] - 1) / 2.0, (tgt_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c_world = tgt_affine @ tgt_c
    # place src center at target center
    origin = tgt_c_world[:3] - A[:3, 0] * src_c[0] - A[:3, 1] * src_c[1] - A[:3, 2] * src_c[2]
    A[:3, 3] = origin

    src_img = nib.Nifti1Image(src_vol.astype(np.float32), affine=A)
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))
    out_img = resample_from_to(src_img, tgt_img, order=order, mode=mode, cval=cval)
    return out_img.get_fdata(dtype=np.float32).astype(np.float32)


def resample_mask_align_xy_to_target(
    src_mask: np.ndarray,
    src_affine: np.ndarray,
    tgt_shape: Tuple[int, int, int],
    tgt_affine: np.ndarray,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Same as resample_b0_align_xy_to_target but with nearest neighbor for masks."""
    try:
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Nibabel not available for resampling: {exc}")

    dr_src = _norm(src_affine[:3, 0])
    dc_src = _norm(src_affine[:3, 1])
    row_dir = tgt_affine[:3, 0]
    col_dir = tgt_affine[:3, 1]
    row_dir_u = row_dir / (np.linalg.norm(row_dir) + 1e-8)
    col_dir_u = col_dir / (np.linalg.norm(col_dir) + 1e-8)
    n_src = max(1, src_mask.shape[2] - 1)
    n_tgt = max(1, tgt_shape[2] - 1)
    z_vec_forced = tgt_affine[:3, 2] * (n_tgt / float(n_src))

    A = np.eye(4, dtype=np.float64)
    A[:3, 0] = row_dir_u * dr_src
    A[:3, 1] = col_dir_u * dc_src
    A[:3, 2] = z_vec_forced
    src_c = np.array([(src_mask.shape[0] - 1) / 2.0, (src_mask.shape[1] - 1) / 2.0, (src_mask.shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c = np.array([(tgt_shape[0] - 1) / 2.0, (tgt_shape[1] - 1) / 2.0, (tgt_shape[2] - 1) / 2.0, 1.0], dtype=np.float64)
    tgt_c_world = tgt_affine @ tgt_c
    origin = tgt_c_world[:3] - A[:3, 0] * src_c[0] - A[:3, 1] * src_c[1] - A[:3, 2] * src_c[2]
    A[:3, 3] = origin

    src_img = nib.Nifti1Image(src_mask.astype(np.float32), affine=A)
    tgt_img = nib.Nifti1Image(np.zeros(tgt_shape, dtype=np.float32), affine=tgt_affine.astype(np.float64))
    out_img = resample_from_to(src_img, tgt_img, order=0, mode=mode, cval=cval)
    out = out_img.get_fdata(dtype=np.float32)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


