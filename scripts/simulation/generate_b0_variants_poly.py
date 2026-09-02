#!/usr/bin/env python3
"""
B0 variant generation using either 2D polynomial or 3D spherical harmonics fitting.

Supports two fitting methods:
1. poly2d: 2D polynomial fitting per slice (like the original B0 fitting process)
2. sh3d: 3D spherical harmonics fitting across the entire volume

Both methods:
- Use only mask ROI for fitting (not rectangular regions)
- Apply adjustments only to high-order terms
- Scale adjustments to target RMS for controlled variation
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dgr.physics.b0_fitting import polynomial_fit_2d_volume, polynomial_fit_2d_slice, spherical_harmonics_fit, FitInputs


def parse_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def find_registered_b0(root: str):
    """Find all b0_to_t2_registered.npz files recursively."""
    pat = os.path.join(root, "**", "b0_to_t2_registered.npz")
    for p in glob.glob(pat, recursive=True):
        yield p


def load_b0_and_metadata(path: str):
    """Load B0 field and metadata from registration file."""
    d = np.load(path, allow_pickle=True)
    
    # Use original registered B0 field for polynomial fitting
    if "b0_registered" in d:
        b0 = d["b0_registered"].astype(np.float32)
        print(f"Using original registered B0 field from {path}")
    elif "b0_fitted_2d" in d:
        b0 = d["b0_fitted_2d"].astype(np.float32)
        print(f"Warning: Using fitted B0 field from {path} (not recommended)")
    else:
        raise ValueError(f"No B0 array found in {path}")
    
    # Extract metadata
    metadata = {
        't2_volume': d['t2_volume'].astype(np.float32) if 't2_volume' in d else None,
        't2_affine': d['t2_affine'] if 't2_affine' in d else None,
        'b0_affine': d['b0_affine'] if 'b0_affine' in d else None,
        'b0_voxel_size': d['b0_voxel_size'] if 'b0_voxel_size' in d else None,
        'fit2d_order': int(d['fit2d_order'].item()) if 'fit2d_order' in d else 6,
        'phase_mask': d['phase_mask'].astype(np.float32) if 'phase_mask' in d else None,
    }
    
    return b0, metadata


def extract_coefficients(b0_field: np.ndarray, mask: np.ndarray, order: int, 
                        voxel_size_mm: tuple, fit_type: str = "poly2d", ridge_lambda: float = 0.0) -> np.ndarray:
    """Extract coefficients using specified fitting method."""
    if fit_type == "poly2d":
        # Use 2D polynomial fitting per slice
        pred_volume, coeffs_per_slice = polynomial_fit_2d_volume(
            b0_field, mask, voxel_size_mm, order, ridge_lambda=ridge_lambda
        )
        return coeffs_per_slice  # Shape: (num_coeffs, nz)
    
    elif fit_type == "sh3d":
        # Use 3D spherical harmonics fitting
        fit_inputs = FitInputs(
            field_map_hz=b0_field,
            mask=mask,
            order=order,
            voxel_size_mm=voxel_size_mm,
            fov_size_mm=None,
        )
        fit_outputs = spherical_harmonics_fit(fit_inputs, ridge_lambda=0.0)
        # Convert 1D SH coefficients to 2D array for consistency
        coeffs_1d = fit_outputs.coeffs
        coeffs_2d = np.tile(coeffs_1d.reshape(-1, 1), (1, b0_field.shape[2]))
        return coeffs_2d  # Shape: (num_coeffs, nz)
    
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}. Must be 'poly2d' or 'sh3d'")


def reconstruct_b0_from_coeffs(coeffs: np.ndarray, shape: tuple, order: int, 
                               voxel_size_mm: tuple, fit_type: str = "poly2d") -> np.ndarray:
    """Reconstruct B0 field from coefficients using specified method."""
    if fit_type == "poly2d":
        # Reconstruct using 2D polynomial coefficients
        nx, ny, nz = shape
        dx, dy, dz = voxel_size_mm
        
        from dgr.physics.b0_fitting import build_coordinate_1d, design_matrix_poly2d
        
        x1d = build_coordinate_1d(nx, dx, None)
        y1d = build_coordinate_1d(ny, dy, None)
        Xmm, Ymm = np.meshgrid(x1d, y1d, indexing="ij")
        
        X_full = design_matrix_poly2d(Xmm.reshape(-1), Ymm.reshape(-1), order)
        
        reconstructed = np.zeros(shape, dtype=np.float32)
        for k in range(nz):
            pred_k = (X_full @ coeffs[:, k]).reshape(nx, ny)
            reconstructed[:, :, k] = pred_k.astype(np.float32)
        
        return reconstructed
    
    elif fit_type == "sh3d":
        # Reconstruct using 3D spherical harmonics coefficients
        nx, ny, nz = shape
        dx, dy, dz = voxel_size_mm
        
        from dgr.physics.b0_fitting import build_coordinate_1d, _design_matrix_real_sh
        
        x1d_mm = build_coordinate_1d(nx, dx, None)
        y1d_mm = build_coordinate_1d(ny, dy, None)
        z1d_mm = build_coordinate_1d(nz, dz, None)
        
        # build_coordinate_1d already includes voxel size, no need to multiply again
        Xcm, Ycm, Zcm = np.meshgrid(x1d_mm / 10.0, y1d_mm / 10.0, z1d_mm / 10.0, indexing="ij")
        
        # Use coefficients from first slice (all slices have same SH coefficients)
        coeffs_1d = coeffs[:, 0]
        
        X = _design_matrix_real_sh(Xcm.flatten(), Ycm.flatten(), Zcm.flatten(), order)
        field_flat = X @ coeffs_1d
        reconstructed = field_flat.reshape(shape).astype(np.float32)
        
        return reconstructed
    
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}. Must be 'poly2d' or 'sh3d'")


def adjust_high_order_coeffs(coeffs: np.ndarray, order: int, adjustment_factor: float, 
                            min_order: int = 3, fit_type: str = "poly2d") -> np.ndarray:
    """Adjust high-order coefficients while keeping low-order ones unchanged."""
    adjusted_coeffs = coeffs.copy()
    
    if fit_type == "poly2d":
        # Use polynomial coefficient grouping
        index_by_order = _poly2d_order_index_groups(order)
        nz = coeffs.shape[1]
        
        for l in range(min_order, order + 1):
            idxs = index_by_order[l]
            if not idxs:
                continue
            coeff_block = coeffs[idxs, :]
            mag = np.median(np.abs(coeff_block), axis=1, keepdims=True)
            mag = np.where(mag > 0, mag, 1e-8)
            noise = np.random.normal(0.0, adjustment_factor, size=coeff_block.shape)
            delta = noise * mag
            adjusted_coeffs[idxs, :] = coeff_block + delta
    
    elif fit_type == "sh3d":
        # Use spherical harmonics coefficient grouping
        coeff_idx = 0
        for l in range(order + 1):
            n_coeffs_l = 2 * l + 1
            
            if l >= min_order:
                adjustment = np.random.normal(0, adjustment_factor, (n_coeffs_l, coeffs.shape[1]))
                adjusted_coeffs[coeff_idx:coeff_idx + n_coeffs_l, :] += adjustment
            
            coeff_idx += n_coeffs_l
    
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}. Must be 'poly2d' or 'sh3d'")
    
    return adjusted_coeffs


def _poly2d_order_index_groups(order: int) -> dict:
    """Return a mapping: total_order -> list of coefficient indices for that order.

    The coefficient order matches design_matrix_poly2d in B0_fitting:
    terms appended in lexicographic order over (i, j) with i>=0, j>=0, i+j<=order.
    For a fixed total order l, indices are NOT contiguous; we must gather them explicitly.
    """
    index_by_order = {l: [] for l in range(order + 1)}
    k = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            l = i + j
            index_by_order[l].append(k)
            k += 1
    return index_by_order


def adjust_high_order_poly_coeffs(coeffs_per_slice: np.ndarray, order: int,
                                 adjustment_factor: float, min_order: int = 3,
                                 mode: str = "relative") -> np.ndarray:
    """Adjust high-order polynomial coefficients while keeping low-order ones unchanged.

    - Uses per-order index groups (non-contiguous) to avoid touching low orders
    - mode="relative": delta = coeff * N(0, adj)
      fallback to per-order median magnitude when coeff ~ 0
    """
    adjusted_coeffs = coeffs_per_slice.copy()
    nz = coeffs_per_slice.shape[1]
    index_by_order = _poly2d_order_index_groups(order)

    for l in range(min_order, order + 1):
        idxs = index_by_order[l]
        if not idxs:
            continue
        coeff_block = coeffs_per_slice[idxs, :]  # shape (n_coeffs_l, nz)
        if mode == "relative":
            mag = np.median(np.abs(coeff_block), axis=1, keepdims=True)  # (n_coeffs_l, 1)
            mag = np.where(mag > 0, mag, 1e-8)
            noise = np.random.normal(0.0, adjustment_factor, size=coeff_block.shape)
            delta = noise * mag
        else:
            delta = np.random.normal(0.0, adjustment_factor, size=coeff_block.shape)
        adjusted_coeffs[idxs, :] = coeff_block + delta
    return adjusted_coeffs


def _scale_adjustment_to_target_rms(base_coeffs: np.ndarray, adjusted_coeffs: np.ndarray,
                                    target_rms_hz: float, shape: tuple, order: int,
                                    voxel_size_mm: tuple, mask: np.ndarray, fit_type: str = "poly2d") -> np.ndarray:
    """Scale only the delta (high-order) adjustment so that the resulting field
    difference has approximately the target RMS (within mask)."""
    delta = adjusted_coeffs - base_coeffs
    # Quick reconstruction for base and base+delta
    b0_base = reconstruct_b0_from_coeffs(base_coeffs, shape, order, voxel_size_mm, fit_type)
    b0_adj = reconstruct_b0_from_coeffs(base_coeffs + delta, shape, order, voxel_size_mm, fit_type)
    diff = (b0_adj - b0_base)[mask]
    rms = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
    if rms <= 0 or target_rms_hz <= 0:
        return adjusted_coeffs
    scale = target_rms_hz / (rms + 1e-8)
    return base_coeffs + delta * scale


def make_out_path(src_path: str, in_root: str, out_root: str, 
                 adjustment_factor: float, variant_id: int, fit_type: str = "poly2d") -> tuple:
    """Create output paths for variant (NPZ and PNG)."""
    rel_dir = os.path.relpath(os.path.dirname(src_path), start=in_root)
    out_dir = os.path.join(out_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = f"b0_{fit_type}_variant_factor{adjustment_factor:g}_id{variant_id:02d}"
    npz_path = os.path.join(out_dir, f"{base_name}.npz")
    png_path = os.path.join(out_dir, f"{base_name}_central_slice.png")
    
    return npz_path, png_path


def save_central_slice_visualization(b0_original: np.ndarray, b0_variant: np.ndarray, 
                                   png_path: str, adj_factor: float, variant_id: int):
    """Save central slice visualization comparing original and variant."""
    mid_slice = b0_original.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original B0
    im0 = axes[0].imshow(b0_original[:, :, mid_slice], cmap='jet', vmin=-400, vmax=400)
    axes[0].set_title('Original B0\n(Central Slice)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Variant B0
    im1 = axes[1].imshow(b0_variant[:, :, mid_slice], cmap='jet', vmin=-400, vmax=400)
    axes[1].set_title(f'Variant B0\n(factor={adj_factor}, id={variant_id})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    diff = b0_variant - b0_original
    im_diff = axes[2].imshow(diff[:, :, mid_slice], cmap='RdBu_r', vmin=-100, vmax=100)
    axes[2].set_title(f'Difference\n(Variant - Original)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate B0 variants by adjusting polynomial or SH coefficients")
    ap.add_argument("--registered_root", type=str, required=True,
                    help="Root of registered B0 outputs")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root to mirror input tree with polynomial variants")
    ap.add_argument("--adjustment_factors", type=str, default="0.001,0.005,0.01,0.02",
                    help="Comma-separated list of adjustment factors for high-order coefficients")
    ap.add_argument("--variants_per_factor", type=int, default=3,
                    help="Number of variants to generate per adjustment factor")
    ap.add_argument("--min_order", type=int, default=3,
                    help="Minimum polynomial order to adjust (keep lower orders unchanged)")
    ap.add_argument("--target_rms_hz", type=float, default=50.0,
                    help="Target RMS(Hz) of (variant - original) inside mask; used to scale adjustments")
    ap.add_argument("--target_rms_per_factor", type=str, default=None,
                    help="Comma-separated target RMS values per adjustment factor (overrides --target_rms_hz)")
    ap.add_argument("--copy_t2", action="store_true", help="Copy t2_volume into each variant")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","float32"])
    ap.add_argument("--max_files", type=int, default=None, help="Stop after generating at most this many variants")
    ap.add_argument("--log_every", type=int, default=50, help="Progress print frequency")
    ap.add_argument("--save_png", action="store_true", help="Save central slice PNG visualization for each variant")
    ap.add_argument("--fit_type", type=str, default="poly2d", choices=["poly2d", "sh3d"],
                    help="Fitting method: 'poly2d' for 2D polynomial per-slice, 'sh3d' for 3D spherical harmonics")
    ap.add_argument("--fit_order", type=int, default=None,
                    help="Override fit order from NPZ file (if not specified, uses fit2d_order from metadata)")
    args = ap.parse_args()

    adjustment_factors = parse_floats(args.adjustment_factors)
    variants_per_factor = args.variants_per_factor
    min_order = args.min_order
    
    # Parse target RMS values
    if args.target_rms_per_factor:
        target_rms_values = parse_floats(args.target_rms_per_factor)
        if len(target_rms_values) != len(adjustment_factors):
            print(f"Warning: target_rms_per_factor length ({len(target_rms_values)}) != adjustment_factors length ({len(adjustment_factors)})")
            print("Using default target_rms_hz for all factors")
            target_rms_values = [args.target_rms_hz] * len(adjustment_factors)
    else:
        target_rms_values = [args.target_rms_hz] * len(adjustment_factors)

    created = 0
    start = time.time()
    
    for src in find_registered_b0(args.registered_root):
        try:
            b0_field, metadata = load_b0_and_metadata(src)
        except Exception as e:
            print(f"Skip {src}: {e}")
            continue
        
        # Create mask: prefer provided phase_mask if available (like original B0 registration)
        if metadata['phase_mask'] is not None:
            mask = metadata['phase_mask'] > 0
            print(f"Using phase_mask for fitting")
        else:
            mask = np.abs(b0_field) > 1e-6
            print(f"Using simple threshold mask (phase_mask not available)")
        
        # Get voxel size - use T2 voxel size (from affine matrix) instead of B0 original voxel size
        if metadata['t2_affine'] is not None:
            t2_affine = metadata['t2_affine']
            # Extract voxel size from T2 affine matrix (magnitude of each column vector)
            voxel_size_mm = (
                float(np.linalg.norm(t2_affine[:3, 0])),
                float(np.linalg.norm(t2_affine[:3, 1])),
                float(np.linalg.norm(t2_affine[:3, 2]))
            )
            print(f"Using T2 voxel size: {voxel_size_mm}")
        else:
            voxel_size_mm = (1.0, 1.0, 1.0)  # Default voxel size
        
        # Get fit order (use override if provided, otherwise use metadata)
        if args.fit_order is not None:
            fit_order = args.fit_order
            print(f"Using override fit order: {fit_order}")
        else:
            fit_order = metadata['fit2d_order']
            print(f"Using fit order from metadata: {fit_order}")
        
        # Extract original coefficients using specified method
        try:
            # Use small ridge regularization for numerical stability
            ridge_lambda = 1e-6 if fit_order >= 8 else 0.0
            original_coeffs = extract_coefficients(b0_field, mask, fit_order, voxel_size_mm, args.fit_type, ridge_lambda)
            print(f"Extracted {args.fit_type} coefficients: shape={original_coeffs.shape}, ridge_lambda={ridge_lambda}")
        except Exception as e:
            print(f"Failed to extract {args.fit_type} coefficients from {src}: {e}")
            continue
        
        # Generate variants
        for i, adj_factor in enumerate(adjustment_factors):
            target_rms = target_rms_values[i]
            for variant_id in range(variants_per_factor):
                npz_path, png_path = make_out_path(src, args.registered_root, args.out_root, 
                                                 adj_factor, variant_id, args.fit_type)
                
                if (not args.overwrite) and os.path.isfile(npz_path):
                    continue
                
                # Adjust high-order coefficients using specified method
                adjusted_coeffs_raw = adjust_high_order_coeffs(
                    original_coeffs, fit_order, adj_factor, min_order, args.fit_type
                )

                # Scale delta to target RMS inside mask
                adjusted_coeffs = _scale_adjustment_to_target_rms(
                    original_coeffs, adjusted_coeffs_raw, target_rms,
                    b0_field.shape, fit_order, voxel_size_mm, mask, args.fit_type
                )

                # Reconstruct B0 field using specified method
                try:
                    b0_variant = reconstruct_b0_from_coeffs(
                        adjusted_coeffs, b0_field.shape, fit_order, voxel_size_mm, args.fit_type
                    )
                except Exception as e:
                    print(f"Failed to reconstruct B0 from adjusted coefficients: {e}")
                    continue
                
                # Cast to target dtype
                if args.dtype == "float16":
                    b0_variant = b0_variant.astype(np.float16)
                else:
                    b0_variant = b0_variant.astype(np.float32)
                
                # Prepare output data
                output_data = {
                    'b0_variant': b0_variant,
                    'meta': {
                        'adjustment_factor': float(adj_factor),
                        'variant_id': int(variant_id),
                        'min_order_adjusted': int(min_order),
                        'fit_order': int(fit_order),
                        'fit_type': args.fit_type,
                        'source_path': src,
                        'original_coeffs': original_coeffs,
                        'adjusted_coeffs': adjusted_coeffs,
                    },
                    'dtype_saved': args.dtype,
                }
                
                if args.copy_t2 and metadata['t2_volume'] is not None:
                    t2_vol = metadata['t2_volume']
                    if args.dtype == "float16":
                        t2_vol = t2_vol.astype(np.float16)
                    output_data['t2_volume'] = t2_vol
                
                # Save variant
                np.savez_compressed(npz_path, **output_data)
                
                # Save PNG visualization if requested
                if args.save_png:
                    try:
                        save_central_slice_visualization(
                            b0_field, b0_variant, png_path, adj_factor, variant_id
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save PNG visualization: {e}")
                
                created += 1
                
                if (created % max(1, int(args.log_every))) == 0:
                    print(f"[progress] variants={created} elapsed={time.time()-start:.1f}s")
                
                if (args.max_files is not None) and (created >= int(args.max_files)):
                    print(f"Reached max_files={args.max_files}, stopping early.")
                    print(f"Done. Created {created} {args.fit_type}-adjusted variants.")
                    return
    
    print(f"Done. Created {created} {args.fit_type}-adjusted variants.")


if __name__ == "__main__":
    main()
