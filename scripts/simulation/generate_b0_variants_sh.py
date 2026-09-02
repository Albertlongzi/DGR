#!/usr/bin/env python3
"""
Generate B0 variants by adjusting spherical harmonic coefficients.

This script takes B0_registration_outputs_sweep files and generates variants by:
1. Extracting SH coefficients from b0_fitted_2d
2. Adjusting only high-order coefficients (order >= 3) 
3. Keeping low-order coefficients (order 0,1,2) unchanged for physical consistency
4. Reconstructing B0 field from modified coefficients

This approach provides more diverse B0 fields while maintaining physical plausibility.
"""

import os
import sys
import glob
import argparse
import time
import math
from typing import Iterable, List, Tuple, Dict
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from scipy.special import lpmv
except ImportError:
    lpmv = None

from dgr.physics.b0_fitting import spherical_harmonics_fit, FitInputs, FitOutputs


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def find_registered_b0(root: str) -> Iterable[str]:
    """Find all b0_to_t2_registered.npz files recursively."""
    pat = os.path.join(root, "**", "b0_to_t2_registered.npz")
    for p in glob.glob(pat, recursive=True):
        yield p


def load_b0_and_metadata(path: str) -> Tuple[np.ndarray, Dict]:
    """Load B0 field and metadata from registration file."""
    d = np.load(path, allow_pickle=True)
    
    # Use original registered B0 field for SH fitting (not the already-fitted one)
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


def extract_sh_coefficients(b0_field: np.ndarray, mask: np.ndarray, order: int, 
                          voxel_size_mm: Tuple[float, float, float]) -> np.ndarray:
    """Extract SH coefficients from B0 field."""
    # Create fit inputs
    fit_inputs = FitInputs(
        field_map_hz=b0_field,
        mask=mask,
        order=order,
        voxel_size_mm=voxel_size_mm,
        fov_size_mm=None,  # Will be computed from voxel_size and shape
    )
    
    # Fit SH coefficients
    fit_outputs = spherical_harmonics_fit(fit_inputs, ridge_lambda=0.0)
    return fit_outputs.coeffs


def reconstruct_b0_from_coeffs(coeffs: np.ndarray, shape: Tuple[int, int, int], 
                              order: int, voxel_size_mm: Tuple[float, float, float]) -> np.ndarray:
    """Reconstruct B0 field from SH coefficients."""
    nx, ny, nz = shape
    dx, dy, dz = voxel_size_mm
    
    # Build coordinate grids
    x1d_mm = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    y1d_mm = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    z1d_mm = np.arange(nz, dtype=np.float64) - (nz - 1) / 2.0
    
    x1d_mm *= dx
    y1d_mm *= dy
    z1d_mm *= dz
    
    Xcm, Ycm, Zcm = np.meshgrid(x1d_mm / 10.0, y1d_mm / 10.0, z1d_mm / 10.0, indexing="ij")
    
    # Build design matrix for all voxels
    from dgr.physics.b0_fitting import _design_matrix_real_sh
    X = _design_matrix_real_sh(Xcm.flatten(), Ycm.flatten(), Zcm.flatten(), order)
    
    # Reconstruct field
    field_flat = X @ coeffs
    field = field_flat.reshape(shape)
    
    return field.astype(np.float32)


def adjust_high_order_coeffs(coeffs: np.ndarray, order: int, 
                            adjustment_factor: float, 
                            min_order: int = 3) -> np.ndarray:
    """Adjust high-order SH coefficients while keeping low-order ones unchanged."""
    adjusted_coeffs = coeffs.copy()
    
    coeff_idx = 0
    for l in range(order + 1):
        n_coeffs_l = 2 * l + 1
        
        if l >= min_order:  # Only adjust high-order coefficients
            # Add random adjustment
            adjustment = np.random.normal(0, adjustment_factor, n_coeffs_l)
            adjusted_coeffs[coeff_idx:coeff_idx + n_coeffs_l] += adjustment
        
        coeff_idx += n_coeffs_l
    
    return adjusted_coeffs


def make_out_path(src_path: str, in_root: str, out_root: str, 
                 adjustment_factor: float, variant_id: int) -> str:
    """Create output path for SH-adjusted variant."""
    rel_dir = os.path.relpath(os.path.dirname(src_path), start=in_root)
    out_dir = os.path.join(out_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = f"b0_sh_variant_factor{adjustment_factor:g}_id{variant_id:02d}.npz"
    return os.path.join(out_dir, base)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate B0 variants by adjusting SH coefficients")
    ap.add_argument("--registered_root", type=str, required=True,
                    help="Root of registered B0 outputs (recursively scanned for b0_to_t2_registered.npz)")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output root to mirror input tree with SH variants")
    ap.add_argument("--adjustment_factors", type=str, default="0.1,0.2,0.3,0.5",
                    help="Comma-separated list of adjustment factors for high-order coefficients")
    ap.add_argument("--variants_per_factor", type=int, default=3,
                    help="Number of variants to generate per adjustment factor")
    ap.add_argument("--min_order", type=int, default=3,
                    help="Minimum SH order to adjust (keep lower orders unchanged)")
    ap.add_argument("--copy_t2", action="store_true", help="Copy t2_volume into each variant")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","float32"])
    ap.add_argument("--max_files", type=int, default=None, help="Stop after generating at most this many variants")
    ap.add_argument("--log_every", type=int, default=50, help="Progress print frequency")
    args = ap.parse_args()

    if lpmv is None:
        raise RuntimeError("scipy is required for SH coefficient adjustment")

    adjustment_factors = parse_floats(args.adjustment_factors)
    variants_per_factor = args.variants_per_factor
    min_order = args.min_order

    created = 0
    start = time.time()
    
    for src in find_registered_b0(args.registered_root):
        try:
            b0_field, metadata = load_b0_and_metadata(src)
        except Exception as e:
            print(f"Skip {src}: {e}")
            continue
        
        # Create mask (assume non-zero values are valid)
        mask = np.abs(b0_field) > 1e-6
        
        # Get voxel size
        if metadata['b0_voxel_size'] is not None:
            voxel_size_mm = tuple(metadata['b0_voxel_size'])
        else:
            voxel_size_mm = (1.0, 1.0, 1.0)  # Default voxel size
        
        # Get fit order
        fit_order = metadata['fit2d_order']
        
        # Extract original SH coefficients
        try:
            original_coeffs = extract_sh_coefficients(b0_field, mask, fit_order, voxel_size_mm)
        except Exception as e:
            print(f"Failed to extract SH coefficients from {src}: {e}")
            continue
        
        # Generate variants
        for adj_factor in adjustment_factors:
            for variant_id in range(variants_per_factor):
                outp = make_out_path(src, args.registered_root, args.out_root, adj_factor, variant_id)
                
                if (not args.overwrite) and os.path.isfile(outp):
                    continue
                
                # Adjust high-order coefficients
                adjusted_coeffs = adjust_high_order_coeffs(
                    original_coeffs, fit_order, adj_factor, min_order
                )
                
                # Reconstruct B0 field
                try:
                    b0_variant = reconstruct_b0_from_coeffs(
                        adjusted_coeffs, b0_field.shape, fit_order, voxel_size_mm
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
                np.savez_compressed(outp, **output_data)
                created += 1
                
                if (created % max(1, int(args.log_every))) == 0:
                    print(f"[progress] variants={created} elapsed={time.time()-start:.1f}s")
                
                if (args.max_files is not None) and (created >= int(args.max_files)):
                    print(f"Reached max_files={args.max_files}, stopping early.")
                    print(f"Done. Created {created} SH-adjusted variants.")
                    return
    
    print(f"Done. Created {created} SH-adjusted variants.")


if __name__ == "__main__":
    main()
