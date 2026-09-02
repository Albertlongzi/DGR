#!/usr/bin/env python3
"""
Generate DWI test set for FSL topup/fugue and diffusion model evaluation.

This script generates a small test set (10% of subjects) with:
- pe_axis=0 only
- Both pe_sign directions (+1 and -1) saved in each NPZ
- Ground truth DWI for quantitative comparison (PSNR, SSIM, NMSE)

Each output NPZ contains:
  - dwi_b50_gt, dwi_b1400_gt: ground truth (undistorted)
  - dwi_b50_in_pos, dwi_b1400_in_pos: distorted with pe_sign=+1
  - dwi_b50_in_neg, dwi_b1400_in_neg: distorted with pe_sign=-1
  - vdm_pos, vdm_neg: VDM for each direction
  - t2: T2 reference image
  - metadata (pe_axis, paths, etc.)
"""

import os
import sys
import glob
import argparse
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dgr.utils.warp import compute_vdm_from_b0_2d_ESP
from dgr.utils.epi_warp import forward_splat_with_fallback


# ----------------------------
# Input roots (defaults)
# ----------------------------
DEFAULT_INPUT_ROOTS = [
    "/path/to/dgr_data/Local_train_data2_preprocessed",
    "/path/to/dgr_data/preprocessed_fastmri_prostate",
    "/path/to/dgr_data/preprocessed_diease_prostate_outputs",
]

DEFAULT_B0_ROOT = "/path/to/dgr_data/B0_variants_poly2/order_12"
DEFAULT_OUTPUT_ROOT = "/path/to/dgr_data/dwi_testset"


@dataclass
class TestSetTask:
    """Task description for processing a single test sample (both pe_sign directions)."""
    npz_path: str
    dataset_label: str
    subject_id: str
    b0_variant_path: str
    b0_folder: str
    pe_axis: int  # Always 0 for this script
    output_root: str
    smooth_sigma: float
    esp_s: float
    npe: int
    pf: float
    r: float
    vdm_pe_src_mm: float
    vdm_pe_tgt_mm: float
    overwrite: bool


def load_dwi_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load low/high b-value DWIs and T2 from a preprocessed NPZ.

    Expected keys (based on preprocessed scripts):
      - dwi_b50
      - dwi_b1400
      - t2
    """
    d = np.load(npz_path, allow_pickle=False)
    try:
        # Common keys across our preprocessed pipelines
        keys_low = ["dwi_b50", "b50", "trace_b50", "tracew_b50"]
        keys_high = ["dwi_b1400", "b1400", "tracew_b1400"]
        keys_t2 = ["t2", "t2_volume", "reconstruction_rss"]

        def _pick(keys: List[str]) -> Optional[np.ndarray]:
            for k in keys:
                if k in d:
                    return np.asarray(d[k]).astype(np.float32)
            return None

        low = _pick(keys_low)
        high = _pick(keys_high)
        t2 = _pick(keys_t2)
        if low is None or high is None or t2 is None:
            raise ValueError(f"Missing low/high DWI or T2 in {npz_path}. Keys: {list(d.keys())}")
        if low.shape != high.shape:
            # Attempt to match depth by truncation/pad on Z only, center-crop/pad H/W if needed
            low, high = _fit_to_same_shape(low, high)
        # Align T2 to DWI shape if needed
        if t2.shape != low.shape:
            t2 = center_align_b0_to_target(t2, low.shape)
        return low, high, t2
    finally:
        d.close()


def _fit_to_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop/pad a and b to the same shape (use min H/W/D)."""
    Ha, Wa, Da = a.shape
    Hb, Wb, Db = b.shape
    H = min(Ha, Hb)
    W = min(Wa, Wb)
    D = min(Da, Db)

    def _center_crop(vol: np.ndarray, Ht: int, Wt: int, Dt: int) -> np.ndarray:
        H0 = max(0, (vol.shape[0] - Ht) // 2)
        W0 = max(0, (vol.shape[1] - Wt) // 2)
        D0 = 0
        cropped = vol[H0:H0 + Ht, W0:W0 + Wt, D0:D0 + Dt]
        if cropped.shape != (Ht, Wt, Dt):
            out = np.zeros((Ht, Wt, Dt), dtype=np.float32)
            h, w, d = cropped.shape
            ph = (Ht - h) // 2
            pw = (Wt - w) // 2
            out[ph:ph + h, pw:pw + w, :d] = cropped
            return out
        return cropped

    return _center_crop(a, H, W, D), _center_crop(b, H, W, D)


def center_align_b0_to_target(b0_vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop/pad B0 volume to match target shape (H, W, Z)."""
    Ht, Wt, Zt = target_shape
    Hb, Wb, Zb = b0_vol.shape
    h = min(Ht, Hb)
    w = min(Wt, Wb)
    z = min(Zt, Zb)

    r0_t = (Ht - h) // 2
    c0_t = (Wt - w) // 2
    z0_t = (Zt - z) // 2
    r0_b = (Hb - h) // 2
    c0_b = (Wb - w) // 2
    z0_b = (Zb - z) // 2

    out = np.zeros((Ht, Wt, Zt), dtype=np.float32)
    out[r0_t:r0_t + h, c0_t:c0_t + w, z0_t:z0_t + z] = b0_vol[r0_b:r0_b + h, c0_b:c0_b + w, z0_b:z0_b + z]
    return out


def generate_dwi_pair_from_b0(
    dwi_low: np.ndarray,
    dwi_high: np.ndarray,
    b0_variant: np.ndarray,
    pe_axis: int,
    pe_sign: float,
    smooth_sigma: float,
    esp_s: float,
    npe: int,
    pf: float,
    r: float,
    vdm_pe_src_mm: float,
    vdm_pe_tgt_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (dwi_b50_in, dwi_b1400_in, vdm_volume) using B0 variant and warping params.
    No additional normalization is applied; optional smoothing on warped images.
    """
    H, W, Z = dwi_low.shape

    dwi_low_out_slices: List[np.ndarray] = []
    dwi_high_out_slices: List[np.ndarray] = []
    vdm_slices: List[np.ndarray] = []

    # Pre-align B0 to DWI shape
    b0_aligned = center_align_b0_to_target(b0_variant, (H, W, Z))

    for k in range(Z):
        b0_slice = b0_aligned[:, :, k]

        # Compute VDM in pixels, then scale to target pixel size if needed
        vdm_px = compute_vdm_from_b0_2d_ESP(
            b0_slice_hz=b0_slice,
            esp_s=esp_s,
            npe=npe,
            pf=pf,
            r=r,
            pe_axis=pe_axis,
            pe_sign=pe_sign,
            return_mm=False,
            pe_pixel_size_mm=vdm_pe_src_mm,
        )
        # Scale VDM by pixel-size ratio to match target grid mm
        px_scale = float(vdm_pe_src_mm) / float(vdm_pe_tgt_mm) if vdm_pe_tgt_mm > 0 else 1.0
        vdm_scaled = (vdm_px * px_scale).astype(np.float32)

        # Warp both low/high DWIs with the same displacement
        dwi_low_raw = forward_splat_with_fallback(
            img_2d_np=dwi_low[:, :, k],
            disp_2d_np=vdm_scaled,
            pe_axis=pe_axis,
            valid_mask_np=None,
            tau=0.2,
            alpha=1,
            hole_thresh=1e-5,
        )
        dwi_high_raw = forward_splat_with_fallback(
            img_2d_np=dwi_high[:, :, k],
            disp_2d_np=vdm_scaled,
            pe_axis=pe_axis,
            valid_mask_np=None,
            tau=0.2,
            alpha=1,
            hole_thresh=1e-5,
        )

        if smooth_sigma and smooth_sigma > 0.0:
            dwi_low_raw = gaussian_filter(dwi_low_raw, sigma=smooth_sigma)
            dwi_high_raw = gaussian_filter(dwi_high_raw, sigma=smooth_sigma)

        dwi_low_out_slices.append(dwi_low_raw.astype(np.float32))
        dwi_high_out_slices.append(dwi_high_raw.astype(np.float32))
        vdm_slices.append(vdm_scaled.astype(np.float32))

    dwi_low_out = np.stack(dwi_low_out_slices, axis=2)
    dwi_high_out = np.stack(dwi_high_out_slices, axis=2)
    vdm_vol = np.stack(vdm_slices, axis=2)
    return dwi_low_out, dwi_high_out, vdm_vol


def build_output_path(
    out_root: str,
    pe_axis: int,
    dataset_label: str,
    subject_id: str,
    b0_folder: str,
    b0_file_base: str,
) -> str:
    """Construct output directory for this combination."""
    axis_dir = f"pe_axis{pe_axis}"
    # Keep dataset label to separate different sources
    out_dir = os.path.join(out_root, axis_dir, dataset_label, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"testpair_{b0_folder}_{b0_file_base}.npz"
    return os.path.join(out_dir, fname)


def _percentile_norm01(vol: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> np.ndarray:
    """Normalize volume to [0,1] using percentile clipping for visualization."""
    v = vol.astype(np.float32)
    lo = np.percentile(v, p1)
    hi = np.percentile(v, p99)
    if hi <= lo:
        hi = lo + 1e-6
    v = (v - lo) / (hi - lo)
    v = np.clip(v, 0.0, 1.0)
    return v.astype(np.float32)


def save_testset_visualization(
    out_png_path: str,
    dwi_b50_gt: np.ndarray,
    dwi_b50_in_pos: np.ndarray,
    dwi_b50_in_neg: np.ndarray,
    dwi_b1400_gt: np.ndarray,
    dwi_b1400_in_pos: np.ndarray,
    dwi_b1400_in_neg: np.ndarray,
    vdm_pos: np.ndarray,
    vdm_neg: np.ndarray,
    t2_vol: np.ndarray,
    title: str = "",
) -> None:
    """Save a central-slice visualization for a generated test pair (both directions)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k = dwi_b50_gt.shape[2] // 2
        
        # Extract central slices
        b50_gt = dwi_b50_gt[:, :, k]
        b50_pos = dwi_b50_in_pos[:, :, k]
        b50_neg = dwi_b50_in_neg[:, :, k]
        b1400_gt = dwi_b1400_gt[:, :, k]
        b1400_pos = dwi_b1400_in_pos[:, :, k]
        b1400_neg = dwi_b1400_in_neg[:, :, k]
        vdm_pos_k = vdm_pos[:, :, k]
        vdm_neg_k = vdm_neg[:, :, k]
        t2_k = t2_vol[:, :, k]

        # Normalize raw data to [0,1] for visualization
        b50_gt_norm = _percentile_norm01(b50_gt)
        b50_pos_norm = _percentile_norm01(b50_pos)
        b50_neg_norm = _percentile_norm01(b50_neg)
        b1400_gt_norm = _percentile_norm01(b1400_gt)
        b1400_pos_norm = _percentile_norm01(b1400_pos)
        b1400_neg_norm = _percentile_norm01(b1400_neg)
        t2_k_norm = _percentile_norm01(t2_k)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        if title:
            fig.suptitle(title, fontsize=14)

        # Row 0: b50 GT, b50 +1, b50 -1, VDM +1
        axes[0, 0].imshow(b50_gt_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 0].set_title("b50 GT"); axes[0, 0].axis("off")
        axes[0, 1].imshow(b50_pos_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("b50 IN (+1)"); axes[0, 1].axis("off")
        axes[0, 2].imshow(b50_neg_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("b50 IN (-1)"); axes[0, 2].axis("off")
        im_vdm_pos = axes[0, 3].imshow(vdm_pos_k, cmap="RdBu_r", vmin=-32, vmax=32)
        axes[0, 3].set_title("VDM (+1)"); axes[0, 3].axis("off")
        plt.colorbar(im_vdm_pos, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # Row 1: b1400 GT, b1400 +1, b1400 -1, VDM -1
        axes[1, 0].imshow(b1400_gt_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("b1400 GT"); axes[1, 0].axis("off")
        axes[1, 1].imshow(b1400_pos_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("b1400 IN (+1)"); axes[1, 1].axis("off")
        axes[1, 2].imshow(b1400_neg_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 2].set_title("b1400 IN (-1)"); axes[1, 2].axis("off")
        im_vdm_neg = axes[1, 3].imshow(vdm_neg_k, cmap="RdBu_r", vmin=-32, vmax=32)
        axes[1, 3].set_title("VDM (-1)"); axes[1, 3].axis("off")
        plt.colorbar(im_vdm_neg, ax=axes[1, 3], fraction=0.046, pad=0.04)

        # Row 2: T2, diff maps
        axes[2, 0].imshow(t2_k_norm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, 0].set_title("T2"); axes[2, 0].axis("off")
        
        # Difference: distorted - GT
        diff_b50_pos = b50_pos_norm - b50_gt_norm
        diff_b50_neg = b50_neg_norm - b50_gt_norm
        axes[2, 1].imshow(diff_b50_pos, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[2, 1].set_title("b50 diff (+1 - GT)"); axes[2, 1].axis("off")
        axes[2, 2].imshow(diff_b50_neg, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[2, 2].set_title("b50 diff (-1 - GT)"); axes[2, 2].axis("off")
        
        # VDM difference (should be opposite)
        vdm_diff = vdm_pos_k + vdm_neg_k  # Should be near zero if symmetric
        im_vdm_diff = axes[2, 3].imshow(vdm_diff, cmap="RdBu_r", vmin=-5, vmax=5)
        axes[2, 3].set_title("VDM sum (should≈0)"); axes[2, 3].axis("off")
        plt.colorbar(im_vdm_diff, ax=axes[2, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Failed to save visualization {out_png_path}: {e}")


def process_testset_task(task: TestSetTask) -> Tuple[bool, str]:
    """
    Process a single test set task (generate both pe_sign directions).
    Returns (success, error_message).
    """
    try:
        # Build output path
        base_name = os.path.basename(task.b0_variant_path)
        b0_base_noext = base_name[:-4] if base_name.lower().endswith(".npz") else base_name
        
        out_path = build_output_path(
            out_root=task.output_root,
            pe_axis=task.pe_axis,
            dataset_label=task.dataset_label,
            subject_id=task.subject_id,
            b0_folder=task.b0_folder,
            b0_file_base=b0_base_noext,
        )
        
        if (not task.overwrite) and os.path.isfile(out_path):
            return (True, "skipped")

        # Load DWI NPZ
        dwi_low, dwi_high, t2_vol = load_dwi_npz(task.npz_path)

        # Load B0 variant
        bd = np.load(task.b0_variant_path, allow_pickle=True)
        try:
            if "b0_variant" not in bd:
                return (False, f"Missing b0_variant in {task.b0_variant_path}")
            b0_variant = np.asarray(bd["b0_variant"]).astype(np.float32)
        finally:
            bd.close()

        # Generate warped DWI pair for pe_sign = +1
        dwi_low_in_pos, dwi_high_in_pos, vdm_pos = generate_dwi_pair_from_b0(
            dwi_low=dwi_low,
            dwi_high=dwi_high,
            b0_variant=b0_variant,
            pe_axis=task.pe_axis,
            pe_sign=1.0,
            smooth_sigma=task.smooth_sigma,
            esp_s=task.esp_s,
            npe=task.npe,
            pf=task.pf,
            r=task.r,
            vdm_pe_src_mm=task.vdm_pe_src_mm,
            vdm_pe_tgt_mm=task.vdm_pe_tgt_mm,
        )

        # Generate warped DWI pair for pe_sign = -1
        dwi_low_in_neg, dwi_high_in_neg, vdm_neg = generate_dwi_pair_from_b0(
            dwi_low=dwi_low,
            dwi_high=dwi_high,
            b0_variant=b0_variant,
            pe_axis=task.pe_axis,
            pe_sign=-1.0,
            smooth_sigma=task.smooth_sigma,
            esp_s=task.esp_s,
            npe=task.npe,
            pf=task.pf,
            r=task.r,
            vdm_pe_src_mm=task.vdm_pe_src_mm,
            vdm_pe_tgt_mm=task.vdm_pe_tgt_mm,
        )

        # Save NPZ with both directions
        payload = {
            # Ground truth (undistorted)
            "dwi_b50_gt": dwi_low.astype(np.float32),
            "dwi_b1400_gt": dwi_high.astype(np.float32),
            # Distorted with pe_sign = +1
            "dwi_b50_in_pos": dwi_low_in_pos.astype(np.float32),
            "dwi_b1400_in_pos": dwi_high_in_pos.astype(np.float32),
            "vdm_pos": vdm_pos.astype(np.float32),
            # Distorted with pe_sign = -1
            "dwi_b50_in_neg": dwi_low_in_neg.astype(np.float32),
            "dwi_b1400_in_neg": dwi_high_in_neg.astype(np.float32),
            "vdm_neg": vdm_neg.astype(np.float32),
            # T2 reference
            "t2": t2_vol.astype(np.float32),
            # Metadata
            "pe_axis": np.int32(task.pe_axis),
            "input_npz_path": task.npz_path,
            "b0_variant_path": task.b0_variant_path,
            "b0_folder": task.b0_folder,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **payload)

        # Save visualization
        out_png = out_path[:-4] + ".png" if out_path.lower().endswith(".npz") else out_path + ".png"
        vis_title = f"{task.dataset_label}/{task.subject_id}\n{task.b0_folder} | pe_axis={task.pe_axis}"
        save_testset_visualization(
            out_png_path=out_png,
            dwi_b50_gt=dwi_low,
            dwi_b50_in_pos=dwi_low_in_pos,
            dwi_b50_in_neg=dwi_low_in_neg,
            dwi_b1400_gt=dwi_high,
            dwi_b1400_in_pos=dwi_high_in_pos,
            dwi_b1400_in_neg=dwi_high_in_neg,
            vdm_pos=vdm_pos,
            vdm_neg=vdm_neg,
            t2_vol=t2_vol,
            title=vis_title,
        )

        return (True, "success")
    except Exception as e:
        return (False, str(e))


def discover_input_npz(input_roots: List[str]) -> List[Tuple[str, str, str]]:
    """
    Recursively discover NPZ files from input roots.
    Returns list of (dataset_label, subject_id, npz_path).
    """
    entries: List[Tuple[str, str, str]] = []
    for root in input_roots:
        if not os.path.isdir(root):
            continue
        dataset_label = os.path.basename(root.rstrip("/"))
        for path in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True):
            # Heuristic subject id: relative path without extension
            rel = os.path.relpath(path, root)
            subject_id = rel.replace("/", "_").replace("\\", "_")
            subject_id = subject_id[:-4] if subject_id.lower().endswith(".npz") else subject_id
            entries.append((dataset_label, subject_id, path))
    return entries


def list_b0_subject_folders(b0_root: str) -> List[str]:
    """List B0 subject folders under b0_root."""
    folders: List[str] = []
    if not os.path.isdir(b0_root):
        return folders
    for name in os.listdir(b0_root):
        p = os.path.join(b0_root, name)
        if os.path.isdir(p):
            folders.append(name)
    folders.sort()
    return folders


def list_b0_variants_in_folder(folder_path: str) -> List[str]:
    """List all B0 variant NPZ files within a B0 subject folder (ignore PNGs)."""
    all_npz = glob.glob(os.path.join(folder_path, "*.npz"))
    return [p for p in all_npz if not p.endswith("_central_slice.png")]


def main():
    ap = argparse.ArgumentParser(
        description="Generate DWI test set for FSL topup/fugue and diffusion model evaluation"
    )
    ap.add_argument("--input_roots", type=str, nargs="*",
                    default=DEFAULT_INPUT_ROOTS,
                    help="List of preprocessed NPZ roots to scan recursively")
    ap.add_argument("--b0_root", type=str, default=DEFAULT_B0_ROOT,
                    help="Root containing B0 variant subject folders")
    ap.add_argument("--output_root", type=str,
                    default=DEFAULT_OUTPUT_ROOT,
                    help="Output root directory")
    ap.add_argument("--test_fraction", type=float, default=0.1,
                    help="Fraction of subjects to include in test set (default: 0.1 = 10%%)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible subject selection")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs")
    ap.add_argument("--smooth_sigma", type=float, default=1.5,
                    help="Gaussian smoothing sigma for warped DWI (0 to disable)")
    ap.add_argument("--num_workers", type=int, default=10,
                    help="Number of parallel worker processes (default: 10)")
    # EPI / VDM parameters
    ap.add_argument("--esp_s", type=float, default=0.00068, help="Echo spacing (s)")
    ap.add_argument("--npe", type=int, default=100, help="Phase encoding steps")
    ap.add_argument("--pf", type=float, default=1.0, help="Partial Fourier factor")
    ap.add_argument("--r", type=float, default=2.0, help="Acceleration factor")
    ap.add_argument("--vdm_pe_src_mm", type=float, default=2.0, help="Source pixel size (mm) for VDM scaling")
    ap.add_argument("--vdm_pe_tgt_mm", type=float, default=0.5625, help="Target pixel size (mm) for VDM scaling")
    args = ap.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Discover input NPZs
    all_inputs = discover_input_npz(args.input_roots)
    print(f"Found {len(all_inputs)} total DWI NPZs across all input roots")
    
    # Sample test_fraction of subjects
    n_test = max(1, int(len(all_inputs) * args.test_fraction))
    inputs = random.sample(all_inputs, n_test)
    print(f"Selected {len(inputs)} subjects ({args.test_fraction*100:.1f}%) for test set")

    # B0 subject folders
    b0_subject_folders = list_b0_subject_folders(args.b0_root)
    if not b0_subject_folders:
        print(f"[WARN] No B0 subject folders found under {args.b0_root}")
        return

    print(f"Found {len(b0_subject_folders)} B0 subject folders")

    # Build task list
    tasks: List[TestSetTask] = []
    print(f"Building task list...")
    
    for dataset_label, subject_id, npz_path in inputs:
        # Select ONE random B0 subject folder
        b0_folder = random.choice(b0_subject_folders)
        folder_path = os.path.join(args.b0_root, b0_folder)
        variant_paths = list_b0_variants_in_folder(folder_path)
        
        if not variant_paths:
            print(f"[WARN] No B0 variants in {folder_path}, skipping {subject_id}")
            continue

        # Select ONE random B0 variant
        b0_variant_path = random.choice(variant_paths)
        
        # Create task for pe_axis=0 only (both pe_sign directions handled in task)
        task = TestSetTask(
            npz_path=npz_path,
            dataset_label=dataset_label,
            subject_id=subject_id,
            b0_variant_path=b0_variant_path,
            b0_folder=b0_folder,
            pe_axis=0,  # Only pe_axis=0
            output_root=args.output_root,
            smooth_sigma=args.smooth_sigma,
            esp_s=args.esp_s,
            npe=args.npe,
            pf=args.pf,
            r=args.r,
            vdm_pe_src_mm=args.vdm_pe_src_mm,
            vdm_pe_tgt_mm=args.vdm_pe_tgt_mm,
            overwrite=args.overwrite,
        )
        tasks.append(task)

    print(f"Total tasks: {len(tasks)}")
    print(f"Using {args.num_workers} worker processes")
    print(f"Output directory: {args.output_root}")

    # Process tasks in parallel
    total_written = 0
    total_skipped = 0
    total_failed = 0
    
    with Pool(processes=args.num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_testset_task, tasks),
            total=len(tasks),
            desc="Generating test pairs"
        ))
    
    # Count results
    for success, msg in results:
        if success:
            if msg == "skipped":
                total_skipped += 1
            else:
                total_written += 1
        else:
            total_failed += 1
            print(f"[ERROR] Task failed: {msg}")

    print(f"\n{'='*60}")
    print(f"Test set generation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_root}")
    print(f"Wrote: {total_written} test NPZs (each with both pe_sign directions)")
    print(f"Skipped (already exist): {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"\nEach NPZ contains:")
    print(f"  - dwi_b50_gt, dwi_b1400_gt (ground truth)")
    print(f"  - dwi_b50_in_pos, dwi_b1400_in_pos, vdm_pos (pe_sign=+1)")
    print(f"  - dwi_b50_in_neg, dwi_b1400_in_neg, vdm_neg (pe_sign=-1)")
    print(f"  - t2 (T2 reference)")


if __name__ == "__main__":
    main()

