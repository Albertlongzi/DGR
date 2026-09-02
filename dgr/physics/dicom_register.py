import os
import re
import glob
import argparse
from typing import Tuple, List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from dgr.physics.dicom_io import load_dicom_stack, save_header_to_txt

try:
    import pydicom
except ImportError as e:
    raise SystemExit("The 'pydicom' package is required. Install with: pip install pydicom") from e


def _parse_list_from_header(line: str) -> List[float]:
    """Extract numeric list from a DICOM header text line, tolerant of delimiters.

    Handles formats like: 'PixelSpacing: [0.5, 0.5]' or 'PixelSpacing = 0.5\\0.5' etc.
    """
    # Replace backslashes with commas, strip brackets
    cleaned = line.replace("\\", ",").replace("[", "").replace("]", "")
    # Extract all float-like patterns
    numbers = re.findall(r"[-+]?\d*\.??\d+(?:[eE][-+]?\d+)?", cleaned)
    return [float(n) for n in numbers]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def read_series_params(sorted_paths: List[str]) -> Tuple[
    Tuple[float, float],
    Optional[float],
    Tuple[float, float, float],
    Tuple[float, float, float, float, float, float],
    float,
]:
    """Read essential fields from the actual DICOM series via pydicom.

    Computes slice spacing from ImagePositionPatient along the slice normal when possible.

    Returns:
        pixel_spacing (row_spacing_mm, col_spacing_mm)
        slice_thickness_mm (may be None)
        image_position_patient_xyz_mm (first slice)
        image_orientation_patient (6 floats: row_dir[3], col_dir[3])
        slice_spacing_mm (derived; fallback to thickness if needed)
    """
    datasets: List["pydicom.dataset.FileDataset"] = []
    for p in sorted_paths:
        try:
            ds = pydicom.dcmread(p)
            datasets.append(ds)
        except Exception:
            continue
    if not datasets:
        raise RuntimeError("No readable DICOM datasets to extract series params.")

    first = datasets[0]
    # Pixel spacing
    try:
        ps = tuple(float(x) for x in first.PixelSpacing)  # type: ignore[attr-defined]
        if len(ps) != 2:
            raise Exception
        pixel_spacing = (ps[0], ps[1])
    except Exception:
        pixel_spacing = (1.0, 1.0)

    # Slice thickness (may be absent)
    slice_thickness: Optional[float]
    try:
        slice_thickness = float(first.SliceThickness)  # type: ignore[attr-defined]
    except Exception:
        slice_thickness = None

    # Orientation and position
    try:
        iop = [float(x) for x in first.ImageOrientationPatient]  # type: ignore[attr-defined]
        image_orientation = (iop[0], iop[1], iop[2], iop[3], iop[4], iop[5])
    except Exception:
        image_orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    try:
        ipp = [float(x) for x in first.ImagePositionPatient]  # type: ignore[attr-defined]
        image_position = (ipp[0], ipp[1], ipp[2])
    except Exception:
        image_position = (0.0, 0.0, 0.0)

    # Derived slice spacing from IPP along slice normal
    rx, ry, rz, cx, cy, cz = image_orientation
    row_dir = np.array([rx, ry, rz], dtype=np.float64)
    col_dir = np.array([cx, cy, cz], dtype=np.float64)
    slice_dir = np.cross(row_dir, col_dir)
    positions: List[float] = []
    for ds in datasets:
        try:
            ipp_s = [float(x) for x in ds.ImagePositionPatient]  # type: ignore[attr-defined]
            positions.append(float(np.dot(np.array(ipp_s, dtype=np.float64), slice_dir)))
        except Exception:
            continue
    slice_spacing = None
    if len(positions) >= 2:
        positions_sorted = sorted(positions)
        diffs = np.diff(positions_sorted)
        diffs = diffs[np.abs(diffs) > 1e-6]
        if diffs.size > 0:
            slice_spacing = float(np.median(np.abs(diffs)))
    if slice_spacing is None:
        # Fallbacks
        try:
            sbs = float(first.SpacingBetweenSlices)  # type: ignore[attr-defined]
            slice_spacing = sbs
        except Exception:
            if slice_thickness is not None:
                slice_spacing = float(slice_thickness)
            else:
                slice_spacing = 1.0

    return pixel_spacing, slice_thickness, image_position, image_orientation, float(slice_spacing)


def build_affine(
    pixel_spacing: Tuple[float, float],
    slice_spacing: float,
    image_position: Tuple[float, float, float],
    image_orientation: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Build voxel-to-world affine using DICOM orientation, spacing, and origin.

    Uses row/column direction cosines from ImageOrientationPatient and slice
    direction via cross product. Maps voxel indices (r, c, k) to patient space.
    """
    dr, dc = float(pixel_spacing[0]), float(pixel_spacing[1])
    dz = float(slice_spacing)
    ox, oy, oz = image_position
    rx, ry, rz, cx, cy, cz = image_orientation
    row_dir = np.array([rx, ry, rz], dtype=np.float64)
    col_dir = np.array([cx, cy, cz], dtype=np.float64)
    slice_dir = np.cross(row_dir, col_dir)
    # Construct affine: columns are scaled direction vectors
    A = np.eye(4, dtype=np.float64)
    A[0:3, 0] = row_dir * dr
    A[0:3, 1] = col_dir * dc
    A[0:3, 2] = slice_dir * dz
    A[0:3, 3] = np.array([ox, oy, oz], dtype=np.float64)
    return A


def _get_slice_position_from_ds(ds: "pydicom.dataset.FileDataset") -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    try:
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
    except Exception:
        pass
    try:
        return float(getattr(ds, "SliceLocation", 0.0) or 0.0)
    except Exception:
        return 0.0


def _project_ipp_along_normal(ipp: Tuple[float, float, float],
                              iop: Tuple[float, float, float, float, float, float]) -> float:
    rx, ry, rz, cx, cy, cz = iop
    row_dir = np.array([rx, ry, rz], dtype=np.float64)
    col_dir = np.array([cx, cy, cz], dtype=np.float64)
    slice_dir = np.cross(row_dir, col_dir)
    return float(np.dot(np.array(ipp, dtype=np.float64), slice_dir))


def _sort_volume_and_paths_by_world_z(volume: np.ndarray, paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Sort slices by world-space position along slice normal (ascending)."""
    if not paths:
        return volume, paths
    # Read first ds to get orientation
    first = pydicom.dcmread(paths[0], stop_before_pixels=True)
    iop = tuple(float(x) for x in getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))  # type: ignore
    positions: List[float] = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            ipp = tuple(float(x) for x in getattr(ds, "ImagePositionPatient", [0, 0, 0]))  # type: ignore
            positions.append(_project_ipp_along_normal(ipp, iop))
        except Exception:
            positions.append(0.0)
    order = sorted(range(len(paths)), key=lambda i: positions[i])
    vol_sorted = np.take(volume, order, axis=2)
    paths_sorted = [paths[i] for i in order]
    return vol_sorted, paths_sorted


def _split_tracew_series(
    volume: np.ndarray,
    sorted_paths: List[str],
) -> Tuple[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str]]]:
    total_slices = volume.shape[2]
    if total_slices % 2 != 0:
        raise RuntimeError(
            f"TRACEW series expected even number of slices, got {total_slices}"
        )

    slice_positions: List[float] = []
    for path in sorted_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        slice_positions.append(round(_get_slice_position_from_ds(ds), 4))

    pos_to_indices: Dict[float, List[int]] = {}
    for idx, pos in enumerate(slice_positions):
        pos_to_indices.setdefault(pos, []).append(idx)

    # Group indices by position; for each position we have two frames (two repeats or two b-values)
    group_a_indices: List[int] = []
    group_b_indices: List[int] = []
    for pos, indices in pos_to_indices.items():
        if len(indices) != 2:
            raise RuntimeError(
                "TRACEW series splitting failed: each slice location must have exactly two frames."
            )
        indices_sorted = sorted(indices)
        group_a_indices.append(indices_sorted[0])
        group_b_indices.append(indices_sorted[1])

    # Sort indices by spatial position to retain consistent superior-inferior ordering
    group_a_indices = [idx for idx, _ in sorted(
        ((idx, slice_positions[idx]) for idx in group_a_indices), key=lambda x: x[1]
    )]
    group_b_indices = [idx for idx, _ in sorted(
        ((idx, slice_positions[idx]) for idx in group_b_indices), key=lambda x: x[1]
    )]

    vol_a = np.take(volume, group_a_indices, axis=2)
    vol_b = np.take(volume, group_b_indices, axis=2)
    # Assign brighter group as b=50 (low b is brighter), dimmer as b=1000
    mean_a = float(np.mean(vol_a))
    mean_b = float(np.mean(vol_b))
    if mean_a >= mean_b:
        vol_b50, vol_b1000 = vol_a, vol_b
        paths_b50 = [sorted_paths[idx] for idx in group_a_indices]
        paths_b1000 = [sorted_paths[idx] for idx in group_b_indices]
    else:
        vol_b50, vol_b1000 = vol_b, vol_a
        paths_b50 = [sorted_paths[idx] for idx in group_b_indices]
        paths_b1000 = [sorted_paths[idx] for idx in group_a_indices]

    return (vol_b50, paths_b50), (vol_b1000, paths_b1000)


def compute_physical_range(shape: Tuple[int, int, int], affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute patient-space bounding box [min_xyz], [max_xyz] for a volume."""
    r_max, c_max, k_max = shape[0] - 1, shape[1] - 1, shape[2] - 1
    corners = np.array([
        [0, 0, 0, 1],
        [r_max, 0, 0, 1],
        [0, c_max, 0, 1],
        [0, 0, k_max, 1],
        [r_max, c_max, 0, 1],
        [r_max, 0, k_max, 1],
        [0, c_max, k_max, 1],
        [r_max, c_max, k_max, 1],
    ], dtype=np.float64).T
    world = affine @ corners
    xyz = world[0:3, :]
    xyz_min = xyz.min(axis=1)
    xyz_max = xyz.max(axis=1)
    return xyz_min, xyz_max


def resample_volume_to_target(source_vol: np.ndarray, source_affine: np.ndarray,
                              target_shape: Tuple[int, int, int], target_affine: np.ndarray) -> np.ndarray:
    """Resample source volume into target grid using nearest-neighbor for simplicity.

    For higher quality, swap to linear interpolation via scipy if available.
    """
    try:
        from scipy.ndimage import map_coordinates
        use_scipy = True
    except Exception:
        use_scipy = False

    # Build mapping from target voxel grid to source voxel coordinates
    # For each target voxel index i_t, we compute world = A_t @ [i_t,1], then i_s = inv(A_s) @ world
    inv_source_affine = np.linalg.inv(source_affine)

    t_rows, t_cols, t_slices = target_shape
    rr = np.arange(t_rows)
    cc = np.arange(t_cols)
    kk = np.arange(t_slices)
    R, C, K = np.meshgrid(rr, cc, kk, indexing="ij")
    ones = np.ones_like(R, dtype=np.float64)
    tgt_homo = np.stack([R.astype(np.float64), C.astype(np.float64), K.astype(np.float64), ones], axis=0)
    world = target_affine @ tgt_homo.reshape(4, -1)
    src_idx_homo = inv_source_affine @ world
    sr = src_idx_homo[0, :]
    sc = src_idx_homo[1, :]
    sk = src_idx_homo[2, :]

    # Using scipy if available (trilinear), else nearest-neighbor with clipping
    if use_scipy:
        coords = np.vstack([sr, sc, sk])
        resampled_flat = map_coordinates(
            source_vol,
            coords,
            order=1,
            mode="nearest",
        )
    else:
        sr_n = np.rint(sr).astype(int)
        sc_n = np.rint(sc).astype(int)
        sk_n = np.rint(sk).astype(int)
        # Clip to bounds (nearest-neighbor)
        sr_n = np.clip(sr_n, 0, source_vol.shape[0] - 1)
        sc_n = np.clip(sc_n, 0, source_vol.shape[1] - 1)
        sk_n = np.clip(sk_n, 0, source_vol.shape[2] - 1)
        resampled_flat = source_vol[sr_n, sc_n, sk_n]

    resampled = resampled_flat.reshape(target_shape)
    return resampled.astype(np.float32)


def _plot_start_mid_end_rows(
    volumes: List[np.ndarray],
    titles: List[str],
    out_path: str,
) -> None:
    """Plot start/mid/end slices for each volume in separate rows."""

    if len(volumes) != len(titles):
        raise ValueError("volumes and titles must have the same length")

    def idx_triplet(size: int) -> Tuple[int, int, int]:
        if size <= 0:
            return (0, 0, 0)
        return (0, size // 2, max(0, size - 1))

    rows = len(volumes)
    plt.figure(figsize=(12, 3 * rows))

    for row_idx, (volume, title) in enumerate(zip(volumes, titles), start=0):
        indices = idx_triplet(volume.shape[2])
        for col_offset, slice_idx in enumerate(indices):
            subplot_position = row_idx * 3 + col_offset + 1
            plt.subplot(rows, 3, subplot_position)
            plt.imshow(volume[:, :, slice_idx], cmap="gray")
            plt.title(f"{title} (k={slice_idx})")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved slice grid to: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def _print_series_summary(
    label: str,
    volume: np.ndarray,
    pixel_spacing: Tuple[float, float],
    slice_thickness: Optional[float],
    slice_spacing: float,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float, float, float],
    affine: np.ndarray,
) -> None:
    print(f"{label} header params:")
    print(f"  PixelSpacing (row, col) mm: {pixel_spacing}")
    print(f"  SliceThickness mm: {slice_thickness}")
    print(f"  Derived SliceSpacing mm: {slice_spacing}")
    print(f"  ImagePositionPatient (x,y,z) mm: {position}")
    print(f"  ImageOrientationPatient (row_xyz, col_xyz): {orientation}")
    print(f"{label} volume shape (H,W,S): {volume.shape}")
    min_xyz, max_xyz = compute_physical_range(volume.shape, affine)
    print(
        f"{label} patient-space range x:[{min_xyz[0]:.2f},{max_xyz[0]:.2f}] "
        f"y:[{min_xyz[1]:.2f},{max_xyz[1]:.2f}] z:[{min_xyz[2]:.2f},{max_xyz[2]:.2f}] mm"
    )


def process_subject(subject_dicoms_dir: str, out_dir: str) -> None:
    adc_dir = os.path.join(subject_dicoms_dir, "AX_DIFFUSION_ADC")
    t2_dir = os.path.join(subject_dicoms_dir, "AX_T2")
    tracew_dir = os.path.join(subject_dicoms_dir, "AX_DIFFUSION_TRACEW")

    if not os.path.isdir(tracew_dir):
        raise FileNotFoundError(
            f"TRACEW series not found for subject: {tracew_dir}"
        )

    os.makedirs(out_dir, exist_ok=True)

    # Load volumes and sort by world z for consistent orientation
    adc_vol, adc_paths, _ = load_dicom_stack(adc_dir)
    adc_vol, adc_paths = _sort_volume_and_paths_by_world_z(adc_vol, adc_paths)
    adc_ref = pydicom.dcmread(adc_paths[0], stop_before_pixels=True)

    t2_vol, t2_paths, _ = load_dicom_stack(t2_dir)
    t2_vol, t2_paths = _sort_volume_and_paths_by_world_z(t2_vol, t2_paths)
    t2_ref = pydicom.dcmread(t2_paths[0], stop_before_pixels=True)

    tracew_vol, tracew_paths, _ = load_dicom_stack(tracew_dir)
    tracew_vol, tracew_paths = _sort_volume_and_paths_by_world_z(tracew_vol, tracew_paths)
    (tracew_b50, tracew_b50_paths), (tracew_b1000, tracew_b1000_paths) = _split_tracew_series(
        tracew_vol, tracew_paths
    )

    # Save header txts for organization
    adc_header_path = os.path.join(out_dir, "adc_header.txt")
    save_header_to_txt(adc_ref, adc_header_path)
    save_header_to_txt(t2_ref, os.path.join(out_dir, "t2_header.txt"))
    tracew_b50_ref = pydicom.dcmread(tracew_b50_paths[0], stop_before_pixels=True)
    tracew_b1000_ref = pydicom.dcmread(tracew_b1000_paths[0], stop_before_pixels=True)
    save_header_to_txt(tracew_b50_ref, os.path.join(out_dir, "tracew_b50_header.txt"))
    save_header_to_txt(tracew_b1000_ref, os.path.join(out_dir, "tracew_b1000_header.txt"))

    # Extract params from DICOM directly
    adc_ps, adc_thk, adc_pos, adc_orient, adc_dz = read_series_params(adc_paths)
    t2_ps, t2_thk, t2_pos, t2_orient, t2_dz = read_series_params(t2_paths)
    b50_ps, b50_thk, b50_pos, b50_orient, b50_dz = read_series_params(tracew_b50_paths)
    b1000_ps, b1000_thk, b1000_pos, b1000_orient, b1000_dz = read_series_params(tracew_b1000_paths)

    # Build affines
    adc_affine = build_affine(adc_ps, adc_dz, adc_pos, adc_orient)
    t2_affine = build_affine(t2_ps, t2_dz, t2_pos, t2_orient)
    b50_affine = build_affine(b50_ps, b50_dz, b50_pos, b50_orient)
    b1000_affine = build_affine(b1000_ps, b1000_dz, b1000_pos, b1000_orient)

    # Print summaries
    _print_series_summary("ADC", adc_vol, adc_ps, adc_thk, adc_dz, adc_pos, adc_orient, adc_affine)
    _print_series_summary("TRACEW b=50", tracew_b50, b50_ps, b50_thk, b50_dz, b50_pos, b50_orient, b50_affine)
    _print_series_summary("TRACEW b=1000", tracew_b1000, b1000_ps, b1000_thk, b1000_dz, b1000_pos, b1000_orient, b1000_affine)
    _print_series_summary("T2", t2_vol, t2_ps, t2_thk, t2_dz, t2_pos, t2_orient, t2_affine)

    # Resample diffusion-derived volumes into T2 space
    adc_in_t2 = resample_volume_to_target(
        source_vol=adc_vol,
        source_affine=adc_affine,
        target_shape=t2_vol.shape,
        target_affine=t2_affine,
    )
    b50_in_t2 = resample_volume_to_target(
        source_vol=tracew_b50,
        source_affine=b50_affine,
        target_shape=t2_vol.shape,
        target_affine=t2_affine,
    )
    b1000_in_t2 = resample_volume_to_target(
        source_vol=tracew_b1000,
        source_affine=b1000_affine,
        target_shape=t2_vol.shape,
        target_affine=t2_affine,
    )

    # Plot grids before and after registration
    pre_grid_path = os.path.join(out_dir, "pre_registration_slices.png")
    _plot_start_mid_end_rows(
        volumes=[t2_vol, adc_vol, tracew_b50, tracew_b1000],
        titles=["T2 (native)", "ADC (native)", "TRACEW b=50 (native)", "TRACEW b=1000 (native)"],
        out_path=pre_grid_path,
    )

    reg_grid_path = os.path.join(out_dir, "registered_dwi_to_t2_slices.png")
    _plot_start_mid_end_rows(
        volumes=[t2_vol, adc_in_t2, b50_in_t2, b1000_in_t2],
        titles=["T2", "ADC→T2", "TRACEW b=50→T2", "TRACEW b=1000→T2"],
        out_path=reg_grid_path,
    )

    # Save volumes and affines for future processing
    npz_path = os.path.join(out_dir, "registered_volumes.npz")
    np.savez_compressed(
        npz_path,
        t2_volume=t2_vol.astype(np.float32),
        adc_volume=adc_vol.astype(np.float32),
        tracew_b50_volume=tracew_b50.astype(np.float32),
        tracew_b1000_volume=tracew_b1000.astype(np.float32),
        dwi_registered=adc_in_t2.astype(np.float32),
        adc_registered=adc_in_t2.astype(np.float32),
        tracew_b50_registered=b50_in_t2.astype(np.float32),
        tracew_b1000_registered=b1000_in_t2.astype(np.float32),
        t2_affine=t2_affine.astype(np.float64),
        dwi_affine=adc_affine.astype(np.float64),
        adc_affine=adc_affine.astype(np.float64),
        tracew_b50_affine=b50_affine.astype(np.float64),
        tracew_b1000_affine=b1000_affine.astype(np.float64),
    )
    print(f"Saved registered volumes to: {npz_path}")


def discover_subjects(dataset_root: str, limit: Optional[int] = 5) -> List[str]:
    subjects = []
    pattern = os.path.join(dataset_root, "subjects", "sub-*/DICOMS")
    for path in sorted(glob.glob(pattern)):
        # ensure required series exist
        if (
            os.path.isdir(os.path.join(path, "AX_DIFFUSION_ADC"))
            and os.path.isdir(os.path.join(path, "AX_T2"))
            and os.path.isdir(os.path.join(path, "AX_DIFFUSION_TRACEW"))
        ):
            subjects.append(path)
        if limit is not None and len(subjects) >= limit:
            break
    return subjects


def main():
    parser = argparse.ArgumentParser(description="Register diffusion-derived series to T2 for fastMRI prostate subjects")
    parser.add_argument("--dataset_root", type=str, default="/path/to/dgr_data/fastmri_prostate_v3",
                        help="Dataset root containing subjects/sub-*/DICOMS")
    parser.add_argument("--output_root", type=str, default=os.path.join(os.getcwd(), "fastmri_registration_outputs"),
                        help="Directory to write outputs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of subjects to process (omit for all)")
    parser.add_argument("--subject_dir", type=str, default=None,
                        help="Optional: process only this subject's DICOMS directory (e.g., .../subjects/sub-001/DICOMS)")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    
    if args.subject_dir is not None:
        dicoms_dir = args.subject_dir
        subject_id = os.path.basename(os.path.dirname(dicoms_dir))
        out_dir = os.path.join(args.output_root, subject_id)
        print(f"Processing single subject {subject_id} ...")
        process_subject(dicoms_dir, out_dir)
        return

    subjects = discover_subjects(args.dataset_root, limit=args.limit)
    if not subjects:
        raise SystemExit("No subjects found with required AX_DIFFUSION_ADC, AX_DIFFUSION_TRACEW, and AX_T2 series.")

    for idx, dicoms_dir in enumerate(subjects, 1):
        subject_id = os.path.basename(os.path.dirname(dicoms_dir))
        out_dir = os.path.join(args.output_root, subject_id)
        print(f"\n[{idx}/{len(subjects)}] Processing {subject_id} ...")
        try:
            process_subject(dicoms_dir, out_dir)
        except Exception as e:
            print(f"  Skipping {subject_id} due to error: {e}")


if __name__ == "__main__":
    main()


