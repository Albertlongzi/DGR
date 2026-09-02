import os
import argparse
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from dgr.physics.b0_field_read import load_b0_map
from dgr.physics.dicom_io import load_dicom_stack
from dgr.physics.dicom_register import read_series_params, build_affine, resample_volume_to_target
from dgr.physics.b0_fitting import polynomial_fit_2d_volume


def build_b0_affine_forced_z(b0_shape: Tuple[int, int, int], b0_voxel: Tuple[float, float, float],
                             t2_shape: Tuple[int, int, int], t2_affine: np.ndarray,
                             swap_xy: bool = True) -> np.ndarray:
    """Build B0 affine with axis-aligned XY (patient x,y) using B0 pixel sizes and
    force the B0 z-index span to linearly match the T2 z-index span.

    - k=0 of B0 -> k=0 plane center of T2
    - k=K-1 of B0 -> k=Nz-1 plane center of T2
    - XY are independent of k (no oblique coupling), preserving simple physical interpolation.
    """
    dr_b0, dc_b0, _ = [float(x) for x in b0_voxel]
    nx_t2, ny_t2, nz_t2 = t2_shape
    t2_z_vec = t2_affine[0:3, 2]
    # Forced z per-index vector for B0
    k_b0 = max(1, b0_shape[2] - 1)
    k_t2 = max(1, nz_t2 - 1)
    z_vec_forced = t2_z_vec * (k_t2 / float(k_b0))

    A = np.eye(4, dtype=np.float64)
    row_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    col_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if swap_xy:
        row_dir, col_dir = col_dir, row_dir
    A[0:3, 0] = row_dir * dr_b0
    A[0:3, 1] = col_dir * dc_b0
    A[0:3, 2] = z_vec_forced

    # Origin so that B0 center (r0,c0) at k=0 maps to T2 center (r_c, c_c) at k=0
    r0_b0 = (b0_shape[0] - 1) / 2.0
    c0_b0 = (b0_shape[1] - 1) / 2.0
    r_c_t2 = (nx_t2 - 1) / 2.0
    c_c_t2 = (ny_t2 - 1) / 2.0
    world_t2_k0_center = (t2_affine @ np.array([r_c_t2, c_c_t2, 0.0, 1.0], dtype=np.float64))[:3]
    origin = world_t2_k0_center - A[0:3, 0] * r0_b0 - A[0:3, 1] * c0_b0
    A[0:3, 3] = origin
    return A


def extract_slice_mask(data_dict: dict, map_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, float]:
    """Extract UNIC_B0Map.Parameters.Mask (or 'mask') for valid-slice detection only.

    Returns a 3D boolean mask and the center z-index (float) of slices with mask.
    This is NOT the phase mask. No phasemask variants are considered here.
    """
    ub = data_dict.get("UNIC_B0Map", None)
    if ub is None:
        raise RuntimeError("UNIC_B0Map not found in MAT data")

    prm = None
    if hasattr(ub, "Parameters"):
        prm = getattr(ub, "Parameters")
    elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
        prm = ub["Parameters"]  # type: ignore[index]
    if prm is None:
        raise RuntimeError("UNIC_B0Map.Parameters not found in MAT data")

    m = None
    try:
        if hasattr(prm, "Mask"):
            m = np.asarray(getattr(prm, "Mask")).squeeze()
        elif hasattr(prm, "mask"):
            m = np.asarray(getattr(prm, "mask")).squeeze()
        elif isinstance(prm, dict) and "Mask" in prm:
            m = np.asarray(prm["Mask"]).squeeze()
        elif isinstance(prm, dict) and "mask" in prm:
            m = np.asarray(prm["mask"]).squeeze()
    except Exception:
        m = None

    if not isinstance(m, np.ndarray) or m.ndim != 3 or m.shape != map_shape:
        raise RuntimeError("Parameters.Mask (or mask) missing or has invalid shape; expected 3D and same as Map")

    mask = m.astype(bool)
    z_any = mask.reshape(-1, map_shape[2]).any(axis=0)
    if np.any(z_any):
        z_idx = np.where(z_any)[0]
        z_center = float((z_idx.min() + z_idx.max()) / 2.0)
    else:
        z_center = float((map_shape[2] - 1) / 2.0)
    return mask, z_center


def extract_phase_mask(data_dict: dict, map_shape: Tuple[int, int, int]) -> np.ndarray:
    """Extract UNIC_B0Map.Parameters.phasemask/phaseMask/Phasemask.

    If absent or invalid, returns an all-ones mask (no cavities).
    """
    ub = data_dict.get("UNIC_B0Map", None)
    if ub is None:
        return np.ones(map_shape, dtype=np.float32)
    prm = None
    if hasattr(ub, "Parameters"):
        prm = getattr(ub, "Parameters")
    elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
        prm = ub["Parameters"]  # type: ignore[index]
    if prm is None:
        return np.ones(map_shape, dtype=np.float32)

    m = None
    try:
        for key in ["phasemask", "phaseMask", "Phasemask"]:
            if hasattr(prm, key):
                m = np.asarray(getattr(prm, key)).squeeze()
                break
        if m is None and isinstance(prm, dict):
            for key in ["phasemask", "phaseMask", "Phasemask"]:
                if key in prm:
                    m = np.asarray(prm[key]).squeeze()
                    break
    except Exception:
        m = None
    if not isinstance(m, np.ndarray) or m.ndim != 3 or m.shape != map_shape:
        return np.ones(map_shape, dtype=np.float32)
    return (m.astype(np.float32) > 0.001).astype(np.float32)


def load_b0_map_with_params(mat_path: str) -> Tuple[np.ndarray, float, Tuple[float, float, float], dict]:
    """Load B0 Map and CentralFreq, plus voxel size from Parameters if available.

    Returns (freq_map_hz, central_freq_hz, voxel_size_mm (dr,dc,dz), full_data_dict)
    """
    # Reuse loader to get Map and CentralFreq
    # We also need the full dict for Parameters.voxelsize and Mask
    # We'll use scipy.io.loadmat directly here for rich access
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy is required") from e
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Extract Map and CentralFeq/CentralFreq as in B0_field_read
    ub = data.get("UNIC_B0Map", None)
    if ub is None:
        raise RuntimeError("UNIC_B0Map not found in MAT file")
    if hasattr(ub, "Map"):
        b0_map = np.asarray(getattr(ub, "Map")).squeeze().astype(np.float32)
    else:
        b0_map = np.asarray(ub["Map"]).squeeze().astype(np.float32)  # type: ignore[index]
    # Central frequency
    central = None
    prm = None
    if hasattr(ub, "Parameters"):
        prm = getattr(ub, "Parameters")
    elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
        prm = ub["Parameters"]  # type: ignore[index]
    if prm is not None:
        for key in ["CentralFeq", "CentralFreq", "centralFeq", "centralFreq"]:
            try:
                if hasattr(prm, key):
                    central = float(getattr(prm, key))
                    break
                if isinstance(prm, dict) and key in prm:
                    central = float(prm[key])
                    break
            except Exception:
                continue
    if central is None:
        raise RuntimeError("Central frequency not found in Parameters")

    # Voxel size
    vox = (1.0, 1.0, 1.0)

    def _fields_of(p) -> list:
        names = []
        try:
            if hasattr(p, "_fieldnames") and p._fieldnames is not None:
                names.extend(list(p._fieldnames))
        except Exception:
            pass
        try:
            if isinstance(p, np.void) and getattr(p, "dtype", None) is not None and p.dtype.names:
                names.extend(list(p.dtype.names))
        except Exception:
            pass
        try:
            if isinstance(p, dict):
                names.extend(list(p.keys()))
        except Exception:
            pass
        return sorted(set(names))

    def _get_vec(p, keys):
        for key in keys:
            try:
                if hasattr(p, key):
                    v = np.asarray(getattr(p, key)).squeeze()
                    return v
                if isinstance(p, dict) and key in p:
                    v = np.asarray(p[key]).squeeze()
                    return v
            except Exception:
                continue
        return None

    if prm is not None:
        # Try direct voxel size keys
        v = _get_vec(prm, ["voxelsize", "voxelSize", "VoxelSize", "Voxelsize", "VoxelSpacing", "Resolution", "resolution", "dxyz", "DeltaXYZ", "PixelSpacing"])
        if v is not None:
            arr = np.asarray(v).ravel()
            if arr.size >= 3:
                vox = (float(arr[0]), float(arr[1]), float(arr[2]))
            elif arr.size == 2:
                # Need slice spacing too
                dz = _get_vec(prm, ["SliceThickness", "sliceThickness", "SliceSpacing", "SpacingBetweenSlices", "dz", "DeltaZ"]) or 1.0
                dz = float(np.asarray(dz).ravel()[0]) if isinstance(dz, (np.ndarray, list, tuple)) else float(dz)
                vox = (float(arr[0]), float(arr[1]), float(dz))
        # If still default or obviously wrong, try derive from FOV and MatrixSize
        if vox == (1.0, 1.0, 1.0) or any(x <= 0 for x in vox):
            fov = _get_vec(prm, ["FOV", "fov", "FoV", "Fov", "FieldOfView"])
            msz = _get_vec(prm, ["MatrixSize", "matrixSize", "ImageSize", "Size", "matrix"])
            dz = _get_vec(prm, ["SliceThickness", "sliceThickness", "SliceSpacing", "SpacingBetweenSlices"]) or 1.0
            try:
                if fov is not None and msz is not None:
                    fov_arr = np.asarray(fov).ravel().astype(float)
                    msz_arr = np.asarray(msz).ravel().astype(float)
                    # Use first two dims for in-plane; third from dz if available
                    if fov_arr.size >= 2 and msz_arr.size >= 2 and msz_arr[0] > 0 and msz_arr[1] > 0:
                        dr = fov_arr[0] / msz_arr[0]
                        dc = fov_arr[1] / msz_arr[1]
                        dz_val = float(np.asarray(dz).ravel()[0]) if isinstance(dz, (np.ndarray, list, tuple)) else float(dz)
                        vox = (float(dr), float(dc), float(dz_val))
            except Exception:
                pass

    freq_map_hz = b0_map.astype(np.float32) * float(central)
    return freq_map_hz, float(central), vox, data


def compute_physical_range(shape: Tuple[int, int, int], affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Register B0 frequency map to T2 with forced Z span mapping")
    parser.add_argument("--mat_path", type=str,
                        default="/path/to/dgr_data/B0_folder/subject_012825_left/scanner/UNIC_B0Mapinphae_3D.mat",
                        help="Path to B0 .mat file")
    parser.add_argument("--t2_dir", type=str,
                        default="/path/to/dgr_data/fastmri_prostate_v3/subjects/sub-001/DICOMS/AX_T2",
                        help="Reference T2 DICOM series directory (ideally max-slice subject)")
    parser.add_argument("--out_dir", type=str, default=os.path.join(os.getcwd(), "B0_registration_outputs"))
    parser.add_argument("--force_z_span", action="store_true", help="Force B0 z-span to match T2 z-span (start->start, end->end)")
    parser.add_argument("--inspect_params_only", action="store_true",
                        help="Only load MAT and print UNIC_B0Map.Parameters fields and phasemask presence, then exit")
    parser.add_argument("--flip_updown", action="store_true", default=True,
                        help="Flip the registered B0 (and phase mask) along the row (up-down) axis before saving/plotting")
    parser.add_argument("--fit2d_order", type=int, default=4,
                        help="Per-slice 2D polynomial fit order on B0 within phase mask (default: 4)")
    parser.add_argument("--fit2d_ridge", type=float, default=0.0,
                        help="Ridge regularization lambda for 2D polynomial fit (default: 0.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # When only inspecting parameter fields, no need to load T2
    if args.inspect_params_only:
        freq_map_hz, central_hz, vox_size, data = load_b0_map_with_params(args.mat_path)
        ub = data.get("UNIC_B0Map", None)
        prm = None
        if ub is not None:
            if hasattr(ub, "Parameters"):
                prm = getattr(ub, "Parameters")
            elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
                prm = ub["Parameters"]  # type: ignore[index]
        fields = []
        try:
            if hasattr(prm, "_fieldnames") and prm._fieldnames is not None:
                fields.extend(list(prm._fieldnames))
            elif isinstance(prm, np.void) and getattr(prm, "dtype", None) is not None and prm.dtype.names:
                fields.extend(list(prm.dtype.names))
            elif isinstance(prm, dict):
                fields.extend(list(prm.keys()))
        except Exception:
            pass
        print("[INSPECT] Parameters fields:", sorted(set(fields)))
        def _report(name: str):
            try:
                if hasattr(prm, name):
                    x = np.asarray(getattr(prm, name)).squeeze()
                elif isinstance(prm, dict) and name in prm:
                    x = np.asarray(prm[name]).squeeze()
                else:
                    print(f"[INSPECT] {name}: NOT FOUND")
                    return
                print(f"[INSPECT] {name}: shape={x.shape} dtype={x.dtype} min={np.min(x)} max={np.max(x)} uniq={np.unique(x)[:10]}")
            except Exception as e:
                print(f"[INSPECT] {name}: ERROR {e}")
        for key in ["phasemask", "phaseMask", "Phasemask", "Mask", "mask", "MatrixSize", "FOV", "voxelsize", "SliceThickness", "SpacingBetweenSlices"]:
            _report(key)
        return

    # Load T2 stack and affine
    t2_vol, t2_paths, _ = load_dicom_stack(args.t2_dir)
    t2_ps, t2_thk, t2_pos, t2_orient, t2_dz = read_series_params(t2_paths)
    t2_affine = build_affine(t2_ps, t2_dz, t2_pos, t2_orient)

    # Compute T2 center world coordinate
    r_c = (t2_vol.shape[0] - 1) / 2.0
    c_c = (t2_vol.shape[1] - 1) / 2.0
    k_c = (t2_vol.shape[2] - 1) / 2.0
    t2_center_world = (t2_affine @ np.array([r_c, c_c, k_c, 1.0], dtype=np.float64))[:3]

    # Load B0 freq map and voxel size; also get data dict to infer mask/z-center
    freq_map_hz, central_hz, vox_size, data = load_b0_map_with_params(args.mat_path)
    slice_mask, z_center_idx = extract_slice_mask(data, freq_map_hz.shape)
    phase_mask_full = extract_phase_mask(data, freq_map_hz.shape)
    # Crop B0 along z strictly to the mask==1 slice span (phase/magnitude mask), not the matrix extent
    z_any = slice_mask.reshape(-1, slice_mask.shape[2]).any(axis=0)
    if np.any(z_any):
        z_idx = np.where(z_any)[0]
        k_min = int(z_idx.min())
        k_max = int(z_idx.max())
    else:
        k_min, k_max = 0, freq_map_hz.shape[2] - 1
    b0_cropped = freq_map_hz[:, :, k_min : k_max + 1].astype(np.float32)
    # crop phase mask to same z-span for later resampling
    phase_mask_cropped = phase_mask_full[:, :, k_min : k_max + 1].astype(np.float32)

    # Extract additional B0 parameters for debug (matrix size, FOV if present)
    b0_matrix_size = None
    b0_fov = None
    ub = data.get("UNIC_B0Map", None)
    prm = None
    if ub is not None:
        if hasattr(ub, "Parameters"):
            prm = getattr(ub, "Parameters")
        elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
            prm = ub["Parameters"]  # type: ignore[index]
    def _extract_vector(p, keys):
        for key in keys:
            try:
                if hasattr(p, key):
                    v = np.asarray(getattr(p, key)).squeeze()
                    return v
                if isinstance(p, dict) and key in p:
                    v = np.asarray(p[key]).squeeze()
                    return v
            except Exception:
                continue
        return None
    if prm is not None:
        ms = _extract_vector(prm, ["MatrixSize", "matrixSize", "ImageSize", "Size", "matrix"])
        if ms is not None:
            try:
                b0_matrix_size = tuple(int(x) for x in np.asarray(ms).ravel()[:3])
            except Exception:
                b0_matrix_size = tuple(float(x) for x in np.asarray(ms).ravel()[:3])  # type: ignore[assignment]
        fv = _extract_vector(prm, ["FOV", "fov", "FoV", "Fov"])
        if fv is not None:
            try:
                b0_fov = tuple(float(x) for x in np.asarray(fv).ravel()[:3])
            except Exception:
                b0_fov = None

    print("[DEBUG] T2 shape (H,W,S):", t2_vol.shape)
    print("[DEBUG] B0 shape (H,W,S):", freq_map_hz.shape)
    print("[DEBUG] T2 spacing (row,col,slc mm):", t2_ps, t2_dz)
    # Show available parameter fields to help debug voxel size lookup
    prm_fields = []
    if prm is not None:
        try:
            if hasattr(prm, "_fieldnames") and prm._fieldnames is not None:
                prm_fields = list(prm._fieldnames)
            elif isinstance(prm, np.void) and getattr(prm, "dtype", None) is not None and prm.dtype.names:
                prm_fields = list(prm.dtype.names)
            elif isinstance(prm, dict):
                prm_fields = list(prm.keys())
        except Exception:
            prm_fields = []
    print("[DEBUG] B0 voxel size (row,col,slc mm):", vox_size)
    print("[DEBUG] Parameters fields:", prm_fields)
    print("[DEBUG] B0 matrix size (Parameters):", b0_matrix_size)
    print("[DEBUG] B0 FOV (Parameters, mm):", b0_fov)
    print("[DEBUG] Central frequency (Hz):", central_hz)
    print("[DEBUG] z_center_idx (from Mask):", z_center_idx)

    # Build B0 affine: XY follow T2 in-plane orientation with B0 pixel sizes; Z is forced to span T2 z-range
    if args.force_z_span:
        b0_affine = build_b0_affine_forced_z(b0_cropped.shape, vox_size, t2_vol.shape, t2_affine)
    else:
        # Fallback to previous center-alignment behavior
        def _build_center(map_shape, voxel_size, t2_center_world, z_center_index, swap_xy=True):
            dr, dc, dz = voxel_size
            r0 = (map_shape[0] - 1) / 2.0
            c0 = (map_shape[1] - 1) / 2.0
            k0 = float(z_center_index)
            A = np.eye(4, dtype=np.float64)
            row_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            col_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if swap_xy:
                row_dir, col_dir = col_dir, row_dir
            A[0:3, 0] = row_dir * dr
            A[0:3, 1] = col_dir * dc
            A[0:3, 2] = np.array([0.0, 0.0, 1.0], dtype=np.float64) * dz
            origin = (
                np.asarray(t2_center_world, dtype=np.float64)
                - A[0:3, 0] * r0
                - A[0:3, 1] * c0
                - A[0:3, 2] * k0
            )
            A[0:3, 3] = origin
            return A
        b0_affine = _build_center(b0_cropped.shape, vox_size, t2_center_world, z_center_idx, swap_xy=True)

    # Resample B0 into T2 space
    b0_in_t2 = resample_volume_to_target(b0_cropped, b0_affine, t2_vol.shape, t2_affine)
    
    # Resample phase mask into T2 space (nearest/threshold)
    try:
        raise Exception("force fallback")
        import nibabel as nib  # type: ignore
        from nibabel.processing import resample_from_to  # type: ignore
        src_img = nib.Nifti1Image(phase_mask_cropped.astype(np.float32), affine=b0_affine.astype(np.float64))
        tgt_img = nib.Nifti1Image(np.zeros(t2_vol.shape, dtype=np.float32), affine=t2_affine.astype(np.float64))
        mask_img = resample_from_to(src_img, tgt_img, order=0, mode="constant", cval=0.0)
        pm = mask_img.get_fdata(dtype=np.float32)
        # VALID only where value equals 1 (other values mean masked out)
        phase_mask_t2 = (np.isclose(pm, 1.0)).astype(np.float32)
    except Exception:
        # fallback: use same resampler and then threshold
        pm = resample_volume_to_target(phase_mask_cropped, b0_affine, t2_vol.shape, t2_affine).astype(np.float32)
        pm = np.rint(pm)
        phase_mask_t2 = (pm == 1.0).astype(np.float32)

    # Optional up-down flip for B0 and phase mask to match network orientation
    if args.flip_updown:
        b0_in_t2 = np.flip(b0_in_t2, axis=0).copy()
        phase_mask_t2 = np.flip(phase_mask_t2, axis=0).copy()

    # Physical ranges
    t2_min, t2_max = compute_physical_range(t2_vol.shape, t2_affine)
    b0_min, b0_max = compute_physical_range(b0_cropped.shape, b0_affine)
    print(f"[DEBUG] T2 patient-space range x:[{t2_min[0]:.2f},{t2_max[0]:.2f}] y:[{t2_min[1]:.2f},{t2_max[1]:.2f}] z:[{t2_min[2]:.2f},{t2_max[2]:.2f}] mm")
    print(f"[DEBUG] B0 patient-space range x:[{b0_min[0]:.2f},{b0_max[0]:.2f}] y:[{b0_min[1]:.2f},{b0_max[1]:.2f}] z:[{b0_min[2]:.2f},{b0_max[2]:.2f}] mm")

    # Write debug info to file
    debug_txt = os.path.join(args.out_dir, "b0_registration_debug.txt")
    with open(debug_txt, "w", encoding="utf-8") as f:
        f.write(f"T2 shape: {t2_vol.shape}\n")
        f.write(f"B0 shape orig: {freq_map_hz.shape}\n")
        f.write(f"B0 z-crop: k_min={k_min} k_max={k_max} -> shape {b0_cropped.shape}\n")
        f.write(f"T2 spacing (row,col,slc mm): {t2_ps} , {t2_dz}\n")
        f.write(f"B0 voxel size (row,col,slc mm): {vox_size}\n")
        f.write(f"Forced Z span mapping: {bool(args.force_z_span)}\n")
        f.write(f"B0 matrix size (Parameters): {b0_matrix_size}\n")
        f.write(f"B0 FOV (Parameters, mm): {b0_fov}\n")
        f.write(f"Central frequency (Hz): {central_hz}\n")
        f.write(f"z_center_idx (from Mask): {z_center_idx}\n")
        f.write(f"T2 range mm: min={t2_min.tolist()} max={t2_max.tolist()}\n")
        f.write(f"B0 range mm: min={b0_min.tolist()} max={b0_max.tolist()}\n")
    print(f"Saved debug: {debug_txt}")

    # Load DWI(ADC), resample to T2, and plot start/mid/end alongside registered B0
    dwi_in_t2 = None
    dwi_dir = os.path.join(os.path.dirname(args.t2_dir), "AX_DIFFUSION_ADC")
    if os.path.isdir(dwi_dir):
        try:
            dwi_vol, dwi_paths, _ = load_dicom_stack(dwi_dir)
            dwi_ps, dwi_thk, dwi_pos, dwi_orient, dwi_dz = read_series_params(dwi_paths)
            dwi_aff = build_affine(dwi_ps, dwi_dz, dwi_pos, dwi_orient)
            dwi_in_t2 = resample_volume_to_target(dwi_vol, dwi_aff, t2_vol.shape, t2_affine)
        except Exception as e:
            print(f"[WARN] Failed to load/resample DWI from {dwi_dir}: {e}")
            dwi_in_t2 = None
    else:
        print(f"[WARN] DWI directory not found: {dwi_dir}")

    # Do per-slice 2D polynomial fitting on B0 within phase mask at T2 resolution
    b0_fit_full, coeffs_per_slice = polynomial_fit_2d_volume(
        field_map_hz=b0_in_t2,
        mask=(phase_mask_t2 > 0),
        voxel_size_mm=(float(t2_ps[0]), float(t2_ps[1]), float(t2_dz)),
        order=int(args.fit2d_order),
        ridge_lambda=float(args.fit2d_ridge),
    )
    b0_fit_full = b0_fit_full.astype(np.float32)

    # Plot T2, B0, DWI start/mid/end (3x3 grid) if DWI available
    idxs = (0, t2_vol.shape[2] // 2, max(0, t2_vol.shape[2] - 1))
    # Determine number of rows: T2, B0, Phase Mask, (optionally) DWI, Fitted B0 (2D)
    nrows = 4  # include fitted row
    if dwi_in_t2 is not None:
        nrows += 1

    plt.figure(figsize=(12, 4 * nrows))
    row_idx = 0
    # Row 1: T2
    for j, k in enumerate(idxs, start=1):
        plt.subplot(nrows, 3, row_idx * 3 + j)
        plt.imshow(t2_vol[:, :, k], cmap="gray")
        plt.title(f"T2 k={k}")
        plt.axis("off")
    row_idx += 1
    # Row 2: B0
    for j, k in enumerate(idxs, start=1):
        plt.subplot(nrows, 3, row_idx * 3 + j)
        plt.imshow(b0_in_t2[:, :, k], cmap="bwr")
        plt.title(f"B0→T2 (Hz) k={k}")
        plt.axis("off")
    row_idx += 1
    # Row 3: Phase Mask
    for j, k in enumerate(idxs, start=1):
        plt.subplot(nrows, 3, row_idx * 3 + j)
        plt.imshow(phase_mask_t2[:, :, k], cmap="gray", vmin=0, vmax=1)
        plt.title(f"Phase Mask k={k}")
        plt.axis("off")
    row_idx += 1
    # Row 4: DWI (if available)
    if dwi_in_t2 is not None:
        for j, k in enumerate(idxs, start=1):
            plt.subplot(nrows, 3, row_idx * 3 + j)
            plt.imshow(dwi_in_t2[:, :, k], cmap="gray")
            plt.title(f"DWI→T2 k={k}")
            plt.axis("off")
        row_idx += 1
    # Row 5: B0 fitted (2D polynomial per slice)
    for j, k in enumerate(idxs, start=1):
        plt.subplot(nrows, 3, row_idx * 3 + j)
        plt.imshow(b0_fit_full[:, :, k], cmap="bwr")
        plt.title(f"B0 2D-Fit (order {args.fit2d_order}) k={k}")
        plt.axis("off")
    row_idx += 1

    plt.tight_layout()
    grid_path = os.path.join(args.out_dir, "t2_b0_phasemask_dwi_start_mid_end.png")
    plt.savefig(grid_path, dpi=150)
    print(f"Saved: {grid_path}")
    try:
        plt.show()
    except Exception:
        pass

    # Save outputs
    npz_path = os.path.join(args.out_dir, "b0_to_t2_registered.npz")
    save_dict = {
        "t2_volume": t2_vol.astype(np.float32),
        "b0_registered": b0_in_t2.astype(np.float32),
        "b0_fitted_2d": b0_fit_full.astype(np.float32),
        "t2_affine": t2_affine.astype(np.float64),
        "b0_affine": b0_affine.astype(np.float64),
        "b0_voxel_size": np.array(vox_size, dtype=np.float32),
        "z_center_idx": np.array([z_center_idx], dtype=np.float32),
        "phase_mask": phase_mask_t2.astype(np.float32),
        "fit2d_order": np.array([int(args.fit2d_order)], dtype=np.int32),
    }
    
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved: {npz_path}")
    try:
        import numpy as _np
        print(f"[SAVE] phase_mask stats: shape={phase_mask_t2.shape} min={_np.min(phase_mask_t2):.3f} max={_np.max(phase_mask_t2):.3f} mean={_np.mean(phase_mask_t2):.3f}")
        print(f"[SAVE] b0_registered stats: shape={b0_in_t2.shape} min={_np.min(b0_in_t2):.1f} max={_np.max(b0_in_t2):.1f} Hz")
    except Exception:
        pass

    # Plot mid-slice comparison
    mid_k = int(round((t2_vol.shape[2] - 1) / 2.0))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(t2_vol[:, :, mid_k], cmap="gray")
    plt.title("T2 (mid slice)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(b0_in_t2[:, :, mid_k], cmap="bwr")
    plt.title("B0→T2 (Hz, mid slice)")
    plt.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "b0_to_t2_mid.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

