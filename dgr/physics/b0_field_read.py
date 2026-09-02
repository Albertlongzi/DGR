import os
from typing import Tuple, Any, Dict, Optional, List
from datetime import datetime

import numpy as np


def _describe_scipy_struct(obj, indent: int = 2) -> None:
    pad = " " * indent
    try:
        # scipy mat_struct-like
        fields = getattr(obj, "_fieldnames", None)
        if fields:
            for f in fields:
                try:
                    val = getattr(obj, f)
                    if isinstance(val, np.ndarray):
                        print(f"{pad}{f}: ndarray shape={val.shape} dtype={val.dtype}")
                    else:
                        print(f"{pad}{f}: type={type(val).__name__}")
                except Exception:
                    print(f"{pad}{f}: <unreadable>")
            return
    except Exception:
        pass
    # numpy void (structured array element)
    if isinstance(obj, np.void) and hasattr(obj, "dtype") and obj.dtype.names:
        for f in obj.dtype.names:
            try:
                val = obj[f]
                if isinstance(val, np.ndarray):
                    print(f"{pad}{f}: ndarray shape={val.shape} dtype={val.dtype}")
                else:
                    print(f"{pad}{f}: type={type(val).__name__}")
            except Exception:
                print(f"{pad}{f}: <unreadable>")


def _inspect_mat_with_scipy(mat_path: str) -> None:
    try:
        from scipy.io import loadmat, whosmat
    except Exception:
        print("scipy not available to inspect mat file")
        return
    try:
        info = whosmat(mat_path)
        print("Top-level variables (whosmat):")
        for name, shape, mtype in info:
            print(f"  - {name}: shape={shape}, type={mtype}")
    except Exception:
        pass
    try:
        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"scipy.loadmat failed: {e}")
        return
    skip = {"__header__", "__version__", "__globals__"}
    print("Detailed variable summary:")
    for k, v in data.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
            if v.dtype.names:
                print(f"    fields: {list(v.dtype.names)}")
        elif hasattr(v, "_fieldnames"):
            print(f"  {k}: struct with fields")
            _describe_scipy_struct(v, indent=4)
        else:
            print(f"  {k}: type={type(v).__name__}")


def _try_load_with_scipy(mat_path: str) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int]], Optional[np.ndarray]]:
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy is required to read non-HDF5 .mat files") from e

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # Extract Map
    map_key_candidates = [
        "Map",
        "B0Map",
        "FreqMap",
        "MapInPhase",
        "MapPhase",
        "B0",
    ]
    map_arr = None
    for key in map_key_candidates:
        if key in data:
            try:
                map_arr = np.asarray(data[key]).squeeze().astype(np.float32)
                break
            except Exception:
                pass
    # Try nested struct UNIC_B0Map
    if map_arr is None and "UNIC_B0Map" in data:
        ub = data["UNIC_B0Map"]
        try:
            if hasattr(ub, "Map"):
                map_arr = np.asarray(getattr(ub, "Map")).squeeze().astype(np.float32)
            elif isinstance(ub, np.void) and ub.dtype.names and "Map" in ub.dtype.names:
                map_arr = np.asarray(ub["Map"]).squeeze().astype(np.float32)
        except Exception:
            map_arr = None
    if map_arr is None:
        raise KeyError("'Map' not found in MAT file (scipy loader)")

    # Extract Parameters
    params = data.get("Parameters", None)
    central = None
    mask = None
    phase_mask = None
    vox = None
    fov = None
    arraysize = None
    if params is not None:
        for key in ["CentralFeq", "CentralFreq", "centralFeq", "centralFreq"]:
            try:
                if hasattr(params, key):
                    central = float(getattr(params, key))
                    break
                if isinstance(params, dict) and key in params:
                    central = float(params[key])
                    break
            except Exception:
                continue
        # PhaseMask preferred
        try:
            for pm_key in ["phasemask", "PhaseMask", "phaseMask", "Phasemask", "phase_mask", "phase"]:
                if hasattr(params, pm_key):
                    phase_mask = np.asarray(getattr(params, pm_key)).astype(bool)
                    break
                if isinstance(params, dict) and pm_key in params:
                    phase_mask = np.asarray(params[pm_key]).astype(bool)
                    break
        except Exception:
            phase_mask = None
        try:
            if hasattr(params, "Mask"):
                mask = np.asarray(getattr(params, "Mask")).astype(bool)
            elif isinstance(params, dict) and "Mask" in params:
                mask = np.asarray(params["Mask"]).astype(bool)
        except Exception:
            mask = None
        try:
            if hasattr(params, "Voxelsize"):
                v = np.asarray(getattr(params, "Voxelsize")).astype(float).reshape(-1)
                if v.size >= 3:
                    vox = (float(v[0]), float(v[1]), float(v[2]))
            elif isinstance(params, dict) and "Voxelsize" in params:
                v = np.asarray(params["Voxelsize"]).astype(float).reshape(-1)
                if v.size >= 3:
                    vox = (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            vox = None
        try:
            info = getattr(getattr(params, "info", None), "img", None)
            if info is None and isinstance(params, dict) and "info" in params:
                info = params["info"]
                if isinstance(info, dict) and "img" in info:
                    info = info["img"]
            if info is not None:
                def _get(a, k):
                    if hasattr(a, k):
                        return getattr(a, k)
                    if isinstance(a, dict) and k in a:
                        return a[k]
                    return None
                dReadoutFOV = _get(info, "dReadoutFOV")
                dPhaseFOV = _get(info, "dPhaseFOV")
                dThickness = _get(info, "dThickness")
                arr = _get(info, "arraysize")
                if dReadoutFOV is not None and dPhaseFOV is not None and dThickness is not None:
                    fov = (float(np.squeeze(dReadoutFOV)), float(np.squeeze(dPhaseFOV)), float(np.squeeze(dThickness)))
                if arr is not None:
                    arr = np.asarray(arr).reshape(-1)
                    if arr.size >= 3:
                        arraysize = (int(arr[0]), int(arr[1]), int(arr[2]))
        except Exception:
            pass
    if central is None and "UNIC_B0Map" in data:
        ub = data["UNIC_B0Map"]
        try:
            prm = None
            if hasattr(ub, "Parameters"):
                prm = getattr(ub, "Parameters")
            elif isinstance(ub, np.void) and ub.dtype.names and "Parameters" in ub.dtype.names:
                prm = ub["Parameters"]
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
                try:
                    for pm_key in ["phasemask", "PhaseMask", "phaseMask", "Phasemask", "phase_mask", "phase"]:
                        if hasattr(prm, pm_key):
                            phase_mask = np.asarray(getattr(prm, pm_key)).astype(bool)
                            break
                        if isinstance(prm, dict) and pm_key in prm:
                            phase_mask = np.asarray(prm[pm_key]).astype(bool)
                            break
                except Exception:
                    pass
                try:
                    if hasattr(prm, "Mask"):
                        mask = np.asarray(getattr(prm, "Mask")).astype(bool)
                    elif isinstance(prm, dict) and "Mask" in prm:
                        mask = np.asarray(prm["Mask"]).astype(bool)
                except Exception:
                    pass
        except Exception:
            pass
    if central is None:
        for key in ["CentralFeq", "CentralFreq", "centralFeq", "centralFreq"]:
            if key in data:
                try:
                    central = float(np.asarray(data[key]).squeeze())
                    break
                except Exception:
                    continue
    if central is None:
        raise KeyError("Failed to locate Parameters.CentralFeq/CentralFreq in MAT file")

    return map_arr, float(central), mask, vox, fov, arraysize, phase_mask


def _try_load_with_h5py(mat_path: str) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int]], Optional[np.ndarray]]:
    try:
        import h5py
    except Exception as e:
        raise RuntimeError("h5py is required to read HDF5 .mat files (v7.3)") from e

    with h5py.File(mat_path, "r") as f:
        def read_dataset(name: str) -> Any:
            if name in f:
                d = f[name]
                if hasattr(d, "__array__"):
                    return np.array(d)
                return d[()]
            return None

        map_arr = None
        if "Map" in f:
            map_arr = np.array(f["Map"])
        else:
            for k in f.keys():
                try:
                    if isinstance(f[k], h5py.Group) and "Map" in f[k]:
                        map_arr = np.array(f[k]["Map"])
                        break
                except Exception:
                    continue
        if map_arr is None:
            raise KeyError("'Map' not found in MAT file (h5py loader)")
        map_arr = np.asarray(map_arr).squeeze().astype(np.float32)

        central = None
        mask = None
        phase_mask = None
        vox = None
        fov = None
        arraysize = None
        params = f.get("Parameters", None)
        if params is not None and isinstance(params, h5py.Group):
            for key in ["CentralFeq", "CentralFreq", "centralFeq", "centralFreq"]:
                if key in params:
                    try:
                        central = float(np.array(params[key]).squeeze())
                        break
                    except Exception:
                        continue
            # PhaseMask
            for pm_key in ["phasemask", "PhaseMask", "phaseMask", "Phasemask", "phase_mask", "phase"]:
                if pm_key in params:
                    try:
                        phase_mask = np.array(params[pm_key]).astype(bool)
                        break
                    except Exception:
                        phase_mask = None
            # Mask
            if "Mask" in params:
                try:
                    mask = np.array(params["Mask"]).astype(bool)
                except Exception:
                    mask = None
            # Voxelsize
            if "Voxelsize" in params:
                try:
                    v = np.array(params["Voxelsize"]).astype(float).reshape(-1)
                    if v.size >= 3:
                        vox = (float(v[0]), float(v[1]), float(v[2]))
                except Exception:
                    vox = None
            info = params.get("info", None)
            if info is not None and isinstance(info, h5py.Group):
                img = info.get("img", None)
                a = info if img is None else img
                def _try_read(name: str):
                    if name in a:
                        try:
                            return float(np.array(a[name]).squeeze())
                        except Exception:
                            return None
                    return None
                dReadoutFOV = _try_read("dReadoutFOV")
                dPhaseFOV = _try_read("dPhaseFOV")
                dThickness = _try_read("dThickness")
                if dReadoutFOV is not None and dPhaseFOV is not None and dThickness is not None:
                    fov = (dReadoutFOV, dPhaseFOV, dThickness)
                if "arraysize" in a:
                    try:
                        arr = np.array(a["arraysize"]).reshape(-1)
                        if arr.size >= 3:
                            arraysize = (int(arr[0]), int(arr[1]), int(arr[2]))
                    except Exception:
                        pass
        if central is None:
            for key in ["CentralFeq", "CentralFreq", "centralFeq", "centralFreq"]:
                if key in f:
                    try:
                        central = float(np.array(f[key]).squeeze())
                        break
                    except Exception:
                        continue
        if central is None:
            raise KeyError("Failed to locate Parameters.CentralFeq/CentralFreq in MAT file (h5py)")

    return map_arr, float(central), mask, vox, fov, arraysize, phase_mask


def load_b0_map(mat_path: str) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int]], Optional[np.ndarray]]:
    """Load Map, CentralFreq, Mask, Voxel size (mm), FOV (mm), arraysize, PhaseMask from a MATLAB .mat file."""
    try:
        return _try_load_with_scipy(mat_path)
    except Exception as e1:
        try:
            return _try_load_with_h5py(mat_path)
        except Exception as e2:
            print(f"Failed to load expected variables. Inspecting MAT contents...\n - scipy error: {e1}\n - h5py error: {e2}")
            _inspect_mat_with_scipy(mat_path)
            raise


def plot_central_slice(freq_map_hz: np.ndarray, title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    if freq_map_hz.ndim != 3:
        raise ValueError("Expected a 3D frequency map")
    k = freq_map_hz.shape[2] // 2
    plt.figure(figsize=(6, 5))
    im = plt.imshow(freq_map_hz[:, :, k], cmap="bwr")
    plt.title(f"{title} (k={k})")
    cbar = plt.colorbar(im)
    cbar.set_label("Hz")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def find_subject_mat_files(root: str) -> List[str]:
    mats: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.mat') and 'b0map' in fn.lower():
                mats.append(os.path.join(dirpath, fn))
    mats.sort()
    return mats


def select_slices_with_mask(mask: np.ndarray) -> np.ndarray:
    # returns boolean index of slices along k with any mask voxel
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    nz = mask.shape[2]
    sel = np.zeros(nz, dtype=bool)
    for k in range(nz):
        sel[k] = np.any(mask[:, :, k])
    return sel


def _derive_subject_label(mat_path: str) -> str:
    parts = [p for p in mat_path.split(os.sep) if p]
    # prefer folder names containing 'subject' or 'subejct' (typo) from the end
    for p in reversed(parts):
        low = p.lower()
        if "subject" in low or "subejct" in low:
            return p
    # fallback: first part from end containing digits
    for p in reversed(parts[:-1]):
        if any(ch.isdigit() for ch in p):
            return p
    # last resort: file stem
    stem = os.path.splitext(os.path.basename(mat_path))[0]
    return stem


def process_all_subjects(root: str, sh_order: int = 12, smooth_sigma_vox: float = 1.0, ridge_lambda: float = 0.0) -> None:
    from dgr.physics.b0_fitting import FitInputs, spherical_harmonics_fit_full, gaussian_smooth_masked
    import matplotlib.pyplot as plt

    mat_files = find_subject_mat_files(root)
    if not mat_files:
        print(f"No MAT files found under {root}")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "B0_fitting_outputs")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "fit_log.txt")

    def write_log(line: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass

    write_log(f"=== Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} order={sh_order} sigma={smooth_sigma_vox} ridge={ridge_lambda} root={root} ===")

    for mp in mat_files:
        try:
            b0_map, central_freq, mask, vox, fov, arraysize, phase_mask = load_b0_map(mp)
        except Exception as e:
            msg = f"[SKIP] {mp}: failed to load ({e})"
            print(msg)
            write_log(msg)
            continue

        subject = _derive_subject_label(mp)
        freq_map_hz = b0_map.astype(np.float32) * float(central_freq)

        # fallbacks
        if mask is None or mask.shape != freq_map_hz.shape:
            mask = np.isfinite(freq_map_hz) & (freq_map_hz != 0)
        if phase_mask is None or phase_mask.shape != freq_map_hz.shape:
            phase_mask_present = False
            fit_mask = mask.astype(bool)
            fit_mask_type = "Mask"
        else:
            phase_mask_present = True
            fit_mask = phase_mask.astype(bool)
            fit_mask_type = "PhaseMask"

        # meta log
        meta_info = (
            f"{subject} meta: map_shape={tuple(freq_map_hz.shape)}, central_freq={central_freq:.3f} Hz, "
            f"vox_mm={vox}, fov_mm={fov}, phase_mask_present={phase_mask_present}"
        )
        write_log(meta_info)

        # slice selection by Mask
        selection_mask_type = "Mask"
        sel = select_slices_with_mask(mask)
        num_slices = int(sel.sum())
        sel_indices = np.where(sel)[0]
        sel_info = (
            f"{subject}: selected {num_slices} slices (of {mask.shape[2]}) by {selection_mask_type}; "
            f"selected_k={list(map(int, sel_indices))}"
        )
        print(sel_info)
        write_log(sel_info)
        if num_slices == 0:
            continue

        # restrict to selected slices
        fmap_sel = freq_map_hz[:, :, sel]
        mask_sel = mask[:, :, sel]
        fit_mask_sel = fit_mask[:, :, sel]

        # Fallback voxel size if unavailable
        if vox is None:
            vox = (1.0, 1.0, 1.0)

        # 1) smoothing in PhaseMask region (or Mask if no PhaseMask)
        smooth_in = fit_mask_sel
        smoothed = gaussian_smooth_masked(fmap_sel, smooth_in, sigma_vox=smooth_sigma_vox)

        # 2) train SH on smoothed values within PhaseMask, predict full matrix
        train_inputs = FitInputs(field_map_hz=smoothed, mask=fit_mask_sel, voxel_size_mm=vox, order=sh_order, fov_size_mm=fov)
        try:
            fit_out, pred_all = spherical_harmonics_fit_full(train_inputs, ridge_lambda=ridge_lambda, standardize_design=True)
        except Exception as e:
            msg = f"[FIT-FAIL] {subject}: {e}"
            print(msg)
            write_log(msg)
            continue

        # 3) compose final volume: inside PhaseMask use smoothed; outside PhaseMask use SH prediction
        composite = np.where(fit_mask_sel, smoothed, pred_all)

        # stats logging (inside selection region only)
        def _masked_stats(a: np.ndarray, m: np.ndarray) -> Dict[str, float]:
            vals = a[m]
            if vals.size == 0:
                return {"mean": np.nan, "std": np.nan}
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        stats_orig = _masked_stats(fmap_sel, fit_mask_sel)
        stats_smooth = _masked_stats(smoothed, fit_mask_sel)
        stats_comp = _masked_stats(composite, fit_mask_sel)
        write_log(f"  stats(orig in fit): mean={stats_orig['mean']:.3f}, std={stats_orig['std']:.3f} Hz")
        write_log(f"  stats(smooth in fit): mean={stats_smooth['mean']:.3f}, std={stats_smooth['std']:.3f} Hz")
        write_log(f"  stats(composite in fit): mean={stats_comp['mean']:.3f}, std={stats_comp['std']:.3f} Hz")

        # plot 3-panel at middle selected slice index
        sel_list = list(map(int, sel_indices))
        k_full = int(sel_list[len(sel_list) // 2])
        k_sel = int(len(sel_list) // 2)
        orig_slice = freq_map_hz[:, :, k_full].astype(np.float32)
        smooth_slice = smoothed[:, :, k_sel].astype(np.float32)
        comp_slice = composite[:, :, k_sel].astype(np.float32)

        # Robust color scaling from the fitting mask region to avoid outlier dominance
        fit_mask_slice_full = fit_mask[:, :, k_full]
        fit_mask_slice_sel = fit_mask_sel[:, :, k_sel]
        vals_masked = []
        if np.any(fit_mask_slice_full):
            vals_masked.append(orig_slice[fit_mask_slice_full])
        if np.any(fit_mask_slice_sel):
            vals_masked.append(smooth_slice[fit_mask_slice_sel])
            vals_masked.append(comp_slice[fit_mask_slice_sel])
        if vals_masked:
            vcat = np.concatenate([v.ravel() for v in vals_masked if v.size > 0])
            vcat = vcat[np.isfinite(vcat)]
            if vcat.size > 0:
                vmin = float(np.percentile(vcat, 2))
                vmax = float(np.percentile(vcat, 98))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    vmin, vmax = float(np.min(vcat)), float(np.max(vcat))
            else:
                allcat = np.concatenate([orig_slice.ravel(), smooth_slice.ravel(), comp_slice.ravel()])
                vmin, vmax = float(np.nanmin(allcat)), float(np.nanmax(allcat))
        else:
            allcat = np.concatenate([orig_slice.ravel(), smooth_slice.ravel(), comp_slice.ravel()])
            vmin, vmax = float(np.nanmin(allcat)), float(np.nanmax(allcat))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.nanmin(orig_slice)), float(np.nanmax(orig_slice))

        import matplotlib.pyplot as plt
        out_path = os.path.join(out_dir, f"{subject}_k{k_full}_orig_smooth_comp.png")
        plt.figure(figsize=(15, 4))
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(orig_slice, cmap="jet", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Original (k={k_full})")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Hz")

        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(smooth_slice, cmap="jet", vmin=vmin, vmax=vmax)
        ax2.set_title("Smoothed (in mask)")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Hz")

        ax3 = plt.subplot(1, 3, 3)
        im3 = ax3.imshow(comp_slice, cmap="jet", vmin=vmin, vmax=vmax)
        ax3.set_title("Final (smooth inside, SH outside)")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Hz")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  saved: {out_path}")
        write_log(f"  saved: {out_path} (k={k_full})")


def main() -> None:
    root = "/path/to/dgr_data/B0_folder"
    process_all_subjects(root, sh_order=12, smooth_sigma_vox=1.0, ridge_lambda=0.0)


if __name__ == "__main__":
    main()


