import os
import glob
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import pydicom
except ImportError as e:
    raise SystemExit("The 'pydicom' package is required. Install with: pip install pydicom") from e


def _find_dicom_files(folder_path: str) -> List[str]:
    """Return a sorted list of DICOM file paths inside folder_path.

    Looks for files with .dcm/.DCM extensions. If none are found, falls back to all files.
    """
    pattern_lower = os.path.join(folder_path, "*.dcm")
    pattern_upper = os.path.join(folder_path, "*.DCM")
    files = glob.glob(pattern_lower) + glob.glob(pattern_upper)
    if not files:
        # Fallback: include all files (some DICOMs may not have .dcm extension)
        files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, f))
        ]
    return sorted(files)


def _read_instance_number(ds: "pydicom.dataset.FileDataset") -> int:
    """Safely read InstanceNumber; default to 0 if missing or invalid."""
    try:
        return int(getattr(ds, "InstanceNumber", 0) or 0)
    except Exception:
        return 0


def _read_slice_position(ds: "pydicom.dataset.FileDataset") -> float:
    """Try to read z-position for sorting if InstanceNumber is unavailable."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    try:
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
    except Exception:
        pass
    try:
        # SliceLocation is less reliable but can be used as fallback
        return float(getattr(ds, "SliceLocation", 0.0) or 0.0)
    except Exception:
        return 0.0


def _apply_rescale(image: np.ndarray, ds: "pydicom.dataset.FileDataset") -> np.ndarray:
    """Apply DICOM rescale slope/intercept if present."""
    slope = getattr(ds, "RescaleSlope", 1) or 1
    intercept = getattr(ds, "RescaleIntercept", 0) or 0
    return (image.astype(np.float32) * float(slope)) + float(intercept)


def load_dicom_stack(folder_path: str) -> Tuple[np.ndarray, List[str], "pydicom.dataset.FileDataset"]:
    """Load a series of DICOM files from a folder and stack into (H, W, S) array.

    - Sorts by InstanceNumber when available; otherwise by spatial position or filename.
    - Applies RescaleSlope/RescaleIntercept if present.

    Returns:
        stack_array: np.ndarray with shape (H, W, S)
        sorted_paths: list of file paths in the order used
        ref_dataset: the dataset of the first slice (for header writing)
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    file_paths = _find_dicom_files(folder_path)
    if not file_paths:
        raise FileNotFoundError(f"No files found in folder: {folder_path}")

    # Read datasets to determine sort order
    datasets: List["pydicom.dataset.FileDataset"] = []
    valid_paths: List[str] = []
    for path in file_paths:
        try:
            ds = pydicom.dcmread(path)
            # Accessing pixel_array later; ensure PhotometricInterpretation loaded
            _ = getattr(ds, "Rows", None)
            datasets.append(ds)
            valid_paths.append(path)
        except Exception:
            # Skip files that can't be read as DICOM
            continue

    if not datasets:
        raise RuntimeError(f"No readable DICOM datasets found in folder: {folder_path}")

    # Sort by preferred keys
    instances = [_read_instance_number(ds) for ds in datasets]
    if any(instances):
        sorted_indices = sorted(range(len(datasets)), key=lambda i: instances[i])
    else:
        positions = [_read_slice_position(ds) for ds in datasets]
        if any(positions):
            sorted_indices = sorted(range(len(datasets)), key=lambda i: positions[i])
        else:
            sorted_indices = sorted(range(len(datasets)), key=lambda i: valid_paths[i])

    sorted_datasets = [datasets[i] for i in sorted_indices]
    sorted_paths = [valid_paths[i] for i in sorted_indices]

    # Extract images, ensuring consistent dimensions
    first_ds = sorted_datasets[0]
    rows = int(getattr(first_ds, "Rows", 0) or 0)
    cols = int(getattr(first_ds, "Columns", 0) or 0)
    if rows <= 0 or cols <= 0:
        raise RuntimeError("Invalid DICOM images with missing Rows/Columns metadata.")

    images: List[np.ndarray] = []
    for ds in sorted_datasets:
        arr = ds.pixel_array  # type: ignore[attr-defined]
        if arr.ndim != 2:
            raise RuntimeError("Expected 2D slices; found non-2D pixel data.")
        if arr.shape != (rows, cols):
            raise RuntimeError(
                f"Inconsistent slice size: expected {(rows, cols)}, got {arr.shape}"
            )
        arr = _apply_rescale(arr, ds)
        images.append(arr)

    # Stack into (H, W, S)
    stack_array = np.stack(images, axis=-1)
    return stack_array, sorted_paths, first_ds


def save_header_to_txt(ds: "pydicom.dataset.FileDataset", output_txt_path: str) -> None:
    """Write DICOM header (without PixelData) to a txt file."""
    ds_copy = ds.copy()
    if "PixelData" in ds_copy:
        del ds_copy.PixelData
    text = ds_copy.__str__()
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_mid_slice(volume_hws: np.ndarray, title: str) -> None:
    """Plot the middle slice along the S dimension for a (H, W, S) volume."""
    if volume_hws.ndim != 3:
        raise ValueError("Expected 3D volume with shape (H, W, S)")
    mid_index = volume_hws.shape[-1] // 2
    slice_2d = volume_hws[:, :, mid_index]
    plt.imshow(slice_2d, cmap="gray")
    plt.title(title)
    plt.axis("off")


def main() -> None:
    # Input folders provided by the user
    diffusion_dir = "/path/to/dgr_data/fastmri_prostate_v3/subjects/sub-001/DICOMS/AX_DIFFUSION_ADC"
    t2_dir = "/path/to/dgr_data/fastmri_prostate_v3/subjects/sub-001/DICOMS/AX_T2"

    # Load stacks
    print("Loading diffusion DICOM series...")
    diffusion_stack, diffusion_paths, diffusion_ref_ds = load_dicom_stack(diffusion_dir)
    print(f"Loaded {len(diffusion_paths)} diffusion slices from: {diffusion_dir}")

    print("Loading T2 DICOM series...")
    t2_stack, t2_paths, t2_ref_ds = load_dicom_stack(t2_dir)
    print(f"Loaded {len(t2_paths)} T2 slices from: {t2_dir}")

    # Print matrix sizes in (W, H, S)
    diffusion_h, diffusion_w, diffusion_s = diffusion_stack.shape
    t2_h, t2_w, t2_s = t2_stack.shape
    print(
        f"Diffusion matrix size (W * H * S): {diffusion_w} * {diffusion_h} * {diffusion_s}"
    )
    print(f"T2 matrix size (W * H * S): {t2_w} * {t2_h} * {t2_s}")

    # Save one header per series (first slice is representative)
    diff_header_txt = os.path.join(os.getcwd(), "diffusion_header.txt")
    t2_header_txt = os.path.join(os.getcwd(), "t2_header.txt")
    save_header_to_txt(diffusion_ref_ds, diff_header_txt)
    save_header_to_txt(t2_ref_ds, t2_header_txt)
    print(f"Saved diffusion header to: {diff_header_txt}")
    print(f"Saved T2 header to: {t2_header_txt}")

    # Plot one slice from each (middle slice)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_mid_slice(diffusion_stack, "Diffusion (mid slice)")
    plt.subplot(1, 2, 2)
    plot_mid_slice(t2_stack, "T2 (mid slice)")
    plt.tight_layout()
    # Save figure for headless environments
    fig_path = os.path.join(os.getcwd(), "diffusion_t2_preview.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved preview figure to: {fig_path}")
    try:
        plt.show()
    except Exception:
        # In case of headless backend, we already saved the figure
        pass


def get_series_volumes_and_headers(diffusion_dir: str, t2_dir: str, out_dir: str):
    """Convenience API to be imported by registration script.

    Returns a tuple: (diffusion_stack, t2_stack, diffusion_header_path, t2_header_path)
    and writes the header text files into out_dir.
    """
    diffusion_stack, _, diffusion_ref_ds = load_dicom_stack(diffusion_dir)
    t2_stack, _, t2_ref_ds = load_dicom_stack(t2_dir)

    os.makedirs(out_dir, exist_ok=True)
    diff_header_txt = os.path.join(out_dir, "diffusion_header.txt")
    t2_header_txt = os.path.join(out_dir, "t2_header.txt")
    save_header_to_txt(diffusion_ref_ds, diff_header_txt)
    save_header_to_txt(t2_ref_ds, t2_header_txt)
    return diffusion_stack, t2_stack, diff_header_txt, t2_header_txt


if __name__ == "__main__":
    main()


