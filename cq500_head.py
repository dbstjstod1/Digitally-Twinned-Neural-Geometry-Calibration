"""Read CQ500 DICOM directly and sample an explicitly positioned numerical head.

No cache, source file, training preprocessing, motion augmentation, or GPU is
used. The first held-out case is selected from the existing series index before
image inspection, using the local project's documented sequential split.

Coordinates are DICOM LPS millimetres (x left, y posterior, z superior).
Arrays are z/y/x. Native HU remain untouched in ``CQ500Head.native_hu``;
attenuation uses ``max(HU, -1000) / 1000 + 1`` times ``mu_water`` in 1/mm,
without an upper HU clip. Linear interpolation samples that attenuation field.

Gantry tilt matters: the third column of the native affine is measured from
ImagePositionPatient, not inferred from the slice normal or SliceThickness.
This preserves the sheared sampling lattice in tilted axial CT. GDCM is used
only to decode HU values; its orthogonalized 3-D image geometry is not used.
DICOM coordinate convention: PS3.3 C.7.6.2 Image Plane Module,
https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class CQ500Head:
    native_hu: np.ndarray
    native_index_xyz_to_lps_mm: np.ndarray
    metadata: dict

    @property
    def centre_lps_mm(self) -> np.ndarray:
        index = (np.asarray(self.native_hu.shape[::-1], dtype=float) - 1.0) / 2.0
        return self.native_index_xyz_to_lps_mm[:3, :3] @ index + self.native_index_xyz_to_lps_mm[:3, 3]


def select_held_out_series(data_root: str | Path, test_index: int = 0) -> tuple[dict, dict]:
    """Read the existing index; never scan the dataset or write an index/cache.

    Matches ``fm3d/dataset_cq500.py``: thickness <= .7 mm, >=64 slices,
    slice count within 50% of the thin-series median, most slices per patient,
    then numeric patient ordering and 150 train / 50 validation / remainder
    test. Stable input order breaks equal-length series ties.
    """
    root = Path(data_root).resolve()
    index_path = root / "_cq500_index.json"
    index_bytes = index_path.read_bytes()
    records = json.loads(index_bytes)
    thin = [r for r in records if r["thickness_mm"] <= .7 and r["n_slices"] >= 64]
    if not thin:
        raise ValueError("Existing CQ500 index contains no eligible thin series")
    median = float(np.median([r["n_slices"] for r in thin]))
    kept = [r for r in thin if abs(r["n_slices"] / median - 1.0) <= .5]
    patients = {}
    for record in kept:
        patient = record["patient"]
        if patient not in patients or record["n_slices"] > patients[patient]["n_slices"]:
            patients[patient] = record
    selected = [patients[p] for p in sorted(patients)]
    test = selected[200:]
    if not isinstance(test_index, (int, np.integer)) or not 0 <= test_index < len(test):
        raise ValueError(f"test_index must be in [0, {len(test) - 1}]")
    record = dict(test[test_index])
    relative = Path(record["path"])
    if relative.is_absolute():
        series_path = relative
    elif "CQ500" in relative.parts:
        series_path = root.joinpath(*relative.parts[relative.parts.index("CQ500") + 1:])
    else:
        series_path = root / relative
    if not series_path.is_dir():
        raise FileNotFoundError(series_path)
    record["resolved_path"] = str(series_path)
    selection = {
        "index_path": str(index_path),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "selection_rule": "thickness<=0.7mm; slices>=64; count within 50% of median; most slices per patient; numeric patient order",
        "thin_series_count_median": median,
        "selected_patients": len(selected),
        "split_counts": {"train": 150, "validation": 50, "test": len(test)},
        "test_index": int(test_index),
        "chosen_before_image_inspection": True,
    }
    return record, selection


def read_cq500_head(data_root: str | Path, test_index: int = 0, *, num_threads: int = 4) -> CQ500Head:
    """Decode a selected native HU series and verify its physical lattice.

    Header residuals above .05 mm, variable orientation, missing slices, and
    nonuniform spacing fail explicitly instead of assigning incorrect geometry.
    Sub-.05-mm decimal rounding in DICOM positions is represented by the affine
    between the first and last image positions and reported in metadata.
    """
    import pydicom
    import SimpleITK as sitk

    record, selection = select_held_out_series(data_root, test_index)
    names = tuple(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(record["resolved_path"], record["series"]))
    if len(names) != record["n_slices"] or len(names) < 2:
        raise ValueError("DICOM slice count differs from the selected index")
    tags = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
            "RescaleSlope", "RescaleIntercept", "Rows", "Columns"]
    headers = [pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags) for path in names]
    positions = np.array([h.ImagePositionPatient for h in headers], dtype=np.float64)
    orientations = np.array([h.ImageOrientationPatient for h in headers], dtype=np.float64)
    pixel_spacings = np.array([h.PixelSpacing for h in headers], dtype=np.float64)
    if not np.allclose(orientations, orientations[0], atol=1e-6, rtol=0):
        raise ValueError("Variable DICOM image orientation is unsupported")
    if not np.allclose(pixel_spacings, pixel_spacings[0], atol=1e-6, rtol=0):
        raise ValueError("Variable DICOM pixel spacing is unsupported")
    if len({(h.Rows, h.Columns) for h in headers}) != 1:
        raise ValueError("DICOM image dimensions vary within the series")
    x_axis, y_axis = orientations[0, :3], orientations[0, 3:]
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    if abs(np.dot(x_axis, y_axis)) > 1e-6:
        raise ValueError("DICOM in-plane axes are not orthogonal")
    normal = np.cross(x_axis, y_axis)
    normal_positions = positions @ normal
    if not np.all(np.diff(normal_positions) > 0):
        raise ValueError("DICOM slices are not strictly ordered along the slice normal")
    slice_step = (positions[-1] - positions[0]) / (len(names) - 1)
    predicted = positions[0] + np.arange(len(names))[:, None] * slice_step
    position_error = np.linalg.norm(positions - predicted, axis=1)
    step_error = np.linalg.norm(np.diff(positions, axis=0) - slice_step, axis=1)
    if max(position_error.max(), step_error.max()) > .05:
        raise ValueError("Irregular/missing DICOM slices exceed the .05 mm affine tolerance")
    affine = np.eye(4, dtype=np.float64)
    # DICOM PixelSpacing is row then column; ImageOrientationPatient gives
    # the direction of increasing column, then increasing row index.
    affine[:3, 0] = x_axis * pixel_spacings[0, 1]
    affine[:3, 1] = y_axis * pixel_spacings[0, 0]
    affine[:3, 2] = slice_step
    affine[:3, 3] = positions[0]
    if np.linalg.det(affine[:3, :3]) <= 0:
        raise ValueError("Invalid or reflected native DICOM lattice")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    reader.SetNumberOfThreads(max(1, int(num_threads)))
    # GDCM can warn about millimetre-decimal position rounding. Geometry is
    # validated against all DICOM headers above instead of relying on its grid.
    image = reader.Execute()
    native_hu = sitk.GetArrayFromImage(image)
    if native_hu.shape != (len(names), headers[0].Rows, headers[0].Columns):
        raise ValueError("Decoded array does not match the indexed DICOM dimensions")
    if not np.isfinite(native_hu).all():
        raise ValueError("Non-finite native HU")
    shape_xyz = np.array(native_hu.shape[::-1], dtype=float)
    corners = np.array([[x, y, z] for x in (-.5, shape_xyz[0] - .5)
                        for y in (-.5, shape_xyz[1] - .5) for z in (-.5, shape_xyz[2] - .5)])
    physical_corners = corners @ affine[:3, :3].T + affine[:3, 3]
    metadata = {
        "dataset": "CQ500", "patient": int(record["patient"]), "series_record": record,
        "selection": selection, "native_array_order": "zyx", "physical_axes": "DICOM LPS: left, posterior, superior",
        "native_shape_zyx": list(native_hu.shape),
        "native_index_xyz_to_lps_mm": affine.tolist(),
        "dicom_pixel_spacing_row_column_mm": pixel_spacings[0].tolist(),
        "dicom_slice_thickness_mm": float(headers[0].SliceThickness),
        "slice_step_lps_mm": slice_step.tolist(),
        "slice_normal_spacing_mm": float(np.dot(slice_step, normal)),
        "slice_position_affine_max_error_mm": float(position_error.max()),
        "slice_step_max_error_mm": float(step_error.max()),
        "native_lps_bounds_mm": [physical_corners.min(axis=0).tolist(), physical_corners.max(axis=0).tolist()],
        "native_hu_range": [float(native_hu.min()), float(native_hu.max())],
        "rescale_slope_values": sorted({float(getattr(h, "RescaleSlope", 1.0)) for h in headers}),
        "rescale_intercept_values": sorted({float(getattr(h, "RescaleIntercept", 0.0)) for h in headers}),
        "source_header_order_sha256": hashlib.sha256("\n".join(Path(p).name for p in names).encode()).hexdigest(),
        "decoder": "SimpleITK/GDCM HU pixels only; geometry independently derived from DICOM headers",
        "simpleitk_version": sitk.Version_VersionString(),
        "pydicom_version": pydicom.__version__,
    }
    head = CQ500Head(native_hu, affine, metadata)
    metadata["default_isocenter_lps_mm"] = head.centre_lps_mm.tolist()
    return head


def resample_cq500_head(
    head: CQ500Head,
    shape_zyx: Sequence[int],
    voxel_mm: float,
    *,
    isocenter_lps_mm: Sequence[float] | None = None,
    center_shift_xyz_mm: Sequence[float] = (0., 0., 0.),
    mu_water: float = .02,
) -> tuple[np.ndarray, dict]:
    """Return float32 attenuation on an isocentre-centred LPS-aligned grid.

    ``isocenter_lps_mm`` fixes which source anatomy lies at simulation xyz=0.
    ``center_shift_xyz_mm`` translates the head in simulation axes: a positive
    z shift moves anatomy superiorly in the simulated volume. It is equivalent
    to subtracting that shift from the chosen LPS isocentre. Both the effective
    isocentre and target bounds are saved, so placement is fully reproducible.
    Target samples are ``(index-(size-1)/2)*voxel_mm`` in each simulation axis.
    Unknown source coverage and out-of-grid source material are reported.
    """
    from scipy.ndimage import affine_transform

    raw_shape = np.asarray(shape_zyx)
    if (raw_shape.shape != (3,) or np.any(raw_shape <= 0)
            or np.any(raw_shape != raw_shape.astype(int))):
        raise ValueError("shape_zyx must contain three positive integers")
    shape = raw_shape.astype(int)
    if not np.isfinite(voxel_mm) or voxel_mm <= 0 or not np.isfinite(mu_water) or mu_water <= 0:
        raise ValueError("voxel_mm and mu_water must be finite and positive")
    iso = head.centre_lps_mm if isocenter_lps_mm is None else np.asarray(isocenter_lps_mm, dtype=float)
    shift = np.asarray(center_shift_xyz_mm, dtype=float)
    if iso.shape != (3,) or shift.shape != (3,) or not np.isfinite(iso).all() or not np.isfinite(shift).all():
        raise ValueError("isocentre and shift must be finite xyz triplets")
    iso = iso - shift
    first_lps = iso - (shape[::-1] - 1) * float(voxel_mm) / 2.0
    inverse = np.linalg.inv(head.native_index_xyz_to_lps_mm[:3, :3])
    # scipy uses zyx index vectors on both sides of the mapping.
    reversal = np.eye(3)[::-1]
    matrix = reversal @ inverse @ (np.eye(3) * voxel_mm) @ reversal
    offset = reversal @ inverse @ (first_lps - head.native_index_xyz_to_lps_mm[:3, 3])
    attenuation = np.maximum(head.native_hu, -1000.0)
    attenuation += 1000.0
    attenuation *= np.float32(mu_water / 1000.0)
    mu = affine_transform(attenuation, matrix, offset=offset, output_shape=tuple(shape),
                          output=np.float32, order=1, mode="constant", cval=0.0, prefilter=False)
    bounds = np.asarray(head.metadata["native_lps_bounds_mm"]) - iso
    target_bounds = np.array([-shape[::-1], shape[::-1]]) * float(voxel_mm) / 2.0
    metadata = dict(head.metadata)
    metadata.update({
        "shape_zyx": shape.tolist(), "voxel_mm": float(voxel_mm), "mu_water_per_mm": float(mu_water),
        "isocenter_lps_mm": iso.tolist(), "requested_center_shift_xyz_mm": shift.tolist(),
        "target_first_voxel_lps_mm": first_lps.tolist(),
        "target_index_zyx_to_native_index_zyx_matrix": matrix.tolist(),
        "target_index_zyx_to_native_index_zyx_offset": offset.tolist(),
        "target_bounds_simulation_xyz_mm": target_bounds.tolist(),
        "native_bounds_simulation_xyz_mm": bounds.tolist(),
        "entire_native_lattice_inside_target_box": bool(np.all(bounds[0] >= target_bounds[0]) and np.all(bounds[1] <= target_bounds[1])),
        "hu_to_attenuation": "mu_water * (max(native_HU, -1000) + 1000) / 1000; no upper clip",
        "interpolation": "linear interpolation of native attenuation on DICOM sheared lattice",
        "outside_source": "zero attenuation (air); unavailable anatomy is not invented",
        "mu_range_per_mm": [float(mu.min()), float(mu.max())],
        "native_z_coverage_warning": "Only anatomy captured in the selected DICOM series is available; air padding does not extend anatomical coverage.",
    })
    return mu, metadata


def load_cq500_head(data_root: str | Path, shape_zyx: Sequence[int], voxel_mm: float,
                   *, test_index: int = 0, **resample_kwargs) -> tuple[np.ndarray, dict]:
    """Convenience wrapper; reuse ``read_cq500_head`` for several target grids."""
    return resample_cq500_head(read_cq500_head(data_root, test_index), shape_zyx, voxel_mm, **resample_kwargs)


def source_coverage_mask(resampled_metadata: dict) -> np.ndarray:
    """True where interpolation uses acquired DICOM voxel centres exclusively.

    Padding is excluded from image-quality metrics. The mask describes source
    CT coverage, independently of CBCT detector visibility or completeness.
    Its 2-D scratch arrays keep memory bounded even for the fine forward grid.
    """
    shape = tuple(resampled_metadata["shape_zyx"])
    native_shape = np.asarray(resampled_metadata["native_shape_zyx"])
    matrix = np.asarray(resampled_metadata["target_index_zyx_to_native_index_zyx_matrix"])
    offset = np.asarray(resampled_metadata["target_index_zyx_to_native_index_zyx_offset"])
    y = np.arange(shape[1], dtype=np.float64)[:, None]
    x = np.arange(shape[2], dtype=np.float64)[None, :]
    mask = np.ones(shape, dtype=bool)
    for axis in range(3):
        plane = matrix[axis, 1] * y + matrix[axis, 2] * x + offset[axis]
        for iz in range(shape[0]):
            coordinate = plane + matrix[axis, 0] * iz
            mask[iz] &= (coordinate >= 0.0) & (coordinate <= native_shape[axis] - 1)
    return mask
