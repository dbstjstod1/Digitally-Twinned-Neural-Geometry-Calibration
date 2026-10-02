"""One segmented, z-dominant validation trajectory around each nominal orbit.

Wobble and linear mountain occupy separate scan intervals. The original mode
translates the complete rig. The optional gantry-rotation mode moves the rig
about the isocenter, preserving source radius, detector distances and K.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from .detector import ExtendedDetector
from .sinespin import build_icono_orbit


DEFAULT_CONFIG = dict(
    perturbation_model="translation",
    n_views=546,
    detector_bin=2,
    scan_angle_deg=220.0,
    sod_mm=750.0,
    sdd_mm=1200.0,
    tilt_amplitude_deg=10.0,
    wobble_peak_to_peak_mm=5.0,
    mountain_height_mm=2.5,
    xy_max_mm=0.1,
    wobble_cycles=1,
    wobble_interval=[0.10, 0.45],
    mountain_interval=[0.55, 0.90],
    padding_vu={"circular": [240, 180], "sinespin": [304, 196]},
)
SCENARIOS = ("nominal", "validation")


def resolve_validation_config(config=None):
    """Resolve legacy settings unchanged; pivot fraction belongs only to pivot mode."""
    cfg = deepcopy(DEFAULT_CONFIG)
    supplied = {} if config is None else deepcopy(config)
    if supplied.get("perturbation_model") == "gantry_pivot_rotation":
        cfg["rotation_pivot_fraction"] = 0.5
    unknown = set(supplied) - set(cfg)
    if unknown:
        raise ValueError(
            "Unknown validation configuration keys: " + str(sorted(unknown))
        )
    cfg.update(supplied)
    return cfg


def _segment_indices(interval, views, name):
    endpoints = np.asarray(interval, dtype=np.float64)
    if (
        endpoints.shape != (2,)
        or not np.isfinite(endpoints).all()
        or not 0 <= endpoints[0] < endpoints[1] <= 1
    ):
        raise ValueError(
            name + " interval must be increasing finite fractions in [0,1]"
        )
    a, b = np.rint(endpoints * (views - 1)).astype(int)
    if b - a < 2:
        raise ValueError(name + " interval must contain at least three acquired views")
    return int(a), int(b)


def segmented_profiles(views, config):
    """Return localized xyz shifts with signed wobble around the nominal.

    Interval endpoints and mountain apex are actual acquired views. Wobble is
    a sine with a smooth zero-endpoint envelope, antisymmetric about the
    interval midpoint. Its sampled peak-to-peak is normalized to its setting. Mountain has two straight lines,
    including the sampled apex. Outside the two intervals all shifts are zero.
    """
    wobble_span = float(config["wobble_peak_to_peak_mm"])
    height, xy = float(config["mountain_height_mm"]), float(config["xy_max_mm"])
    cycles = config["wobble_cycles"]
    if (
        not np.isfinite([wobble_span, height, xy]).all()
        or min(wobble_span, height, xy) < 0
    ):
        raise ValueError("Require finite nonnegative z range and xy bound")
    if not np.isfinite(cycles) or int(cycles) != cycles or cycles < 1:
        raise ValueError("Require a positive integer wobble cycle count")
    a, b = _segment_indices(config["wobble_interval"], views, "Wobble")
    c, d = _segment_indices(config["mountain_interval"], views, "Mountain")
    wi, mi = config["wobble_interval"], config["mountain_interval"]
    if not (wi[1] <= mi[0] or mi[1] <= wi[0]) or not (b <= c or d <= a):
        raise ValueError("Wobble and mountain intervals must not overlap")
    if b - a < 4 * int(cycles):
        raise ValueError("Wobble needs at least four acquired steps per cycle")
    shift = np.zeros((views, 3), dtype=np.float64)
    wobble, mountain = np.zeros(views), np.zeros(views)
    segment_id = np.zeros(views, dtype=np.int64)
    u = np.linspace(0.0, 1.0, b - a + 1)
    wave = np.sin(2 * np.pi * cycles * u) * np.sin(np.pi * u) ** 2
    # Enforce antisymmetry at float precision without shifting zero endpoints.
    wave = 0.5 * (wave - wave[::-1])
    wave[[0, -1]] = 0.0
    wobble[a : b + 1] = 0.5 * wobble_span * wave / np.max(np.abs(wave))
    apex = (c + d) // 2
    mountain[c : apex + 1] = np.linspace(0.0, height, apex - c + 1)
    mountain[apex : d + 1] = np.linspace(height, 0.0, d - apex + 1)
    for lo, hi, label in ((a, b, 1), (c, d, 2)):
        u = np.linspace(0.0, 1.0, hi - lo + 1)
        envelope = np.sin(np.pi * u) ** 2
        shift[lo : hi + 1, 0] = xy * envelope * np.sin(2 * np.pi * u)
        shift[lo : hi + 1, 1] = xy * envelope * np.sin(4 * np.pi * u + 0.4)
        shift[[lo, hi], :2] = 0.0
        segment_id[lo : hi + 1] = label
    shift[:, 2] = wobble + mountain
    segments = [
        dict(kind="wobble", start_view=a, end_view=b),
        dict(kind="mountain", start_view=c, end_view=d, apex_view=apex),
    ]
    return dict(
        progress=np.linspace(0.0, 1.0, views),
        translation_xyz_mm=shift,
        wobble_z_mm=wobble,
        mountain_z_mm=mountain,
        segment_id=segment_id,
        segments=segments,
    )


def build_validation_family(kind, config=None):
    """Return nominal and perturbed geometries with localized motion profiles.

    ``translation`` retains the original equal source/detector displacement.
    ``gantry_rotation`` rotates source, detector and detector axes together
    about the view's horizontal detector-column axis through the isocenter.
    Its angle is chosen to match the same exact sampled source-z waveform.
    Source radius and K remain fixed, so xy motion follows spherical geometry
    and is not controlled by the translation-only ``xy_max_mm`` setting.
    ``gantry_pivot_rotation`` rotates the complete rig about a parallel axis
    at q = rotation_pivot_fraction * nominal_source. This is a gantry-frame
    pivot, not a globally fixed world point. Source-z keeps the same waveform,
    but the detector no longer tracks isocenter. K/SDD remain constant; the
    origin-relative effective translation is induced by the pivot rotation.
    """
    cfg = resolve_validation_config(config)
    if cfg["perturbation_model"] not in (
        "translation",
        "gantry_rotation",
        "gantry_pivot_rotation",
    ):
        raise ValueError(
            "Perturbation model must be translation, gantry_rotation or gantry_pivot_rotation"
        )
    pivot_fraction = float(cfg.get("rotation_pivot_fraction", 0.0))
    if not np.isfinite(pivot_fraction) or not 0.0 <= pivot_fraction < 1.0:
        raise ValueError("Rotation pivot fraction must be finite and in [0,1)")
    if kind not in ("circular", "sinespin"):
        raise ValueError("Nominal kind must be circular or sinespin")
    if (
        not np.isfinite(cfg["n_views"])
        or int(cfg["n_views"]) != cfg["n_views"]
        or cfg["n_views"] < 3
    ):
        raise ValueError("Require at least three integer views")
    finite = [
        cfg[k] for k in ("scan_angle_deg", "sod_mm", "sdd_mm", "tilt_amplitude_deg")
    ]
    if not np.isfinite(finite).all():
        raise ValueError("Nominal geometry settings must be finite")
    nominal = build_icono_orbit(
        kind,
        n_views=cfg["n_views"],
        detector_bin=cfg["detector_bin"],
        scan_angle_deg=cfg["scan_angle_deg"],
        sod_mm=cfg["sod_mm"],
        sdd_mm=cfg["sdd_mm"],
        tilt_amplitude_deg=0.0 if kind == "circular" else cfg["tilt_amplitude_deg"],
    )
    profiles = segmented_profiles(nominal.n_views, cfg)
    shift = profiles["translation_xyz_mm"]
    if cfg["perturbation_model"] == "translation":
        validation = replace(
            nominal,
            kind=kind + "_segmented_validation",
            source_positions=nominal.source_positions + shift,
            module_centers=nominal.module_centers + shift,
        )
        # In translation mode, nominal tilt/sod describe the commanded orbit;
        # explicit physical source/detector arrays describe the acquisition.
    else:
        z_shift = shift[:, 2].copy()
        nominal_tilt = np.deg2rad(nominal.tilt_deg)
        lever_arm = (1.0 - pivot_fraction) * nominal.sod_mm
        target_sine = np.sin(nominal_tilt) + z_shift / lever_arm
        if np.any(np.abs(target_sine) >= 1.0):
            raise ValueError(
                "Requested source z reaches or exceeds the fixed-radius orbit pole"
            )
        tilt_change = np.arcsin(target_sine) - nominal_tilt
        # Exact identity outside the active intervals, including their ends.
        tilt_change[z_shift == 0.0] = 0.0
        axis = nominal.col_vectors.copy()
        angle = -tilt_change
        # Active, right-handed physical-xyz Rodrigues rotation. With this
        # horizontal column axis, negative angle increases the source tilt.
        skew = np.zeros((nominal.n_views, 3, 3), dtype=np.float64)
        skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
        cosine, sine = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
        rotation = (
            cosine * np.eye(3)[None]
            + (1.0 - cosine) * axis[:, :, None] * axis[:, None, :]
            + sine * skew
        )
        rotate = lambda values: np.einsum("vij,vj->vi", rotation, values)
        pivot = pivot_fraction * nominal.source_positions
        rig_translation = pivot - rotate(pivot)
        source = rotate(nominal.source_positions)
        detector = rotate(nominal.module_centers)
        is_pivot = cfg["perturbation_model"] == "gantry_pivot_rotation"
        if is_pivot:
            # Rotate about q using lever vectors, retaining the source-z prescription.
            source = rotate(nominal.source_positions - pivot) + pivot
            detector = rotate(nominal.module_centers - pivot) + pivot
            zero = z_shift == 0.0
            source[zero] = nominal.source_positions[zero]
            detector[zero] = nominal.module_centers[zero]
        validation = replace(
            nominal,
            kind=kind
            + (
                "_segmented_pivot_validation"
                if is_pivot
                else "_segmented_orbit_validation"
            ),
            source_positions=source,
            module_centers=detector,
            row_vectors=rotate(nominal.row_vectors),
            col_vectors=rotate(nominal.col_vectors),
            tilt_deg=nominal.tilt_deg + np.rad2deg(tilt_change),
        )
        source_delta = validation.source_positions - nominal.source_positions
        profiles.update(
            perturbation_model=cfg["perturbation_model"],
            # Retain this legacy preview key as source displacement only.
            # No independent rig translation is present in this mode.
            translation_xyz_mm=source_delta,
            source_displacement_xyz_mm=source_delta,
            detector_displacement_xyz_mm=validation.module_centers
            - nominal.module_centers,
            rig_translation_xyz_mm=(
                rig_translation if is_pivot else np.zeros_like(source_delta)
            ),
            requested_source_z_displacement_mm=z_shift,
            rotation_axis_xyz=axis,
            rotation_angle_deg=np.rad2deg(angle),
            rotation_matrix_xyz=rotation,
            rotation_pivot_xyz_mm=pivot if is_pivot else np.zeros(3),
            nominal_tilt_deg=nominal.tilt_deg.copy(),
            actual_tilt_deg=validation.tilt_deg.copy(),
            xy_policy=(
                "Geometry-induced xy displacement at fixed source radius; "
                "xy_max_mm and ad hoc xy translation are not applied"
            ),
        )
        if is_pivot:
            profiles.update(
                rotation_pivot_fraction=pivot_fraction,
                xy_policy="Geometry-induced xy motion about the gantry-frame offset pivot; no independently prescribed translation",
                independent_translation_prescribed=False,
                effective_translation_note="Nonzero translation in the nominal-origin 9DoF chart is induced by rotation about q; it is not an additional independent GT perturbation",
            )
    return {
        name: ExtendedDetector(geometry, cfg["padding_vu"][kind])
        for name, geometry in (("nominal", nominal), ("validation", validation))
    }, profiles
