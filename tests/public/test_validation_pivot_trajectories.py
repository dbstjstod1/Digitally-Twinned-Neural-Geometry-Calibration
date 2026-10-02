"""Independent CPU checks for a rigid rig rotation about an offset gantry pivot."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from geocal.parameters import (
    compose_effective_parameters,
    effective_parameters_from_pmat,
)
from geocal.coordinates import geometry_to_pmat
from geocal.camera import decompose_physical_camera
from geocal.trajectories import build_validation_family


class ValidationPivotTrajectoryTests(unittest.TestCase):
    def family(self, kind, **settings):
        return build_validation_family(
            kind, dict(perturbation_model="gantry_pivot_rotation", **settings)
        )

    def test_exact_source_z_localization_and_independent_rigid_pivot_rotation(self):
        for kind in ("circular", "sinespin"):
            for views in (41, 546):
                family, profile = self.family(
                    kind, n_views=views, rotation_pivot_fraction=0.5
                )
                _, baseline = build_validation_family(kind, {"n_views": views})
                nominal, truth = family["nominal"], family["validation"]
                dz = truth.source_positions[:, 2] - nominal.source_positions[:, 2]
                with self.subTest(kind=kind, views=views):
                    np.testing.assert_allclose(
                        dz, baseline["translation_xyz_mm"][:, 2], atol=7e-13, rtol=0.0
                    )
                    self.assertAlmostEqual(float(np.ptp(dz)), 5.0, places=11)
                    self.assertAlmostEqual(float(dz.min()), -2.5, places=11)
                    self.assertAlmostEqual(float(dz.max()), 2.5, places=11)
                    zero = profile["requested_source_z_displacement_mm"] == 0.0
                    self.assertTrue(np.any(zero))
                    for field in (
                        "source_positions",
                        "module_centers",
                        "row_vectors",
                        "col_vectors",
                    ):
                        np.testing.assert_array_equal(
                            getattr(truth, field)[zero], getattr(nominal, field)[zero]
                        )
                    # SciPy exponentiation is independent of the generator's Rodrigues implementation.
                    rotation = Rotation.from_rotvec(
                        profile["rotation_axis_xyz"]
                        * np.deg2rad(profile["rotation_angle_deg"])[:, None]
                    ).as_matrix()
                    pivot = 0.5 * nominal.source_positions
                    np.testing.assert_allclose(
                        profile["rotation_matrix_xyz"], rotation, atol=8e-15, rtol=0.0
                    )
                    np.testing.assert_array_equal(
                        profile["rotation_pivot_xyz_mm"], pivot
                    )
                    for field in ("source_positions", "module_centers"):
                        original = getattr(nominal, field)
                        expected = (
                            np.einsum("vij,vj->vi", rotation, original - pivot) + pivot
                        )
                        np.testing.assert_allclose(
                            getattr(truth, field), expected, atol=7e-13, rtol=0.0
                        )
                    for field in ("row_vectors", "col_vectors"):
                        expected = np.einsum(
                            "vij,vj->vi", rotation, getattr(nominal, field)
                        )
                        np.testing.assert_allclose(
                            getattr(truth, field), expected, atol=8e-15, rtol=0.0
                        )
                    np.testing.assert_allclose(
                        np.linalg.norm(truth.source_positions - pivot, axis=1),
                        375.0,
                        atol=5e-13,
                    )
                    self.assertFalse(profile["independent_translation_prescribed"])

    def test_fixed_intrinsics_sdd_and_detector_axes_without_forced_isocenter_tracking(
        self,
    ):
        for kind in ("circular", "sinespin"):
            family, profile = self.family(kind, n_views=61)
            nominal, truth = family["nominal"], family["validation"]
            with self.subTest(kind=kind):
                normal = np.cross(truth.row_vectors, truth.col_vectors)
                separation = truth.module_centers - truth.source_positions
                np.testing.assert_allclose(
                    np.linalg.norm(separation, axis=1), 1200.0, atol=1e-11
                )
                np.testing.assert_allclose(separation, 1200.0 * normal, atol=1e-11)
                frame = np.stack(
                    (truth.col_vectors, -truth.row_vectors, normal), axis=-1
                )
                np.testing.assert_allclose(
                    frame.transpose(0, 2, 1) @ frame,
                    np.broadcast_to(np.eye(3), frame.shape),
                    atol=2e-14,
                )
                np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=2e-14)
                cn = decompose_physical_camera(
                    geometry_to_pmat(nominal, dtype=np.float64)
                )
                cg = decompose_physical_camera(
                    geometry_to_pmat(truth, dtype=np.float64)
                )
                np.testing.assert_allclose(cg["K_mm"], cn["K_mm"], atol=4e-10, rtol=0.0)
                np.testing.assert_allclose(
                    cg["source_xyz_mm"], truth.source_positions, atol=2e-10
                )
                # The new pivot must not quietly re-aim the central ray toward isocenter.
                miss = np.linalg.norm(np.cross(truth.source_positions, normal), axis=1)
                self.assertGreater(float(miss.max()), 2.4)
                center_uv = (
                    truth.projection_matrices()[:, :2, 3]
                    / truth.projection_matrices()[:, 2, 3, None]
                )
                panel_center = np.array(
                    [(truth.detector_cols - 1) / 2, (truth.detector_rows - 1) / 2]
                )
                self.assertGreater(
                    float(np.linalg.norm(center_uv - panel_center, axis=1).max()), 6.0
                )

    def test_effective_translation_is_induced_by_pivot_and_all_nine_recompose_P(self):
        center = np.array([12.0, -5.0, 42.0])
        for kind in ("circular", "sinespin"):
            family, profile = self.family(kind, n_views=61, rotation_pivot_fraction=0.5)
            nominal, truth = family["nominal"], family["validation"]
            pn = geometry_to_pmat(nominal, center_internal_mm=center, dtype=np.float64)
            pt = geometry_to_pmat(truth, center_internal_mm=center, dtype=np.float64)
            extracted = effective_parameters_from_pmat(
                pt, pn, center_internal_mm=center
            )
            motion = extracted["parameters_9"]
            rotation = Rotation.from_rotvec(
                profile["rotation_axis_xyz"]
                * np.deg2rad(profile["rotation_angle_deg"])[:, None]
            ).as_matrix()
            inverse = rotation.transpose(0, 2, 1)
            pivot = 0.5 * nominal.source_positions
            expected_translation = pivot - np.einsum("vij,vj->vi", inverse, pivot)
            with self.subTest(kind=kind):
                np.testing.assert_allclose(motion[:, :3], 0.0, atol=4e-10, rtol=0.0)
                np.testing.assert_allclose(
                    motion[:, 3:6], expected_translation, atol=4e-10, rtol=0.0
                )
                self.assertGreater(float(np.abs(motion[:, 3:6]).max()), 2.4)
                np.testing.assert_allclose(
                    Rotation.from_euler("xyz", motion[:, 6:], degrees=True).as_matrix(),
                    inverse,
                    atol=4e-14,
                    rtol=0.0,
                )
                # Object-side t = -G.T * rig_t, not the independently specified translation.
                np.testing.assert_allclose(
                    motion[:, 3:6],
                    -np.einsum(
                        "vij,vj->vi", inverse, profile["rig_translation_xyz_mm"]
                    ),
                    atol=4e-10,
                    rtol=0.0,
                )
                composed = compose_effective_parameters(
                    pn, motion, center_internal_mm=center
                )
                normalized = pt / extracted["projective_scale"][:, None, None]
                relative = np.linalg.norm(
                    composed - normalized, axis=(1, 2)
                ) / np.linalg.norm(normalized, axis=(1, 2))
                self.assertLess(float(relative.max()), 1e-12)
                # Independently verify the equivalent apparent-object transform on pixel P.
                transform = np.broadcast_to(np.eye(4), (61, 4, 4)).copy()
                transform[:, :3, :3] = inverse
                transform[:, :3, 3] = expected_translation
                np.testing.assert_allclose(
                    truth.projection_matrices(),
                    nominal.projection_matrices() @ transform,
                    atol=2e-9,
                    rtol=0.0,
                )

    def test_projection_matches_independent_ray_plane_intersections(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [107.5, 0.0, -28.0],
                [-72.0, 18.0, -52.0],
                [0.0, -108.0, 94.0],
            ]
        )
        center = np.array([12.0, -5.0, 42.0])
        world = np.c_[(points + center)[:, [0, 2, 1]], np.ones(len(points))]
        for kind in ("circular", "sinespin"):
            family, _ = self.family(kind, n_views=61, rotation_pivot_fraction=0.5)
            geometry = family["validation"]
            pmat = geometry_to_pmat(
                geometry, center_internal_mm=center, dtype=np.float64
            )
            for view in (0, 9, 14, 22, 38, 43, 51, 60):
                with self.subTest(kind=kind, view=view):
                    source, detector = (
                        geometry.source_positions[view],
                        geometry.module_centers[view],
                    )
                    row, col = geometry.row_vectors[view], geometry.col_vectors[view]
                    normal = np.cross(row, col)
                    rays = points - source
                    intersection = (
                        source
                        + rays
                        * (np.dot(detector - source, normal) / (rays @ normal))[:, None]
                    )
                    expected = np.column_stack(
                        (
                            (intersection - detector) @ col,
                            (intersection - detector) @ row,
                        )
                    )
                    expected += [
                        0.5 * geometry.detector_width_mm,
                        0.5 * geometry.detector_height_mm,
                    ]
                    projected = world @ pmat[view].T
                    np.testing.assert_allclose(
                        projected[:, :2] / projected[:, 2:],
                        expected,
                        atol=3e-10,
                        rtol=0.0,
                    )

    def test_isocenter_limit_and_zero_amplitude_recover_existing_geometry(self):
        for kind in ("circular", "sinespin"):
            pivot, pivot_profile = self.family(
                kind, n_views=61, rotation_pivot_fraction=0.0
            )
            existing, _ = build_validation_family(
                kind, dict(perturbation_model="gantry_rotation", n_views=61)
            )
            zero, profile = self.family(
                kind,
                n_views=61,
                rotation_pivot_fraction=0.5,
                wobble_peak_to_peak_mm=0.0,
                mountain_height_mm=0.0,
            )
            with self.subTest(kind=kind):
                for field in (
                    "source_positions",
                    "module_centers",
                    "row_vectors",
                    "col_vectors",
                    "tilt_deg",
                ):
                    np.testing.assert_allclose(
                        getattr(pivot["validation"], field),
                        getattr(existing["validation"], field),
                        atol=5e-13,
                        rtol=0.0,
                    )
                    np.testing.assert_array_equal(
                        getattr(zero["validation"], field),
                        getattr(zero["nominal"], field),
                    )
                np.testing.assert_allclose(
                    pivot["validation"].projection_matrices(),
                    existing["validation"].projection_matrices(),
                    atol=2e-9,
                    rtol=0.0,
                )
                np.testing.assert_array_equal(
                    pivot_profile["rig_translation_xyz_mm"], 0.0
                )
                np.testing.assert_array_equal(profile["rotation_angle_deg"], 0.0)

    def test_default_pivot_and_invalid_fraction_or_unreachable_z(self):
        for kind in ("circular", "sinespin"):
            default, _ = self.family(kind, n_views=61)
            explicit, _ = self.family(kind, n_views=61, rotation_pivot_fraction=0.5)
            np.testing.assert_array_equal(
                default["validation"].source_positions,
                explicit["validation"].source_positions,
            )
            for fraction in (-1e-9, 1.0, 1.2, np.nan, np.inf, -np.inf):
                with self.subTest(kind=kind, fraction=fraction), self.assertRaises(
                    ValueError
                ):
                    self.family(kind, rotation_pivot_fraction=fraction)
            with self.subTest(kind=kind, fraction="unreachable"), self.assertRaises(
                ValueError
            ):
                self.family(kind, rotation_pivot_fraction=0.999)
        # A nonzero pivot fraction must not silently alter the legacy modes.
        for mode in ("translation", "gantry_rotation"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                build_validation_family(
                    "circular",
                    dict(perturbation_model=mode, rotation_pivot_fraction=0.5),
                )


if __name__ == "__main__":
    unittest.main()
