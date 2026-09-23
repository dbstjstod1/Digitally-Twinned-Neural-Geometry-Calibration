"""Physical camera decomposition without an aim-at-isocentre assumption."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_geometry import geometry_to_pmat
from physical_camera import (compose_physical_camera, compose_physical_nine,
                             decompose_physical_camera, physical_camera_residuals)
from sinespin_geometry import build_icono_orbit


class PhysicalCameraTests(unittest.TestCase):
    def setUp(self):
        self.geometry = build_icono_orbit("sinespin", detector_bin=2, n_views=37)
        self.nominal_p = geometry_to_pmat(self.geometry, dtype=np.float64)

    def test_source_and_proper_orientation_match_independent_acquisition_geometry(self):
        camera = decompose_physical_camera(self.nominal_p)
        geometry = self.geometry
        np.testing.assert_allclose(camera["source_xyz_mm"], geometry.source_positions, atol=1e-12)
        np.testing.assert_allclose(camera["col_xyz"], geometry.col_vectors, atol=1e-15)
        np.testing.assert_allclose(camera["row_xyz"], geometry.row_vectors, atol=1e-15)
        np.testing.assert_allclose(camera["normal_xyz"], geometry.normals, atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(camera["Q_camera_to_physical"]), 1., atol=1e-15)
        # Independently derived apparatus orientation, not the P decomposition.
        expected_q = Rotation.from_euler("xyz", np.column_stack((
            -90. - geometry.tilt_deg, np.zeros(geometry.n_views), geometry.theta_deg)), degrees=True).as_matrix()
        np.testing.assert_allclose(camera["Q_camera_to_physical"], expected_q, atol=1e-15)
        expected_k = [geometry.sdd_mm, geometry.detector_width_mm/2, geometry.detector_height_mm/2]
        np.testing.assert_allclose(camera["intrinsics_f_cu_cv_mm"], np.tile(expected_k, (geometry.n_views, 1)), atol=8e-13)

    def test_exact_roundtrip_nullspace_and_ray_coordinates_with_noncentral_sources(self):
        rng = np.random.default_rng(584)
        camera = decompose_physical_camera(self.nominal_p)
        views = self.geometry.n_views
        source = camera["source_xyz_mm"] + rng.uniform(-7., 7., (views, 3))
        rotation = Rotation.from_rotvec(rng.uniform(-.06, .06, (views, 3))).as_matrix()
        q = rotation @ camera["Q_camera_to_physical"]
        k = camera["K_mm"].copy()
        k[:, 0, 0] += 3.
        k[:, 1, 1] -= 2.
        k[:, 0, 1] += .17
        k[:, 0, 2] += rng.uniform(-4., 4., views)
        k[:, 1, 2] += rng.uniform(-4., 4., views)
        center = np.array([12., -5., 42.])
        scales = rng.uniform(.2, 3., views) * rng.choice([-1., 1.], views)
        p = compose_physical_camera(source, q, k, center_internal_mm=center, projective_scale=scales)
        decoded = decompose_physical_camera(p, center_internal_mm=center)
        np.testing.assert_allclose(decoded["source_xyz_mm"], source, atol=1e-12)
        np.testing.assert_allclose(decoded["Q_camera_to_physical"], q, atol=2e-15)
        np.testing.assert_allclose(decoded["K_mm"], k, atol=2e-12)
        np.testing.assert_allclose(decoded["projective_scale"], scales, atol=3e-15)
        rebuilt = compose_physical_camera(decoded["source_xyz_mm"], decoded["Q_camera_to_physical"],
                                          decoded["K_mm"], center_internal_mm=center,
                                          projective_scale=decoded["projective_scale"])
        np.testing.assert_allclose(rebuilt, p, atol=8e-10)
        # An independent homogeneous null-space calculation recovers C.
        for i in [0, 13, 27, 36]:
            _, _, vh = np.linalg.svd(p[i])
            world_c = vh[-1, :3]/vh[-1, 3]
            np.testing.assert_allclose(world_c[[0, 2, 1]]-center, source[i], atol=2e-9)
        # Direct ray coordinates in the declared detector basis, including skew.
        points = rng.uniform(-50., 50., (20, 3))
        world = (points+center)[:, [0, 2, 1]]
        projected = p @ np.c_[world, np.ones(len(points))].T
        uv = projected[:, :2] / projected[:, 2, None]
        rays = points[None] - source[:, None]
        col = np.einsum("vpi,vi->vp", rays, q[:, :, 0])
        row = np.einsum("vpi,vi->vp", rays, -q[:, :, 1])
        depth = np.einsum("vpi,vi->vp", rays, q[:, :, 2])
        expected_u = k[:, 0, 0, None]*col/depth + k[:, 0, 1, None]*row/depth + k[:, 0, 2, None]
        expected_v = k[:, 1, 1, None]*row/depth + k[:, 1, 2, None]
        np.testing.assert_allclose(uv, np.stack((expected_u, expected_v), axis=1), atol=2e-13)
        # This is deliberately not an isocentric/aim-at-origin camera.
        angle = np.linalg.norm(np.cross(q[:, :, 2], -source/np.linalg.norm(source, axis=1)[:, None]), axis=1)
        self.assertGreater(angle.max(), .04)

    def test_gt_relative_rotation_vector_is_in_fixed_physical_axes(self):
        truth = decompose_physical_camera(self.nominal_p)
        np.testing.assert_allclose(physical_camera_residuals(truth, truth)["residuals9"], 0., atol=1e-14)
        views = self.geometry.n_views
        omega = np.tile([.7, -1.2, .3], (views, 1))
        q = Rotation.from_rotvec(omega, degrees=True).as_matrix() @ truth["Q_camera_to_physical"]
        source = truth["source_xyz_mm"] + [1., -2., 3.]
        k = truth["K_mm"].copy()
        k[:, 0, 0] += 2.; k[:, 1, 1] += 2.
        k[:, 0, 2] += 3.; k[:, 1, 2] -= 4.
        estimated = decompose_physical_camera(compose_physical_camera(source, q, k))
        result = physical_camera_residuals(estimated, truth)
        np.testing.assert_allclose(result["orientation_rotvec_physical_deg"], omega, atol=4e-14)
        expected = np.tile([1., -2., 3., .7, -1.2, .3, 2., 3., -4.], (views, 1))
        np.testing.assert_allclose(result["residuals9"], expected, atol=2e-12)
        relative = q @ truth["Q_camera_to_physical"].transpose(0, 2, 1)
        angle_trace = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1., 1.)))
        np.testing.assert_allclose(result["orientation_error_deg"], angle_trace, atol=2e-12)

    def test_source_orientation_and_intrinsics_are_separate_physical_quantities(self):
        camera = decompose_physical_camera(self.nominal_p)
        views = self.geometry.n_views
        parameters = np.column_stack((camera["source_xyz_mm"], np.zeros((views, 3)), camera["intrinsics_f_cu_cv_mm"]))
        parameters[:, 5] = .5  # A physical-z rotation about each camera's own source.
        parameters[:, 6:] += [2., 1., -3.]
        p = compose_physical_nine(parameters, reference_Q=camera["Q_camera_to_physical"])
        changed = decompose_physical_camera(p)
        np.testing.assert_allclose(changed["source_xyz_mm"], camera["source_xyz_mm"], atol=1e-12)
        np.testing.assert_allclose(changed["intrinsics_f_cu_cv_mm"], parameters[:, 6:], atol=1e-12)
        residual = physical_camera_residuals(changed, camera)
        np.testing.assert_allclose(residual["orientation_rotvec_physical_deg"], np.tile([0., 0., .5], (views, 1)), atol=3e-14)

    def test_full_k_retains_float32_anisotropy_that_nine_component_summary_omits(self):
        camera = decompose_physical_camera(self.nominal_p.astype(np.float32))
        p_full = compose_physical_camera(camera["source_xyz_mm"], camera["Q_camera_to_physical"], camera["K_mm"],
                                         projective_scale=camera["projective_scale"])
        np.testing.assert_allclose(p_full, self.nominal_p.astype(np.float32), atol=4e-10)
        self.assertGreater(np.abs(camera["focal_anisotropy_mm"]).max(), 1e-6)
        parameters = np.column_stack((camera["source_xyz_mm"], np.zeros((len(p_full), 3)), camera["intrinsics_f_cu_cv_mm"]))
        p_nine = compose_physical_nine(parameters, reference_Q=camera["Q_camera_to_physical"])
        shared = decompose_physical_camera(p_nine)
        self.assertLess(np.abs(shared["focal_anisotropy_mm"]).max(), 2e-12)
        self.assertLess(np.abs(shared["skew_mm"]).max(), 2e-12)

    def test_invalid_reflected_camera_and_focal_are_rejected(self):
        camera = decompose_physical_camera(self.nominal_p)
        reflected = camera["Q_camera_to_physical"].copy()
        reflected[:, :, 1] *= -1
        with self.assertRaises(ValueError):
            compose_physical_camera(camera["source_xyz_mm"], reflected, camera["K_mm"])
        invalid_k = camera["K_mm"].copy()
        invalid_k[:, 0, 0] = -1
        with self.assertRaises(ValueError):
            compose_physical_camera(camera["source_xyz_mm"], camera["Q_camera_to_physical"], invalid_k)


if __name__ == "__main__":
    unittest.main()
