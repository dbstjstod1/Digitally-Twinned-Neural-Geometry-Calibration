"""Independent CPU checks for noncircular calibration coordinates and 9 DoF support."""
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from calibration_geometry import (centered_source_positions, geometry_to_pmat,
                                  pmat_to_pixel)
from DoF_transform import apply_9DoF_transform_effective
from fast_projectors import RT_PARAM, modular_arrays_geocal
from sinespin_geometry import build_icono_orbit


class CalibrationGeometryTests(unittest.TestCase):
    def test_world_mm_projection_against_independent_ray_plane_intersection(self):
        geometry = build_icono_orbit(detector_bin=4, n_views=17)
        center = np.array([12., -5., 42.])
        pmat = geometry_to_pmat(geometry, center_internal_mm=center, dtype=np.float64)
        points = np.array([[0., 0., 0.], [31., -47., 63.], [-72., 18., -52.]])
        world = (points + center)[:, [0, 2, 1]]
        for view in (0, 4, 8, 12, 16):
            source, detector = geometry.source_positions[view], geometry.module_centers[view]
            row, col = geometry.row_vectors[view], geometry.col_vectors[view]
            normal = np.cross(row, col)
            for point, point_world in zip(points, world):
                ray = point - source
                intersection = source + ray * (np.dot(detector-source, normal) / np.dot(ray, normal))
                expected_mm = np.array([np.dot(intersection-detector, col),
                                        np.dot(intersection-detector, row)])
                expected_mm += [.5*geometry.detector_width_mm, .5*geometry.detector_height_mm]
                projected = pmat[view] @ np.r_[point_world, 1.]
                np.testing.assert_allclose(projected[:2]/projected[2], expected_mm, atol=1e-10)

    def test_conversion_recovers_physical_arrays_and_box_center(self):
        geometry = build_icono_orbit(detector_bin=2, n_views=37)
        for center in (np.zeros(3), np.array([12., -5., 42.])):
            # Origins refer to voxel-box edges. First voxel centre adds half a voxel.
            shape = np.array([929, 929, 801])
            spacing = .2
            origin = center - shape * spacing / 2
            pmat = geometry_to_pmat(geometry, center_internal_mm=center, dtype=np.float64)
            recovered = modular_arrays_geocal(
                torch.tensor(pmat), torch.zeros((geometry.n_views, 2)),
                nu=geometry.detector_cols, nv=geometry.detector_rows,
                du=geometry.pixel_width, dv=geometry.pixel_height,
                ureverse=-1, vreverse=-1,
                roi=RT_PARAM(0, 0, geometry.detector_cols, geometry.detector_rows),
                recon_type=1, ori_nu=geometry.detector_cols, ori_nv=geometry.detector_rows,
                imsx=int(shape[0]), imsy=int(shape[1]), imsz=int(shape[2]),
                dx=spacing, dy=spacing, dz=spacing, X0=origin[0], Y0=origin[1], Z0=origin[2])
            for actual, expected in zip(recovered, geometry.modular_arrays(np.float64)):
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-10)
            np.testing.assert_allclose(centered_source_positions(pmat, center_internal_mm=center),
                                       geometry.source_positions, atol=1e-10)
            # Include negative homogeneous scale: source and pixel coordinates are unchanged.
            recovered_pixels = pmat_to_pixel(-3.7*pmat, du=geometry.pixel_width,
                                             dv=geometry.pixel_height, center_internal_mm=center)
            np.testing.assert_allclose(recovered_pixels/-3.7, geometry.projection_matrices(),
                                       atol=2e-10)

    def test_exact_sinespin_lies_in_existing_effective_9dof_family(self):
        options = dict(n_views=546, scan_angle_deg=220., detector_bin=2)
        nominal = build_icono_orbit("circular", **options)
        truth = build_icono_orbit("sinespin", **options)
        np.testing.assert_array_equal(nominal.theta_deg, truth.theta_deg)
        p0 = torch.tensor(geometry_to_pmat(nominal))
        geo = torch.zeros((truth.n_views, 7))
        zeros = torch.zeros((truth.n_views, 3))
        # Apparatus tilts by -beta about its column. Right-multiplied object motion
        # therefore rotates by +beta about the same column; no K or translation change.
        rotations = Rotation.from_rotvec(np.deg2rad(truth.tilt_deg)[:, None] * truth.col_vectors)
        euler = rotations.as_euler("xyz", degrees=True)
        self.assertLess(float(np.abs(euler).max()), 9.)
        args = dict(nx=929, ny=929, nz=801, dx=.2, dy=.2, dz=.2,
                    X0=-92.9, Y0=-92.9, Z0=-80.1, use_inverse_right_multiply=0)
        exact, _ = apply_9DoF_transform_effective(p0, geo, zeros, zeros,
                                                 torch.tensor(euler, dtype=torch.float32), **args)
        zero, _ = apply_9DoF_transform_effective(p0, geo, zeros, zeros, zeros, **args)
        rng = np.random.default_rng(782)
        points = np.c_[rng.uniform(-70., 70., (1000, 3)), np.ones(1000)]

        def detector_coordinates(matrix):
            matrix = pmat_to_pixel(matrix, du=truth.pixel_width, dv=truth.pixel_height)
            projected = np.einsum("vij,pj->vpi", matrix, points)
            return projected[..., :2]/projected[..., 2, None]

        target = truth.projection_matrices() @ points.T
        target = (target[:, :2]/target[:, 2, None]).transpose(0, 2, 1)
        error = np.linalg.norm(detector_coordinates(exact.numpy())-target, axis=-1)
        self.assertLess(float(error.max()), 2e-4)  # Float32 QR/Euler reconstruction.
        self.assertLess(float(np.sqrt(np.mean(error**2))), 4e-5)
        self.assertLess(float(np.max(np.linalg.norm(detector_coordinates(zero.numpy())-
                                                   detector_coordinates(p0.numpy()), axis=-1))), 2e-4)
        source_error = centered_source_positions(exact.numpy())-truth.source_positions
        self.assertLess(float(np.linalg.norm(source_error, axis=-1).max()), 4e-4)
        initial_error = detector_coordinates(p0.numpy())-target
        self.assertGreater(float(np.sqrt(np.mean(np.sum(initial_error**2, axis=-1)))), 10.)


if __name__ == "__main__":
    unittest.main()
