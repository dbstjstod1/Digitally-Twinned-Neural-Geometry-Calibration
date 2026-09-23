"""Independent smooth nine-component GT and physical modular detector geometry.

Splines generate evaluation labels only; the estimator remains the vanilla MLP.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from calibration_gauge import compose_effective_parameters, PARAMETER_NAMES
from calibration_geometry import geometry_to_pmat
from physical_camera import decompose_physical_camera


def spline_motion9(views, config):
    if views < 2:
        raise ValueError('Require at least two views')
    knots = int(config['knots'])
    amplitudes = np.asarray(config['amplitudes9'], dtype=float)
    if knots < 4 or amplitudes.shape != (9,) or not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError('Require >=4 knots and nine finite positive amplitudes')
    rng = np.random.default_rng(config['seed'])
    x = np.linspace(0., 1., knots)
    y = rng.uniform(-1., 1., (knots, 9))
    query = np.linspace(0., 1., views)
    spline = CubicSpline(x, y, bc_type='natural')
    motion = spline(query)
    offset = motion.mean(axis=0)
    scale = amplitudes / np.max(np.abs(motion-offset), axis=0)
    return (motion-offset)*scale, dict(knot_progress=x.tolist(),
        knot_values_applied=((y-offset)*scale).tolist(), boundary_condition='natural',
        centering='Subtract acquired-view mean; scale acquired-view maximum absolute value',
        parameter_names=list(PARAMETER_NAMES), **config)


class SplineGeometry:
    """Flat detector with varying principal point/SDD and arbitrary rigid poses."""
    kind = 'spline9'

    def __init__(self, nominal, config):
        self.nominal = nominal
        self.motion9, self.spline_record = spline_motion9(nominal.n_views, config)
        self.world_pmat = compose_effective_parameters(geometry_to_pmat(nominal, dtype=np.float64), self.motion9)
        camera = decompose_physical_camera(self.world_pmat)
        self.source_positions = camera['source_xyz_mm']
        self.row_vectors, self.col_vectors = camera['row_xyz'], camera['col_xyz']
        self.normals = camera['normal_xyz']
        self.focal_mm, self.cu_mm, self.cv_mm = camera['intrinsics_f_cu_cv_mm'].T
        self.module_centers = (self.source_positions + self.focal_mm[:, None]*self.normals
            + (nominal.detector_width_mm/2-self.cu_mm)[:, None]*self.col_vectors
            + (nominal.detector_height_mm/2-self.cv_mm)[:, None]*self.row_vectors)
        self.tilt_deg = np.zeros(nominal.n_views)  # No single sine-tilt parameter exists.

    def __getattr__(self, name):
        return getattr(self.nominal, name)

    def modular_arrays(self, dtype=np.float32):
        return tuple(np.ascontiguousarray(a, dtype=dtype) for a in (
            self.source_positions, self.module_centers, self.row_vectors, self.col_vectors))

    def projection_matrices(self):
        # Rebuild from actual physical detector vectors, independently of saved P.
        displacement = self.module_centers-self.source_positions
        distance = np.einsum('vi,vi->v', displacement, self.normals)
        u0 = (self.detector_cols-1)/2-np.einsum('vi,vi->v', displacement, self.col_vectors)/self.pixel_width
        v0 = (self.detector_rows-1)/2-np.einsum('vi,vi->v', displacement, self.row_vectors)/self.pixel_height
        u = distance[:, None]/self.pixel_width*self.col_vectors+u0[:, None]*self.normals
        v = distance[:, None]/self.pixel_height*self.row_vectors+v0[:, None]*self.normals
        linear = np.stack((u, v, self.normals), axis=1)
        return np.concatenate((linear, -np.einsum('vij,vj->vi', linear, self.source_positions)[..., None]), axis=2)

    def detector_visibility(self, points):
        q = np.einsum('vij,pj->vpi', self.projection_matrices(), np.c_[points, np.ones(len(points))])
        uv = q[..., :2]/q[..., 2:]
        return ((q[..., 2] > 0) & (uv[..., 0] >= -.5) & (uv[..., 0] <= self.detector_cols-.5)
                & (uv[..., 1] >= -.5) & (uv[..., 1] <= self.detector_rows-.5))

    def record(self):
        return dict(kind=self.kind, views=self.n_views, scan_angle_deg=self.scan_angle_deg,
            angle_step_deg=float(self.theta_deg[1]-self.theta_deg[0]),
            detector_shape_vu=[self.detector_rows, self.detector_cols],
            pixel_vu_mm=[self.pixel_height, self.pixel_width],
            detector_extent_vu_mm=[self.detector_height_mm, self.detector_width_mm],
            nominal_sod_mm=self.nominal.sod_mm, nominal_sdd_mm=self.nominal.sdd_mm,
            actual_source_radius_min_max_mm=[float(x) for x in (
                np.linalg.norm(self.source_positions, axis=1).min(), np.linalg.norm(self.source_positions, axis=1).max())],
            source_to_plane_distance_min_max_mm=[float(self.focal_mm.min()), float(self.focal_mm.max())],
            principal_point='Varies per view; see spline_motion9.npy and truth_geometry.npz',
            isocenter_constraint=False, distances_are_assumed=True, spline=self.spline_record)
