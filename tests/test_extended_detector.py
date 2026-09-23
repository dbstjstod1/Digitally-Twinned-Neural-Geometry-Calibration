import unittest
import numpy as np
from run_sinespin_calibration import geometries
from calibration_geometry import geometry_to_pmat
from calibration_gauge import effective_parameters_from_pmat
from denseball_landmarks import project_landmarks
from ball_phantom_fov import require_box_fov


class ExtendedDetectorTests(unittest.TestCase):
    def test_padding_changes_only_pixel_origin_not_physical_rays_or_motion(self):
        config=dict(seed=20260923,knots=8,amplitudes9=[2.]*9)
        original=geometries(31,2,spline_config=config)
        extended=geometries(31,2,spline_config=config,detector_padding_vu=(240,180))
        points=np.random.default_rng(4).uniform(-110,110,(35,3))
        for a,b in zip(original,extended):
            for left,right in zip(a.modular_arrays(),b.modular_arrays()):np.testing.assert_array_equal(left,right)
            np.testing.assert_allclose(project_landmarks(b.projection_matrices(),points),
                project_landmarks(a.projection_matrices(),points)+[180,240],rtol=0,atol=1e-10)
            self.assertEqual(b.detector_rows,a.detector_rows+480)
        motion=effective_parameters_from_pmat(geometry_to_pmat(extended[1],dtype=np.float64),
                                             geometry_to_pmat(extended[0],dtype=np.float64))['parameters_9']
        np.testing.assert_allclose(motion,original[1].motion9,rtol=0,atol=1e-9)
        require_box_fov({'truth':extended[1]},(651,643,643),.4)


if __name__=='__main__':unittest.main()
