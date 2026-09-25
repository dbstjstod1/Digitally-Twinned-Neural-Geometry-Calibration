"""Noncircular nominal, static intrinsic recovery, and shared-variable gradients."""
import unittest
import numpy as np
import torch
from calibration_geometry import geometry_to_pmat, pmat_to_pixel
from calibration_gauge import effective_parameters_from_pmat
from run_sinespin_calibration import geometries, apply_motion
from spline_calibration_geometry import spline_motion9
from spline_motion_model import SharedIntrinsicBSplineMotion9

CONFIG = dict(seed=20260925, knots=8,
              amplitudes9=[0,0,0,.5,.5,.35,.1,.08,.08],
              bias9=[.3,1.,-.25,.15,-.1,.2,.03,-.02,.01])


class SineSpinGeocalTests(unittest.TestCase):
    def test_nonzero_nominal_tilt_and_rays(self):
        nom, truth = geometries(61,2,spline_config=CONFIG,nominal_kind='sinespin',
                                detector_padding_vu=(240,180))
        self.assertGreater(np.max(abs(nom.tilt_deg)),9.9)
        pn, pt = geometry_to_pmat(nom,dtype=np.float64), geometry_to_pmat(truth,dtype=np.float64)
        recovered=effective_parameters_from_pmat(pt,pn)['parameters_9']
        np.testing.assert_allclose(recovered,truth.motion9,atol=2e-10)
        np.testing.assert_allclose(truth.motion9[:,:3],np.tile(CONFIG['bias9'][:3],(61,1)),atol=0)
        np.testing.assert_allclose(pmat_to_pixel(pt,du=truth.pixel_width,dv=truth.pixel_height),
                                   truth.projection_matrices(),atol=1e-9)
        p,m=apply_motion(torch.tensor(pn),torch.tensor(truth.motion9,dtype=torch.float32),{},
                         physical=True,shape=(651,643,643),voxel=.4)
        xyz=np.c_[np.random.default_rng(7).uniform(-100,100,(23,3)),np.ones(23)]
        a=p.detach().numpy()@xyz.T;b=pt@xyz.T
        np.testing.assert_allclose(a[:,:2]/a[:,2:],b[:,:2]/b[:,2:],atol=3e-4,rtol=0)

    def test_shared_parameter_count_gradients_and_archive(self):
        model=SharedIntrinsicBSplineMotion9(61,20,ts_max_mm=3,tp_max_mm=3,rot_max_deg=1,dtype=torch.float64)
        self.assertEqual(sum(p.numel() for p in model.parameters()),123)
        ids=torch.arange(61)
        w=torch.linspace(.1,1,61*9,dtype=torch.float64).reshape(61,9)
        (model(ids)*w).sum().backward()
        torch.testing.assert_close(model.raw_intrinsic.grad,w[:,:3].sum(0)*model.scales[:3])
        torch.testing.assert_close(model.raw_rigid.grad,model.basis.T@w[:,3:]*model.scales[3:])
        with torch.no_grad():
            model.raw_intrinsic.copy_(torch.tensor([.2,-.3,.1]))
            model.raw_rigid.copy_(torch.arange(120).reshape(20,6)/100.)
        curve=model(ids)
        torch.testing.assert_close(curve[:,:3],curve[:1,:3].expand(61,-1),rtol=0,atol=0)
        torch.testing.assert_close(curve,model.basis@model.physical_coefficients(),atol=1e-14,rtol=1e-14)
        self.assertTrue((curve.abs()<=model.scales).all())
        restored=SharedIntrinsicBSplineMotion9(61,20,dtype=torch.float64)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(ids),curve,atol=0,rtol=0)
        self.assertEqual(model.get_config()['parameter_count'],123)

    def test_shared_camera_gradient_noncircular_nominal(self):
        nom,_=geometries(31,2,nominal_kind='sinespin')
        p0=torch.tensor(geometry_to_pmat(nom,dtype=np.float64))
        model=SharedIntrinsicBSplineMotion9(31,8,ts_max_mm=3,tp_max_mm=3,rot_max_deg=1)
        xyz=torch.tensor(np.c_[np.random.default_rng(2).uniform(-70,70,(19,3)),np.ones(19)],dtype=torch.float32)
        p,_=apply_motion(p0,model(torch.arange(31)),{},physical=True)
        q=p@xyz.T;(q[:,:2]/q[:,2:]).square().mean().backward()
        self.assertTrue(torch.isfinite(model.raw_intrinsic.grad).all())
        self.assertTrue((model.raw_intrinsic.grad.abs()>1e-6).all())
        self.assertTrue(torch.isfinite(model.raw_rigid.grad).all())
        self.assertTrue((model.raw_rigid.grad.abs().sum(0)>1e-6).all())

    def test_invalid_bias_or_negative_amplitude(self):
        for extra in [dict(bias9=[0]*8),dict(bias9=[float('nan')]*9),dict(amplitudes9=[-1]*9)]:
            with self.assertRaises(ValueError):spline_motion9(31,dict(CONFIG,**extra))


if __name__=='__main__':unittest.main()
