"""Checks for genuinely frozen K/rigid blocks, fresh graphs and Adam resume."""
import copy
import unittest
import numpy as np
import torch
from alternating_spline import SplitBSplineMotion9, AlternatingSplineAdam
from spline_motion_model import BSplineMotion9


class AlternatingSplineTests(unittest.TestCase):
    def make(self):
        return SplitBSplineMotion9(31,8,dtype=torch.float64)

    def loss(self, model, block):
        motion=model(torch.arange(31),block=block)
        # Coupled observation: pose and intrinsics both affect each measurement.
        return ((motion[:,:3]+.4*motion[:,3:6]+.2*motion[:,6:] - 1.3)**2).mean()

    def update(self, model, optimizer, block):
        optimizer.zero_grad(set_to_none=True)
        value=self.loss(model,block)
        value.backward()
        optimizer.step(block)
        return float(value.detach())

    def test_same_spline_space_values_and_gradient(self):
        joint=BSplineMotion9(31,8,dtype=torch.float64)
        split=self.make()
        values=torch.linspace(-.1,.2,72,dtype=torch.float64).reshape(8,9)
        with torch.no_grad():
            joint.raw_coefficients.copy_(values)
            split.raw_intrinsic.copy_(values[:,:3])
            split.raw_rigid.copy_(values[:,3:])
        ids=torch.tensor([0,30,15,3,3])
        torch.testing.assert_close(joint(ids),split(ids),rtol=0,atol=0)
        self.assertEqual(joint.get_config(),split.get_config())
        joint(ids).square().sum().backward()
        split(ids).square().sum().backward()
        torch.testing.assert_close(joint.raw_coefficients.grad,
            torch.cat((split.raw_intrinsic.grad,split.raw_rigid.grad),dim=-1),rtol=0,atol=0)

    def test_inactive_parameter_curve_and_moments_frozen_after_both_momenta_exist(self):
        model=self.make();optimizer=AlternatingSplineAdam(model,.001)
        for _ in range(4):
            for block in optimizer.blocks:self.update(model,optimizer,block)
        for block in optimizer.blocks:
            inactive='intrinsic' if block=='rigid' else 'rigid'
            param=optimizer.parameters[inactive]
            before=param.detach().clone()
            curve=model(torch.arange(31)).detach().clone()
            other_state=copy.deepcopy(optimizer.optimizers[inactive].state_dict())
            self.update(model,optimizer,block)
            self.assertIsNone(param.grad)
            self.assertTrue(torch.equal(param,before))
            after=model(torch.arange(31)).detach()
            ids=slice(0,3) if inactive=='intrinsic' else slice(3,9)
            self.assertTrue(torch.equal(curve[:,ids],after[:,ids]))
            state=optimizer.optimizers[inactive].state_dict()
            for key,value in other_state['state'][0].items():
                torch.testing.assert_close(state['state'][0][key],value,rtol=0,atol=0)
        self.assertEqual(optimizer.updates,dict(rigid=5,intrinsic=5))
        self.assertEqual(optimizer.inactive_checks,10)

    def test_fresh_intrinsic_objective_after_rigid_update(self):
        model=self.make();opt=AlternatingSplineAdam(model,.001)
        before=self.loss(model,'intrinsic')
        stale=torch.autograd.grad(before,model.raw_intrinsic)[0]
        self.update(model,opt,'rigid')
        after=self.loss(model,'intrinsic')
        fresh=torch.autograd.grad(after,model.raw_intrinsic)[0]
        self.assertLess(float(after.detach()),float(before.detach()))
        self.assertFalse(torch.equal(stale,fresh))

    def test_checkpoint_preserves_both_moments_and_next_updates(self):
        model=self.make();opt=AlternatingSplineAdam(model,.001)
        for _ in range(3):
            for block in opt.blocks:self.update(model,opt,block)
        restored=self.make();other=AlternatingSplineAdam(restored,.001)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        for block in opt.blocks:
            a=self.update(model,opt,block);b=self.update(restored,other,block)
            self.assertEqual(a,b)
            torch.testing.assert_close(model.raw_coefficients,restored.raw_coefficients,rtol=0,atol=0)
        self.assertEqual(opt.updates,other.updates)
        self.assertEqual(opt.inactive_checks,other.inactive_checks)

    def test_camera_intrinsics_and_source_freeze_in_physical_geometry(self):
        from sinespin_geometry import build_icono_orbit
        from calibration_geometry import geometry_to_pmat
        from calibration_gauge import decompose_world_pmat
        from run_sinespin_calibration import apply_motion
        model=self.make();opt=AlternatingSplineAdam(model,.001)
        nominal=geometry_to_pmat(build_icono_orbit('circular',n_views=31,scan_angle_deg=220,detector_bin=2),dtype=np.float64)
        def camera():
            with torch.no_grad():
                p,_=apply_motion(torch.tensor(nominal),model(torch.arange(31)),{},physical=True)
            return decompose_world_pmat(p.numpy())
        before=camera()
        self.update(model,opt,'rigid')
        after_rigid=camera()
        np.testing.assert_allclose(before['K'],after_rigid['K'],rtol=0,atol=4e-4)
        self.update(model,opt,'intrinsic')
        after_k=camera()
        np.testing.assert_allclose(after_rigid['source_world_mm'],after_k['source_world_mm'],rtol=0,atol=4e-4)

    def test_shared_gradient_is_rejected(self):
        model=self.make();opt=AlternatingSplineAdam(model,.001)
        self.loss(model,None).backward()
        with self.assertRaises(RuntimeError): opt.step('rigid')
        with self.assertRaises(ValueError): model(torch.arange(31),block='unknown')


if __name__=='__main__': unittest.main()
