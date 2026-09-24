"""Replay is exact, and refuses any unexpected RNG consumption."""
import copy
import unittest
import torch
from resume_bspline_shuffle import EpochShuffleReplay
from spline_motion_model import BSplineMotion9


class ReplayTests(unittest.TestCase):
    def test_next_permutations_match_uninterrupted(self):
        original=torch.randperm
        torch.manual_seed(9)
        expected=[original(31,device='cpu') for _ in range(17)]
        torch.manual_seed(9)
        replay=EpochShuffleReplay(original,seed=9,views=31,completed_epochs=7)
        for value in expected[7:]:
            self.assertTrue(torch.equal(value,replay(31,device='cpu')))
        self.assertEqual(replay.calls,10)

    def test_refuse_rng_drift(self):
        torch.manual_seed(9);torch.rand(1)
        replay=EpochShuffleReplay(torch.randperm,seed=9,views=31,completed_epochs=7)
        with self.assertRaises(ValueError):replay(31,device='cpu')

    def test_adam_next_epochs_exact_after_restore(self):
        def make():
            model=BSplineMotion9(31,8,dtype=torch.float64)
            return model,torch.optim.Adam(model.parameters(),lr=.001)
        model,opt=make()
        target=torch.sin(torch.linspace(0,4,31,dtype=torch.float64))[:,None]
        original=torch.randperm
        def epoch(model,opt):
            for batch in torch.randperm(31,device='cpu').split(4):
                opt.zero_grad(set_to_none=True)
                (model(batch)-target[batch]).square().mean().backward();opt.step()
        torch.manual_seed(9)
        for i in range(12):
            epoch(model,opt)
            if i==6:state=copy.deepcopy(model.state_dict());optim=copy.deepcopy(opt.state_dict())
        expected=model.raw_coefficients.detach().clone()
        restored,other=make();restored.load_state_dict(state);other.load_state_dict(optim)
        torch.manual_seed(9)
        replay=EpochShuffleReplay(original,seed=9,views=31,completed_epochs=7)
        try:
            torch.randperm=replay
            for _ in range(5):epoch(restored,other)
        finally:torch.randperm=original
        torch.testing.assert_close(restored.raw_coefficients,expected,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
