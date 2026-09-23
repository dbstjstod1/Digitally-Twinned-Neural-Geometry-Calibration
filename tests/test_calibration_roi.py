"""Check detector-coordinate crops, loss gradients, and observation provenance."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from calibration_roi import ProjectionROI
from calibration_losses import build_loss


class ProjectionROITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'roi.json'
        self.record=dict(target_sha256='observations',detector_shape_vu=[9,11],
                         boxes_xyxy=[[1,2,8,8],[3,1,10,7]],definition='test')

    def make_roi(self):
        self.path.write_text(json.dumps(self.record))
        return ProjectionROI(self.path,views=2,rows=9,cols=11,
                             target_sha256='observations',device='cpu')

    def test_shuffled_views_use_correct_crop_and_keep_integer_counts(self):
        roi=self.make_roi()
        image=torch.arange(198).reshape(2,9,11)
        actual=roi.crop(image.flip(0),torch.tensor([1,0]))
        expected=torch.stack([image[1,1:7,3:10],image[0,2:8,1:8]])
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(actual.dtype,torch.int64)

    def test_signed_lncc_gradients_match_explicit_crops_and_exclude_plate(self):
        roi=self.make_roi();torch.manual_seed(10)
        pred=torch.rand(2,9,11,dtype=torch.float64,requires_grad=True)
        other=pred.detach().clone().requires_grad_()
        target=torch.rand_like(pred)
        fn=build_loss('signed_lncc',kernel_size=3)
        actual=roi.loss(fn,pred,target,torch.tensor([0,1]))
        expected=fn(torch.stack([other[0,2:8,1:8],other[1,1:7,3:10]]),
                    torch.stack([target[0,2:8,1:8],target[1,1:7,3:10]]))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        grad=torch.autograd.grad(actual,pred)[0]
        torch.testing.assert_close(grad,torch.autograd.grad(expected,other)[0],rtol=0,atol=0)
        self.assertEqual(float(grad[0,:2].abs().sum()),0.)
        self.assertGreater(float(grad.abs().sum()),0.)
        altered=target.clone();altered[0,:2]=1000
        torch.testing.assert_close(roi.loss(fn,pred,altered,torch.tensor([0,1])),actual,rtol=0,atol=0)

    def test_invalid_provenance_and_boxes_rejected(self):
        for key,value in [('target_sha256','wrong'),('detector_shape_vu',[10,11]),
                          ('boxes_xyxy',[[1,2,8,8],[3,1,12,7]]),
                          ('boxes_xyxy',[[1,2,8,8],[3,1,9,7]]),
                          ('boxes_xyxy',[[1.,2,8,8],[3,1,10,7]])]:
            saved=self.record[key];self.record[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.make_roi()
            self.record[key]=saved


if __name__=='__main__':unittest.main()
