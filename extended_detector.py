"""Virtual detector extension with unchanged source, plane, axes and pixel pitch."""
import numpy as np


class ExtendedDetector:
    def __init__(self, geometry, padding_vu):
        padding=np.asarray(padding_vu)
        if padding.shape!=(2,) or not np.issubdtype(padding.dtype,np.integer) or np.any(padding<0):
            raise ValueError('Padding must be two nonnegative integer pixel counts')
        self.base=geometry
        self.padding_vu=padding.tolist()
        self.detector_rows=geometry.detector_rows+2*int(padding[0])
        self.detector_cols=geometry.detector_cols+2*int(padding[1])

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def detector_height_mm(self):return self.detector_rows*self.pixel_height

    @property
    def detector_width_mm(self):return self.detector_cols*self.pixel_width

    def projection_matrices(self):
        v,u=self.padding_vu
        shift=np.array([[1.,0.,u],[0.,1.,v],[0.,0.,1.]])
        return shift@self.base.projection_matrices()

    def detector_visibility(self,points):
        q=np.einsum('vij,pj->vpi',self.projection_matrices(),np.c_[points,np.ones(len(points))])
        uv=q[...,:2]/q[...,2:]
        return ((q[...,2]>0)&(uv[...,0]>=-.5)&(uv[...,0]<=self.detector_cols-.5)
                &(uv[...,1]>=-.5)&(uv[...,1]<=self.detector_rows-.5))

    def record(self):
        record=self.base.record().copy()
        record.update(detector_shape_vu=[self.detector_rows,self.detector_cols],
            detector_extent_vu_mm=[self.detector_height_mm,self.detector_width_mm],
            virtual_detector_padding_vu=self.padding_vu,
            virtual_detector_definition='Symmetric pixel extension; original source, plane centre, axes and pixel pitch preserved')
        return record
