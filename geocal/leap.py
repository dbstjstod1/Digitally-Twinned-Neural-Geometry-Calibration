"""Independent Joseph data generation with the pinned LEAP build."""

import numpy as np


def configure_leap(geo, shape, voxel, device):
    from leapctype import tomographicModels

    ct = tomographicModels()
    ct.set_gpu(device.index or 0)
    ct.print_warnings = False
    ct.print_cost = False
    if not hasattr(ct, "get_forceJosephModularBackprojection"):
        raise RuntimeError(
            "This experiment requires the Joseph-pinned LEAP build; see examples/README.md."
        )
    if not ct.set_forceJosephModular(True) or not ct.get_forceJosephModular():
        raise RuntimeError("LEAP could not pin the forward projector to Joseph.")
    if not ct.get_forceJosephModularBackprojection():
        raise RuntimeError("LEAP could not pin the backprojector to Joseph.")
    if not ct.set_modularbeam(
        geo.n_views,
        geo.detector_rows,
        geo.detector_cols,
        geo.pixel_height,
        geo.pixel_width,
        *geo.modular_arrays()
    ):
        raise RuntimeError("LEAP rejected the detector geometry.")
    nz, ny, nx = shape
    if not ct.set_volume(nx, ny, nz, voxel, voxel, 0.0, 0.0, 0.0):
        raise RuntimeError("LEAP rejected the centered reconstruction grid.")
    ct.set_volumeDimensionOrder(1)  # LEAP ZYX
    ct.set_diameterFOV(float(np.hypot(nx, ny) * voxel))
    ct.set_offsetScan(False)
    ct.set_truncatedScan(False)
    return ct
