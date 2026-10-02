"""Shared physical-coordinate projection and correction adapters."""

from pathlib import Path
import hashlib
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def projector_kwargs(g, shape=(651, 643, 643), voxel=0.4):
    from .geometry import RT_PARAM

    return dict(
        nu=g.detector_cols,
        nv=g.detector_rows,
        du=g.pixel_width,
        dv=g.pixel_height,
        imsx=shape[2],
        imsy=shape[1],
        imsz=shape[0],
        dx=voxel,
        dy=voxel,
        dz=voxel,
        X0=-shape[2] * voxel / 2,
        Y0=-shape[1] * voxel / 2,
        Z0=-shape[0] * voxel / 2,
        ureverse=-1,
        vreverse=-1,
        roi=RT_PARAM(0, 0, g.detector_cols, g.detector_rows),
        recon_type=1,
        ori_nu=g.detector_cols,
        ori_nv=g.detector_rows,
    )


def apply_motion(
    nominal, raw, bounds, *, shape=(651, 643, 643), voxel=0.4, physical=False
):
    import torch
    from .transforms import apply_9DoF_transform_effective, motion9_to_ts_tp_rot

    if physical:
        ts, tp, rot = raw[..., :3], raw[..., 3:6], raw[..., 6:9]
    else:
        ts, tp, rot, _ = motion9_to_ts_tp_rot(raw, **bounds)
    geo = torch.zeros((len(nominal), 7), device=nominal.device)
    p, _ = apply_9DoF_transform_effective(
        nominal,
        geo,
        ts,
        tp,
        rot,
        nx=shape[2],
        ny=shape[1],
        nz=shape[0],
        dx=voxel,
        dy=voxel,
        dz=voxel,
        X0=-shape[2] * voxel / 2,
        Y0=-shape[1] * voxel / 2,
        Z0=-shape[0] * voxel / 2,
        use_inverse_right_multiply=0,
    )
    return p.reshape(-1, 3, 4), torch.cat((ts, tp, rot), dim=-1)


def project(volume, p, g, *, voxel=0.4):
    import torch
    from .projector import sinoproj_joseph

    return sinoproj_joseph(
        smat=volume[None, None],
        Pmat=p,
        geo_parameter=torch.zeros((len(p), 7), device=p.device),
        geo_stitch=torch.zeros((len(p), 2), device=p.device),
        **projector_kwargs(g, tuple(volume.shape), voxel),
    )


def load_volume(path, device, shape=(651, 643, 643)):
    import torch

    if Path(path).stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f"Raw file size does not match float32 shape {shape}")
    volume = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    return torch.from_numpy(np.array(volume)).to(device)


def project_points(p, xyz):
    hom = np.column_stack((xyz, np.ones(len(xyz))))
    q = np.einsum("vij,pj->vpi", p, hom)
    return q[..., :2] / q[..., 2:3]
