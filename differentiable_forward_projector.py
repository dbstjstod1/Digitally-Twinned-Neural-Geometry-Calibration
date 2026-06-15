from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------
# Differentiable projector (ray-marching)
# ---------------------------

@dataclass
class RT_PARAM:
    x: int
    y: int
    width: int
    height: int


def _swap_yz(vec: torch.Tensor) -> torch.Tensor:
    """
    English comments only.
    Swap y and z axes to match your CUDA convention.
    vec: (...,3) in (x,y,z)
    """
    x = vec[..., 0:1]
    y = vec[..., 1:2]
    z = vec[..., 2:3]
    return torch.cat([x, z, y], dim=-1)


def _ray_box_intersect(
    o: torch.Tensor,
    d: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
    eps_dir: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Slab method ray-box intersection.
    o, d: (R, 3)
    box_min, box_max: (3,)
    Returns: tmin, tmax, hit (R,)
    """
    d_safe = torch.where(d.abs() < eps_dir, torch.full_like(d, eps_dir), d)
    inv_d = 1.0 / d_safe

    t0 = (box_min[None, :] - o) * inv_d
    t1 = (box_max[None, :] - o) * inv_d

    t_small = torch.minimum(t0, t1)
    t_big = torch.maximum(t0, t1)

    tmin = torch.max(t_small, dim=-1).values
    tmax = torch.min(t_big, dim=-1).values
    hit = tmax > tmin
    return tmin, tmax, hit


def _world_to_grid_norm(
    pts: torch.Tensor,
    *,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    X0: float, Y0: float, Z0: float,
    align_corners: bool,
) -> torch.Tensor:
    """
    English comments only.
    Convert world coords (x,y,z) to grid_sample normalized coords in [-1, 1].
    pts: (..., 3)
    """
    ix = (pts[..., 0] - X0) / dx - 0.5
    iy = (pts[..., 1] - Y0) / dy - 0.5
    iz = (pts[..., 2] - Z0) / dz - 0.5

    if align_corners:
        x_norm = 2.0 * ix / max(nx - 1, 1) - 1.0
        y_norm = 2.0 * iy / max(ny - 1, 1) - 1.0
        z_norm = 2.0 * iz / max(nz - 1, 1) - 1.0
    else:
        x_norm = (2.0 * ix + 1.0) / float(nx) - 1.0
        y_norm = (2.0 * iy + 1.0) / float(ny) - 1.0
        z_norm = (2.0 * iz + 1.0) / float(nz) - 1.0

    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def _build_jk_grid(nu: int, nv: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Returns jp2, kp2 each (R,) where R=nu*nv.
    """
    jj = torch.arange(nu, device=device, dtype=dtype)
    kk = torch.arange(nv, device=device, dtype=dtype)
    k_grid, j_grid = torch.meshgrid(kk, jj, indexing="ij")  # (nv,nu)
    return j_grid.reshape(-1), k_grid.reshape(-1)


def _apply_roi_mapping(
    jp2: torch.Tensor,
    kp2: torch.Tensor,
    *,
    roi: RT_PARAM,
    ori_nu: int,
    ori_nv: int,
    nu: int,
    nv: int,
    recon_type: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comments only.
    """
    jp = jp2
    kp = kp2
    if recon_type == 1:
        jp = jp2 - float(roi.x)
        kp = kp2 - float(ori_nv - (roi.height + roi.y))
    elif recon_type == 0:
        jp = jp2
        kp = kp2 + float(ori_nv - (roi.height + roi.y))
    return jp, kp


def _pmat_decompose_K_torch(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    English comments only.
    Torch version of your MATLAB PmatDecompose() for K extraction.

    Input:
      A: (3,3) = P[:, :3]
    Output:
      K: (3,3) upper-triangular (intrinsic-like)
    """
    # MATLAB:
    # P_left = flipud(A)^T
    P_left = torch.flip(A, dims=[0]).transpose(0, 1)  # (3,3)

    # [Q, R33] = qr(P_left)
    # torch.linalg.qr returns Q, R (R is upper-triangular)
    Q, R33 = torch.linalg.qr(P_left)

    # MATLAB K construction:
    # K(1,1)=R33(3,3); K(1,2)=R33(2,3); K(1,3)=R33(1,3)
    # K(2,1)=R33(3,2); K(2,2)=R33(2,2); K(2,3)=R33(1,2)
    # K(3,1)=R33(3,1); K(3,2)=R33(2,1); K(3,3)=R33(1,1)
    K = torch.zeros((3, 3), device=A.device, dtype=A.dtype)
    K[0, 0] = R33[2, 2]
    K[0, 1] = R33[1, 2]
    K[0, 2] = R33[0, 2]
    K[1, 0] = R33[2, 1]
    K[1, 1] = R33[1, 1]
    K[1, 2] = R33[0, 1]
    K[2, 0] = R33[2, 0]
    K[2, 1] = R33[1, 0]
    K[2, 2] = R33[0, 0]

    # MATLAB sign fixes to make diag(K) positive:
    # If K(j,j) < 0, flip K(:,j) and corresponding row of R (not needed here for f,u0,v0).
    # We apply only K column flips (sufficient for K-based intrinsics).
    for j in range(3):
        s = torch.where(K[j, j] < 0.0, torch.tensor(-1.0, device=A.device, dtype=A.dtype),
                        torch.tensor(1.0, device=A.device, dtype=A.dtype))
        K[:, j] = K[:, j] * s

    # Avoid exactly-zero focal due to numerical issues
    K[0, 0] = torch.where(K[0, 0].abs() < eps, eps * torch.sign(K[0, 0] + eps), K[0, 0])
    K[1, 1] = torch.where(K[1, 1].abs() < eps, eps * torch.sign(K[1, 1] + eps), K[1, 1])

    return K


def sinoproj_rdsh_pinv_raycast_dominant(
    *,
    smat: torch.Tensor,              # (1,1,imsz,imsy,imsx)
    Pmat: torch.Tensor,              # (B,12) or (B,3,4)
    geo_parameter: torch.Tensor,     # (B,7)
    geo_stitch: torch.Tensor,        # (B,2)
    nu: int, nv: int, du: float, dv: float,
    imsx: int, imsy: int, imsz: int,
    dx: float, dy: float, dz: float,
    X0: float, Y0: float, Z0: float,
    ureverse: int, vreverse: int,
    roi: RT_PARAM,
    recon_type: int,
    n_samples: int = 128,
    chunk_size: int = 8192,
    ori_nu: Optional[int] = None,
    ori_nv: Optional[int] = None,
    reg: float = 1e-6,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    English comments only.
    Differentiable ray-marching forward projector.
    Output: (B, nv, nu)

    Change vs original:
      - f, un, vn are recomputed from updated P (MATLAB-like PmatDecompose)
      - No longer taken from P_I
    """
    assert smat.ndim == 5 and smat.shape[0] == 1 and smat.shape[1] == 1, "smat must be (1,1,D,H,W)"
    device = smat.device

    if ori_nu is None:
        ori_nu = nu
    if ori_nv is None:
        ori_nv = nv

    if Pmat.ndim == 2 and Pmat.shape[1] == 12:
        Pmat_ = Pmat.view(-1, 3, 4)
    elif Pmat.ndim == 3 and Pmat.shape[1:] == (3, 4):
        Pmat_ = Pmat
    else:
        raise ValueError("Pmat must be (B,12) or (B,3,4)")

    B = Pmat_.shape[0]
    Rpix = nu * nv

    jp2, kp2 = _build_jk_grid(nu, nv, device=device, dtype=torch.float32)
    jp, kp = _apply_roi_mapping(jp2, kp2, roi=roi, ori_nu=ori_nu, ori_nv=ori_nv, nu=nu, nv=nv, recon_type=recon_type)

    extra_u = float(ori_nu - (roi.x + roi.width))

    box_min = torch.tensor([X0, Y0, Z0], device=device, dtype=torch.float32)
    box_max = torch.tensor([X0 + dx * imsx, Y0 + dy * imsy, Z0 + dz * imsz], device=device, dtype=torch.float32)

    out = torch.zeros((B, nv, nu), device=device, dtype=torch.float32)
    I3 = torch.eye(3, device=device, dtype=torch.float32)

    vol = smat.to(torch.float32)
    ks = (torch.arange(n_samples, device=device, dtype=torch.float32) + 0.5)  # (n_samples,)
    
    for b in range(B):
        P = Pmat_[b].to(torch.float32)  # (3,4)
        A = P[:, :3]                    # (3,3)
        t = P[:, 3]                     # (3,)
        A_reg = A + reg * I3

        # Recompute K from updated P (MATLAB PmatDecompose logic)
        K = _pmat_decompose_K_torch(A, eps=1e-8)
        f = K[0, 0]     # K(1,1)
        un = K[0, 2]    # K(1,3)
        vn = K[1, 2]    # K(2,3)
        
        # print(f"[b={b}] f={f.detach().cpu().item():.6f}, un={un.detach().cpu().item():.6f}, vn={vn.detach().cpu().item():.6f}")
        # import pdb
        # pdb.set_trace()
        
        src_world = geo_parameter[b, 0:3].to(torch.float32)
        w_geo = geo_parameter[b, 5].to(torch.float32)
        
        stitch_u = geo_stitch[b, 0].to(torch.float32) # u0,v0 shift from stitch info
        stitch_v = geo_stitch[b, 1].to(torch.float32)

        if int(ureverse) == 1:
            u = (float(nu) - jp + 0.5 + extra_u) * float(du)
        else:
            u = (jp + 0.5 + stitch_u + extra_u) * float(du)

        if int(vreverse) == 1:
            v = (float(nv) - kp + 0.5 + stitch_v) * float(dv)
        else:
            v = (kp + 0.5 + stitch_v) * float(dv)

        detdist = torch.sqrt((u - un) ** 2 + (v - vn) ** 2 + 1e-12)
        vec_scale = torch.sqrt(f * f + detdist * detdist + 1e-12)

        w_map = w_geo.expand_as(u)
        tmp = torch.stack([u * w_geo, v * w_geo, w_map], dim=-1)
        cal = tmp - t[None, :]  # (R,3)

        soln = torch.linalg.solve(A_reg[None, :, :].expand(Rpix, 3, 3), cal)  # (R,3)

        vec = soln - src_world[None, :]
        vec_norm = torch.sqrt((vec * vec).sum(dim=-1) + 1e-12)
        vec_unit = vec / vec_norm[:, None]
        diff = vec_unit * vec_scale[:, None]

        src_vol = _swap_yz(src_world[None, :]).expand(Rpix, 3)
        diff_vol = _swap_yz(diff)

        d_norm = torch.sqrt((diff_vol * diff_vol).sum(dim=-1) + 1e-12)
        d = diff_vol / d_norm[:, None]

        tmin, tmax, hit = _ray_box_intersect(src_vol, d, box_min, box_max)

        seg_len = (tmax - tmin).clamp_min(0.0)
        step = seg_len / float(n_samples)

        proj_flat = torch.zeros((Rpix,), device=device, dtype=torch.float32)

        for r0 in range(0, Rpix, chunk_size):
            r1 = min(r0 + chunk_size, Rpix)

            o_c = src_vol[r0:r1]
            d_c = d[r0:r1]
            tmin_c = tmin[r0:r1]
            step_c = step[r0:r1]
            hit_c = hit[r0:r1]

            t_s = tmin_c[:, None] + step_c[:, None] * ks[None, :]
            pts = o_c[:, None, :] + d_c[:, None, :] * t_s[:, :, None]

            grid = _world_to_grid_norm(
                pts,
                nx=imsx, ny=imsy, nz=imsz,
                dx=dx, dy=dy, dz=dz,
                X0=X0, Y0=Y0, Z0=Z0,
                align_corners=align_corners,
            )

            grid = grid.permute(1, 0, 2).contiguous()       # (n_samples,Rc,3)
            grid = grid[:, None, :, :].unsqueeze(0)         # (1,n_samples,1,Rc,3)

            samp = F.grid_sample(
                vol, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=align_corners
            )  # (1,1,n_samples,1,Rc)

            samp = samp[0, 0, :, 0, :]                      # (n_samples,Rc)
            integral = samp.sum(dim=0) * step_c             # (Rc,)
            integral = torch.where(hit_c, integral, torch.zeros_like(integral))
            proj_flat[r0:r1] = integral

        out[b] = proj_flat.view(nv, nu)

    return out.to(smat.dtype)
