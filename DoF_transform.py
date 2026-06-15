import torch
from typing import Tuple

# ============================================================
# Rotation helpers
# ============================================================

def _rot_xyz_deg_world(rot_deg_world: torch.Tensor) -> torch.Tensor:
    """
    English comments only.
    R = Rz * Ry * Rx with world-axis rotations.
    rot_deg_world: (B,3) = (rx, ry, rz) in degrees.
    Returns: (B,3,3)
    """
    rot = torch.deg2rad(rot_deg_world.to(torch.float32))
    rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)

    return Rz @ (Ry @ Rx)


def _perm_S_internal_to_world(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    English comments only.
    INTERNAL(x,y,z) = (WORLD x, WORLD z, WORLD y).
    This means WORLD = S * INTERNAL for column vectors.
    """
    S = torch.tensor([
        [1.0, 0.0, 0.0],  # x_w = x_i
        [0.0, 0.0, 1.0],  # y_w = z_i
        [0.0, 1.0, 0.0],  # z_w = y_i
    ], device=device, dtype=dtype)
    return S


def _rot_from_internal_euler_to_world(rot_internal_deg: torch.Tensor) -> torch.Tensor:
    """
    English comments only.
    Build rotation in INTERNAL axes, then convert it to WORLD axes:
      R_world = S * R_internal * S^T
    rot_internal_deg: (B,3) degrees, interpreted as INTERNAL-axis Euler
                      in the SAME order used by _rot_xyz_deg_world (Rz*Ry*Rx).
    """
    B = rot_internal_deg.shape[0]
    device = rot_internal_deg.device
    dtype = torch.float32

    R_int = _rot_xyz_deg_world(rot_internal_deg.to(dtype))  # (B,3,3)

    S = _perm_S_internal_to_world(device=device, dtype=dtype)  # (3,3)
    S = S.view(1, 3, 3).expand(B, 3, 3)

    R_w = S @ R_int @ S.transpose(1, 2)
    return R_w


def _swap_yz_internal_to_world(v: torch.Tensor) -> torch.Tensor:
    """
    English comments only.
    INTERNAL(x,y,z) = (WORLD x, WORLD z, WORLD y) in your pipeline.
    v: (...,3) in INTERNAL
    Returns: (...,3) in WORLD
    """
    return torch.stack([v[..., 0], v[..., 2], v[..., 1]], dim=-1)


# ============================================================
# P decomposition (MATLAB-style) for Eq4
# ============================================================

def _pmat_decompose_KRt_batch(P: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Decompose pinhole projection matrix:
      P: (B,3,4)
    Returns:
      K: (B,3,3) with K[2,2]=1 and diag positive
      R: (B,3,3) (world->camera rotation)
      t: (B,3)   (camera translation in camera coords), such that P = K [R|t]
    """
    B = P.shape[0]
    device = P.device
    dtype = P.dtype

    A = P[:, :, :3]         # (B,3,3)
    p4 = P[:, :, 3]         # (B,3)

    P_left = torch.flip(A, dims=[1]).transpose(1, 2)  # (B,3,3)
    Q, R33 = torch.linalg.qr(P_left)  # (B,3,3), (B,3,3)

    R = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    R[:, 0, 0] = Q[:, 0, 2]
    R[:, 0, 1] = Q[:, 1, 2]
    R[:, 0, 2] = Q[:, 2, 2]
    R[:, 1, 0] = Q[:, 0, 1]
    R[:, 1, 1] = Q[:, 1, 1]
    R[:, 1, 2] = Q[:, 2, 1]
    R[:, 2, 0] = Q[:, 0, 0]
    R[:, 2, 1] = Q[:, 1, 0]
    R[:, 2, 2] = Q[:, 2, 0]

    K = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = R33[:, 2, 2]
    K[:, 0, 1] = R33[:, 1, 2]
    K[:, 0, 2] = R33[:, 0, 2]
    K[:, 1, 0] = R33[:, 2, 1]
    K[:, 1, 1] = R33[:, 1, 1]
    K[:, 1, 2] = R33[:, 0, 1]
    K[:, 2, 0] = R33[:, 2, 0]
    K[:, 2, 1] = R33[:, 1, 0]
    K[:, 2, 2] = R33[:, 0, 0]

    for j in range(3):
        s = torch.where(K[:, j, j] < 0.0, -torch.ones((B,), device=device, dtype=dtype),
                        torch.ones((B,), device=device, dtype=dtype))
        K[:, :, j] = K[:, :, j] * s[:, None]
        R[:, j, :] = R[:, j, :] * s[:, None]

    k22 = K[:, 2, 2].clamp_min(eps)
    K = K / k22[:, None, None]
    p4 = p4 / k22[:, None]

    K[:, 0, 0] = torch.where(K[:, 0, 0].abs() < eps, eps * torch.sign(K[:, 0, 0] + eps), K[:, 0, 0])
    K[:, 1, 1] = torch.where(K[:, 1, 1].abs() < eps, eps * torch.sign(K[:, 1, 1] + eps), K[:, 1, 1])

    t = torch.linalg.solve(K, p4)  # (B,3)
    return K, R, t

def _pmat_decompose_KRt_batch_nominalK(
    P: torch.Tensor,
    f: torch.Tensor,
    un: torch.Tensor,
    vn: torch.Tensor,
    eps: float = 1e-8,
    use_scale_correction: bool = True,
):
    if P.ndim != 3 or P.shape[1:] != (3, 4):
        raise ValueError(f"P must have shape (B,3,4), but got {tuple(P.shape)}")

    B = P.shape[0]
    device = P.device
    dtype = P.dtype

    def _expand_param(x, name: str):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=device, dtype=dtype)
        else:
            x = x.to(device=device, dtype=dtype)

        if x.ndim == 0:
            x = x.expand(B)
        elif x.ndim == 1:
            if x.shape[0] == 1:
                x = x.expand(B)
            elif x.shape[0] != B:
                raise ValueError(f"{name} must be scalar, (1,), or (B,), but got {tuple(x.shape)}")
        else:
            raise ValueError(f"{name} must be scalar, (1,), or (B,), but got {tuple(x.shape)}")
        return x

    f = _expand_param(f, "f")
    un = _expand_param(un, "un")
    vn = _expand_param(vn, "vn")

    K_fixed = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    K_fixed[:, 0, 0] = f
    K_fixed[:, 1, 1] = f
    K_fixed[:, 0, 2] = un
    K_fixed[:, 1, 2] = vn
    K_fixed[:, 2, 2] = 1.0

    A = P[:, :, :3]
    p4 = P[:, :, 3]

    K_inv = torch.linalg.inv(K_fixed)
    M = torch.bmm(K_inv, A)
    c = torch.bmm(K_inv, p4.unsqueeze(-1)).squeeze(-1)

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = torch.bmm(U, Vh)

    detR = torch.det(R)
    neg_mask = detR < 0

    if neg_mask.any():
        fix = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        fix[neg_mask, 2, 2] = -1.0
        R = torch.bmm(torch.bmm(U, fix), Vh)

    # Best-fit isotropic scale for M ~= sR
    scale = (R * M).sum(dim=(1, 2)) / (R * R).sum(dim=(1, 2))
    scale = scale.clamp_min(eps)

    if use_scale_correction:
        t = c / scale.unsqueeze(1)
    else:
        t = c

    return K_fixed, R, t, scale

def _compute_source_from_P(P: torch.Tensor, eps_det: float = 1e-12) -> torch.Tensor:
    """
    English comments only.
    Compute camera center (source) from P = [A|t]:
      C = -inv(A) @ t
    P: (B,3,4)
    Returns: (B,3) source in WORLD coords.
    """
    A = P[:, :, :3]          # (B,3,3)
    t = P[:, :, 3:4]         # (B,3,1)

    detA = torch.linalg.det(A)
    use_pinv = detA.abs() < eps_det

    A_inv = torch.zeros_like(A)
    if use_pinv.any():
        A_inv[use_pinv] = torch.linalg.pinv(A[use_pinv])
    if (~use_pinv).any():
        A_inv[~use_pinv] = torch.linalg.inv(A[~use_pinv])

    C = -torch.bmm(A_inv, t)[:, :, 0]  # (B,3)
    return C

# ============================================================
# Mean-K helper
# ============================================================

@torch.no_grad()
# def compute_mean_K0_from_P(P0: torch.Tensor) -> torch.Tensor:
#     """
#     English comments only.
#     Compute mean intrinsic K from all views.

#     P0: (V,12) or (V,3,4)
#     Return: (3,3) mean K
#     """
#     if P0.ndim == 2 and P0.shape[1] == 12:
#         P0m = P0.view(-1, 3, 4)
#     elif P0.ndim == 3 and P0.shape[1:] == (3, 4):
#         P0m = P0
#     else:
#         raise ValueError("P0 must be (V,12) or (V,3,4)")

#     K_all, _, _ = _pmat_decompose_KRt_batch(P0m.to(torch.float32), eps=1e-8)

#     fu_mean = K_all[:, 0, 0].mean()
#     fv_mean = K_all[:, 1, 1].mean()
#     u0_mean = K_all[:, 0, 2].mean()
#     v0_mean = K_all[:, 1, 2].mean()

#     K_mean = torch.zeros((3, 3), device=P0m.device, dtype=torch.float32)
#     K_mean[0, 0] = fu_mean
#     K_mean[1, 1] = fv_mean
#     K_mean[0, 2] = u0_mean
#     K_mean[1, 2] = v0_mean
#     K_mean[2, 2] = 1.0
#     return K_mean

def compute_mean_K0_from_P(P0: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """
    English comments only.
    Compute mean intrinsic K from all views using all K entries.

    P0: (V,12) or (V,3,4)
    Return: (3,3) mean K
    """
    if P0.ndim == 2 and P0.shape[1] == 12:
        P0m = P0.view(-1, 3, 4)
    elif P0.ndim == 3 and P0.shape[1:] == (3, 4):
        P0m = P0
    else:
        raise ValueError("P0 must be (V,12) or (V,3,4)")

    K_all, _, _ = _pmat_decompose_KRt_batch(P0m.to(torch.float32), eps=1e-8)

    # Mean over all views using all K entries
    K_mean = K_all.mean(dim=0)

    if verbose:
        print("[K_mean from all K entries]")
        print(K_mean.detach().cpu().numpy())

        print("\n[K entry statistics]")
        for i in range(3):
            for j in range(3):
                x = K_all[:, i, j]
                print(
                    f"K[{i},{j}]: "
                    f"mean={x.mean().item():.8f}, "
                    f"std={x.std(unbiased=False).item():.8f}, "
                    f"min={x.min().item():.8f}, "
                    f"max={x.max().item():.8f}"
                )

    return K_mean
# ============================================================
# 6DoF
# ============================================================

def apply_6DoF_transform(
    P0: torch.Tensor,               # (B,12) or (B,3,4)
    geo_old: torch.Tensor,          # (B,7) = [cx,cy,cz,U,V,W,idx]
    tp_internal: torch.Tensor,      # (B,3) object translation in projector INTERNAL coords (your convention)
    rot_internal_deg: torch.Tensor, # (B,3) object rotation in projector INTERNAL coords (your convention)
    *,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    X0: float, Y0: float, Z0: float,
    use_inverse_right_multiply: int = 0,  # keep for compatibility; usually 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comments only.

    Eq.(3)-style 6DoF: build P_new via baseline decomposition and extrinsic update,
    so that the motion parameters are effectively interpreted in detector(camera) coordinates
    after multiplying by E0 = [R0|t0] (world->camera).

    This matches the spirit of Eq.(3): P_motion = K0 * (E0 * Tobj)[:3,:4]
    (No K(ts) and no Tsrc in 6DoF.)
    """
    # ---- Shape checks ----
    if P0.ndim == 2 and P0.shape[1] == 12:
        P0m = P0.view(-1, 3, 4)
    elif P0.ndim == 3 and P0.shape[1:] == (3, 4):
        P0m = P0
    else:
        raise ValueError("P0 must be (B,12) or (B,3,4)")

    if geo_old.ndim != 2 or geo_old.shape[1] < 7:
        raise ValueError("geo_old must be (B,7)")

    B = P0m.shape[0]
    device = P0m.device
    dtype = torch.float32

    # ---- Baseline decomposition: P0 = K0 [R0|t0] ----
    K0, R0, t0 = _pmat_decompose_KRt_batch(P0m.to(dtype), eps=1e-8)

    # ---- INTERNAL -> WORLD (your y-z swap convention) ----
    tp_w = _swap_yz_internal_to_world(tp_internal).to(device=device, dtype=dtype)

    # ---- Center pivot (WORLD) ----
    cx_i = float(X0 + dx * (0.5 * nx))
    cy_i = float(Y0 + dy * (0.5 * ny))
    cz_i = float(Z0 + dz * (0.5 * nz))
    c_world = torch.tensor([cx_i, cz_i, cy_i], device=device, dtype=dtype).view(1, 3).expand(B, 3)

    # ---- Rotation (INTERNAL Euler -> WORLD) ----
    Robj = _rot_from_internal_euler_to_world(rot_internal_deg)  # (B,3,3)

    # ---- Translation with pivot correction ----
    Rc = torch.bmm(Robj, c_world[:, :, None])[:, :, 0]  # (B,3)
    t_eff = tp_w + c_world - Rc                          # (B,3)

    # ---- Choose Tobj or its inverse for right-multiply compatibility ----
    if int(use_inverse_right_multiply) == 1:
        R_use = Robj.transpose(1, 2)
        t_use = -torch.bmm(R_use, t_eff[:, :, None])[:, :, 0]
    else:
        R_use = Robj
        t_use = t_eff

    Tobj = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    Tobj[:, :3, :3] = R_use
    Tobj[:, :3, 3] = t_use
    Tobj[:, 3, 3] = 1.0

    # ---- Build baseline extrinsic E0 and update: E = E0 @ Tobj ----
    E0 = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    E0[:, :3, :3] = R0.to(dtype)
    E0[:, :3, 3]  = t0.to(dtype)
    E0[:, 3, 3]   = 1.0

    E = torch.bmm(E0, Tobj)  # (B,4,4)

    # ---- Rebuild P_new: P = K0 @ E[:3,:4] ----
    P_new = torch.bmm(K0.to(dtype), E[:, :3, :4])  # (B,3,4)
    P_new_flat = P_new.reshape(B, 12)

    # ---- Update geo: source only; keep U,V,W,idx unchanged ----
    source_new = _compute_source_from_P(P_new, eps_det=1e-12)  # (B,3)

    geo_old_f = geo_old.to(device=device, dtype=dtype)
    geo_new = torch.cat([source_new, geo_old_f[:, 3:7]], dim=1).to(geo_old.dtype)

    return P_new_flat, geo_new


def motion6_to_tp_rot(
    p6: torch.Tensor,
    tp_max_mm: float = 50.0,
    rot_max_deg: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Map unconstrained p6 -> bounded (tp, rot) using tanh scaling.
    """
    raw = p6.to(torch.float32)
    tp_raw = raw[..., 0:3]
    rot_raw = raw[..., 3:6]

    tp = tp_max_mm * torch.tanh(tp_raw / tp_max_mm)
    rot_deg = rot_max_deg * torch.tanh(rot_raw / rot_max_deg)
    return tp, rot_deg, raw


# ============================================================
# 9DoF (Eq4) : update source only, keep U/V/W/idx unchanged
# ============================================================

def apply_9DoF_transform(
    P0: torch.Tensor,               # (B,12) or (B,3,4)
    geo_old: torch.Tensor,          # (B,7) = [cx,cy,cz,U,V,W,idx]
    ts_internal: torch.Tensor,      # (B,3)
    tp_internal: torch.Tensor,      # (B,3)
    rot_internal_deg: torch.Tensor, # (B,3)
    *,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    X0: float, Y0: float, Z0: float,
    use_inverse_right_multiply: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comments only.

    Build P_new with Eq4-style construction (same as your previous code),
    but update geo only for source:
      geo_new[:,0:3] = camera center computed from P_new
      geo_new[:,3:7] = geo_old[:,3:7] 그대로 유지 (U,V,w_geo,idx fixed)
    """
    if P0.ndim == 2 and P0.shape[1] == 12:
        P0m = P0.view(-1, 3, 4)
    elif P0.ndim == 3 and P0.shape[1:] == (3, 4):
        P0m = P0
    else:
        raise ValueError("P0 must be (B,12) or (B,3,4)")

    if geo_old.ndim != 2 or geo_old.shape[1] < 7:
        raise ValueError("geo_old must be (B,7)")

    B = P0m.shape[0]
    device = P0m.device
    dtype = torch.float32

    # ---- Baseline decomposition ----
    K0, R0, t0 = _pmat_decompose_KRt_batch(P0m.to(dtype), eps=1e-8)
    ts0_x = K0[:, 0, 2]
    ts0_y = K0[:, 1, 2]
    ts0_z = K0[:, 0, 0]

    # ---- INTERNAL -> WORLD ----
    ts_w = _swap_yz_internal_to_world(ts_internal).to(device=device, dtype=dtype) # source translation
    tp_w = _swap_yz_internal_to_world(tp_internal).to(device=device, dtype=dtype) # object translation

    # ---- Center pivot (WORLD) ----
    cx_i = float(X0 + dx * (0.5 * nx))
    cy_i = float(Y0 + dy * (0.5 * ny))
    cz_i = float(Z0 + dz * (0.5 * nz))
    c_world = torch.tensor([cx_i, cz_i, cy_i], device=device, dtype=dtype).view(1, 3).expand(B, 3)

    Robj = _rot_from_internal_euler_to_world(rot_internal_deg)  # (B,3,3)
    Rc = torch.bmm(Robj, c_world[:, :, None])[:, :, 0]
    t_eff = tp_w + c_world - Rc

    # ---- Tobj selection ----
    if int(use_inverse_right_multiply) == 1:
        R_use = Robj.transpose(1, 2)
        t_use = -torch.bmm(R_use, t_eff[:, :, None])[:, :, 0]
    else:
        R_use = Robj
        t_use = t_eff

    Tobj = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    Tobj[:, :3, :3] = R_use
    Tobj[:, :3, 3] = t_use
    Tobj[:, 3, 3] = 1.0

    # ---- Source motion Tsrc (WORLD) ----
    Tsrc = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    Tsrc[:, :3, :3] = torch.eye(3, device=device, dtype=dtype)[None].expand(B, 3, 3)
    Tsrc[:, :3, 3] = -ts_w
    Tsrc[:, 3, 3] = 1.0

    # ---- ts_world -> camera coords for K(ts) ----
    ts_cam = torch.bmm(R0.to(dtype), ts_w[:, :, None])[:, :, 0]
    
    # ts_abs_x = ts0_x + ts_cam[:, 0] 
    # ts_abs_y = ts0_y + ts_cam[:, 1]
    # ts_abs_z = ts0_z + ts_cam[:, 2]
    
    # SH revised for consistent change between K and Source coordinate varying ts
    ts_abs_x = ts0_x - ts_cam[:, 0]
    ts_abs_y = ts0_y + ts_cam[:, 1]    
    ts_abs_z = ts0_z - ts_cam[:, 2]

    Kts = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    Kts[:, 0, 0] = ts_abs_z # It derived from K[:,0,0]
    Kts[:, 1, 1] = ts_abs_z # It does not preserve original K0[:,1,1], just use same value with K0[:,0,0]
    Kts[:, 0, 2] = ts_abs_x
    Kts[:, 1, 2] = ts_abs_y
    Kts[:, 2, 2] = 1.0
    
    # import pdb
    # pdb.set_trace()

    # ---- Build extrinsic and P_new ----
    E0 = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    E0[:, :3, :3] = R0.to(dtype)
    E0[:, :3, 3] = t0.to(dtype)
    E0[:, 3, 3] = 1.0

    E = torch.bmm(torch.bmm(E0, Tsrc), Tobj) # Tsrc and T obj map to camera coords due to E0
    P_new = torch.bmm(Kts, E[:, :3, :4])  # (B,3,4)
    P_new_flat = P_new.reshape(B, 12)

    # ---- Update source only; keep U,V,W,idx unchanged ----
    source_new = _compute_source_from_P(P_new, eps_det=1e-12)  # (B,3)

    geo_old_f = geo_old.to(device=device, dtype=dtype)
    geo_new = torch.cat([source_new, geo_old_f[:, 3:7]], dim=1).to(geo_old.dtype)

    return P_new_flat, geo_new

def apply_9DoF_transform_effective(
    P0: torch.Tensor,               # (B,12) or (B,3,4)
    geo_old: torch.Tensor,          # (B,7) = [cx,cy,cz,U,V,W,idx]
    ts_internal: torch.Tensor,      # (B,3) = effective source-like K parameters
    tp_internal: torch.Tensor,      # (B,3) = effective rigid translation
    rot_internal_deg: torch.Tensor, # (B,3) = effective rigid rotation
    skew: torch.Tensor,
    *,
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    X0: float, Y0: float, Z0: float,
    use_inverse_right_multiply: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English comments only.

    Effective 9DoF model:
      - ts_internal controls K only (3 DoF):
          ts_x -> principal point u
          ts_y -> principal point v
          ts_z -> shared focal-like scale
      - tp_internal and rot_internal_deg control effective rigid motion only

    Build:
      P_new = Kts @ (E0 @ Tobj)[:3,:4]

    This is NOT the original physical 9DoF Eq.(4) model.
    This is an optimization-friendly effective model with:
      - effective K correction from ts
      - effective rigid extrinsic correction from tp, rot
    """
    # ---- Shape checks ----
    if P0.ndim == 2 and P0.shape[1] == 12:
        P0m = P0.view(-1, 3, 4)
    elif P0.ndim == 3 and P0.shape[1:] == (3, 4):
        P0m = P0
    else:
        raise ValueError("P0 must be (B,12) or (B,3,4)")

    if geo_old.ndim != 2 or geo_old.shape[1] < 7:
        raise ValueError("geo_old must be (B,7)")

    if ts_internal.ndim != 2 or ts_internal.shape[1] != 3:
        raise ValueError("ts_internal must be (B,3)")

    if tp_internal.ndim != 2 or tp_internal.shape[1] != 3:
        raise ValueError("tp_internal must be (B,3)")

    if rot_internal_deg.ndim != 2 or rot_internal_deg.shape[1] != 3:
        raise ValueError("rot_internal_deg must be (B,3)")

    B = P0m.shape[0]
    device = P0m.device
    dtype = torch.float32

    # ---- Baseline decomposition ----
    K0, R0, t0 = _pmat_decompose_KRt_batch(P0m.to(dtype), eps=1e-8)

    # Baseline intrinsic terms
    fu0 = K0[:, 0, 0]
    fv0 = K0[:, 1, 1]
    u00 = K0[:, 0, 2]
    v00 = K0[:, 1, 2]

    # Optional: preserve non-main K entries
    s0 = K0[:, 0, 1]   # skew-like term
    k20 = K0[:, 2, 0]
    k21 = K0[:, 2, 1]
    k22 = K0[:, 2, 2]

    # ---- INTERNAL -> WORLD ----
    ts_w = _swap_yz_internal_to_world(ts_internal).to(device=device, dtype=dtype)
    tp_w = _swap_yz_internal_to_world(tp_internal).to(device=device, dtype=dtype)

    # ---- ts_world for effective K ----
    # We interpret ts as source-like K parameters only.
    # Shared scale from z, principal point shifts from x/y
    u0_new = u00 + ts_w[:, 0]
    v0_new = v00 + ts_w[:, 1]

    # Shared focal-like scale
    # We preserve isotropic scale here: fu_new and fv_new move together.
    scale_shared_u = fu0 + ts_w[:, 2]
    scale_shared_v = fv0 + ts_w[:, 2]

    # Optional: if you want exact common scale from a single baseline,
    # you may replace the two lines above with a shared baseline such as
    # f0 = 0.5 * (fu0 + fv0), then fu_new = fv_new = f0 - ts_cam[:,2].
    fu_new = scale_shared_u
    fv_new = scale_shared_v
    
    # Skew correction
    s0_new = s0 + skew[:, 0]
    
    # ---- Directly assemble effective Kts ----
    Kts = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    Kts[:, 0, 0] = fu_new
    # Kts[:, 0, 1] = s0
    Kts[:, 0, 1] = s0_new
    Kts[:, 0, 2] = u0_new
    Kts[:, 1, 1] = fv_new
    Kts[:, 1, 2] = v0_new
    Kts[:, 2, 0] = k20
    Kts[:, 2, 1] = k21
    Kts[:, 2, 2] = k22

    # import pdb
    # pdb.set_trace()
    # ---- Center pivot (WORLD) ----
    cx_i = float(X0 + dx * (0.5 * nx))
    cy_i = float(Y0 + dy * (0.5 * ny))
    cz_i = float(Z0 + dz * (0.5 * nz))
    c_world = torch.tensor([cx_i, cz_i, cy_i], device=device, dtype=dtype).view(1, 3).expand(B, 3)

    # ---- Rotation (INTERNAL Euler -> WORLD) ----
    Robj = _rot_from_internal_euler_to_world(rot_internal_deg)

    # ---- Translation with pivot correction ----
    Rc = torch.bmm(Robj, c_world[:, :, None])[:, :, 0]
    t_eff = tp_w + c_world - Rc

    # ---- Choose Tobj or its inverse ----
    if int(use_inverse_right_multiply) == 1:
        R_use = Robj.transpose(1, 2)
        t_use = -torch.bmm(R_use, t_eff[:, :, None])[:, :, 0]
    else:
        R_use = Robj
        t_use = t_eff

    Tobj = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    Tobj[:, :3, :3] = R_use
    Tobj[:, :3, 3] = t_use
    Tobj[:, 3, 3] = 1.0

    # ---- Build baseline extrinsic E0 ----
    E0 = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    E0[:, :3, :3] = R0.to(dtype)
    E0[:, :3, 3] = t0.to(dtype)
    E0[:, 3, 3] = 1.0

    # ---- Effective rigid correction only ----
    E = torch.bmm(E0, Tobj)

    # ---- Rebuild P_new ----
    P_new = torch.bmm(Kts, E[:, :3, :4])
    P_new_flat = P_new.reshape(B, 12)

    # ---- Update source only; keep U,V,W,idx unchanged ----
    # This is the effective source implied by final P_new.
    source_new = _compute_source_from_P(P_new, eps_det=1e-12)

    geo_old_f = geo_old.to(device=device, dtype=dtype)
    
    # ---- Update W_geo changed ----
    # c_wgeo = geo_old_f[:, 5] / (fu0 + 1e-8) # Assume different CT device share similar c_wgeo
    # w_geo_new = c_wgeo * fu_new
    # geo_old_f[:, 5] = w_geo_new
    
    geo_new = torch.cat([source_new, geo_old_f[:, 3:7]], dim=1).to(geo_old.dtype)

    return P_new_flat, geo_new

def motion9_to_ts_tp_rot(
    p9: torch.Tensor,
    ts_max_mm: float = 50.0,
    tp_max_mm: float = 50.0,
    rot_max_deg: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Map unconstrained p9 -> bounded (ts,tp,rot) using tanh scaling.
    """
    raw = p9.to(torch.float32)
    ts_raw = raw[..., 0:3]
    tp_raw = raw[..., 3:6]
    rot_raw = raw[..., 6:9]

    # ts = ts_max_mm * torch.tanh(ts_raw / ts_max_mm)
    # tp = tp_max_mm * torch.tanh(tp_raw / tp_max_mm)
    # rot_deg = rot_max_deg * torch.tanh(rot_raw / rot_max_deg)
    
    ts = ts_max_mm * torch.tanh(ts_raw)
    tp = tp_max_mm * torch.tanh(tp_raw)
    rot_deg = rot_max_deg * torch.tanh(rot_raw)

    return ts, tp, rot_deg, raw

def motion10_to_ts_tp_rot_skew(
    p10: torch.Tensor,
    ts_max_mm: float = 50.0,
    tp_max_mm: float = 50.0,
    rot_max_deg: float = 20.0,
    skew_max: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.
    Map unconstrained p10 -> bounded (ts,tp,rot,skew) using tanh scaling.
    """
    raw = p10.to(torch.float32)
    ts_raw = raw[..., 0:3]
    tp_raw = raw[..., 3:6]
    rot_raw = raw[..., 6:9]
    skew_raw = raw[..., 9:10]
    
    ts = ts_max_mm * torch.tanh(ts_raw)
    tp = tp_max_mm * torch.tanh(tp_raw)
    rot_deg = rot_max_deg * torch.tanh(rot_raw)
    skew = skew_max * torch.tanh(skew_raw)

    return ts, tp, rot_deg, skew, raw
