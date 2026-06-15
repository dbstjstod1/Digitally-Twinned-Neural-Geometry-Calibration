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
# P decomposition (MATLAB-style)
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
# 9DoF (effective) : update source only, keep U/V/W/idx unchanged
# ============================================================

def apply_9DoF_transform_effective(
    P0: torch.Tensor,               # (B,12) or (B,3,4)
    geo_old: torch.Tensor,          # (B,7) = [cx,cy,cz,U,V,W,idx]
    ts_internal: torch.Tensor,      # (B,3) = effective source-like K parameters
    tp_internal: torch.Tensor,      # (B,3) = effective rigid translation
    rot_internal_deg: torch.Tensor, # (B,3) = effective rigid rotation
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

    Pure 9-DoF: there is no intrinsic skew DoF. The skew-like term K[0,1] is
    fixed to zero in the effective Kts.
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

    # Preserve non-main K entries
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
    fu_new = fu0 + ts_w[:, 2]
    fv_new = fv0 + ts_w[:, 2]

    # ---- Directly assemble effective Kts ----
    # Pure 9-DoF: the skew-like term K[0,1] is fixed to zero.
    Kts = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    Kts[:, 0, 0] = fu_new
    Kts[:, 0, 1] = 0.0
    Kts[:, 0, 2] = u0_new
    Kts[:, 1, 1] = fv_new
    Kts[:, 1, 2] = v0_new
    Kts[:, 2, 0] = k20
    Kts[:, 2, 1] = k21
    Kts[:, 2, 2] = k22

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

    ts = ts_max_mm * torch.tanh(ts_raw)
    tp = tp_max_mm * torch.tanh(tp_raw)
    rot_deg = rot_max_deg * torch.tanh(rot_raw)

    return ts, tp, rot_deg, raw
