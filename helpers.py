"""Raw-volume I/O, gantry export, and scan-geometry helpers."""
import os
from typing import Tuple

import numpy as np


# ---------------------------
# IO helpers
# ---------------------------
def reverse_flag(v: int) -> int:
    # English comments only.
    return 1 if v > 0 else 0


def load_raw_f32_memmap(path: str, shape: tuple) -> np.memmap:
    # English comments only.
    n_elem = int(np.prod(shape))
    file_bytes = os.path.getsize(path)
    expected_bytes = n_elem * 4
    if file_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch: {path}\n"
            f" got={file_bytes} bytes, expected={expected_bytes} bytes, shape={shape}"
        )
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape, order="C")


# ============================================================
# Gantry writer helpers
#   - PI recompute from P
#   - geo recompute like C
#   - BUT: isocenter init can be circlefit (A-option)
# ============================================================

def _pmat_decompose_K_numpy(P33: np.ndarray) -> np.ndarray:
    """
    English comments only.
    MATLAB-like decomposition to obtain K from P33 (3x3).
    """
    P_left = np.flipud(P33).T
    _, R33 = np.linalg.qr(P_left)

    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 0] = R33[2, 2]
    K[0, 1] = R33[1, 2]
    K[0, 2] = R33[0, 2]
    K[1, 0] = R33[2, 1]
    K[1, 1] = R33[1, 1]
    K[1, 2] = R33[0, 1]
    K[2, 0] = R33[2, 0]
    K[2, 1] = R33[1, 0]
    K[2, 2] = R33[0, 0]

    for j in range(3):
        if K[j, j] < 0:
            K[:, j] *= -1.0

    return K


def _recompute_PI_from_P_numpy(P_12: np.ndarray, eps_det: float = 1e-12) -> np.ndarray:
    """
    English comments only.
    Recompute PI (V,12) from P (V,12).
    """
    V = P_12.shape[0]
    PI = np.zeros((V, 12), dtype=np.float64)

    for v in range(V):
        P = P_12[v].reshape(3, 4).astype(np.float64)
        P33 = P[:, :3]

        detA = np.linalg.det(P33)
        if abs(detA) < eps_det:
            P33_inv = np.linalg.pinv(P33)
        else:
            P33_inv = np.linalg.inv(P33)

        PI[v, 0:9] = P33_inv.reshape(-1)

        K = _pmat_decompose_K_numpy(P33)
        PI[v, 9] = K[0, 0]
        PI[v, 10] = K[0, 2]
        PI[v, 11] = K[1, 2]

    return PI.astype(np.float32)


def _project_point_uv_w_numpy(
    P_12: np.ndarray,
    Sxyz: np.ndarray,
    dethalf: float,
    dethalf_V: float,
    pitch: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    English comments only.
    Project Sxyz to all views and compute U,V,W.
    """
    Vn = P_12.shape[0]
    S_h = np.array([Sxyz[0], Sxyz[1], Sxyz[2], 1.0], dtype=np.float64)

    U = np.zeros((Vn,), dtype=np.float64)
    Vv = np.zeros((Vn,), dtype=np.float64)
    W = np.zeros((Vn,), dtype=np.float64)

    for i in range(Vn):
        P = P_12[i].reshape(3, 4).astype(np.float64)
        proj = P @ S_h
        x, y, z = proj[0], proj[1], proj[2]
        U[i] = (dethalf - (x / z)) / pitch
        Vv[i] = (dethalf_V - (y / z)) / pitch
        W[i] = z

    return U, Vv, W


def _std_numpy(x: np.ndarray) -> float:
    """
    English comments only.
    Population std.
    """
    x = x.astype(np.float64)
    m = float(np.mean(x))
    v = float(np.mean((x - m) ** 2))
    return float(np.sqrt(v))


def _spos_calc_numpy(P_12: np.ndarray, eps_det: float = 1e-12) -> np.ndarray:
    """
    English comments only.
    Spos_Calc equivalent: C = -inv(P33)*P31
    """
    Vn = P_12.shape[0]
    C = np.zeros((Vn, 3), dtype=np.float64)

    for k in range(Vn):
        P = P_12[k].reshape(3, 4).astype(np.float64)
        P33 = P[:, :3]
        P31 = P[:, 3:4]

        detA = np.linalg.det(P33)
        if abs(detA) < eps_det:
            P33_inv = np.linalg.pinv(P33)
        else:
            P33_inv = np.linalg.inv(P33)

        C31 = P33_inv @ P31
        C[k, 0] = -C31[0, 0]
        C[k, 1] = -C31[1, 0]
        C[k, 2] = -C31[2, 0]

    return C


# ---------------------------
# A-option: short-scan circle-fitting isocenter init
# ---------------------------

def _circle_fit_kasa_xy_numpy(x: np.ndarray, y: np.ndarray, *, rcond: float = None) -> Tuple[float, float, float]:
    """
    English comments only.
    Kasa circle fit:
      x^2 + y^2 + A x + B y + C = 0
    center = (-A/2, -B/2)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise ValueError("Need at least 3 points for circle fit.")

    D = np.stack([x, y, np.ones_like(x)], axis=1)
    rhs = -(x * x + y * y)

    sol, _, _, _ = np.linalg.lstsq(D, rhs, rcond=rcond)
    A, B, Cc = float(sol[0]), float(sol[1]), float(sol[2])

    a = -0.5 * A
    b = -0.5 * B
    r2 = a * a + b * b - Cc
    r = float(np.sqrt(max(r2, 0.0)))
    return a, b, r


def _isocent_calc_shortscan_circlefit_numpy(
    C: np.ndarray,
    *,
    y_mode: str = "mean",
    outlier_reject: bool = True,
    reject_sigma: float = 3.0,
) -> np.ndarray:
    """
    English comments only.
    Circle fit on XZ plane, and Y from mean/median of cy.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("C must be (V,3).")

    cx = C[:, 0].copy()
    cy = C[:, 1].copy()
    cz = C[:, 2].copy()

    if outlier_reject and C.shape[0] >= 10:
        x0, z0, r0 = _circle_fit_kasa_xy_numpy(cx, cz)
        rr = np.sqrt((cx - x0) ** 2 + (cz - z0) ** 2)
        resid = rr - r0
        m = float(np.mean(resid))
        s = float(np.sqrt(np.mean((resid - m) ** 2)) + 1e-12)

        keep = np.abs(resid - m) <= (reject_sigma * s)
        if np.count_nonzero(keep) >= 3:
            cx, cy, cz = cx[keep], cy[keep], cz[keep]

    X0, Z0, _ = _circle_fit_kasa_xy_numpy(cx, cz)

    if y_mode == "mean":
        Y0 = float(np.mean(cy))
    elif y_mode == "median":
        Y0 = float(np.median(cy))
    else:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    return np.array([X0, Y0, Z0], dtype=np.float64)


# ---------------------------
# Legacy opposite-pair (kept for debugging)
# ---------------------------

def _isocent_calc_numpy(C: np.ndarray) -> np.ndarray:
    """
    English comments only.
    Legacy opposite-pair LS (requires true 180-deg pairs).
    """
    Vn = C.shape[0]
    if (Vn % 2) != 0:
        raise ValueError("IsoCent_Calc assumes even NLAM.")

    loop = Vn // 2
    matsize = Vn * 3 // 2

    cx = C[:, 0]
    cy = C[:, 1]
    cz = C[:, 2]

    A = np.zeros((matsize, 3), dtype=np.float64)
    b = np.zeros((matsize,), dtype=np.float64)

    for i in range(loop):
        dcx = cx[i + loop] - cx[i]
        dcy = cy[i + loop] - cy[i]
        dcz = cz[i + loop] - cz[i]

        A[i, 0] = dcy
        A[i, 1] = -dcx
        A[i, 2] = 0.0
        b[i] = cx[i] * dcy - cy[i] * dcx

        A[i + loop, 0] = dcz
        A[i + loop, 1] = 0.0
        A[i + loop, 2] = -dcx
        b[i + loop] = cx[i] * dcz - cz[i] * dcx

        A[i + 2 * loop, 0] = 0.0
        A[i + 2 * loop, 1] = dcz
        A[i + 2 * loop, 2] = -dcy
        b[i + 2 * loop] = cy[i] * dcz - cz[i] * dcy

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol.astype(np.float64)


def _isocent_optimization_numpy(
    P_12: np.ndarray,
    IsoCent_init: np.ndarray,
    *,
    dethalf: float,
    dethalf_V: float,
    pitch: float,
    iter_num: int = 100,
    step_mm: float = 0.1,
    step_mag: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    English comments only.
    Minimize std(U) by updating Z then X.
    """
    U0, _, _ = _project_point_uv_w_numpy(P_12, IsoCent_init, dethalf, dethalf_V, pitch)
    std0 = _std_numpy(U0)

    Xs = float(IsoCent_init[0])
    Ys = float(IsoCent_init[1])
    Zs = float(IsoCent_init[2])

    # Optimize Z
    step2 = float(step_mm)
    dir2 = 1.0
    prev = None
    for l in range(int(iter_num)):
        Zs = Zs + step2 * dir2
        Uc, _, _ = _project_point_uv_w_numpy(
            P_12,
            np.array([Xs, Ys, Zs], dtype=np.float64),
            dethalf,
            dethalf_V,
            pitch,
        )
        cur = _std_numpy(Uc)

        if (l == 0) and (cur > std0):
            dir2 = -1.0
        if (l > 0) and (prev is not None) and (cur > prev):
            dir2 = -dir2
            step2 = step2 * float(step_mag)
        prev = cur

    # Optimize X
    step1 = float(step_mm)
    dir1 = 1.0
    prev = None
    for l in range(int(iter_num)):
        Xs = Xs + step1 * dir1
        Uc, _, _ = _project_point_uv_w_numpy(
            P_12,
            np.array([Xs, Ys, Zs], dtype=np.float64),
            dethalf,
            dethalf_V,
            pitch,
        )
        cur = _std_numpy(Uc)

        if (l == 0) and (cur > std0):
            dir1 = -1.0
        if (l > 0) and (prev is not None) and (cur > prev):
            dir1 = -dir1
            step1 = step1 * float(step_mag)
        prev = cur

    Iso_opt = np.array([Xs, Ys, Zs], dtype=np.float64)
    U_opt, V_opt, W_arr = _project_point_uv_w_numpy(P_12, Iso_opt, dethalf, dethalf_V, pitch)
    return U_opt, V_opt, W_arr, Iso_opt

def recompute_geo_like_C_numpy(
    P_12: np.ndarray,
    geo_old_7: np.ndarray,
    *,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
    iter_num: int = 100,
    step_mm: float = 0.1,
    step_mag: float = 0.75,
    isocenter_method: str = "circlefit",
    y_mode: str = "mean",
    outlier_reject: bool = True,
) -> np.ndarray:
    """
    English comments only.
    Recompute geo_param:
      - source C from P
      - isocenter init:
          "plane_circlefit" : 3D plane fit -> plane circle fit -> 3D center
          "circlefit"       : XZ-plane circle fit
          "oppositepair"    : legacy opposite-pair LS
      - optimize X/Z to minimize std(U)
      - compute U,V,W from optimized isocenter
    """
    Vn = P_12.shape[0]
    if geo_old_7.shape[0] != Vn or geo_old_7.shape[1] < 7:
        raise ValueError("geo_old_7 must be (V,7) and match P_12.")

    pitch = float(du)
    dethalf = 0.5 * float(nu_ori) * float(du)
    dethalf_V = 0.5 * float(nv_ori) * float(dv)

    C = _spos_calc_numpy(P_12)

    if isocenter_method == "plane_circlefit":
        Iso = _isocent_calc_plane_circlefit_numpy(
            C,
            outlier_reject=outlier_reject,
            reject_sigma=3.0,
        )
    elif isocenter_method == "circlefit":
        Iso = _isocent_calc_shortscan_circlefit_numpy(
            C,
            y_mode=y_mode,
            outlier_reject=outlier_reject,
            reject_sigma=3.0,
        )
    elif isocenter_method == "oppositepair":
        Iso = _isocent_calc_numpy(C)
    else:
        raise ValueError(f"Unsupported isocenter_method: {isocenter_method}")

    U_opt, V_opt, W_arr, _ = _isocent_optimization_numpy(
        P_12,
        Iso,
        dethalf=dethalf,
        dethalf_V=dethalf_V,
        pitch=pitch,
        iter_num=int(iter_num),
        step_mm=float(step_mm),
        step_mag=float(step_mag),
    )

    geo_new = np.zeros((Vn, 7), dtype=np.float64)
    geo_new[:, 0:3] = C
    geo_new[:, 3] = U_opt
    geo_new[:, 4] = V_opt
    geo_new[:, 5] = W_arr
    geo_new[:, 6] = geo_old_7[:, 6]  # keep idx

    return geo_new.astype(np.float32)


def write_gantry_file_33_recompute_PI_and_recompute_geoC(
    out_path: str,
    P_12: np.ndarray,
    geo_old_7: np.ndarray,
    stitch_2: np.ndarray,
    *,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
    iso_iter_num: int = 100,
    iso_step_mm: float = 0.1,
    iso_step_mag: float = 0.75,
    isocenter_method: str = "circlefit",
    y_mode: str = "mean",
    outlier_reject: bool = True,
) -> None:
    """
    English comments only.
    Write 33 tokens/line:
      P(12) + g(6 recomputed) + idx(1) + PI(12 recomputed) + u0 v0 (2)
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    V = P_12.shape[0]

    geo_7 = recompute_geo_like_C_numpy(
        P_12,
        geo_old_7,
        nu_ori=int(nu_ori),
        nv_ori=int(nv_ori),
        du=float(du),
        dv=float(dv),
        iter_num=int(iso_iter_num),
        step_mm=float(iso_step_mm),
        step_mag=float(iso_step_mag),
        isocenter_method=isocenter_method,
        y_mode=y_mode,
        outlier_reject=outlier_reject,
    )

    PI_12 = _recompute_PI_from_P_numpy(P_12)

    # Use CRLF to match legacy Windows gantry files.
    with open(out_path, "w", encoding="ascii", newline="") as f:
        for v in range(V):
            P = P_12[v]
            g = geo_7[v, 0:6]
            idx = int(round(float(geo_7[v, 6])))
            PI = PI_12[v]
            u0 = float(stitch_2[v, 0])
            v0 = float(stitch_2[v, 1])

            tokens = []

            # 12 P tokens
            tokens.extend([f"{float(x):.6f}" for x in P])

            # 6 geo tokens
            tokens.extend([f"{float(x):.6f}" for x in g])

            # 1 idx token as integer string
            tokens.append(str(idx))

            # 12 PI tokens
            tokens.extend([f"{float(x):.6f}" for x in PI])

            # 2 stitch tokens
            tokens.append(f"{u0:.6f}")
            tokens.append(f"{v0:.6f}")

            if len(tokens) != 33:
                raise ValueError(f"Token count mismatch: {len(tokens)} != 33")

            f.write(" ".join(tokens) + "\r\n")


def _fit_orbit_center_plane_circle_3d_numpy(
    C: np.ndarray,
    *,
    outlier_reject: bool = True,
    reject_sigma: float = 3.0,
    return_debug: bool = False,
):
    """
    English comments only.

    Fit a best-fit plane to 3D source positions, project them onto that plane,
    fit a 2D circle in the plane coordinates, and map the center back to 3D.

    Args:
        C:
            (V,3) source positions in 3D.
        outlier_reject:
            If True, reject outliers once using radial residuals in plane-circle fit.
        reject_sigma:
            Sigma threshold for outlier rejection.
        return_debug:
            If True, also return debug information.

    Returns:
        center_3d:
            (3,) fitted orbit center in 3D.
        radius:
            Circle radius in the fitted plane.
        normal:
            (3,) unit normal of the fitted plane.

        If return_debug=True, also returns a dict.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("C must have shape (V,3).")
    if C.shape[0] < 3:
        raise ValueError("Need at least 3 points for 3D plane/circle fitting.")

    # Step 1. Best-fit plane by PCA/SVD
    plane_origin = np.mean(C, axis=0)
    X = C - plane_origin[None, :]

    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    plane_u = Vt[0].copy()
    plane_v = Vt[1].copy()
    normal = Vt[2].copy()

    plane_u /= (np.linalg.norm(plane_u) + 1e-12)
    plane_v /= (np.linalg.norm(plane_v) + 1e-12)
    normal /= (np.linalg.norm(normal) + 1e-12)

    if np.dot(np.cross(plane_u, plane_v), normal) < 0:
        plane_v = -plane_v

    # Step 2. Project to plane coordinates
    x2d = X @ plane_u
    y2d = X @ plane_v

    # Step 3. 2D circle fit
    if outlier_reject and C.shape[0] >= 10:
        cx0, cy0, r0 = _circle_fit_kasa_xy_numpy(x2d, y2d)

        rr = np.sqrt((x2d - cx0) ** 2 + (y2d - cy0) ** 2)
        resid = rr - r0

        m = float(np.mean(resid))
        s = float(np.sqrt(np.mean((resid - m) ** 2)) + 1e-12)

        mask_used = np.abs(resid - m) <= (reject_sigma * s)

        if np.count_nonzero(mask_used) >= 3:
            x_use = x2d[mask_used]
            y_use = y2d[mask_used]
        else:
            mask_used = np.ones_like(x2d, dtype=bool)
            x_use = x2d
            y_use = y2d
    else:
        mask_used = np.ones_like(x2d, dtype=bool)
        x_use = x2d
        y_use = y2d

    cx_2d, cy_2d, radius = _circle_fit_kasa_xy_numpy(x_use, y_use)

    # Step 4. Map back to 3D
    center_3d = plane_origin + cx_2d * plane_u + cy_2d * plane_v

    if not return_debug:
        return center_3d.astype(np.float64), float(radius), normal.astype(np.float64)

    debug = {
        "plane_origin": plane_origin.astype(np.float64),
        "plane_u": plane_u.astype(np.float64),
        "plane_v": plane_v.astype(np.float64),
        "normal": normal.astype(np.float64),
        "points_2d": np.stack([x2d, y2d], axis=1).astype(np.float64),
        "points_2d_used": np.stack([x_use, y_use], axis=1).astype(np.float64),
        "mask_used": mask_used,
        "center_2d": np.array([cx_2d, cy_2d], dtype=np.float64),
        "radius": float(radius),
    }
    return center_3d.astype(np.float64), float(radius), normal.astype(np.float64), debug


def _isocent_calc_plane_circlefit_numpy(
    C: np.ndarray,
    *,
    outlier_reject: bool = True,
    reject_sigma: float = 3.0,
) -> np.ndarray:
    """
    English comments only.

    Estimate 3D isocenter by:
      1) best-fit plane in 3D
      2) circle fit in that plane
      3) map circle center back to 3D
    """
    center_3d, _, _ = _fit_orbit_center_plane_circle_3d_numpy(
        C,
        outlier_reject=outlier_reject,
        reject_sigma=reject_sigma,
        return_debug=False,
    )
    return center_3d.astype(np.float64)

def _decompose_projection_camera_numpy(
    P34: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    English comments only.

    Decompose a 3x4 projection matrix P into normalized K, R, t, source C, and raw projective scale.

    Model:
        P = s * K * [R | t]
    where:
        K[2,2] = 1 after normalization,
        R is a proper rotation,
        t is metric-consistent with K,
        C = -R^T t.

    Returns:
        K_norm:
            (3,3) intrinsic matrix normalized so that K[2,2] = 1.
        R:
            (3,3) proper rotation matrix.
        t:
            (3,) translation vector in camera model Xc = R Xw + t.
        C:
            (3,) source position in world coordinates.
        scale_raw:
            Raw projective scale embedded in P.
    """
    P34 = np.asarray(P34, dtype=np.float64).reshape(3, 4)
    A = P34[:, :3]
    p4 = P34[:, 3]

    # Step 1. Get a raw K-like matrix from the left 3x3 block.
    K_raw = _pmat_decompose_K_numpy(A).astype(np.float64)

    if abs(K_raw[2, 2]) < eps:
        raise ValueError("K_raw[2,2] is too small to normalize.")

    # Normalize K so that K[2,2] = 1.
    scale_raw = float(K_raw[2, 2])
    K_norm = K_raw / scale_raw

    # Step 2. Remove K and estimate the closest rotation.
    A_norm = A / scale_raw
    M = np.linalg.inv(K_norm) @ A_norm

    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        fix = np.eye(3, dtype=np.float64)
        fix[2, 2] = -1.0
        R = U @ fix @ Vt

    # Step 3. Best-fit isotropic scale between M and R.
    denom = float(np.sum(R * R))
    if denom < eps:
        raise ValueError("Rotation fit denominator is too small.")

    scale_rot = float(np.sum(R * M) / denom)
    if abs(scale_rot) < eps:
        raise ValueError("Estimated rotation scale is too small.")

    # Step 4. Translation in camera coordinates.
    c_vec = np.linalg.inv(K_norm) @ (p4 / scale_raw)
    t = c_vec / scale_rot

    # Step 5. Source position in world coordinates.
    C = -R.T @ t

    return K_norm, R, t, C, scale_raw


def _compute_beamcenter_3d_from_P_numpy(
    P34: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    English comments only.

    Compute the true beam center in 3D from a projection matrix.

    Beam center is defined as:
        the foot of the perpendicular from the source to the detector plane.

    In camera coordinates:
        source = (0,0,0)
        detector plane = Z = f
        beam center = (0,0,f)

    Therefore in world coordinates:
        B_world = C + f * n_world
    where:
        C is the source position,
        n_world is the detector normal / central-ray direction in world coordinates,
        f is the focal length from normalized K.

    Returns:
        B_world:
            (3,) beam center 3D point on the detector plane.
        C_world:
            (3,) source position in world coordinates.
        n_world:
            (3,) detector normal / central-ray direction in world coordinates.
        f_mm:
            Focal length in normalized intrinsic coordinates.
        un:
            Principal point u in the same detector-length unit.
        vn:
            Principal point v in the same detector-length unit.
    """
    K, R, t, C, _ = _decompose_projection_camera_numpy(P34, eps=eps)

    fu = float(K[0, 0])
    fv = float(K[1, 1])
    f_mm = 0.5 * (fu + fv)

    un = float(K[0, 2])
    vn = float(K[1, 2])

    # For Xc = R Xw + t, the world direction corresponding to camera +Z
    # is the third row of R (equivalently R^T @ [0,0,1]).
    n_world = R[2, :].copy()
    n_norm = float(np.linalg.norm(n_world))
    if n_norm < eps:
        raise ValueError("Detector normal norm is too small.")
    n_world /= n_norm

    B_world = C + f_mm * n_world

    return B_world, C, n_world, f_mm, un, vn


def _compute_beamcenter_uv_w_numpy(
    P_12: np.ndarray,
    *,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    English comments only.

    Compute source position and true beam center U,V,W for all views.

    U,V are computed by projecting the true 3D beam center point back with the raw P.
    This keeps the result fully consistent with the raw projective scale of P.

    Returns:
        C_all:
            (V,3) source positions.
        U_all:
            (V,) detector u index-like coordinates, same convention as existing code.
        V_all:
            (V,) detector v index-like coordinates, same convention as existing code.
        W_all:
            (V,) third homogeneous component after projecting the 3D beam center with raw P.
    """
    Vn = P_12.shape[0]

    dethalf_u = 0.5 * float(nu_ori) * float(du)
    dethalf_v = 0.5 * float(nv_ori) * float(dv)

    C_all = np.zeros((Vn, 3), dtype=np.float64)
    U_all = np.zeros((Vn,), dtype=np.float64)
    V_all = np.zeros((Vn,), dtype=np.float64)
    W_all = np.zeros((Vn,), dtype=np.float64)

    for i in range(Vn):
        P = P_12[i].reshape(3, 4).astype(np.float64)

        B_world, C_world, _, _, _, _ = _compute_beamcenter_3d_from_P_numpy(P, eps=eps)

        C_all[i, :] = C_world

        Bh = np.array([B_world[0], B_world[1], B_world[2], 1.0], dtype=np.float64)
        proj = P @ Bh
        x, y, z = float(proj[0]), float(proj[1]), float(proj[2])

        if abs(z) < eps:
            raise ValueError(f"Beam center projection z is too small at view {i}.")

        # Keep exactly the same sign/origin convention as the existing code.
        U_all[i] = (dethalf_u - (x / z)) / float(du)
        V_all[i] = (dethalf_v - (y / z)) / float(dv)
        W_all[i] = z

    return C_all, U_all, V_all, W_all


def recompute_geo_like_C_numpy_beamcenter(
    P_12: np.ndarray,
    geo_old_7: np.ndarray,
    *,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
) -> np.ndarray:
    """
    English comments only.

    Recompute geo_param using the true beam center instead of projected isocenter.

    geo_new columns:
      0:3 -> source position Cx,Cy,Cz
      3   -> beam center U
      4   -> beam center V
      5   -> beam center W
      6   -> keep original idx
    """
    Vn = P_12.shape[0]
    if geo_old_7.shape[0] != Vn or geo_old_7.shape[1] < 7:
        raise ValueError("geo_old_7 must be (V,7) and match P_12.")

    C_all, U_all, V_all, W_all = _compute_beamcenter_uv_w_numpy(
        P_12,
        nu_ori=int(nu_ori),
        nv_ori=int(nv_ori),
        du=float(du),
        dv=float(dv),
        eps=1e-12,
    )

    geo_new = np.zeros((Vn, 7), dtype=np.float64)
    geo_new[:, 0:3] = C_all
    geo_new[:, 3] = U_all
    geo_new[:, 4] = V_all
    geo_new[:, 5] = W_all
    geo_new[:, 6] = geo_old_7[:, 6]

    return geo_new.astype(np.float32)
