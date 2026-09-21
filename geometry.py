"""Shared scan geometry and analytic nominal orbit for calibration and export."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from helpers import recompute_geo_like_C_numpy, recompute_geo_like_C_numpy_beamcenter


@dataclass
class RT_PARAM:
    x: int
    y: int
    width: int
    height: int


@dataclass
class ReconConfig:
    # English comments only.
    NLAM: int = 800
    ScanAngle_deg: float = 360.0
    StartAngle_deg: float = -4.8

    nu: int = 776
    nv: int = 1264
    du: float = 0.228
    dv: float = 0.228

    ori_nu: Optional[int] = None
    ori_nv: Optional[int] = None

    imsx: int = 1002
    imsy: int = 1002
    imsz: int = 982
    dx: float = 0.2
    dy: float = 0.2
    dz: float = 0.2

    # Volume origin offset
    x_pos: float = 0.0
    y_pos: float = 0.0
    z_pos: float = 0.0

    ureverse_raw: int = -1
    vreverse_raw: int = -1
    recon_type: int = 1

    # Legacy fields accepted by existing experiment scripts/checkpoint configs.
    # Joseph samples voxel planes; n_samples and chunk_size have no effect.
    n_samples: int = 128
    chunk_size: int = 8192
    projector: str = "joseph"

    def __post_init__(self):
        if self.projector != "joseph":
            raise ValueError("Only the Joseph projector is supported (projector='joseph').")

    # Computed world origin for voxel grid
    X0: float = 0.0
    Y0: float = 0.0
    Z0: float = 0.0


def build_nominal_orbit_from_geometry(
    *,
    n_views: int,
    scan_angle_deg: float,
    start_angle_deg: float,
    k_nominal: float,
    un_nominal: float,
    vn_nominal: float,
    SOD: float,
    SDD: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    X0: float,
    Y0: float,
    Z0: float,
    nu_ori: int,
    nv_ori: int,
    du: float,
    dv: float,
    orbit_axis: str = "y",
    clockwise_sign: float = -1.0,
    include_endpoint: bool = False,
    use_beamcenter_geo: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English comments only.

    Build nominal projection matrices without using initial P matrices.

    Returns:
        P_nominal_flat:
            (V,12) nominal projection matrices.
        geo_nominal:
            (V,7) geometry parameters.
        geo_stitch:
            (V,2) zero stitch offsets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V = int(n_views)
    orbit_axis = orbit_axis.lower()

    # ------------------------------------------------------------
    # Step 1. Define isocenter in WORLD coordinates.
    # ------------------------------------------------------------
    cx_i = float(X0 + dx * (0.5 * nx))
    cy_i = float(Y0 + dy * (0.5 * ny))

    # Keep the same z0 convention as the existing projector/gantry setup.
    # Do not use Z0 + dz * (0.5 * nz).
    cz_i = float(Z0)

    # Match the existing convention:
    # INTERNAL(x,y,z) = WORLD(x,z,y)
    iso_world = torch.tensor(
        [cx_i, cz_i, cy_i],
        device=device,
        dtype=dtype,
    )

    print("[analytic nominal Pmat center]")
    print(f"  X0, Y0, Z0       = {X0:.6f}, {Y0:.6f}, {Z0:.6f}")
    print(f"  cx_i, cy_i, cz_i = {cx_i:.6f}, {cy_i:.6f}, {cz_i:.6f}")
    print(f"  iso_world        = {iso_world.detach().cpu().numpy()}")

    # ------------------------------------------------------------
    # Step 2. Build angular positions.
    # ------------------------------------------------------------
    if include_endpoint and V > 1:
        step_deg = float(scan_angle_deg) / float(V - 1)
    else:
        step_deg = float(scan_angle_deg) / float(V)

    angles_deg = (
        float(start_angle_deg)
        + torch.arange(V, device=device, dtype=dtype) * (float(clockwise_sign) * step_deg)
    )
    theta = angles_deg * (torch.pi / 180.0)

    # ------------------------------------------------------------
    # Step 3. Build source trajectory.
    # ------------------------------------------------------------
    source = torch.zeros((V, 3), device=device, dtype=dtype)

    if orbit_axis == "y":
        # Source rotates on the XZ plane around WORLD-y axis.
        source[:, 0] = iso_world[0] + float(SOD) * torch.sin(theta)
        source[:, 1] = iso_world[1]
        source[:, 2] = iso_world[2] - float(SOD) * torch.cos(theta)

        up_world = torch.tensor(
            [0.0, 1.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    elif orbit_axis == "x":
        # Source rotates on the YZ plane around WORLD-x axis.
        source[:, 0] = iso_world[0]
        source[:, 1] = iso_world[1] + float(SOD) * torch.sin(theta)
        source[:, 2] = iso_world[2] - float(SOD) * torch.cos(theta)

        up_world = torch.tensor(
            [1.0, 0.0, 0.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    elif orbit_axis == "z":
        # Source rotates on the XY plane around WORLD-z axis.
        source[:, 0] = iso_world[0] + float(SOD) * torch.sin(theta)
        source[:, 1] = iso_world[1] - float(SOD) * torch.cos(theta)
        source[:, 2] = iso_world[2]

        up_world = torch.tensor(
            [0.0, 0.0, 1.0],
            device=device,
            dtype=dtype,
        ).view(1, 3).repeat(V, 1)

    else:
        raise ValueError(f"Unsupported orbit_axis: {orbit_axis}")

    # ------------------------------------------------------------
    # Step 4. Build camera/detector coordinate frame.
    #
    # Convention:
    #   Xc = R Xw + t
    #   source maps to camera origin
    #   central ray direction maps to +Z camera axis
    # ------------------------------------------------------------
    z_cam_world = iso_world.view(1, 3) - source
    z_cam_world = z_cam_world / (
        torch.linalg.norm(z_cam_world, dim=1, keepdim=True) + 1e-12
    )

    x_cam_world = torch.cross(up_world, z_cam_world, dim=1)
    x_cam_world = x_cam_world / (
        torch.linalg.norm(x_cam_world, dim=1, keepdim=True) + 1e-12
    )

    y_cam_world = torch.cross(z_cam_world, x_cam_world, dim=1)
    y_cam_world = y_cam_world / (
        torch.linalg.norm(y_cam_world, dim=1, keepdim=True) + 1e-12
    )

    # Rows of R are camera axes expressed in WORLD coordinates.
    R = torch.stack(
        [x_cam_world, y_cam_world, z_cam_world],
        dim=1,
    )

    # t = -R C
    t = -torch.bmm(R, source[:, :, None])[:, :, 0]

    # ------------------------------------------------------------
    # Step 5. Build nominal intrinsic matrix K.
    # ------------------------------------------------------------
    K = torch.zeros((V, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = float(k_nominal)
    K[:, 1, 1] = float(k_nominal)
    K[:, 0, 2] = float(un_nominal)
    K[:, 1, 2] = float(vn_nominal)
    K[:, 2, 2] = 1.0

    # ------------------------------------------------------------
    # Step 6. Build projection matrices.
    # ------------------------------------------------------------
    E = torch.zeros((V, 3, 4), device=device, dtype=dtype)
    E[:, :, :3] = R
    E[:, :, 3] = t

    P_nominal = torch.bmm(K, E)
    P_nominal_flat = P_nominal.reshape(V, 12)

    # ------------------------------------------------------------
    # Step 7. Recompute geo from generated P.
    # ------------------------------------------------------------
    P_np = P_nominal_flat.detach().cpu().numpy().astype(np.float32)

    geo_old_np = np.zeros((V, 7), dtype=np.float32)
    geo_old_np[:, 6] = np.arange(V, dtype=np.float32)

    if use_beamcenter_geo:
        geo_np = recompute_geo_like_C_numpy_beamcenter(
            P_12=P_np,
            geo_old_7=geo_old_np,
            nu_ori=int(nu_ori),
            nv_ori=int(nv_ori),
            du=float(du),
            dv=float(dv),
        )
    else:
        geo_np = recompute_geo_like_C_numpy(
            P_12=P_np,
            geo_old_7=geo_old_np,
            nu_ori=int(nu_ori),
            nv_ori=int(nv_ori),
            du=float(du),
            dv=float(dv),
            iter_num=100,
            step_mm=0.1,
            step_mag=0.75,
            isocenter_method="circlefit",
            y_mode="mean",
            outlier_reject=False,
        )

    geo_nominal = torch.from_numpy(geo_np).to(device=device, dtype=dtype)

    # No gantry file dependency, so stitch offsets are zero.
    geo_stitch = torch.zeros((V, 2), device=device, dtype=dtype)

    # Keep SDD as explicit input for logging/debugging.
    _ = float(SDD)

    return P_nominal_flat, geo_nominal, geo_stitch
