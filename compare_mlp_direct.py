"""
Head-to-head: MLP (MotionNetHash) vs direct per-view 9-DoF baseline.

For the views the direct run has finished so far (parsed from its logs), this
recomputes the MLP's per-view loss with the SAME loss / ROI / projector and
prints a side-by-side comparison + summary stats.
"""
import re
import glob
import numpy as np
import torch
from monai.losses import LocalNormalizedCrossCorrelationLoss

from AI_Geocal import ReconConfig, RT_PARAM, build_nominal_orbit_from_geometry
from helpers import load_raw_f32_memmap, reverse_flag
from DoF_transform import apply_9DoF_transform_effective, motion9_to_ts_tp_rot
from differentiable_forward_projector import sinoproj_rdsh_pinv_raycast_dominant
from models.MotionNetHash import MotionNetHash_9DoF

MLP_DIR = "result_denseball/9DoF_analytic_nominalP_10/no_initP_k_un_vn_SOD_SDD_vs1"
CKPT = f"{MLP_DIR}/motion_model_ep0100.pth"
VOLUME = "./open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"
PROJ = "./Denseball_proj_480.raw"
LOGS = ["logs/direct_gpu0.log", "logs/direct_gpu1.log"]


def parse_direct_logs():
    pat = re.compile(r"\[view (\d+)\].*final=([0-9.eE+-]+)")
    out = {}
    for lg in LOGS:
        try:
            with open(lg) as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        out[int(m.group(1))] = float(m.group(2))
        except FileNotFoundError:
            pass
    return out


def main():
    direct = parse_direct_logs()
    views = sorted(direct.keys())
    if not views:
        raise SystemExit("no completed direct views yet")
    print(f"comparing {len(views)} completed views", flush=True)

    device = torch.device("cuda")
    cfg = ReconConfig(
        NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0,
        nu=776, nv=1264, ori_nu=776, ori_nv=1264, du=0.228, dv=0.228,
        imsx=929, imsy=929, imsz=801, dx=0.2, dy=0.2, dz=0.2,
        ureverse_raw=1, vreverse_raw=1, recon_type=1, n_samples=256, chunk_size=16384,
    )
    cfg.X0 = -0.5 * cfg.imsx * cfg.dx; cfg.Y0 = -0.5 * cfg.imsy * cfg.dy; cfg.Z0 = 0.0
    roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)

    vol = torch.from_numpy(np.asarray(
        load_raw_f32_memmap(VOLUME, (cfg.imsz, cfg.imsy, cfg.imsx))).copy()
    ).to(device, torch.float32)
    smat = vol[None, None]
    proj_mm = load_raw_f32_memmap(PROJ, (cfg.NLAM, cfg.nv, cfg.nu))
    urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)

    P_nom, geo_nom, st = build_nominal_orbit_from_geometry(
        n_views=cfg.NLAM, scan_angle_deg=cfg.ScanAngle_deg, start_angle_deg=cfg.StartAngle_deg,
        k_nominal=650.0, un_nominal=34.0, vn_nominal=15.0, SOD=443.0, SDD=650.0,
        nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
        X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, nu_ori=cfg.ori_nu, nv_ori=cfg.ori_nv,
        du=cfg.du, dv=cfg.dv, orbit_axis="y", clockwise_sign=-1.0,
        include_endpoint=False, use_beamcenter_geo=False, device=device, dtype=torch.float32,
    )

    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=2, kernel_size=31, kernel_type="rectangular", reduction="mean").to(device)
    u0, u1, v0, v1 = 26, cfg.nu, 0, 1182

    model = MotionNetHash_9DoF(n_views=cfg.NLAM).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)["model_state"])
    model.eval()

    idx_t = torch.tensor(views, device=device)
    with torch.no_grad():
        p9 = model(idx_t)
        ts, tp, rot, _ = motion9_to_ts_tp_rot(p9, ts_max_mm=10.0, tp_max_mm=10.0, rot_max_deg=10.0)

    mlp_loss = {}
    with torch.no_grad():
        for bi in range(0, len(views), 4):
            sl = slice(bi, bi + 4)
            vv = views[bi:bi + 4]
            P_new, geo_new = apply_9DoF_transform_effective(
                P0=P_nom[idx_t[sl]], geo_old=geo_nom[idx_t[sl]],
                ts_internal=ts[sl], tp_internal=tp[sl], rot_internal_deg=rot[sl],
                nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, use_inverse_right_multiply=0)
            pred = sinoproj_rdsh_pinv_raycast_dominant(
                smat=smat, Pmat=P_new, geo_parameter=geo_new, geo_stitch=st[idx_t[sl]],
                nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv,
                imsx=cfg.imsx, imsy=cfg.imsy, imsz=cfg.imsz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, ureverse=urev, vreverse=vrev, roi=roi,
                recon_type=cfg.recon_type, n_samples=cfg.n_samples, chunk_size=cfg.chunk_size,
                ori_nu=cfg.ori_nu, ori_nv=cfg.ori_nv, align_corners=False)
            tgt = torch.from_numpy(np.asarray(proj_mm[vv]).copy()).to(device, torch.float32)
            for k, v in enumerate(vv):
                l = 1.0 + lncc(pred[k:k+1].unsqueeze(1)[:, :, v0:v1, u0:u1].float(),
                               tgt[k:k+1].unsqueeze(1)[:, :, v0:v1, u0:u1].float())
                mlp_loss[v] = float(l)

    print(f"\n{'view':>5} | {'MLP':>9} | {'direct':>9} | {'Δ(d-MLP)':>9}")
    print("-" * 42)
    dm, dd = [], []
    for v in views:
        m, d = mlp_loss[v], direct[v]
        dm.append(m); dd.append(d)
        print(f"{v:5d} | {m:9.5f} | {d:9.5f} | {d-m:+9.5f}")
    dm, dd = np.array(dm), np.array(dd)
    print("-" * 42)
    print(f"\nSUMMARY over {len(views)} views (same loss/ROI/projector):")
    print(f"  MLP    mean={dm.mean():.5f}  std={dm.std():.5f}  max={dm.max():.5f}")
    print(f"  direct mean={dd.mean():.5f}  std={dd.std():.5f}  max={dd.max():.5f}")
    print(f"  direct better on {(dd < dm).sum()}/{len(views)} views; "
          f"MLP better on {(dm < dd).sum()}/{len(views)}")
    print(f"  direct 'stuck' (>0.72): {(dd > 0.72).sum()} views | "
          f"MLP (>0.72): {(dm > 0.72).sum()} views")


if __name__ == "__main__":
    main()
