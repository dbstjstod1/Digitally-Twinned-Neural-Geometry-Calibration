"""Shared 4T Denseball geometry and the recorded research recipes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME_PATH = ROOT / (
    "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_"
    "balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"
)
PROJECTIONS_PATH = ROOT / "Denseball_proj_480.raw"
DIRECT_OUT_DIR = ROOT / "result_denseball" / "joseph_direct_vs1"
NOMINAL_GEOMETRY = dict(
    k_nominal=650.0, un_nominal=34.0, vn_nominal=15.0,
    SOD=443.0, SDD=650.0, nominal_orbit_axis="y",
    nominal_clockwise_sign=-1.0, nominal_include_endpoint=False,
    use_beamcenter_geo=False,
)
MOTION_BOUNDS = dict(ts_max_mm=10.0, tp_max_mm=10.0, rot_max_deg=10.0)
TRAINING_RECIPE = dict(
    epochs=100, batch_size=4, lr=1e-3, use_amp=False, seed=0,
    loss_type="lncc", save_every=5,
)
DIRECT_RECIPE = dict(n_iters=150, lr=5e-2, seed=0)


def make_config():
    from geometry import ReconConfig

    return ReconConfig(
        NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0,
        nu=776, nv=1264, ori_nu=776, ori_nv=1264,
        du=0.228, dv=0.228, imsx=929, imsy=929, imsz=801,
        dx=0.2, dy=0.2, dz=0.2,
        ureverse_raw=1, vreverse_raw=1, recon_type=1,
    )


def make_roi(cfg):
    from geometry import RT_PARAM

    return RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)


def training_dir(view_step=1):
    return ROOT / "result_denseball" / f"joseph_mlp_vs{view_step}"
