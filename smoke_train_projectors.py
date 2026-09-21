"""End-to-end smoke of `train_motion_hash_model` with a chosen projector, on the shipped 4T
Denseball case, comparing the per-epoch training loss with the ORIGINAL run's logged history
(result_denseball/.../no_initP_k_un_vn_SOD_SDD_vs8, view_step=8, seed 0, batch 4, lr 1e-3).

    CUDA_VISIBLE_DEVICES=1 python smoke_train_projectors.py --projector joseph --epochs 5
    CUDA_VISIBLE_DEVICES=1 python smoke_train_projectors.py --projector raymarch_triton --epochs 5

A full A/B (the original took 90 min for 100 epochs at view_step=8 on a 3090):

    python smoke_train_projectors.py --projector joseph --epochs 100 --view_step 8
    python smoke_train_projectors.py --projector joseph --epochs 100 --view_step 1   # = vs1
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AI_Geocal import ReconConfig, RT_PARAM, train_motion_hash_model

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--projector", default="joseph")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--view_step", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = ReconConfig(NLAM=480, ScanAngle_deg=360.0, StartAngle_deg=180.0, nu=776, nv=1264,
                  ori_nu=776, ori_nv=1264, du=0.228, dv=0.228, imsx=929, imsy=929, imsz=801,
                  dx=0.2, dy=0.2, dz=0.2, ureverse_raw=1, vreverse_raw=1, recon_type=1,
                  n_samples=256, chunk_size=16384, projector=args.projector)
roi = RT_PARAM(x=0, y=0, width=cfg.nu, height=cfg.nv)
out_dir = args.out or os.path.join(
    HERE, "result_denseball", "9DoF_analytic_nominalP_10",
    f"fastproj_{args.projector}_vs{args.view_step}_ep{args.epochs}")
train_motion_hash_model(
    cfg=cfg,
    volume_path=os.path.join(HERE, "open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_"
                                   "balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw"),
    proj_meas_path=os.path.join(HERE, "Denseball_proj_480.raw"),
    roi=roi, out_dir=out_dir, epochs=args.epochs, batch_size=4, lr=1e-3, use_amp=False,
    seed=0, ts_max_mm=10.0, tp_max_mm=10.0, rot_max_deg=10.0, loss_type="lncc",
    save_every=args.epochs, view_step=args.view_step, k_nominal=650.0, un_nominal=34.0,
    vn_nominal=15.0, SOD=443.0, SDD=650.0, nominal_orbit_axis="y",
    nominal_clockwise_sign=-1.0, nominal_include_endpoint=False, use_beamcenter_geo=False)

ref = os.path.join(HERE, "result_denseball", "9DoF_analytic_nominalP_10",
                   f"no_initP_k_un_vn_SOD_SDD_vs{args.view_step}", "loss_history.csv")
if os.path.exists(ref):
    new = list(csv.DictReader(open(os.path.join(out_dir, "loss_history.csv"))))
    old = list(csv.DictReader(open(ref)))
    print(f"\nper-epoch loss, {args.projector} (this run) vs original ray march (logged, "
          f"RTX 3090):")
    print("  epoch      new_loss  new_s/epoch     orig_loss  orig_s/epoch")
    for a, b in zip(new, old):
        print(f"  {int(a['epoch']):5d}  {float(a['loss']):12.6f}  {float(a['epoch_time_sec']):11.1f}"
              f"  {float(b['loss']):12.6f}  {float(b['epoch_time_sec']):12.1f}")
