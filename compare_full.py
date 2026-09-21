"""Rescore both neural and direct final motions with the same Joseph projector."""

import argparse
import csv
import json
from pathlib import Path

from configs.denseball import DIRECT_OUT_DIR, PROJECTIONS_PATH, VOLUME_PATH, training_dir
from run_single_sample import _latest_ckpt


def _read_metadata(mlp_dir, direct_dir):
    """Require fresh Joseph results before comparing training traces or motions."""
    import torch

    metadata_path = direct_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError("direct results lack Joseph metadata; run and merge a fresh Joseph baseline")
    direct = json.loads(metadata_path.read_text())
    checkpoint = torch.load(_latest_ckpt(mlp_dir), map_location="cpu", weights_only=True)
    if direct.get("projector") != "joseph" or checkpoint.get("cfg", {}).get("projector") != "joseph":
        raise ValueError("both runs must identify Joseph as their training projector")
    ignored = {"n_samples", "chunk_size"}
    for name, value in direct["cfg"].items():
        if name not in ignored and checkpoint["cfg"].get(name) != value:
            raise ValueError(f"training geometries differ at {name}")
    for group in ("nominal_geometry", "motion_bounds"):
        for name, value in direct[group].items():
            if checkpoint.get(name) != value:
                raise ValueError(f"training settings differ at {name}")
    expected_crop = [26, direct["cfg"]["nu"], 0, min(1182, direct["cfg"]["nv"])]
    if direct["loss"] != {"name": "1+LNCC", "kernel_size": 31, "crop_uv": expected_crop}:
        raise ValueError("baseline loss settings differ from the neural training recipe")
    if direct["roi"] != {"x": 0, "y": 0, "width": direct["cfg"]["nu"], "height": direct["cfg"]["nv"]}:
        raise ValueError("comparison requires the full-detector projection ROI")
    if "roi" in checkpoint and checkpoint["roi"] != direct["roi"]:
        raise ValueError("neural and direct projection ROIs differ")
    return direct, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", type=Path, default=training_dir(1))
    parser.add_argument("--direct-dir", type=Path, default=DIRECT_OUT_DIR)
    parser.add_argument("--volume", type=Path, default=VOLUME_PATH)
    parser.add_argument("--projections", type=Path, default=PROJECTIONS_PATH)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    for path in (args.volume, args.projections):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataclasses import fields
    from monai.losses import LocalNormalizedCrossCorrelationLoss
    from AI_Geocal_direct import _build_nominal
    from geometry import ReconConfig, RT_PARAM
    from helpers import load_raw_f32_memmap, reverse_flag
    from DoF_transform import apply_9DoF_transform_effective
    from fast_projectors import sinoproj_joseph

    metadata, checkpoint = _read_metadata(args.mlp_dir, args.direct_dir)
    field_names = {field.name for field in fields(ReconConfig)}
    cfg = ReconConfig(**{key: value for key, value in metadata["cfg"].items() if key in field_names})
    roi = RT_PARAM(**metadata["roi"])
    V = cfg.NLAM
    arrays = []
    for directory in (args.mlp_dir, args.direct_dir):
        motions = [np.load(directory / f"motion_{name}.npy", allow_pickle=False)
                   for name in ("ts_mm", "tp_mm", "rot_deg")]
        if any(value.shape != (V, 3) or not np.isfinite(value).all() for value in motions):
            raise ValueError(f"expected finite ({V}, 3) motion arrays in {directory}")
        arrays.append(motions)
    (m_ts, m_tp, m_rot), (d_ts, d_tp, d_rot) = arrays
    with (args.mlp_dir / "loss_history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    mlp_ep = np.asarray([int(row["epoch"]) for row in rows])
    mlp_curve = np.asarray([float(row["loss"]) for row in rows])
    if not rows or int(checkpoint["epoch"]) != int(mlp_ep[-1]):
        raise ValueError("latest neural checkpoint does not match the completed training history")
    direct_mean_iter = np.load(args.direct_dir / "mean_loss_vs_iter.npy", allow_pickle=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Joseph scoring requires a CUDA GPU")
    device = torch.device("cuda")
    volume = torch.from_numpy(np.asarray(load_raw_f32_memmap(
        args.volume, (cfg.imsz, cfg.imsy, cfg.imsx))).copy()).to(device, torch.float32)
    smat = volume[None, None]
    proj_mm = load_raw_f32_memmap(args.projections, (V, cfg.nv, cfg.nu))
    P_nom, geo_nom, stitch = _build_nominal(cfg, device=device, **metadata["nominal_geometry"])
    urev, vrev = reverse_flag(cfg.ureverse_raw), reverse_flag(cfg.vreverse_raw)
    u0, u1, v0, v1 = metadata["loss"]["crop_uv"]
    lncc = LocalNormalizedCrossCorrelationLoss(
        spatial_dims=2, kernel_size=31, kernel_type="rectangular", reduction="mean").to(device)

    def score_motion(label, motions):
        losses = np.zeros(V, dtype=np.float64)
        ts, tp, rot = [torch.as_tensor(value, device=device) for value in motions]
        with torch.no_grad():
            for start in range(0, V, args.batch_size):
                stop = min(start + args.batch_size, V)
                sl = slice(start, stop)
                P_new, geo_new = apply_9DoF_transform_effective(
                    P0=P_nom[sl], geo_old=geo_nom[sl], ts_internal=ts[sl],
                    tp_internal=tp[sl], rot_internal_deg=rot[sl],
                    nx=cfg.imsx, ny=cfg.imsy, nz=cfg.imsz,
                    dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                    X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, use_inverse_right_multiply=0)
                prediction = sinoproj_joseph(
                    smat=smat, Pmat=P_new, geo_parameter=geo_new, geo_stitch=stitch[sl],
                    nu=cfg.nu, nv=cfg.nv, du=cfg.du, dv=cfg.dv,
                    imsx=cfg.imsx, imsy=cfg.imsy, imsz=cfg.imsz,
                    dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                    X0=cfg.X0, Y0=cfg.Y0, Z0=cfg.Z0, ureverse=urev, vreverse=vrev,
                    roi=roi, recon_type=cfg.recon_type, ori_nu=cfg.ori_nu, ori_nv=cfg.ori_nv)
                target = torch.from_numpy(np.asarray(proj_mm[sl]).copy()).to(device, torch.float32)
                for offset in range(stop - start):
                    losses[start + offset] = 1.0 + float(lncc(
                        prediction[offset:offset+1, None, v0:v1, u0:u1].float(),
                        target[offset:offset+1, None, v0:v1, u0:u1].float()))
                print(f"[{label}] Joseph scoring {stop}/{V}", flush=True)
        return losses

    mlp_final = score_motion("MLP", arrays[0])
    direct_final = score_motion("direct", arrays[1])
    out_dir = args.out_dir or args.direct_dir
    plot_dir = out_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stuck = 0.72
    lines = [f"All {V} final motions rescored with Joseph, 1+LNCC kernel31, crop {metadata['loss']['crop_uv']}",
             f"MLP directory: {args.mlp_dir.resolve()}", f"Direct directory: {args.direct_dir.resolve()}",
             f"Volume: {args.volume.resolve()}", f"Projections: {args.projections.resolve()}",
             "Training traces precede updates; final scores below use the exported final motions.",
             f"{'':14}{'MLP':>12}{'direct':>12}"]
    for label, operation in (("mean loss", np.mean), ("std loss", np.std),
                             ("median loss", np.median), ("max loss", np.max)):
        lines.append(f"{label:14}{operation(mlp_final):12.5f}{operation(direct_final):12.5f}")
    lines.append(f"{'views >0.72':14}{int((mlp_final > stuck).sum()):12d}{int((direct_final > stuck).sum()):12d}")
    lines.append(f"MLP better on {(mlp_final < direct_final).sum()}/{V} views; "
                 f"direct better on {(direct_final < mlp_final).sum()}/{V}")
    for label, directory in (("MLP", args.mlp_dir), ("direct", args.direct_dir)):
        timing = directory / "training_time.txt"
        if timing.is_file():
            lines.append(f"\n[{label} timing]\n{timing.read_text()}")
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "comparison_summary.txt").write_text(summary + "\n")
    np.save(out_dir / "mlp_per_view_loss.npy", mlp_final)
    np.save(out_dir / "direct_per_view_loss.npy", direct_final)

    # ---- plots ----
    # Epochs and per-view iterations have separate axes and are not equal compute.
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(mlp_ep, mlp_curve, color="C0")
    ax[0].set_xlabel("MLP epoch"); ax[0].set_ylabel("mean 1 + LNCC")
    ax[0].set_title("Neural training (Joseph)"); ax[0].grid(alpha=0.3)
    ax[1].plot(np.arange(1, len(direct_mean_iter) + 1), direct_mean_iter, color="C1")
    ax[1].set_xlabel("Direct per-view iteration"); ax[1].set_ylabel("mean 1 + LNCC")
    ax[1].set_title("Direct training (Joseph)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{plot_dir}/convergence.png", dpi=130); plt.close(fig)

    # 2) per-view final loss + histogram
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(mlp_final, label=f"MLP (mean {mlp_final.mean():.4f})", color="C0", lw=0.9)
    ax[0].plot(direct_final, label=f"direct (mean {direct_final.mean():.4f})", color="C1", lw=0.9)
    ax[0].axhline(stuck, color="r", ls="--", lw=0.7, label=f"stuck thr {stuck}")
    ax[0].set_xlabel("view index"); ax[0].set_ylabel("final 1+LNCC")
    ax[0].set_title("Per-view final loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    bins = np.linspace(min(mlp_final.min(), direct_final.min()),
                       max(mlp_final.max(), direct_final.max()), 40)
    ax[1].hist(mlp_final, bins=bins, alpha=0.6, label="MLP", color="C0")
    ax[1].hist(direct_final, bins=bins, alpha=0.6, label="direct", color="C1")
    ax[1].set_xlabel("final 1+LNCC"); ax[1].set_ylabel("# views")
    ax[1].set_title("Final-loss distribution"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{plot_dir}/per_view_loss.png", dpi=130); plt.close(fig)

    # 3) motion params
    names = ["ts (mm)", "tp (mm)", "rot (deg)"]
    comps = ["x/u", "y/v", "z/f"]
    arrs_m = [m_ts, m_tp, m_rot]; arrs_d = [d_ts, d_tp, d_rot]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for r in range(3):
        for c in range(3):
            a = axes[r, c]
            a.plot(arrs_m[r][:, c], color="C0", lw=0.8, label="MLP")
            a.plot(arrs_d[r][:, c], color="C1", lw=0.8, label="direct")
            a.set_title(f"{names[r]} [{comps[c]}]"); a.grid(alpha=0.3)
            if r == 2: a.set_xlabel("view index")
            if r == 0 and c == 0: a.legend(fontsize=8)
    fig.suptitle("Recovered 9-DoF motion parameters: MLP vs direct")
    fig.tight_layout(); fig.savefig(f"{plot_dir}/motion_params.png", dpi=130); plt.close(fig)

    print(f"\n[plots] saved to {plot_dir}/", flush=True)


if __name__ == "__main__":
    main()
