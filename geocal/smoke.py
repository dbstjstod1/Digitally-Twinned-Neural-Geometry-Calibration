"""Exercise the CUDA calibration workflow on a small generated volume."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .coordinates import geometry_to_pmat, pmat_to_pixel
from .pipeline import apply_motion, project, sha256
from .prepare import save_json
from .sinespin import build_icono_orbit
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import torch

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    z, y, x = np.mgrid[-31:32:2, -31:32:2, -31:32:2]
    centers = np.array([[-11, 8, -9], [12, -7, 10], [3, 12, 9.0]])
    volume = sum(
        0.03 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 18)
        for cx, cy, cz in centers
    ).astype(np.float32)
    volume_path = out / "reference.raw"
    volume.tofile(volume_path)
    tensor = torch.from_numpy(volume).to(device)
    g = build_icono_orbit("sinespin", n_views=8, detector_bin=16)
    nominal = torch.from_numpy(geometry_to_pmat(g)).to(device)
    bounds = dict(ts_max_mm=1.0, tp_max_mm=3.0, rot_max_deg=0.6)
    raw = torch.full((8, 9), 0.03, device=device, requires_grad=True)
    truth, _ = apply_motion(nominal, raw, bounds, shape=volume.shape, voxel=2.0)
    projection = project(tensor, truth, g, voxel=2.0)
    (gradient,) = torch.autograd.grad(projection.square().mean(), raw)
    if not bool(torch.isfinite(gradient).all()) or not bool(
        (gradient.abs().sum(0) > 0).all()
    ):
        raise RuntimeError("Missing or nonfinite gradient in the nine-parameter chain")
    inp = out / "input"
    inp.mkdir()
    np.save(inp / "target_projections.npy", projection.detach().cpu().numpy())
    for name, p in [("nominal", nominal), ("truth", truth)]:
        array = p.detach().cpu().numpy()
        np.save(inp / f"P_{name}_world_mm.npy", array)
        np.save(
            inp / f"P_{name}_pixel.npy",
            pmat_to_pixel(array, du=g.pixel_width, dv=g.pixel_height),
        )
    save_json(
        inp / "landmarks.json",
        dict(landmarks=[dict(xyz_mm=c.tolist()) for c in centers]),
    )
    save_json(
        inp / "loss_roi.json",
        dict(
            definition="Full detector for software smoke test",
            target_sha256=sha256(inp / "target_projections.npy"),
            detector_shape_vu=[g.detector_rows, g.detector_cols],
            boxes_xyxy=[[0, 0, g.detector_cols, g.detector_rows]] * 8,
        ),
    )
    save_json(
        inp / "input.json",
        dict(
            volume=dict(
                shape_zyx=list(volume.shape), voxel_mm=2.0, sha256=sha256(volume_path)
            ),
            detector=dict(
                detector_rows=g.detector_rows,
                detector_cols=g.detector_cols,
                pixel_width=g.pixel_width,
                pixel_height=g.pixel_height,
            ),
            sha256={p.name: sha256(p) for p in inp.iterdir()},
        ),
    )
    recipe = dict(
        seed=1,
        batch_size=4,
        view_step=2,
        bounds=bounds,
        hash_grid=dict(
            n_levels=8,
            n_features_per_level=2,
            log2_hashmap_size=15,
            base_resolution=16,
            per_level_scale=(128 / 16) ** (1 / 7),
        ),
        kernel_size=31,
        smooth_dr=1e-5,
        intrinsic_l1_weight=0.01,
        intrinsic_scale_mm=1.0,
        lr_final_factor=0.05,
        stages=[dict(epochs=2, lr=0.003), dict(epochs=2, lr=0.0003)],
    )
    train(inp, volume_path, out / "proposed", recipe, device)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "geocal.evaluate",
            "--input-dir",
            str(inp),
            "--run-dir",
            str(out / "proposed"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    p = np.load(out / "proposed/P_proposed_world_mm.npy")
    if p.shape != (8, 3, 4) or not np.isfinite(p).all():
        raise RuntimeError("Invalid exported projection matrices")
    report = dict(
        passed=True,
        views=8,
        parameters_per_view=9,
        gradient_l1_per_parameter=gradient.abs().sum(0).cpu().tolist(),
        scope="Generated-volume software check; not paper validation",
    )
    save_json(out / "smoke.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
