"""Fit all nine per-view corrections using the fixed simulation recipe."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .coordinates import pmat_to_pixel
from .pipeline import apply_motion, load_volume, project, sha256


def train(input_dir, volume_path, out_dir, recipe, device):
    import torch
    from types import SimpleNamespace
    from .losses import SignedLNCC
    from .models.MotionNetHash import MotionNetHash_9DoF
    from .roi import ProjectionROI

    meta = json.loads((input_dir / "input.json").read_text())
    shape, voxel = tuple(meta["volume"]["shape_zyx"]), meta["volume"]["voxel_mm"]
    if sha256(volume_path) != meta["volume"]["sha256"]:
        raise ValueError("Reference volume differs from prepared inputs")
    for name in ("target_projections.npy", "P_nominal_world_mm.npy", "loss_roi.json"):
        if sha256(input_dir / name) != meta["sha256"][name]:
            raise ValueError(f"Prepared input changed: {name}")
    g = SimpleNamespace(**meta["detector"])
    volume = load_volume(volume_path, device, shape)
    target = torch.from_numpy(np.load(input_dir / "target_projections.npy")).to(device)
    nominal = torch.from_numpy(np.load(input_dir / "P_nominal_world_mm.npy")).to(device)
    views = len(nominal)
    if tuple(target.shape) != (views, g.detector_rows, g.detector_cols):
        raise ValueError("Projection and geometry dimensions differ")
    roi = ProjectionROI(
        input_dir / "loss_roi.json",
        views=views,
        rows=g.detector_rows,
        cols=g.detector_cols,
        target_sha256=meta["sha256"]["target_projections.npy"],
        device=device,
    )
    if min(roi.width, roi.height) < recipe["kernel_size"]:
        raise ValueError("Loss ROI is smaller than the LNCC window")
    loss_fn = SignedLNCC(recipe["kernel_size"], recipe["smooth_dr"]).to(device)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            dict(
                input_sha256=sha256(input_dir / "input.json"),
                source_sha256={
                    str(p.relative_to(Path(__file__).parent)): sha256(p)
                    for p in Path(__file__).parent.rglob("*.py")
                },
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device),
            ),
            indent=2,
        )
        + "\n"
    )
    train_idx = torch.arange(0, views, recipe["view_step"], device=device)
    all_idx = torch.arange(views, device=device)
    state = None
    history = []
    for stage_id, stage in enumerate(recipe["stages"], 1):
        # Match the original separate runs: reseed, construct the model, then load
        # only learned weights. Adam moments restart at the refinement stage.
        torch.manual_seed(recipe["seed"])
        np.random.seed(recipe["seed"])
        model = MotionNetHash_9DoF(n_views=views, **recipe["hash_grid"]).to(device)
        torch.nn.init.zeros_(model.net.model[-1].weight)
        torch.nn.init.zeros_(model.net.model[-1].bias)
        if state is not None:
            model.load_state_dict(state)
        optimizer = torch.optim.Adam(model.parameters(), lr=stage["lr"])
        for epoch in range(1, stage["epochs"] + 1):
            phase = (epoch - 1) / max(1, stage["epochs"] - 1)
            lr = stage["lr"] * (
                recipe["lr_final_factor"]
                + (1 - recipe["lr_final_factor"]) * (1 + np.cos(np.pi * phase)) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            permutation = train_idx[torch.randperm(len(train_idx), device=device)]
            total = 0.0
            for batch in permutation.split(recipe["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                p, motion = apply_motion(
                    nominal[batch],
                    model(batch),
                    recipe["bounds"],
                    shape=shape,
                    voxel=voxel,
                )
                prediction = project(volume, p, g, voxel=voxel)
                image_loss = roi.loss(loss_fn, prediction, target[batch], batch)
                penalty = (
                    recipe["intrinsic_l1_weight"]
                    * (motion[:, :3] / recipe["intrinsic_scale_mm"]).abs().mean()
                )
                loss = image_loss + penalty
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite calibration loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            history.append(
                dict(stage=stage_id, epoch=epoch, loss=total / len(train_idx), lr=lr)
            )
            with (out_dir / "loss_history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[-1]))
                writer.writeheader()
                writer.writerows(history)
            print(
                f'Stage {stage_id}, epoch {epoch}/{stage["epochs"]}: loss={history[-1]["loss"]:.7f}',
                flush=True,
            )
            if epoch % 10 == 0 or epoch == stage["epochs"]:
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        stage=stage_id,
                        epoch=epoch,
                        recipe=recipe,
                    ),
                    out_dir / f"checkpoint_stage{stage_id}.pt",
                )
        state = model.state_dict()
    model.eval()
    with torch.no_grad():
        p, motion = apply_motion(
            nominal, model(all_idx), recipe["bounds"], shape=shape, voxel=voxel
        )
    np.save(out_dir / "P_proposed_world_mm.npy", p.cpu().numpy())
    np.save(
        out_dir / "P_proposed_pixel.npy",
        pmat_to_pixel(p.cpu().numpy(), du=g.pixel_width, dv=g.pixel_height),
    )
    np.save(out_dir / "motion9.npy", motion.cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("examples/calibration.json")
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    if (
        recipe["batch_size"] < 1
        or recipe["view_step"] < 1
        or not recipe["stages"]
        or any(s["epochs"] < 1 or s["lr"] <= 0 for s in recipe["stages"])
        or recipe["intrinsic_scale_mm"] <= 0
        or recipe["intrinsic_l1_weight"] < 0
    ):
        parser.error("Invalid training configuration")
    import torch

    torch.cuda.set_device(args.gpu)
    train(
        args.input_dir,
        args.volume,
        args.out_dir,
        recipe,
        torch.device("cuda", args.gpu),
    )


if __name__ == "__main__":
    main()
