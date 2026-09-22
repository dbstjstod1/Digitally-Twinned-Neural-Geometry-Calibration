"""Post-hoc raw-9 geometry gradient check on the unmodified small ball phantom.

This diagnostic supplies no training initialization or optimization parameters.
It tests both tilt peaks near, but deliberately away from, the true geometry.
The objective is clean-target normalized squared error accumulated in float64;
forward projection and the actual training parameter chain remain float32.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_sinespin_calibration import apply_motion, geometries, load_volume, project, sha256
from calibration_geometry import centered_source_positions

NAMES = ("ts_x_raw", "ts_y_raw", "ts_z_raw", "tp_x_raw", "tp_y_raw", "tp_z_raw",
         "rot_x_raw", "rot_y_raw", "rot_z_raw")


def vector_comparison(analytic, finite):
    an, fn = np.linalg.norm(analytic), np.linalg.norm(finite)
    return {
        "relative_l2_error": float(np.linalg.norm(analytic-finite)/max(an, fn, 1e-30)),
        "cosine": None if an*fn == 0 else float(np.dot(analytic, finite)/(an*fn)),
        "analytic_l2_norm": float(an), "finite_difference_l2_norm": float(fn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"result_sinespin/ball_calibration/input")
    parser.add_argument("--run-dir", type=Path, default=ROOT/"result_sinespin/ball_calibration/baseline_rot15_seed0")
    parser.add_argument("--output", type=Path, default=ROOT/"result_sinespin/ball_calibration/gradient_audit/gradient_check.json")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--views", type=int, nargs="+", default=[136, 409])
    parser.add_argument("--steps", type=float, nargs=2, default=[.001, .002])
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.gpu != 1 or os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Use physical GPU 1 with CUDA_VISIBLE_DEVICES unset.")
    if any(not np.isfinite(value) or value <= 0 for value in args.steps) or args.steps[0] == args.steps[1]:
        parser.error("Supply two distinct positive finite raw-parameter step sizes.")
    metadata = json.loads((args.input_dir/"experiment.json").read_text())
    experiment = json.loads((args.run_dir/"experiment.json").read_text())
    if experiment["input"] != metadata:
        raise ValueError("Run/input metadata mismatch.")
    for name, expected in experiment["source_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f"Current training source differs from the recorded run: {name}")
    shape, voxel = tuple(metadata["volume"]["shape_zyx"]), metadata["volume"]["voxel_mm"]
    volume_path = Path(metadata["volume"]["path"])
    if sha256(volume_path) != metadata["volume"]["sha256"]:
        raise ValueError("Reference raw changed.")
    if sha256(args.input_dir/"clean_projections.npy") != metadata["clean_sha256"]:
        raise ValueError("Independent clean target changed.")
    circle, sine = geometries(metadata["truth"]["views"], detector_bin=2)
    if [sine.detector_rows, sine.detector_cols] != metadata["truth"]["detector_shape_vu"]:
        raise ValueError("This diagnostic expects the recorded detector bin 2 data.")
    if any(view < 0 or view >= sine.n_views for view in args.views):
        raise ValueError("Diagnostic view is outside the acquisition.")
    bounds = experiment["recipe"]["bounds"]
    nominal_array = np.load(args.input_dir/"P_nominal_world_mm.npy")
    if sha256(args.input_dir/"P_nominal_world_mm.npy") != metadata["nominal_pmat_sha256"]:
        raise ValueError("Circular input matrix changed.")
    clean = np.load(args.input_dir/"clean_projections.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise RuntimeError("Physical GPU 1 is unavailable.")
    torch.cuda.set_device(1)
    device = torch.device("cuda", 1)
    started = time.perf_counter()
    volume = load_volume(volume_path, device, shape)
    cases = []
    for view in args.views:
        # Truth only chooses a post-hoc test location near a noncircular pose.
        # A fixed seeded perturbation in every raw coordinate prevents the
        # zero-gradient perfect-fit case; these values never enter training.
        euler = Rotation.from_rotvec(np.deg2rad(sine.tilt_deg[view])*sine.col_vectors[view]).as_euler("xyz", degrees=True)
        if np.any(np.abs(euler) >= bounds["rot_max_deg"]):
            raise ValueError("True diagnostic tilt exceeds the recorded Euler bounds.")
        raw_numpy = rng.normal(0., .055, size=9)
        raw_numpy[6:] += np.arctanh(euler/bounds["rot_max_deg"])
        raw0 = torch.tensor(raw_numpy[None], dtype=torch.float32, device=device)
        nominal = torch.from_numpy(nominal_array[view:view+1]).to(device)
        target = torch.from_numpy(np.array(clean[view:view+1])).to(device).double()
        energy = target.square().mean()
        if not float(energy) > 0:
            raise ValueError("The fixed target must have nonzero energy.")

        def objective(raw):
            matrix, _ = apply_motion(nominal, raw, bounds, shape=shape, voxel=voxel)
            pred = project(volume, matrix, sine, voxel=voxel)
            return (pred.double()-target).square().mean()/energy

        raw = raw0.clone().requires_grad_(True)
        loss = objective(raw)
        loss.backward()
        analytic = raw.grad.detach().cpu().numpy()[0].astype(np.float64)
        if not np.isfinite(analytic).all() or np.linalg.norm(analytic) == 0:
            raise RuntimeError("The analytic raw gradient is nonfinite or identically zero.")
        with torch.no_grad():
            repeat = float(objective(raw0))
            matrix, physical_motion = apply_motion(nominal, raw0, bounds, shape=shape, voxel=voxel)
        if float(loss.detach()) < 1e-8:
            raise RuntimeError("The diagnostic landed too near a perfect fit.")
        random_directions = rng.normal(size=(2, 9))
        random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
        directions = np.r_[np.eye(9), random_directions]
        scales = []
        with torch.no_grad():
            for step in args.steps:
                derivatives = []
                for direction in directions:
                    offset = torch.tensor(step*direction[None], dtype=torch.float32, device=device)
                    plus, minus = float(objective(raw0+offset)), float(objective(raw0-offset))
                    derivatives.append((plus-minus)/(2*step))
                finite = np.asarray(derivatives[:9])
                basis = []
                for name, an, fd in zip(NAMES, analytic, finite):
                    basis.append(dict(name=name, analytic=float(an), finite_difference=float(fd),
                                      absolute_error=float(abs(an-fd)),
                                      symmetric_relative_error=float(abs(an-fd)/max(abs(an),abs(fd),1e-30))))
                significant = np.abs(analytic) > 1e-3*np.linalg.norm(analytic)
                scales.append(dict(raw_step=step, **vector_comparison(analytic, finite), basis=basis,
                    significant_basis_signs_agree=bool(np.all(np.sign(analytic[significant])==np.sign(finite[significant]))),
                    random_directions=[dict(direction=d.tolist(), analytic=float(analytic@d),
                                            finite_difference=float(fd))
                                       for d,fd in zip(random_directions,derivatives[9:])]))
        fd0 = np.array([entry["finite_difference"] for entry in scales[0]["basis"]])
        fd1 = np.array([entry["finite_difference"] for entry in scales[1]["basis"]])
        consistency = vector_comparison(fd0,fd1)["relative_l2_error"]
        case = dict(view=view, acquisition_tilt_deg=float(sine.tilt_deg[view]),
                    raw_state=raw0.cpu().numpy()[0].tolist(), physical_motion=physical_motion.cpu().numpy()[0].tolist(),
                    derived_source_xyz_mm=centered_source_positions(matrix.cpu().numpy())[0].tolist(),
                    objective_value=float(loss.detach()), objective_repeat_abs_difference=abs(float(loss.detach())-repeat),
                    step_comparison=scales, fd_step_consistency_relative_l2=consistency,
                    both_steps_vector_relative_error_below_5_percent=all(item["relative_l2_error"]<.05 for item in scales),
                    fd_steps_agree_within_5_percent=consistency<.05)
        cases.append(case)
        print(json.dumps({"view":view,"loss":case["objective_value"],
                          "steps":[{"h":item["raw_step"],"rel":item["relative_l2_error"],"cos":item["cosine"]} for item in scales],
                          "step_consistency":consistency}),flush=True)
    torch.cuda.synchronize(device)
    result = dict(
        purpose="Independent post-hoc derivative diagnostic; no parameter or checkpoint selection for training",
        objective="Mean squared projection error / mean clean-target energy; float64 scalar accumulation, actual float32 projector/parameter chain",
        phantom="Unmodified supplied raw; no blurring, thresholding, resizing, or fitted attenuation scale",
        state="Representable sineSpin raw rotation plus PCG64 seed perturbation N(0,.055^2) in all nine raw coordinates; not the perfect fit",
        caveat="Sampled sharp phantom has interpolation kinks. Compare both prescribed steps; disagreement alone does not prove a gradient bug. No best-step selection or global gradient-correctness claim.",
        raw_parameter_names=list(NAMES), bounds=bounds, seed=args.seed, views=args.views, steps=args.steps,
        gpu=1, gpu_name=torch.cuda.get_device_name(1), elapsed_seconds=time.perf_counter()-started,
        input_sha256=sha256(args.input_dir/"experiment.json"), volume_sha256=metadata["volume"]["sha256"],
        clean_sha256=metadata["clean_sha256"], run_source_sha256=experiment["source_sha256"],
        diagnostic_sha256=sha256(Path(__file__).resolve()), cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Wrote",args.output,flush=True)


if __name__ == "__main__":
    main()
