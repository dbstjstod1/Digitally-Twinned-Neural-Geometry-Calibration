"""Historical Denseball LNCC tilt capture range; never supplies training parameters.

At two fixed acquisition views, sweep a known rigid family about the detector
column. This uses truth only to interpret the objective landscape after the
independent projection-only experiment has been defined.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from configs.denseball import VOLUME_PATH
from run_sinespin_calibration import (ROOT, apply_motion,
                                     geometries, load_volume, project, sha256)
from sim_sinespin_recon import save_json

INPUT = ROOT/'result_sinespin/denseball_calibration/input'
LEGACY_SHAPE = (801, 929, 929)
LEGACY_VOXEL_MM = .2


def summarize(angles, values, truth_tilt):
    values = np.asarray(values)
    zero = int(np.argmin(np.abs(angles)))
    best = int(np.argmin(values))
    truth = int(np.argmin(np.abs(angles-truth_tilt)))
    local = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))+1
    return dict(minimum_tilt_deg=float(angles[best]), minimum_loss=float(values[best]),
                zero_tilt_loss=float(values[zero]), nearest_truth_loss=float(values[truth]),
                zero_left_loss=float(values[zero-1]), zero_right_loss=float(values[zero+1]),
                local_minima_tilt_deg=angles[local].tolist(),
                nearest_local_minimum_to_zero_deg=(float(angles[local[np.argmin(np.abs(angles[local]))]])
                                                  if len(local) else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--input-dir', type=Path, default=INPUT)
    p.add_argument('--out-dir', type=Path, default=ROOT/'result_sinespin/denseball_calibration/capture_diagnostic')
    args = p.parse_args()
    if args.gpu != 1:
        p.error('This bounded diagnostic is restricted to physical GPU 1')
    metadata = json.loads((args.input_dir/'experiment.json').read_text())
    if (tuple(metadata['volume']['shape_zyx']) != LEGACY_SHAPE
            or metadata['volume']['voxel_mm'] != LEGACY_VOXEL_MM):
        raise ValueError('This historical diagnostic requires the original (801,929,929), 0.2 mm Denseball data')
    if sha256(VOLUME_PATH) != metadata['volume']['sha256']:
        raise ValueError('Original Denseball reference differs from the prepared experiment')
    import torch
    from monai.losses import LocalNormalizedCrossCorrelationLoss
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    circle, sine = geometries(views=metadata['truth']['views'], detector_bin=2)
    if [sine.detector_rows, sine.detector_cols] != metadata['truth']['detector_shape_vu']:
        raise ValueError('Diagnostic expects the prepared detector bin 2 geometry')
    volume = load_volume(VOLUME_PATH, device, LEGACY_SHAPE)
    target = np.load(args.input_dir/'target_projections.npy', mmap_mode='r')
    nominal = np.load(args.input_dir/'P_nominal_world_mm.npy')
    angles = np.arange(-12., 12.001, .5)
    selected = [int(np.argmax(sine.tilt_deg)), int(np.argmin(sine.tilt_deg))]
    levels = (1, 2, 4, 8)
    loss = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=31,
              kernel_type='rectangular', reduction='none').to(device)
    results = []
    started = time.perf_counter()
    with torch.no_grad():
        for view in selected:
            rotations = Rotation.from_rotvec(np.deg2rad(angles)[:, None]*sine.col_vectors[view])
            euler = rotations.as_euler('xyz', degrees=True)
            if np.any(np.abs(euler) >= 10.):
                raise ValueError('Candidate Euler correction exceeds the original per-axis bounds')
            raw = np.zeros((len(angles), 9), dtype=np.float32)
            raw[:, 6:] = np.arctanh(euler/10.)
            raw = torch.from_numpy(raw).to(device)
            base = torch.from_numpy(nominal[view:view+1]).to(device)
            reference = torch.from_numpy(np.array(target[view])).to(device)
            scores = {str(level): [] for level in levels}
            l2 = []
            for start in range(0, len(angles), 4):
                part = raw[start:start+4]
                matrices, _ = apply_motion(base.expand(len(part), -1, -1), part,
                    dict(ts_max_mm=10., tp_max_mm=10., rot_max_deg=10.),
                    shape=LEGACY_SHAPE, voxel=LEGACY_VOXEL_MM)
                pred = project(volume, matrices, sine, voxel=LEGACY_VOXEL_MM)
                ref = reference[None].expand_as(pred)
                l2.extend(torch.sqrt(((pred-ref)**2).sum((1, 2))/ref.square().sum((1, 2))).cpu().tolist())
                for level in levels:
                    a, b = pred[:, None], ref[:, None]
                    if level > 1:
                        a = torch.nn.functional.avg_pool2d(a, level, level)
                        b = torch.nn.functional.avg_pool2d(b, level, level)
                    values = 1.+loss(a, b).flatten(1).mean(1)
                    scores[str(level)].extend(values.cpu().tolist())
            scores['mean_4_2_1'] = np.mean([scores[str(level)] for level in (4, 2, 1)], axis=0).tolist()
            results.append(dict(view=view, theta_deg=float(sine.theta_deg[view]),
                                truth_tilt_deg=float(sine.tilt_deg[view]), losses=scores,
                                projection_relative_l2=l2,
                                summary={key:summarize(angles, value, sine.tilt_deg[view])
                                         for key, value in scores.items()}))
            print(json.dumps(dict(view=view, summary=results[-1]['summary'])), flush=True)
    torch.cuda.synchronize(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(definition='Evaluation-only one-dimensional rigid tilt sweep; no diagnostic parameters enter training',
                  nominal_tilt_deg=0., candidate_tilt_deg=angles.tolist(), levels=list(levels),
                  phantom='Unmodified original 0.2 mm Denseball volume',
                  loss='1+MONAI squared LNCC31, full detector; average pooling at each level',
                  gpu=1, elapsed_seconds=time.perf_counter()-started,
                  input_metadata_sha256=sha256(args.input_dir/'experiment.json'),
                  script_sha256=sha256(Path(__file__)), views=results)
    save_json(args.out_dir/'metrics.json', record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, item in zip(axes, results):
        for key, values in item['losses'].items():
            ax.plot(angles, values, label='Mean levels 4,2,1' if key=='mean_4_2_1' else f'Level {key}')
        ax.axvline(0., color='black', linestyle=':', label='Circular initialization')
        ax.axvline(item['truth_tilt_deg'], color='green', linestyle='--', label='True tilt')
        ax.set(xlabel='Candidate tilt [degrees]', ylabel='Projection LNCC loss',
               title=f"View {item['view']}; true tilt {item['truth_tilt_deg']:.3f}°")
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    fig.suptitle('Denseball capture diagnostic: fixed rigid tilt family; no training supervision')
    fig.savefig(args.out_dir/'capture_landscape.png', dpi=160)
    plt.close(fig)
    print(f'Saved {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
