"""Post-hoc regional error analysis of the saved Shepp--Logan experiment.

This is a protocol comparison: view count and scan arc differ as well as tilt.
It is neither a noise-power-spectrum measurement nor an isolated tilt ablation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sinespin_geometry import SineSpinGeometry


def analyze(folder, radius_mm=65.0):
    metadata = json.loads((folder / 'metrics.json').read_text())
    gt = np.load(folder / 'ground_truth.npy', mmap_mode='r')
    spacing = metadata['grid']['voxel_mm']
    z, y, x = [(np.arange(n) - (n - 1) / 2) * spacing for n in gt.shape]
    X, Y = np.meshgrid(x, y, indexing='xy')
    radial = X*X + Y*Y <= radius_mm**2
    xy = np.stack((X[radial], Y[radial]), axis=-1)
    intervals = []
    for name, record in metadata['geometry'].items():
        with np.load(folder / (name + '_geometry.npz')) as poses:
            geometry = SineSpinGeometry(
                kind=name, source_positions=poses['source_positions'],
                module_centers=poses['module_centers'], row_vectors=poses['row_vectors'],
                col_vectors=poses['col_vectors'], theta_deg=poses['theta_deg'],
                tilt_deg=poses['tilt_deg'], detector_rows=record['detector_shape_vu'][0],
                detector_cols=record['detector_shape_vu'][1],
                pixel_height=record['pixel_vu_mm'][0], pixel_width=record['pixel_vu_mm'][1],
                sod_mm=record['sod_mm'], sdd_mm=record['sdd_mm'],
                scan_angle_deg=record['scan_angle_deg'])
        intervals.append(geometry.longitudinal_intervals(xy, chunk_size=2048))
    lower = np.maximum.reduce([limits[:, 0] for limits in intervals])
    upper = np.minimum.reduce([limits[:, 1] for limits in intervals])
    truth = np.asarray(gt[:, radial])
    visible = (z[:, None] >= lower) & (z[:, None] <= upper)
    valid = visible & (truth > 0)
    reconstructions = {name: np.load(folder / ('recon_' + name + '.npy'), mmap_mode='r')
                       for name in ('circular', 'sinespin')}
    differences = {name: np.asarray(volume[:, radial]) - truth
                   for name, volume in reconstructions.items()}
    rows = []
    # Show every consecutive slab, including poorly reconstructed end regions.
    for lo in range(-100, 100, 20):
        slab = ((z >= lo) & (z < lo + 20))[:, None]
        mask = slab & valid
        denominator = np.linalg.norm(truth[mask].astype(np.float64))
        errors = {name: float(np.linalg.norm(diff[mask].astype(np.float64)) / denominator)
                  for name, diff in differences.items()}
        rows.append(dict(z_lower_mm=lo, z_upper_mm=lo + 20, voxel_count=int(mask.sum()),
                         relative_rmse=errors,
                         relative_reduction_percent=100*(1-errors['sinespin']/errors['circular'])))
    result = dict(analysis='Post-hoc analysis of existing 160-iteration reconstructions',
                  roi=dict(radius_mm=radius_mm, tissue='GT > 0',
                           visibility='Every view of both protocols', slab_width_mm=20),
                  caveat='Protocol differences include 200/220-degree arcs and 496/546 views; this does not isolate tilt or measure NPS.',
                  slabs=rows)
    (folder / 'sampling_review.json').write_text(json.dumps(result, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    labels = [f"{r['z_lower_mm']}:{r['z_upper_mm']}" for r in rows]
    positions = np.arange(len(rows))
    for ax in axes:
        for offset, (name, color) in zip((-.2, .2), (('circular', '#1565c0'), ('sinespin', '#dd8500'))):
            values = [100*r['relative_rmse'][name] for r in rows]
            ax.bar(positions + offset, values, width=.38, label=name, color=color)
        ax.set_xticks(positions, labels, rotation=45, ha='right')
        ax.set(xlabel='z slab [mm]', ylabel='Relative RMSE [%]')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend()
    axes[0].set_title('All slabs, including end-region errors')
    axes[1].set(xlim=(.5, 8.5), ylim=(0, 4), title='Interior slabs, same measurements')
    fig.suptitle('Shepp–Logan | shared every-view visibility, radius <=65 mm | post-hoc protocol comparison')
    fig.savefig(folder / 'sampling_review.png', dpi=160)
    plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, default=Path('result_sinespin/shepp_logan_fig3'))
    args = parser.parse_args()
    result = analyze(args.folder)
    for row in result['slabs']:
        print(f"z={row['z_lower_mm']:4}:{row['z_upper_mm']:4} mm, "
              f"circular={100*row['relative_rmse']['circular']:.3f}%, "
              f"sinespin={100*row['relative_rmse']['sinespin']:.3f}%")
