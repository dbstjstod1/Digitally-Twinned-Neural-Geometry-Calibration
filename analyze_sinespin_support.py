"""CPU-only source-plane and detector-support diagnostics; no image masking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from sinespin_geometry import build_icono_orbit


SOURCE_URL = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7837589/'


def halfspace_z_intervals(equations, xy):
    """Intersect n.xyz+b<=0 half-spaces along z; return [point,lower/upper]."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    a = equations[:, 2]
    b = xy @ equations[:, :2].T + equations[:, 3]
    pos, neg, zero = a > 1e-12, a < -1e-12, np.abs(a) <= 1e-12
    lo = np.max(-b[:, neg]/a[neg], axis=1) if neg.any() else np.full(len(xy), -np.inf)
    hi = np.min(-b[:, pos]/a[pos], axis=1) if pos.any() else np.full(len(xy), np.inf)
    result = np.column_stack((lo, hi))
    result[(lo > hi) | np.any(b[:, zero] > 1e-10, axis=1)] = np.nan
    return result


def intersect_intervals(first, second):
    result = np.column_stack((np.maximum(first[:, 0], second[:, 0]),
                              np.minimum(first[:, 1], second[:, 1])))
    result[result[:, 0] > result[:, 1]] = np.nan
    return result


def interval_record(interval):
    return dict(z_lower_mm=float(interval[0]), z_upper_mm=float(interval[1]),
                height_mm=float(interval[1]-interval[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('result_sinespin/cq500_fig9'))
    args = parser.parse_args()
    # A tetrahedron has z in [0,1-x-y]; outside its xy triangle is empty.
    test_eq = ConvexHull(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])).equations
    check = halfspace_z_intervals(test_eq, [[.2,.3], [0,0], [-.1,.5]])
    np.testing.assert_allclose(check[:2], [[0,.5], [0,1]], atol=1e-12)
    assert np.isnan(check[2]).all()
    geometries = dict(circular_200=build_icono_orbit('circular'),
                      circular_220=build_icono_orbit('circular', n_views=546, scan_angle_deg=220),
                      sinespin_220=build_icono_orbit('sinespin'))
    sine = geometries['sinespin_220']
    equations = ConvexHull(sine.source_positions).equations
    names = ('center', 'x+65', 'x-65', 'y+65', 'y-65')
    cardinal = np.array([[0,0], [65,0], [-65,0], [0,65], [0,-65]], dtype=float)
    angles = np.arange(360, dtype=float)
    ring = 65*np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    xy = np.concatenate((cardinal, ring))
    hull = halfspace_z_intervals(equations, xy)
    detector = {name: g.longitudinal_intervals(xy) for name,g in geometries.items()}
    both = intersect_intervals(hull, detector['sinespin_220'])
    points = {name: dict(xy_mm=cardinal[i].tolist(), source_hull=interval_record(hull[i]),
                        hull_and_all_view_visibility=interval_record(both[i]),
                        detector={arm: interval_record(v[i]) for arm,v in detector.items()})
              for i,name in enumerate(names)}
    dense = build_icono_orbit('sinespin', n_views=5461)
    dense_hull = halfspace_z_intervals(ConvexHull(dense.source_positions).equations, xy)
    heights = both[5:, 1]-both[5:, 0]
    result = dict(
        model=dict(sod_mm=750, sdd_mm=1200, distances_assumed=True, sine_phase_deg=0,
                   sine_tilt_amplitude_deg=10, detector_extent_uv_mm=[397.936,292.908],
                   coordinates='LPS-compatible xyz: left, posterior, superior; origin at isocenter',
                   arcs_views={n:[g.scan_angle_deg,g.n_views] for n,g in geometries.items()}),
        diagnostic='Closure of source-hull plane-intersection region, intersected with every-view detector visibility',
        primary_source_url=SOURCE_URL, paper_url='https://doi.org/10.1117/1.JMI.11.4.043503',
        caveats=[
            'Sampled-source hull approximates a continuous connected orbit; finite views do not establish exact completeness.',
            'Hull membership and point visibility do not guarantee exact reconstruction with truncated projections.',
            'All-view visibility is a conservative chosen criterion, not a universal necessary reconstruction condition.',
            'Circular source hulls degenerate to z=0; approximate FDK can produce volumes outside that plane.',
            'This is not the manufacturer Grangeat support rule, not a fitted 120-mm mask, and not an image reconstruction.',
            'No head images are read or masked. LPS sagittal is x=0 (horizontal y); coronal is y=0 (horizontal x).'],
        cardinal_points=points,
        radius65=dict(azimuth_deg=angles.tolist(), source_hull_z_mm=hull[5:].tolist(),
                      intersection_z_mm=both[5:].tolist(), height_mm=heights.tolist(),
                      height_min_mm=float(heights.min()), height_max_mm=float(heights.max()),
                      height_mean_mm=float(heights.mean()), min_azimuth_deg=float(angles[heights.argmin()]),
                      max_azimuth_deg=float(angles[heights.argmax()]),
                      detector_height_statistics={arm: dict(min_mm=float(np.diff(v[5:]).min()),
                          max_mm=float(np.diff(v[5:]).max()), mean_mm=float(np.diff(v[5:]).mean()))
                          for arm,v in detector.items()},
                      intervals_changed_by_detector=int(np.any(np.abs(hull[5:]-both[5:])>1e-8,axis=1).sum())),
        dense_validation=dict(source_count=5461, max_interval_endpoint_change_mm=float(np.max(np.abs(hull-dense_hull)))),
        selfcheck='Analytic unit tetrahedron intervals and empty transverse support passed')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'support_geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    horizontal = np.linspace(-110, 110, 441)
    colors = {'circular_200':'#1565c0', 'circular_220':'#7a399c', 'sinespin_220':'#d97700'}
    for ax, sagittal in zip(axes, (True, False)):
        zeros = np.zeros_like(horizontal)
        points_xy = np.column_stack((zeros,horizontal) if sagittal else (horizontal,zeros))
        for name,g in geometries.items():
            bounds = g.longitudinal_intervals(points_xy)
            style = '--' if name == 'circular_220' else '-'
            ax.plot(horizontal,bounds[:,0],color=colors[name],ls=style,label=name+' all-view detector')
            ax.plot(horizontal,bounds[:,1],color=colors[name],ls=style)
        region = intersect_intervals(halfspace_z_intervals(equations,points_xy),
                                     sine.longitudinal_intervals(points_xy))
        ax.fill_between(horizontal,region[:,0],region[:,1],color='#388e3c',alpha=.22,
                        label='Sine source hull + all-view visibility (diagnostic)')
        ax.axhline(0,color='gray',lw=.6)
        for position in (-65,65): ax.axvline(position,color='gray',lw=.6,ls=':')
        ax.set(title='Sagittal: x=0' if sagittal else 'Coronal: y=0', ylim=(-110,110),
               xlabel='y, posterior [mm]' if sagittal else 'x, left [mm]', ylabel='z, superior [mm]')
        ax.set_aspect('equal'); ax.grid(alpha=.15)
    fig.suptitle('Geometric support diagnostics | assumed SOD/SDD 750/1200 mm\n'
                 'Source-plane condition is not sufficient for truncated data; not a manufacturer mask')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8)
    fig.tight_layout(rect=[0,.11,1,.9])
    fig.savefig(args.out_dir/'support_geometry.png',dpi=160)
    plt.close(fig)
    print('Tetrahedron self-check passed; source-hull/detector intersection heights:')
    for name in names:
        print(f"  {name}: {points[name]['hull_and_all_view_visibility']['height_mm']:.4f} mm")
    print(f"  radius65 range: {heights.min():.4f}..{heights.max():.4f} mm; mean {heights.mean():.4f} mm")
    print(f"  dense-orbit max boundary change: {np.max(np.abs(hull-dense_hull)):.6f} mm")
    print(args.out_dir/'support_geometry.png')


if __name__ == '__main__':
    main()
