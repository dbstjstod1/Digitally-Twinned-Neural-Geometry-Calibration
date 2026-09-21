"""CPU figures from CQ500 reference and actual reconstructed attenuation arrays."""
from pathlib import Path

import numpy as np


ARMS = ('circular_200', 'circular_220', 'sinespin_220')
LABELS = {'truth': 'CQ500 reference',
          'circular_200': 'Circular 200° / 496 views',
          'circular_220': 'Circular 220° / 546 views',
          'sinespin_220': 'Sine Spin 220° / 546 views'}


def _edges(axis):
    axis = np.asarray(axis)
    if axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0):
        raise ValueError('Plot axes must contain increasing voxel-centre coordinates')
    step = np.diff(axis)
    if not np.allclose(step, step[0]):
        raise ValueError('imshow requires regularly spaced voxel centres')
    return float(axis[0] - step[0]/2), float(axis[-1] + step[-1]/2)


def _occupied_bounds(mask, axes, margin):
    """Full 3D occupied extent, so central-slice crops cannot miss the head ends."""
    bounds = []
    for dim, axis in enumerate(axes):
        hits = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != dim)))
        box = _edges(axis)
        half = (axis[1] - axis[0])/2
        bounds.append((max(box[0], float(axis[hits[0]] - half - margin)),
                       min(box[1], float(axis[hits[-1]] + half + margin))) if len(hits) else box)
    return bounds


def figures(out, truth_mu, recons, masks, axes_mm, mu_water, inputs):
    """Use saved reconstructions directly; masks only modify head_masked.png.

    The representative head_reconstruction.png frames reference HU > -500 plus
    15 mm of margin, not a reconstruction-support crop. Unavailable CT anatomy
    remains absent. Skull-base bounds follow the input's prespecified region.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    z, y, x = [np.asarray(a) for a in axes_mm]
    shape = tuple(len(a) for a in (z, y, x))
    if truth_mu.shape != shape or not np.isfinite(mu_water) or mu_water <= 0:
        raise ValueError('Reference shape or attenuation scale is invalid')
    for name in ARMS:
        if recons[name].shape != shape or masks[name].shape != shape:
            raise ValueError(f'{name}: reconstruction/mask shape differs from reference')
    roi = inputs['fixed_skullbase_region']
    roi_z = np.asarray(roi['z_mm'], dtype=float)
    radius = float(roi['radius_mm'])
    if (roi_z.shape != (2,) or not np.all(np.isfinite(roi_z)) or roi_z[1] <= roi_z[0]
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid prespecified skull-base region')
    xi, yi = int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))
    planes = [(f'Sagittal: x={x[xi]:g} mm', y, lambda a: a[:, :, xi], 'y, posterior [mm]', 1),
              (f'Coronal: y={y[yi]:g} mm', x, lambda a: a[:, yi, :], 'x, left [mm]', 2)]
    names = ('truth',) + ARMS
    # Slice before HU conversion, avoiding four additional full-volume arrays.
    volumes = {'truth': truth_mu, **recons}
    views = [[np.asarray(slicer(volumes[name]))*(1000/mu_water)-1000 for name in names]
             for _, _, slicer, _, _ in planes]
    head_bounds = _occupied_bounds(truth_mu > 0.5*mu_water, (z, y, x), 15.0)
    union = np.logical_or.reduce([masks[name] for name in ARMS])
    detector_bounds = _occupied_bounds(union, (z, y, x), 10.0)
    overview_bounds = [(min(h[0], d[0]), max(h[1], d[1]))
                       for h, d in zip(head_bounds, detector_bounds)]
    del union
    zoom_z = (float(roi_z[0] - 20), float(roi_z[1] + 20))
    zoom_x = (-radius - 10, radius + 10)
    shift = inputs.get('requested_center_shift_xyz_mm', [0, 0, 0])[2]

    def outline(ax, horizontal, visible):
        if np.any(visible) and not np.all(visible):
            ax.contour(horizontal, z, visible, levels=[.5], colors=['#22bbdd'], linewidths=.65)

    for mode in ('reconstruction', 'masked', 'skullbase', 'fov_overview'):
        detail = mode == 'skullbase'
        fig, axs = plt.subplots(2, 4, figsize=(17, 7 if detail else 9), constrained_layout=True)
        for row, (plane, horizontal, slicer, xlabel, dim) in enumerate(planes):
            extent = [*_edges(horizontal), *_edges(z)]
            for col, name in enumerate(names):
                ax = axs[row, col]
                view = views[row][col]
                if mode == 'masked' and name != 'truth':
                    view = np.where(slicer(masks[name]), view, -1000)
                ax.imshow(view, origin='lower', extent=extent, cmap='gray', vmin=-110, vmax=210,
                          interpolation='nearest')
                if name != 'truth' and mode in ('reconstruction', 'fov_overview'):
                    outline(ax, horizontal, slicer(masks[name]))
                ax.axhline(0, color='#ffdf00', ls='--', lw=.7)
                if not detail:
                    ax.axhspan(*roi_z, color='#ff8800', alpha=.055)
                bounds = overview_bounds if mode == 'fov_overview' else head_bounds
                ax.set(title=LABELS[name], xlabel=xlabel, ylabel=plane+'\nz, superior [mm]',
                       xlim=zoom_x if detail else (_edges(horizontal) if mode == 'fov_overview' else bounds[dim]),
                       ylim=zoom_z if detail else bounds[0])
        method = {
            'reconstruction': 'WHOLE ACQUIRED HEAD: reference bounds + margin; raw reconstructions',
            'masked': 'WHOLE HEAD FRAME: display-only every-view detector mask',
            'skullbase': f'SKULL-BASE ZOOM: z={zoom_z[0]:g} to {zoom_z[1]:g} mm; not the full FOV',
            'fov_overview': 'FULL HEAD + DETECTOR EXTENT: raw reconstructions',
        }[mode]
        note = ('cyan: every-view visibility; ' if mode in ('reconstruction', 'fov_overview') else '')
        note += 'yellow: circular source plane; '
        if not detail:
            note += f'orange: ROI z={roi_z[0]:g} to {roi_z[1]:g} mm; '
        fig.suptitle(f'CQ500 | {method}\nActual Joseph LS reconstructions | head shift {shift:+g} mm | '
                     f'HU window [-110, 210]\n{note}visibility is not manufacturer reconstruction support', fontsize=12)
        fig.savefig(out/f'head_{mode}.png', dpi=160)
        if mode == 'reconstruction':
            fig.savefig(out/'head_raw.png', dpi=160)
        plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for row, (plane, horizontal, _, xlabel, _) in enumerate(planes):
        extent = [*_edges(horizontal), *_edges(z)]
        for col, name in enumerate(names):
            ax = axs[row, col]
            is_truth = name == 'truth'
            view = views[row][col] if is_truth else views[row][col] - views[row][0]
            im = ax.imshow(view, origin='lower', extent=extent,
                           cmap='gray' if is_truth else 'RdBu_r',
                           vmin=-110 if is_truth else -100, vmax=210 if is_truth else 100,
                           interpolation='nearest')
            ax.set(title=LABELS[name]+('' if is_truth else ' error'), xlabel=xlabel,
                   ylabel=plane+'\nz, superior [mm]', xlim=zoom_x, ylim=zoom_z)
    fig.colorbar(im, ax=axs[:, 1:], label='Reconstruction minus reference [HU]', shrink=.7)
    fig.suptitle(f'Actual reconstruction errors: skull-base zoom z={zoom_z[0]:g} to {zoom_z[1]:g} mm\n'
                 'Same physical slices and fixed error window; blue indicates negative error')
    fig.savefig(out/'head_skullbase_error.png', dpi=160)
    plt.close(fig)
