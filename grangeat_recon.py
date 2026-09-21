"""Direct numerical inversion of Grangeat's first-derivative Radon relation.

Reference: Grangeat (1991), doi:10.1007/BFb0084509, equations (2.7),
(3.1), (3.3), (3.4), and (4.1). No iterative solver, fitted image scale,
learned redundancy, or FDK approximation is used here.

The continuous inversion is exact with complete plane data. Finite samples,
finite detector extent, differentiation and interpolation introduce errors.
Missing planes are explicitly tracked; a partial angular integral is NOT an
exact reconstruction. A detector-edge check detects truncation but cannot
certify absence of unknown anatomy outside a real detector.
"""
from __future__ import annotations

import os
import time
import numpy as np


def hemisphere_quadrature(n_polar=64, n_azimuth=256):
    """Gauss-Legendre in n_z in [0,1], periodic azimuth; weights sum to 2*pi."""
    if n_polar < 2 or n_azimuth < 4:
        raise ValueError('Require at least 2 polar and 4 azimuth samples')
    nodes, weights = np.polynomial.legendre.leggauss(n_polar)
    nz = (nodes+1)/2
    az = 2*np.pi*np.arange(n_azimuth)/n_azimuth
    r = np.sqrt(1-nz*nz)
    normals = np.stack((r[:, None]*np.cos(az), r[:, None]*np.sin(az),
                        np.broadcast_to(nz[:, None], (n_polar, n_azimuth))), -1).reshape(-1, 3)
    area = np.repeat(weights/2, n_azimuth)*(2*np.pi/n_azimuth)
    return normals, area


def source_hull_mask(source_positions, shape, voxel):
    """Necessary plane-intersection condition for the sampled open source curve.

    The hull does not assert detector completeness. Interpolating consecutive
    sources describes a connected polygonal path; its projection onto any
    normal has range [min n.S, max n.S], without closing its endpoints.
    """
    from scipy.spatial import ConvexHull
    sources = np.asarray(source_positions)
    if np.linalg.matrix_rank(sources-sources.mean(0)) < 3:
        return np.zeros(shape, dtype=bool)
    equations = ConvexHull(sources).equations
    z, y, x = [(np.arange(n)-(n-1)/2)*voxel for n in shape]
    X, Y = np.meshgrid(x, y, indexing='xy')
    xy = np.stack((X.ravel(), Y.ravel()), axis=-1)
    limits = np.empty((len(xy), 2))
    a = equations[:, 2]
    positive, negative, zero = a > 1e-12, a < -1e-12, np.abs(a) <= 1e-12
    for start in range(0, len(xy), 2048):
        b = xy[start:start+2048]@equations[:, :2].T+equations[:, 3]
        lo = np.max(-b[:, negative]/a[negative], axis=1)
        hi = np.min(-b[:, positive]/a[positive], axis=1)
        outside = np.any(b[:, zero] >= 0, axis=1)
        lo[outside], hi[outside] = np.inf, -np.inf
        limits[start:start+2048] = np.stack((lo, hi), axis=-1)
    limits = limits.reshape(len(y), len(x), 2)
    return (z[:, None, None] > limits[None, :, :, 0]+1e-7) & (z[:, None, None] < limits[None, :, :, 1]-1e-7)


def detector_derivatives(projections, geometry):
    """Cosine preweight and physical-mm detector gradients, Eq. (4.1)."""
    g = geometry
    a = np.asarray(projections, dtype=np.float32)
    if a.shape != (g.n_views, g.detector_rows, g.detector_cols) or not np.isfinite(a).all():
        raise ValueError('Projection shape or values are invalid')
    u = (np.arange(g.detector_cols)-(g.detector_cols-1)/2)*g.pixel_width
    v = (np.arange(g.detector_rows)-(g.detector_rows-1)/2)*g.pixel_height
    cosine = g.sdd_mm/np.sqrt(g.sdd_mm**2+v[:, None]**2+u[None, :]**2)
    # Zero continuation is used ONLY for the separately labelled truncated-data
    # estimate. Padding includes its boundary derivative; simply dropping that
    # derivative would not be the derivative of the detector Radon transform.
    weighted = np.pad(a*cosine.astype(np.float32)[None], ((0, 0), (2, 2), (2, 2)))
    dv, du = np.gradient(weighted, g.pixel_height, g.pixel_width, axis=(1, 2), edge_order=1)
    return weighted, np.ascontiguousarray(du), np.ascontiguousarray(dv)


def grangeat_view_numpy(projection, geometry, view, normals, line_step_mm=None):
    """CPU reference: line integral of detector gradient / h^2 for one view."""
    from scipy.ndimage import map_coordinates
    g = geometry
    if np.shape(projection) != (g.detector_rows, g.detector_cols):
        raise ValueError('Projection shape differs from detector')
    u = (np.arange(g.detector_cols)-(g.detector_cols-1)/2)*g.pixel_width
    v = (np.arange(g.detector_rows)-(g.detector_rows-1)/2)*g.pixel_height
    w = np.pad(np.asarray(projection)*g.sdd_mm/np.sqrt(g.sdd_mm**2+v[:, None]**2+u[None, :]**2), 2)
    dv, du = np.gradient(w, g.pixel_height, g.pixel_width, edge_order=1)
    step = min(g.pixel_width, g.pixel_height) if line_step_mm is None else float(line_step_mm)
    if step <= 0:
        raise ValueError('Line quadrature step must be positive')
    length = np.hypot(g.detector_width_mm+4*g.pixel_width, g.detector_height_mm+4*g.pixel_height)
    count = int(np.ceil(length/step)); step = length/count
    t = (np.arange(count)-(count-1)/2)*step
    normals = np.asarray(normals)
    result = np.full(len(normals), np.nan)
    for i, normal in enumerate(normals):
        a, b, c = normal@g.col_vectors[view], normal@g.row_vectors[view], normal@g.normals[view]
        h = np.hypot(a, b)
        if h < 1e-8:
            continue
        ca, sa = a/h, b/h
        s = -g.sdd_mm*c/h
        uu, vv = ca*s-sa*t, sa*s+ca*t
        ij = np.stack((vv/g.pixel_height+(g.detector_rows+3)/2,
                       uu/g.pixel_width+(g.detector_cols+3)/2))
        if not np.any((ij[0]>=2)&(ij[0]<=g.detector_rows+1)&(ij[1]>=2)&(ij[1]<=g.detector_cols+1)):
            continue
        result[i] = step*np.sum(ca*map_coordinates(du, ij, order=1, mode='constant', cval=0, prefilter=False)
                               +sa*map_coordinates(dv, ij, order=1, mode='constant', cval=0, prefilter=False))/h**2
    return result


def _select_cuda(gpu):
    from numba import cuda
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        devices = visible.split(',')
        if str(gpu) not in devices:
            raise ValueError(f'Physical GPU {gpu} is not in CUDA_VISIBLE_DEVICES={visible}')
        device = devices.index(str(gpu))
    else:
        device = gpu
    cuda.select_device(device)
    return cuda


def _cuda_kernels():
    """Lazy CUDA import keeps mathematical helpers and CPU tests GPU-free."""
    from numba import cuda
    import math

    @cuda.jit(device=True)
    def sample(a, view, row, col):
        nr, nc = a.shape[1], a.shape[2]
        if row < 0 or col < 0 or row > nr-1 or col > nc-1:
            return 0.0
        ir, ic = min(int(row), nr-2), min(int(col), nc-2)
        fr, fc = row-ir, col-ic
        return ((1-fr)*((1-fc)*a[view, ir, ic]+fc*a[view, ir, ic+1])
                +fr*((1-fc)*a[view, ir+1, ic]+fc*a[view, ir+1, ic+1]))

    @cuda.jit
    def intermediate(du, dv, weighted, normals, cols, rows, central, D, pu, pv,
                     line_count, step, edge_tol, values, complete):
        q = cuda.grid(1)
        nv = cols.shape[0]
        if q >= normals.shape[0]*nv:
            return
        i, view = q//nv, q % nv
        a = normals[i, 0]*cols[view, 0]+normals[i, 1]*cols[view, 1]+normals[i, 2]*cols[view, 2]
        b = normals[i, 0]*rows[view, 0]+normals[i, 1]*rows[view, 1]+normals[i, 2]*rows[view, 2]
        c = normals[i, 0]*central[view, 0]+normals[i, 1]*central[view, 1]+normals[i, 2]*central[view, 2]
        h = math.sqrt(a*a+b*b)
        values[i, view] = math.nan
        complete[i, view] = False
        if h < 1e-8:
            return
        ca, sa = a/h, b/h
        s = -D*c/h
        nr, nc = du.shape[1], du.shape[2]
        total, hits, truncated = 0.0, 0, False
        for j in range(line_count):
            t = (j-(line_count-1)/2)*step
            col = (ca*s-sa*t)/pu+(nc-1)/2
            row = (sa*s+ca*t)/pv+(nr-1)/2
            if 0 <= col <= nc-1 and 0 <= row <= nr-1:
                if 2 <= col <= nc-3 and 2 <= row <= nr-3:
                    hits += 1
                total += ca*sample(du, view, row, col)+sa*sample(dv, view, row, col)
                if col < 3.5 or col > nc-4.5 or row < 3.5 or row > nr-4.5:
                    if abs(sample(weighted, view, row, col)) > edge_tol:
                        truncated = True
        if hits:
            values[i, view] = total*step/(h*h)
            complete[i, view] = not truncated

    @cuda.jit
    def inverse(rpp, valid, normals, weights, p0, dp, voxel, nz, ny, nx, out, coverage, all_present):
        index = cuda.grid(1)
        if index >= out.size:
            return
        iz, rem = index//(ny*nx), index % (ny*nx)
        iy, ix = rem//nx, rem % nx
        x, y, z = (ix-(nx-1)/2)*voxel, (iy-(ny-1)/2)*voxel, (iz-(nz-1)/2)*voxel
        total, area, all_area = 0.0, 0.0, 0.0
        present = 0
        for i in range(normals.shape[0]):
            coord = (normals[i, 0]*x+normals[i, 1]*y+normals[i, 2]*z-p0)/dp
            ip = int(math.floor(coord))
            all_area += weights[i]
            if 0 <= ip < rpp.shape[1]-1 and valid[i, ip] and valid[i, ip+1]:
                fraction = coord-ip
                total += weights[i]*((1-fraction)*rpp[i, ip]+fraction*rpp[i, ip+1])
                area += weights[i]
                present += 1
        out[index] = -total/(4*math.pi*math.pi)
        coverage[index] = area/all_area
        all_present[index] = present == normals.shape[0]

    @cuda.jit
    def plane_rebin(du, dv, weighted, normals, source_p, cols, central, p0, dp,
                    D, pu, pv, line_count, step, edge_tol, measured, complete):
        q = cuda.grid(1)
        np_offset = measured.shape[1]
        if q >= measured.size:
            return
        i, ip = q//np_offset, q % np_offset
        p = p0+ip*dp
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
        nr, nc = du.shape[1], du.shape[2]
        total, complete_total = 0.0, 0.0
        crossings, complete_crossings = 0, 0
        for view in range(source_p.shape[1]-1):
            p_left, p_right = source_p[i, view], source_p[i, view+1]
            if abs(p_right-p_left) < 1e-8:
                continue
            f = (p-p_left)/(p_right-p_left)
            # Half-open segments avoid double-counting a shared source vertex.
            if f < 0 or f > 1 or (f == 1 and view < source_p.shape[1]-2):
                continue
            # Interpolate the detector pose and re-orthogonalise. The source
            # crossing uses the connected polygonal path, never an endpoint wrap.
            ex = (1-f)*central[view, 0]+f*central[view+1, 0]
            ey = (1-f)*central[view, 1]+f*central[view+1, 1]
            ez = (1-f)*central[view, 2]+f*central[view+1, 2]
            length = math.sqrt(ex*ex+ey*ey+ez*ez)
            ex, ey, ez = ex/length, ey/length, ez/length
            ux = (1-f)*cols[view, 0]+f*cols[view+1, 0]
            uy = (1-f)*cols[view, 1]+f*cols[view+1, 1]
            uz = (1-f)*cols[view, 2]+f*cols[view+1, 2]
            dot = ux*ex+uy*ey+uz*ez
            ux, uy, uz = ux-dot*ex, uy-dot*ey, uz-dot*ez
            length = math.sqrt(ux*ux+uy*uy+uz*uz)
            ux, uy, uz = ux/length, uy/length, uz/length
            # row = col cross central, consistent with SineSpinGeometry.
            vx, vy, vz = uy*ez-uz*ey, uz*ex-ux*ez, ux*ey-uy*ex
            a, b, c = nx*ux+ny*uy+nz*uz, nx*vx+ny*vy+nz*vz, nx*ex+ny*ey+nz*ez
            h = math.sqrt(a*a+b*b)
            if h < 1e-8:
                continue
            ca, sa, s = a/h, b/h, -D*c/h
            line, hits, truncated = 0.0, 0, False
            for j in range(line_count):
                t = (j-(line_count-1)/2)*step
                col = (ca*s-sa*t)/pu+(nc-1)/2
                row = (sa*s+ca*t)/pv+(nr-1)/2
                if 0 <= col <= nc-1 and 0 <= row <= nr-1:
                    if 2 <= col <= nc-3 and 2 <= row <= nr-3:
                        hits += 1
                    d_left = ca*sample(du, view, row, col)+sa*sample(dv, view, row, col)
                    d_right = ca*sample(du, view+1, row, col)+sa*sample(dv, view+1, row, col)
                    line += (1-f)*d_left+f*d_right
                    if col < 3.5 or col > nc-4.5 or row < 3.5 or row > nr-4.5:
                        if ((f < 1 and abs(sample(weighted, view, row, col)) > edge_tol) or
                                (f > 0 and abs(sample(weighted, view+1, row, col)) > edge_tol)):
                            truncated = True
            if hits:
                value = line*step/(h*h)
                total += value
                crossings += 1
                if not truncated:
                    complete_total += value
                    complete_crossings += 1
        measured[i, ip] = total/crossings if crossings else math.nan
        complete[i, ip] = complete_total/complete_crossings if complete_crossings else math.nan
    return intermediate, inverse, plane_rebin


def grangeat_derivatives_gpu(projections, geometry, normals, *, line_step_mm=None,
                             edge_tolerance=1e-5, gpu=1):
    """Return R'(n,n.S_view) and detector-edge completeness diagnostic."""
    cuda = _select_cuda(gpu)
    kernel, _, _ = _cuda_kernels()
    weighted, du, dv = detector_derivatives(projections, geometry)
    g = geometry
    step = min(g.pixel_width, g.pixel_height) if line_step_mm is None else float(line_step_mm)
    if step <= 0:
        raise ValueError('Line quadrature step must be positive')
    length = np.hypot(g.detector_width_mm+4*g.pixel_width, g.detector_height_mm+4*g.pixel_height)
    count = int(np.ceil(length/step)); step = length/count
    arrays = [cuda.to_device(np.ascontiguousarray(a, dtype=np.float32)) for a in
              (du, dv, weighted, normals, g.col_vectors, g.row_vectors, g.normals)]
    values = cuda.device_array((len(normals), g.n_views), dtype=np.float32)
    complete = cuda.device_array(values.shape, dtype=np.bool_)
    kernel[(values.size+127)//128, 128](*arrays, g.sdd_mm, g.pixel_width, g.pixel_height,
                                       count, step, edge_tolerance, values, complete)
    cuda.synchronize()
    return values.copy_to_host(), complete.copy_to_host()


def grangeat_plane_rebin_gpu(projections, geometry, normals, offsets, *, line_step_mm=None,
                             edge_tolerance=1e-5, gpu=1):
    """Eq. (6.7)/(6.8): find each target plane's source intersections first.

    Interpolate neighbouring projection gradients at the target detector line,
    retaining the densely sampled detector coordinate. Interpolating only the
    derived diagonal samples R'(n,n.S_view) discards that information and gives
    a much coarser SOD*dtheta plane-offset sampling. Both are consistent in the
    continuous limit; this ordering substantially reduces finite-view aliasing.
    """
    cuda = _select_cuda(gpu)
    _, _, kernel = _cuda_kernels()
    weighted, du, dv = detector_derivatives(projections, geometry)
    g = geometry
    step = min(g.pixel_width, g.pixel_height) if line_step_mm is None else float(line_step_mm)
    if step <= 0:
        raise ValueError('Line quadrature step must be positive')
    length = np.hypot(g.detector_width_mm+4*g.pixel_width, g.detector_height_mm+4*g.pixel_height)
    count = int(np.ceil(length/step)); step = length/count
    source_p = np.asarray(normals)@g.source_positions.T
    arrays = [cuda.to_device(np.ascontiguousarray(a, dtype=np.float32)) for a in
              (du, dv, weighted, normals, source_p, g.col_vectors, g.normals)]
    measured = cuda.device_array((len(normals), len(offsets)), dtype=np.float32)
    complete = cuda.device_array(measured.shape, dtype=np.float32)
    kernel[(measured.size+127)//128, 128](*arrays, float(offsets[0]), float(offsets[1]-offsets[0]),
                                        g.sdd_mm, g.pixel_width, g.pixel_height, count, step,
                                        edge_tolerance, measured, complete)
    cuda.synchronize()
    return measured.copy_to_host(), complete.copy_to_host()


def rebin_radon_derivative(source_positions, normals, values, complete, offsets):
    """Interpolate along each monotone branch of an OPEN trajectory.

    Several source intersections measure the same plane. Average available
    untruncated branches with weights summing to one. No start/end connection,
    extrapolation, zero-filled missing planes, or density-based gain fit.
    """
    positions = np.asarray(normals)@np.asarray(source_positions).T
    offsets = np.asarray(offsets)
    result = np.full((len(normals), len(offsets)), np.nan, dtype=np.float64)
    for i, p in enumerate(positions):
        signs = np.sign(np.diff(p))
        turns = np.flatnonzero(signs[:-1]*signs[1:] <= 0)+1
        bounds = np.r_[0, turns, len(p)-1]
        total = np.zeros(len(offsets)); counts = np.zeros(len(offsets))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            ids = np.arange(lo, hi+1)
            if len(ids) < 2:
                continue
            if p[ids[0]] > p[ids[-1]]:
                ids = ids[::-1]
            # Never interpolate across an unmeasured or truncated plane sample.
            good = complete[i, ids] & np.isfinite(values[i, ids])
            starts = np.flatnonzero(good & ~np.r_[False, good[:-1]])
            stops = np.flatnonzero(good & ~np.r_[good[1:], False])+1
            for start, stop in zip(starts, stops):
                sub = ids[start:stop]
                if len(sub) < 2:
                    continue
                increasing = np.r_[True, np.diff(p[sub]) > 1e-8]
                sub = sub[increasing]
                if len(sub) < 2:
                    continue
                take = (offsets >= p[sub[0]]) & (offsets <= p[sub[-1]])
                total[take] += np.interp(offsets[take], p[sub], values[i, sub])
                counts[take] += 1
        np.divide(total, counts, out=result[i], where=counts > 0)
    return result


def inverse_radon_gpu(second_derivative, normals, weights, offsets, shape, voxel, *, gpu=1,
                      return_complete=False):
    """Hemisphere Radon inverse and measured angular-area fraction per voxel."""
    cuda = _select_cuda(gpu)
    _, kernel, _ = _cuda_kernels()
    a = np.asarray(second_derivative)
    valid = np.isfinite(a)
    if a.shape != (len(normals), len(offsets)):
        raise ValueError('Radon data shape differs from quadrature')
    if not np.allclose(np.diff(offsets), offsets[1]-offsets[0]) or offsets[1] <= offsets[0]:
        raise ValueError('Radon offsets must be uniformly increasing')
    if not np.isclose(np.sum(weights), 2*np.pi):
        raise ValueError('Hemisphere weights must sum to 2*pi')
    arrays = [cuda.to_device(np.ascontiguousarray(v)) for v in (
        np.nan_to_num(a, nan=0).astype(np.float32), valid,
        np.asarray(normals, dtype=np.float32), np.asarray(weights, dtype=np.float32))]
    count = int(np.prod(shape))
    out = cuda.device_array(count, dtype=np.float32)
    coverage = cuda.device_array(count, dtype=np.float32)
    all_present = cuda.device_array(count, dtype=np.bool_)
    kernel[(count+127)//128, 128](*arrays, float(offsets[0]), float(offsets[1]-offsets[0]),
                                float(voxel), *map(int, shape), out, coverage, all_present)
    cuda.synchronize()
    result = (out.copy_to_host().reshape(shape), coverage.copy_to_host().reshape(shape))
    return (*result, all_present.copy_to_host().reshape(shape)) if return_complete else result


def reconstruct_grangeat(projections, geometry, shape, voxel, *, n_polar=64,
                         n_azimuth=256, radon_step_mm=1., line_step_mm=None, gpu=1,
                         progress=print):
    """Projection -> weighted line derivative -> plane rebinning -> Radon inverse.

    ``reconstruction`` contains NaN wherever any sampled required direction is
    unavailable. ``partial_reconstruction`` contains the unnormalised measured
    angular integral for inspection; it is not an exact extension outside support.
    ``coverage`` is a sampled data diagnostic, not a continuous completeness proof.
    """
    start = time.perf_counter()
    normals, weights = hemisphere_quadrature(n_polar, n_azimuth)
    if radon_step_mm <= 0 or voxel <= 0:
        raise ValueError('Physical sampling intervals must be positive')
    half = np.linalg.norm(np.asarray(shape)*voxel/2)
    np_half = int(np.ceil(half/radon_step_mm))+2
    offsets = np.arange(-np_half, np_half+1)*radon_step_mm
    progress(f'[Grangeat] {len(normals)} plane normals: target-plane/source intersections and detector line derivatives')
    observed, rebinned = grangeat_plane_rebin_gpu(projections, geometry, normals, offsets,
                                                line_step_mm=line_step_mm, gpu=gpu)
    line_seconds = time.perf_counter()-start
    second = np.gradient(rebinned, radon_step_mm, axis=1, edge_order=2)
    # A derivative stencil cannot validate the central missing sample by accident.
    second[~np.isfinite(rebinned)] = np.nan
    rebin_seconds = time.perf_counter()-start-line_seconds
    progress('[Grangeat] direct 3D Radon inversion; no reconstruction iterations')
    partial, coverage, all_present = inverse_radon_gpu(second, normals, weights, offsets, shape, voxel,
                                                      gpu=gpu, return_complete=True)
    hull = source_hull_mask(geometry.source_positions, shape, voxel)
    strict = np.where(all_present & hull, partial, np.nan)
    progress('[Grangeat] separate finite-detector estimate; zero continuation is not exact data')
    observed_second = np.gradient(observed, radon_step_mm, axis=1, edge_order=2)
    observed_second[~np.isfinite(observed)] = np.nan
    estimate, estimate_coverage = inverse_radon_gpu(observed_second, normals, weights, offsets, shape, voxel, gpu=gpu)
    return dict(reconstruction=strict, partial_reconstruction=partial, coverage=coverage,
                finite_detector_estimate=estimate, estimate_plane_coverage=estimate_coverage, source_hull_mask=hull,
                diagnostics=dict(method='Grangeat first-derivative Radon reconstruction',
                    reference='https://doi.org/10.1007/BFb0084509',
                    equations=['2.7', '3.1', '3.3', '3.4', '4.1', '6.7', '6.8'],
                    rebinning='Target plane/source intersection first, projection-gradient angular interpolation at the target detector line',
                    reconstruction_iterations=0, data_fitted_scale=False, positivity_clipping=False,
                    hemisphere_normals=len(normals), n_polar=n_polar, n_azimuth=n_azimuth,
                    angular_weight_sum=float(weights.sum()), radon_step_mm=radon_step_mm,
                    line_step_mm=line_step_mm or min(geometry.pixel_width, geometry.pixel_height),
                    line_seconds=line_seconds, rebin_seconds=rebin_seconds,
                    total_seconds=time.perf_counter()-start,
                    finite_rebinned_fraction=float(np.isfinite(rebinned).mean()),
                    complete_voxel_fraction=float(np.isfinite(strict).mean()),
                    source_hull_voxel_fraction=float(hull.mean()),
                    completeness='All sampled directions present, detector-edge truncation test, and sampled-source hull interior; not a continuous support proof',
                    missing_data='NaN in strict output; partial output sums measured directions without renormalisation',
                    finite_detector_estimate='Detector data continued by zero including boundary derivatives; truncation makes this an approximation, never certified exact',
                    manufacturer_implementation=False))
