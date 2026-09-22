"""Reproducible photon-counting noise for dimensionless CT line integrals.

For each detector pixel and view, N ~ Poisson(I0 * exp(-p)). The measured line
integral is -log(N / I0), with only zero counts replaced by 0.5 before the log.
Counts above I0 produce negative line integrals and are deliberately preserved.
This model assumes an exact incident fluence; it does not add flat-field noise,
electronic noise, scatter, or correlations between detector pixels.
"""
from __future__ import annotations

import operator

import numpy as np


DEFAULT_I0 = 44000.0
UINT32_MAX = int(np.iinfo(np.uint32).max)


def _validate_i0(i0: float) -> float:
    i0 = float(i0)
    if not np.isfinite(i0) or not 0 < i0 <= UINT32_MAX:
        raise ValueError("i0 must be finite, positive, and within the uint32 count range.")
    return i0


def counts_to_line_integrals(counts: np.ndarray, *, i0: float = DEFAULT_I0) -> np.ndarray:
    """Convert integer counts to float32 post-log data, correcting only zeros."""
    i0 = _validate_i0(i0)
    counts = np.asarray(counts)
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Photon counts must have an integer dtype.")
    if np.any(counts < 0) or np.any(counts > UINT32_MAX):
        raise ValueError("Photon counts must be within the uint32 count range.")
    corrected = counts.astype(np.float64)
    corrected[counts == 0] = 0.5
    # Difference of logs also stays finite for very small positive I0.
    return (np.log(i0) - np.log(corrected)).astype(np.float32)


def poisson_noisy_projections(
    clean: np.ndarray,
    *,
    i0: float = DEFAULT_I0,
    seed: int = 0,
    chunk_views: int = 8,
    return_counts: bool = False,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Return ``(noisy_float32, statistics, optional_uint32_counts)`` on CPU.

    The first array axis indexes views. A single NumPy Generator with PCG64
    consumes samples in C order, so changing ``chunk_views`` does not change the
    noise realization. The input is never modified. Reproduction also records
    the NumPy version because sampler implementations can vary across versions.
    """
    i0 = _validate_i0(i0)
    try:
        seed, chunk_views = operator.index(seed), operator.index(chunk_views)
    except TypeError as exc:
        raise ValueError("seed and chunk_views must be integers.") from exc
    if seed < 0 or chunk_views <= 0:
        raise ValueError("seed must be nonnegative and chunk_views must be positive.")
    clean = np.asarray(clean)
    if (clean.ndim < 1 or clean.size == 0 or not np.issubdtype(clean.dtype, np.number)
            or np.issubdtype(clean.dtype, np.complexfloating)):
        raise ValueError("clean must be a nonempty real numeric array indexed by view.")
    noisy = np.empty(clean.shape, dtype=np.float32)
    counts_output = np.empty(clean.shape, dtype=np.uint32) if return_counts else None
    rng = np.random.Generator(np.random.PCG64(seed))
    count_min, count_max, zero_count, negative_count = UINT32_MAX, 0, 0, 0
    clean_min, clean_max, noisy_min, noisy_max = np.inf, -np.inf, np.inf, -np.inf
    expected_sum = sampled_sum = squared_error = 0.0
    for start in range(0, len(clean), chunk_views):
        stop = min(start + chunk_views, len(clean))
        block = np.asarray(clean[start:stop], dtype=np.float64, order="C")
        if not np.isfinite(block).all():
            raise ValueError("Clean line integrals must be finite.")
        # Computing the Beer-Lambert rate in log space avoids intermediate
        # overflow for small I0 and large negative line integrals.
        log_rate = np.log(i0) - block
        if np.any(log_rate > np.log(UINT32_MAX)):
            raise ValueError("Expected photon counts exceed the uint32 storage range.")
        with np.errstate(under="ignore"):
            rate = np.exp(log_rate)
        counts = rng.poisson(rate)
        if int(counts.max()) > UINT32_MAX:
            raise ValueError("A sampled count exceeds uint32; counts were not wrapped or clipped.")
        measurement = counts_to_line_integrals(counts, i0=i0)
        noisy[start:stop] = measurement
        if counts_output is not None:
            counts_output[start:stop] = counts
        count_min = min(count_min, int(counts.min()))
        count_max = max(count_max, int(counts.max()))
        zero_count += int(np.count_nonzero(counts == 0))
        negative_count += int(np.count_nonzero(measurement < 0))
        clean_min, clean_max = min(clean_min, float(block.min())), max(clean_max, float(block.max()))
        noisy_min = min(noisy_min, float(measurement.min()))
        noisy_max = max(noisy_max, float(measurement.max()))
        expected_sum += float(rate.sum(dtype=np.float64))
        sampled_sum += float(counts.sum(dtype=np.float64))
        squared_error += float(np.square(measurement.astype(np.float64) - block).sum())
    statistics = {
        "model": "N ~ Poisson(i0 * exp(-clean)); noisy = -log(max(N, 0.5) / i0)",
        "i0_photons_per_detector_pixel_per_view": i0,
        "seed": seed,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "sample_order": "C order, views then detector pixels; invariant to chunk_views",
        "shape": list(clean.shape),
        "sample_count": int(clean.size),
        "zero_count_policy": "Replace only N=0 by 0.5 before log; positive counts unchanged",
        "negative_postlog_values_preserved": True,
        "count_min": count_min,
        "count_max": count_max,
        "zero_count": zero_count,
        "negative_postlog_count": negative_count,
        "mean_expected_counts": expected_sum / clean.size,
        "mean_sampled_counts": sampled_sum / clean.size,
        "clean_min_max": [clean_min, clean_max],
        "noisy_min_max": [noisy_min, noisy_max],
        "postlog_noise_rmse": float(np.sqrt(squared_error / clean.size)),
    }
    return noisy, statistics, counts_output
