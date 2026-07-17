"""Distributional similarity: Q-Q / ECDF / PDF data, quantile bias, wet-day frequency bias.

Q-Q/ECDF/PDF are domain-pooled (flattened across whatever dims are present --
lat, lon, time, realization) rather than per-pixel: a handful of summary
distribution comparisons are the practical, readable deliverable, not 2880
per-pixel plots. Quantile bias and wet-day frequency bias, by contrast, are
inherently spatial (they're meant to be mapped), so those stay per-pixel.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sbc_qdm.validate import wet_day_frequency
from sbc_qdm.verify.aggregate import iter_spatial_blocks


def _flatten_valid(da: xr.DataArray, max_samples: int = 5_000_000, seed: int = 0) -> np.ndarray:
    """Flatten to non-NaN values, streamed block-by-block (never materializing
    a full 33-year x 51-member x full-domain array as one allocation) and
    subsampled per-block so the final concatenated array stays bounded --
    even a NaN-filtered full record can still be several GiB, which this
    machine doesn't reliably have free. Order doesn't matter for quantile/
    ECDF/histogram use, so per-block random subsampling is safe.
    """
    rng = np.random.default_rng(seed)
    blocks = list(iter_spatial_blocks(da))
    per_block_cap = max(1000, max_samples // max(len(blocks), 1))

    parts = []
    for blk in blocks:
        arr = blk.values.ravel()
        arr = arr[~np.isnan(arr)]
        if arr.size > per_block_cap:
            idx = rng.choice(arr.size, size=per_block_cap, replace=False)
            arr = arr[idx]
        parts.append(arr)
    return np.concatenate(parts) if parts else np.array([])


def qq_pairs(model: xr.DataArray, ref: xr.DataArray, n_quantiles: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Domain-pooled sorted quantile pairs for a Q-Q plot: (quantiles, model_values, ref_values)."""
    quantiles = np.linspace(0, 1, n_quantiles)
    model_q = np.quantile(_flatten_valid(model), quantiles)
    ref_q = np.quantile(_flatten_valid(ref), quantiles)
    return quantiles, model_q, ref_q


def ecdf(da: xr.DataArray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Domain-pooled empirical CDF: (sorted_x, F(x)), subsampled to n_points."""
    sorted_vals = np.sort(_flatten_valid(da))
    n = len(sorted_vals)
    y = np.arange(1, n + 1) / n
    if n > n_points:
        idx = np.linspace(0, n - 1, n_points).astype(int)
        return sorted_vals[idx], y[idx]
    return sorted_vals, y


def pdf_histogram(da: xr.DataArray, bins: int = 60, value_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Domain-pooled density histogram: (bin_centers, density).

    A plain histogram rather than a KDE -- precipitation's point-mass at zero
    (dry days) would be smeared out by kernel smoothing into a misleadingly
    continuous-looking density near zero.
    """
    flat = _flatten_valid(da)
    counts, edges = np.histogram(flat, bins=bins, range=value_range, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def quantile_bias(
    model: xr.DataArray,
    ref: xr.DataArray,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90, 0.95),
    sample_dims: tuple[str, ...] = ("time", "realization"),
) -> xr.DataArray:
    """Per-pixel bias (model - ref) at each requested quantile of the pooled daily distribution.

    Returns a DataArray with a new 'quantile' dim holding the requested quantiles.
    """
    model_dims = [d for d in sample_dims if d in model.dims]
    ref_dims = [d for d in sample_dims if d in ref.dims]

    biases = []
    for q in quantiles:
        model_q = model.quantile(q, dim=model_dims, skipna=True).drop_vars("quantile", errors="ignore")
        ref_q = ref.quantile(q, dim=ref_dims, skipna=True).drop_vars("quantile", errors="ignore")
        biases.append(model_q - ref_q)

    qdim = xr.DataArray(list(quantiles), dims="quantile", name="quantile")
    return xr.concat(biases, dim=qdim)


def wet_day_frequency_bias(model: xr.DataArray, ref: xr.DataArray, threshold_mm: float = 1.0) -> xr.DataArray:
    """Per-pixel wet-day frequency bias: model's fraction of wet days minus ref's."""
    return wet_day_frequency(model, threshold_mm) - wet_day_frequency(ref, threshold_mm)
