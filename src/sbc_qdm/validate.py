"""Skill diagnostics comparing raw vs QDM-corrected forecasts against CHIRPS.

Metrics (computed from the leave-one-year-out corrected hindcast):
  - mean bias maps, pre- vs post-correction (ensemble mean vs CHIRPS)
  - wet-day frequency (fraction of days above a rain/no-rain threshold)
  - CRPS / CRPSS of the ensemble, pre- vs post-correction
  - rank histograms (ensemble reliability)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

REALIZATION_DIM = "realization"


def bias_maps(ref: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray) -> xr.Dataset:
    """Time-mean bias (ensemble mean minus CHIRPS) before and after correction."""
    raw_mean = raw.mean(REALIZATION_DIM) if REALIZATION_DIM in raw.dims else raw
    corrected_mean = corrected.mean(REALIZATION_DIM) if REALIZATION_DIM in corrected.dims else corrected

    raw_bias = (raw_mean - ref).mean("time")
    corrected_bias = (corrected_mean - ref).mean("time")
    return xr.Dataset({"raw_bias": raw_bias, "corrected_bias": corrected_bias})


def wet_day_frequency(da: xr.DataArray, threshold_mm: float = 1.0) -> xr.DataArray:
    """Fraction of time/ensemble steps above `threshold_mm`."""
    dims = [d for d in ("time", REALIZATION_DIM) if d in da.dims]
    return (da > threshold_mm).mean(dims)


def crps_ensemble(obs: xr.DataArray, ensemble: xr.DataArray, realization_dim: str = REALIZATION_DIM) -> xr.DataArray:
    """CRPS of an ensemble forecast against a deterministic observation.

    Uses the sorted-ensemble form of the Gneiting & Raftery (2007, eq. 21)
    estimator -- O(n log n) via sorting instead of the naive O(n^2) pairwise
    sum -- so this stays cheap even at 51 members over the full domain.
    """

    def _crps(obs_val, ens):
        ens_sorted = np.sort(ens, axis=-1)
        n = ens.shape[-1]
        term1 = np.mean(np.abs(ens_sorted - obs_val[..., None]), axis=-1)
        weights = 2 * np.arange(1, n + 1) - n - 1
        term2 = np.sum(weights * ens_sorted, axis=-1) / (n**2)
        return term1 - term2

    # apply_ufunc requires a core dim (realization) to be a single chunk;
    # regrid.py leaves it chunked in blocks for memory-bounded regridding, so
    # it must be rechunked before this call (lat/lon can stay chunked, they're
    # not core dims here).
    if realization_dim in ensemble.dims:
        ensemble = ensemble.chunk({realization_dim: -1})

    return xr.apply_ufunc(
        _crps,
        obs,
        ensemble,
        input_core_dims=[[], [realization_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def crps_skill_score(ref: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray) -> xr.Dataset:
    """Time-mean CRPS pre/post correction, and the CRPS skill score relative to raw."""
    crps_raw = crps_ensemble(ref, raw).mean("time")
    crps_corrected = crps_ensemble(ref, corrected).mean("time")
    crpss = 1 - crps_corrected / crps_raw
    return xr.Dataset({"crps_raw": crps_raw, "crps_corrected": crps_corrected, "crpss": crpss})


def rank_histogram(ref: xr.DataArray, ensemble: xr.DataArray, realization_dim: str = REALIZATION_DIM) -> xr.DataArray:
    """Histogram of CHIRPS' rank within the ensemble, pooled over time/lat/lon.

    A flat histogram indicates a well-calibrated ensemble; U-shaped means
    under-dispersion, dome-shaped means over-dispersion.
    """
    n_members = ensemble.sizes[realization_dim]
    rank = (ensemble < ref).sum(realization_dim).where(ref.notnull())

    flat = rank.values.ravel()
    flat = flat[~np.isnan(flat)].astype(int)
    counts = np.bincount(flat, minlength=n_members + 1)
    return xr.DataArray(counts, dims=["rank"], coords={"rank": np.arange(n_members + 1)}, name="rank_histogram")
