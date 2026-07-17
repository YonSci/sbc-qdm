"""MBE, MAE, PBIAS, RMSE, SD ratio, CV ratio.

Every function takes a `model` DataArray (ensemble mean already taken, or
pass an ensemble DataArray plus `realization_dim` to average it here) and a
`ref` (CHIRPS) DataArray sharing the same sample dimension, and reduces over
`sample_dim` -- "time" pooled over the whole daily record for the daily
scale, or "year" for monthly/JJAS-aggregated arrays (see aggregate.py).
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.verify.aggregate import ensemble_mean


def mean_bias_error(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    model = ensemble_mean(model, realization_dim)
    return (model - ref).mean(sample_dim, skipna=True)


def mean_absolute_error(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    model = ensemble_mean(model, realization_dim)
    return abs(model - ref).mean(sample_dim, skipna=True)


def percentage_bias(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    """PBIAS = 100 * sum(model - ref) / sum(ref) (Gupta et al. 1999 convention)."""
    model = ensemble_mean(model, realization_dim)
    return 100.0 * (model - ref).sum(sample_dim, skipna=True) / ref.sum(sample_dim, skipna=True)


def root_mean_square_error(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    model = ensemble_mean(model, realization_dim)
    return ((model - ref) ** 2).mean(sample_dim, skipna=True) ** 0.5


def sd_ratio(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    """std(model) / std(ref) over sample_dim -- ~1 means the model reproduces observed variability."""
    model = ensemble_mean(model, realization_dim)
    return model.std(sample_dim, skipna=True) / ref.std(sample_dim, skipna=True)


def cv_ratio(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization") -> xr.DataArray:
    """Coefficient-of-variation (std/mean) ratio, model over ref."""
    model = ensemble_mean(model, realization_dim)
    model_cv = model.std(sample_dim, skipna=True) / model.mean(sample_dim, skipna=True)
    ref_cv = ref.std(sample_dim, skipna=True) / ref.mean(sample_dim, skipna=True)
    return model_cv / ref_cv
