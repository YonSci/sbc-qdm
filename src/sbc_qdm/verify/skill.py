"""Deterministic monthly/seasonal skill vs climatology: ACC, Spearman anomaly
correlation, RMSE skill score, interannual variability ratio.

All four operate on the pre-aggregated (year[, month], lat, lon) arrays from
aggregate.monthly_totals()/jjas_totals() -- i.e. this is *temporal* skill,
correlating/scoring each pixel's inter-annual series against observations
across the `sample_dim` ("year"). The complementary *spatial* skill (pattern
correlation across pixels for a fixed time) lives in spatial.py.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sbc_qdm.verify.aggregate import anomaly, ensemble_mean


def anomaly_correlation(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "year", realization_dim: str = "realization") -> xr.DataArray:
    """Pearson ACC between model and ref anomalies (each from its own climatology), across sample_dim."""
    model = ensemble_mean(model, realization_dim)
    model_anom = anomaly(model, sample_dim=sample_dim)
    ref_anom = anomaly(ref, sample_dim=sample_dim)
    return xr.corr(model_anom, ref_anom, dim=sample_dim)


def _rank_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    from scipy.stats import rankdata

    # apply_ufunc requires the core dim (here, the correlation sample dim) to
    # be a single chunk; aggregate.py's unstack() can leave "year"/"month"
    # dask-chunked, so it must be rechunked before this call.
    da = da.chunk({dim: -1})

    return xr.apply_ufunc(
        rankdata,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"nan_policy": "omit"},
    )


def spearman_anomaly_correlation(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "year", realization_dim: str = "realization") -> xr.DataArray:
    """Spearman (rank) correlation of model vs ref anomalies across sample_dim.

    Implemented as a Pearson correlation of within-pixel ranks -- the
    standard equivalence -- so it vectorizes over the whole grid via
    xr.corr() instead of looping scipy.stats.spearmanr per pixel.
    """
    model = ensemble_mean(model, realization_dim)
    model_anom = anomaly(model, sample_dim=sample_dim)
    ref_anom = anomaly(ref, sample_dim=sample_dim)
    model_rank = _rank_along_dim(model_anom, sample_dim)
    ref_rank = _rank_along_dim(ref_anom, sample_dim)
    return xr.corr(model_rank, ref_rank, dim=sample_dim)


def rmse_skill_score(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "year", realization_dim: str = "realization") -> xr.DataArray:
    """RMSESS = 1 - RMSE(model)/RMSE(climatology), where the climatology "forecast"
    is always the long-term ref mean (its RMSE reduces to std(ref)). Positive
    means the model beats naive climatology.
    """
    model = ensemble_mean(model, realization_dim)
    rmse_model = ((model - ref) ** 2).mean(sample_dim, skipna=True) ** 0.5
    rmse_clim = ref.std(sample_dim, skipna=True)
    return 1 - rmse_model / rmse_clim


def interannual_variability_ratio(model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "year", realization_dim: str = "realization") -> xr.DataArray:
    """std(model)/std(ref) across years -- same formula as deterministic.sd_ratio,
    named separately because at this (monthly/JJAS) scale the sample dim IS
    the inter-annual series, which is what this metric is conventionally
    about (does the model reproduce year-to-year variability, not just
    within-season day-to-day variability).
    """
    model = ensemble_mean(model, realization_dim)
    return model.std(sample_dim, skipna=True) / ref.std(sample_dim, skipna=True)
