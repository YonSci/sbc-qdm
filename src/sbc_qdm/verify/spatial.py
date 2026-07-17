"""Spatial performance: pattern correlation, spatial RMSE, spatial correlation.

The complement to skill.py's *temporal* skill (per-pixel, across years):
these correlate/score **across pixels for a fixed time step** (day, month
instance, or year at JJAS scale), producing a time series rather than a map
-- answering "does the model get the spatial structure right on any given
day/month/year", not "is any given pixel biased over the long run".

"Spatial correlation" and "spatial pattern correlation" are close synonyms in
common usage; they're kept as two distinct functions here because the
standard distinction matters -- but note Pearson correlation is invariant to
any *additive constant* shift (it already centers internally), so removing
each time step's own instantaneous spatial mean before correlating would be
a no-op, not a different metric. The real distinction is anomalies from each
**pixel's long-term climatology** (a per-pixel constant across time, not a
per-time constant across pixels): pattern correlation asks "does the model
get today's/this month's departure from normal in the right place", which
plain spatial correlation of raw values (dominated by the shared wet/dry
climatological gradient every day looks similar) cannot distinguish.
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.verify.aggregate import ensemble_mean

SPATIAL_DIMS = ("lat", "lon")


def spatial_correlation(model: xr.DataArray, ref: xr.DataArray, spatial_dims: tuple[str, ...] = SPATIAL_DIMS, realization_dim: str = "realization") -> xr.DataArray:
    """Pearson correlation of raw field values across space, per time step."""
    model = ensemble_mean(model, realization_dim)
    return xr.corr(model, ref, dim=list(spatial_dims))


def spatial_pattern_correlation(
    model: xr.DataArray,
    ref: xr.DataArray,
    model_climatology: xr.DataArray,
    ref_climatology: xr.DataArray,
    spatial_dims: tuple[str, ...] = SPATIAL_DIMS,
    realization_dim: str = "realization",
) -> xr.DataArray:
    """Pearson correlation of spatial *anomaly* patterns (each pixel's own long-term
    climatology removed first), per time step.

    model_climatology/ref_climatology: per-pixel (lat, lon) long-term means,
    e.g. `ref.mean("time")` for an overall-record climatology, or a
    month-specific climatology if you want each time step compared only
    against its own calendar month's normal.
    """
    model = ensemble_mean(model, realization_dim)
    model_anom = model - model_climatology
    ref_anom = ref - ref_climatology
    return xr.corr(model_anom, ref_anom, dim=list(spatial_dims))


def spatial_rmse(model: xr.DataArray, ref: xr.DataArray, spatial_dims: tuple[str, ...] = SPATIAL_DIMS, realization_dim: str = "realization") -> xr.DataArray:
    """RMSE across space, per time step."""
    model = ensemble_mean(model, realization_dim)
    return ((model - ref) ** 2).mean(spatial_dims, skipna=True) ** 0.5
