"""Daily -> monthly / JJAS-seasonal aggregation, climatology, and tercile categories.

Everything downstream (skill scores, RPSS/BSS/ROC) is computed by pooling
across hindcast **years** for a fixed calendar month (monthly scale) or for
the JJAS block (seasonal scale) -- so both aggregations reshape the daily
"time" axis into an explicit "year" sample dimension (plus "month" for the
monthly case) rather than leaving it as a datetime axis.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

JJAS_MONTHS = (6, 7, 8, 9)


def iter_spatial_blocks(da: xr.DataArray, block: int = 10):
    """Yield da sliced into (lat, lon) tiles of at most `block` x `block` pixels.

    Domain-pooling functions (Q-Q/ECDF/PDF, wet/dry spells) need every value
    across the full multi-year record, but calling .values on the whole
    lazy array at once forces a single multi-GiB allocation (33 years x 51
    members x full domain) -- this streams it block-by-block instead,
    matching the lat/lon tiling regrid.py/qdm.py already chunk by.
    """
    lat_size = da.sizes.get("lat", 1)
    lon_size = da.sizes.get("lon", 1)
    for lat0 in range(0, lat_size, block):
        for lon0 in range(0, lon_size, block):
            sel = {}
            if "lat" in da.dims:
                sel["lat"] = slice(lat0, lat0 + block)
            if "lon" in da.dims:
                sel["lon"] = slice(lon0, lon0 + block)
            yield da.isel(sel)


def ensemble_mean(da: xr.DataArray, realization_dim: str = "realization") -> xr.DataArray:
    """Ensemble mean if a realization dim is present, else da unchanged (already deterministic)."""
    return da.mean(realization_dim) if realization_dim in da.dims else da


def monthly_totals(da: xr.DataArray) -> xr.DataArray:
    """Per-(year, month) total, summed over days. Returns dims (..., year, month, ...).

    min_count=1 means a calendar month that's entirely NaN (a masked
    water/ocean pixel) stays NaN, while a month with only a few missing days
    still sums the valid ones.
    """
    monthly = da.resample(time="1MS").sum(min_count=1)
    monthly = monthly.assign_coords(
        year=("time", monthly["time"].dt.year.values),
        month=("time", monthly["time"].dt.month.values),
    )
    return monthly.set_index(time=["year", "month"]).unstack("time")


def jjas_totals(da: xr.DataArray) -> xr.DataArray:
    """Per-year JJAS (Jun-Sep) total, summed over days. Returns dims (..., year, ...)."""
    jjas = da.sel(time=da["time"].dt.month.isin(JJAS_MONTHS))
    grouped = jjas.groupby(jjas["time"].dt.year).sum("time", min_count=1)
    return grouped.rename({"year": "year"})


def climatology(da: xr.DataArray, sample_dim: str = "year") -> xr.DataArray:
    """Mean over the sample dimension (years), keeping any other dims (e.g. month)."""
    return da.mean(sample_dim, skipna=True)


def anomaly(da: xr.DataArray, clim: xr.DataArray | None = None, sample_dim: str = "year") -> xr.DataArray:
    """da minus its climatological mean (computed from da itself if clim isn't given)."""
    if clim is None:
        clim = climatology(da, sample_dim=sample_dim)
    return da - clim


def tercile_thresholds(da: xr.DataArray, sample_dim: str = "year") -> xr.DataArray:
    """Lower (33rd pct) / upper (67th pct) tercile thresholds, new 'tercile_edge' dim.

    Computed from da's own distribution across `sample_dim` -- pass the
    *observed* (CHIRPS) array here, since terciles for RPSS/BSS/ROC are
    defined from the observational climatology, not the model's.
    """
    lower = da.quantile(1 / 3, dim=sample_dim, skipna=True).drop_vars("quantile", errors="ignore")
    upper = da.quantile(2 / 3, dim=sample_dim, skipna=True).drop_vars("quantile", errors="ignore")
    edge = xr.DataArray(["lower", "upper"], dims="tercile_edge", name="tercile_edge")
    return xr.concat([lower, upper], dim=edge)


def tercile_category(da: xr.DataArray, thresholds: xr.DataArray) -> xr.DataArray:
    """Classify each sample into 0=below-normal, 1=near-normal, 2=above-normal."""
    lower = thresholds.sel(tercile_edge="lower", drop=True)
    upper = thresholds.sel(tercile_edge="upper", drop=True)
    cat = xr.ones_like(da)
    cat = xr.where(da < lower, 0, cat)
    cat = xr.where(da > upper, 2, cat)
    return cat.where(da.notnull())


def ensemble_tercile_probabilities(
    ensemble_da: xr.DataArray, thresholds: xr.DataArray, realization_dim: str = "realization"
) -> xr.DataArray:
    """Fraction of ensemble members in each tercile category, new 'category' dim ('below','near','above')."""
    lower = thresholds.sel(tercile_edge="lower", drop=True)
    upper = thresholds.sel(tercile_edge="upper", drop=True)
    prob_below = (ensemble_da < lower).mean(realization_dim)
    prob_above = (ensemble_da > upper).mean(realization_dim)
    prob_near = 1 - prob_below - prob_above
    category = xr.DataArray(["below", "near", "above"], dims="category", name="category")
    probs = xr.concat([prob_below, prob_near, prob_above], dim=category)
    return probs.where(ensemble_da.isel({realization_dim: 0}, drop=True).notnull())
