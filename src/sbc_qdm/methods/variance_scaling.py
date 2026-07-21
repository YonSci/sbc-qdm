"""Variance Scaling bias correction.

Extends Delta Change with a variance correction (Chen et al. 2011;
Teutschbein & Seibert 2012): first removes the mean bias, then rescales the
resulting anomalies by the ratio of standard deviations, so both the mean
*and* the spread of the corrected series match the reference -- something
Linear Scaling / Delta Change don't attempt.

    corrected = ref_mean + (raw - hist_mean) * (ref_std / hist_std)
    corrected = max(corrected, 0)   -- precipitation can't be negative

Trained state per month: (hist_mean, hist_std, ref_mean, ref_std), pooled
over hindcast years + ensemble members.
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims

TrainedVarianceScaling = dict[int, tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]]


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)
    hist_dims = sample_dims(hist_m)

    hist_mean = hist_m.mean(hist_dims, skipna=True)
    hist_std = hist_m.std(hist_dims, skipna=True)
    ref_mean = ref_m.mean("time", skipna=True)
    ref_std = ref_m.std("time", skipna=True)
    return hist_mean.compute(), hist_std.compute(), ref_mean.compute(), ref_std.compute()


def train_variance_scaling(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedVarianceScaling:
    """Fit per-month (hist_mean, hist_std, ref_mean, ref_std) for every calendar month in `hist`."""
    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedVarianceScaling = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m)
    return trained


def apply_variance_scaling(sim: xr.DataArray, trained: TrainedVarianceScaling, cfg: dict) -> xr.DataArray:
    """Apply the trained per-month mean+variance correction to `sim`, month by month."""
    corrected_months = []
    for month, (hist_mean, hist_std, ref_mean, ref_std) in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        safe_hist_std = hist_std.where(hist_std != 0, 1.0)
        ratio = xr.where(hist_std != 0, ref_std / safe_hist_std, 1.0)
        corrected_m = (ref_mean + (sim_m - hist_mean) * ratio).clip(min=0)
        corrected_months.append(corrected_m)

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
