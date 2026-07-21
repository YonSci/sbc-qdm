"""Linear Scaling bias correction.

Standard multiplicative monthly-mean-ratio correction for precipitation
(Lenderink et al. 2007; Teutschbein & Seibert 2012): a single scaling factor
per calendar month per pixel, corrects the mean but nothing about the shape
of the distribution (variance, extremes, wet-day frequency are all left
untouched -- see variance_scaling.py / power_transformation.py for methods
that go further).

    factor = mean(ref_month) / mean(hist_month)   -- pooled over hindcast years + members
    corrected = raw * factor
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims

TrainedLinearScaling = dict[int, xr.DataArray]


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray) -> xr.DataArray:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)

    ref_mean = ref_m.mean("time", skipna=True)
    hist_mean = hist_m.mean(sample_dims(hist_m), skipna=True)
    safe_hist_mean = hist_mean.where(hist_mean != 0, 1.0)
    factor = xr.where(hist_mean != 0, ref_mean / safe_hist_mean, 1.0)
    return factor.compute()


def train_linear_scaling(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedLinearScaling:
    """Fit one multiplicative scaling factor per calendar month present in `hist`."""
    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedLinearScaling = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m)
    return trained


def apply_linear_scaling(sim: xr.DataArray, trained: TrainedLinearScaling, cfg: dict) -> xr.DataArray:
    """Apply the trained per-month scaling factor to `sim`, month by month."""
    corrected_months = []
    for month, factor in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append(sim_m * factor)

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
