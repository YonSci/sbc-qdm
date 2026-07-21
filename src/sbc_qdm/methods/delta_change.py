"""Delta Change bias correction.

Additive monthly-mean-difference correction, the standard counterpart to
Linear Scaling's multiplicative ratio (both correct only the mean; contrast
with variance_scaling.py, which also corrects spread).

Classical "delta change" / "change factor" method is usually framed around a
historical-vs-future climate scenario: apply a model's *projected change*
(future model climatology minus historical model climatology) onto the
*observed* historical series, so the corrected series inherits the
observations' own variability/sequencing rather than the model's. That
framing doesn't map cleanly onto this pipeline's leave-one-year-out
validation of a single historical period (there is no separate "future"
scenario here), so it's adapted to the same per-day/per-member correction
form as the other methods:

    delta = mean(ref_month) - mean(hist_month)   -- pooled over hindcast years + members
    corrected = max(raw + delta, 0)

In this single-period setting it reduces to the additive counterpart of
Linear Scaling's multiplicative form (kept as a separate method here because
that ratio-vs-difference contrast is itself a standard comparison point in
the bias-correction literature for precipitation).
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims

TrainedDeltaChange = dict[int, xr.DataArray]


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray) -> xr.DataArray:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)

    ref_mean = ref_m.mean("time", skipna=True)
    hist_mean = hist_m.mean(sample_dims(hist_m), skipna=True)
    delta = ref_mean - hist_mean
    return delta.compute()


def train_delta_change(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedDeltaChange:
    """Fit one additive monthly delta per calendar month present in `hist`."""
    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedDeltaChange = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m)
    return trained


def apply_delta_change(sim: xr.DataArray, trained: TrainedDeltaChange, cfg: dict) -> xr.DataArray:
    """Apply the trained per-month additive delta to `sim`, month by month."""
    corrected_months = []
    for month, delta in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append((sim_m + delta).clip(min=0))

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
