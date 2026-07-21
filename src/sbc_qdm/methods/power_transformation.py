"""Power Transformation bias correction.

Precipitation-specific method (Leander & Buishand 2007) that corrects both
the mean and the coefficient of variation (std/mean) -- unlike Variance
Scaling's additive spread correction, this one works multiplicatively via a
fitted exponent, which keeps values non-negative without needing to clip and
handles the right-skewed shape of daily precipitation better.

For each pixel/month, solve for an exponent b (root-find, since
CV(hist**b) is monotonic in b for positively-skewed precipitation data) so
that:

    CV(hist**b) == CV(ref)

then a scale factor so the mean matches too:

    a = mean(ref) / mean(hist**b)
    corrected = a * raw**b

Trained state per month: (a, b), both per-pixel DataArrays.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import brentq

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims

TrainedPowerTransformation = dict[int, tuple[xr.DataArray, xr.DataArray]]


def _cv(x: np.ndarray) -> float:
    mean = np.nanmean(x)
    if mean == 0:
        return 0.0
    return np.nanstd(x) / mean


def _solve_b(hist_1d: np.ndarray, ref_cv: float) -> float:
    """Root-find b such that CV(hist_1d**b) == ref_cv, bracketing/expanding as needed."""
    hist_1d = hist_1d[~np.isnan(hist_1d)]
    if hist_1d.size == 0 or np.all(hist_1d == 0) or ref_cv == 0:
        return 1.0

    def objective(b: float) -> float:
        return _cv(hist_1d**b) - ref_cv

    lo, hi = 0.1, 10.0
    f_lo, f_hi = objective(lo), objective(hi)
    # Expand the bracket a few times if the root isn't inside [0.1, 10] yet --
    # CV(hist**b) is monotonically increasing in b for positive, right-skewed
    # precipitation data, so widening in the direction of the sign change finds it.
    expansions = 0
    while f_lo * f_hi > 0 and expansions < 6:
        if abs(f_lo) < abs(f_hi):
            lo /= 2
            f_lo = objective(lo)
        else:
            hi *= 2
            f_hi = objective(hi)
        expansions += 1

    if f_lo * f_hi > 0:
        return 1.0  # couldn't bracket a root (e.g. degenerate pixel) -- fall back to identity exponent

    return brentq(objective, lo, hi, xtol=1e-4, maxiter=100)


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)
    hist_dims = sample_dims(hist_m)

    ref_cv = xr.apply_ufunc(
        lambda x: _cv(x[~np.isnan(x)]),
        ref_m,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    b = xr.apply_ufunc(
        _solve_b,
        hist_m,
        ref_cv,
        input_core_dims=[hist_dims, []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()

    ref_mean = ref_m.mean("time", skipna=True).compute()
    hist_pow_mean = (hist_m**b).mean(hist_dims, skipna=True).compute()
    safe_hist_pow_mean = hist_pow_mean.where(hist_pow_mean != 0, 1.0)
    a = xr.where(hist_pow_mean != 0, ref_mean / safe_hist_pow_mean, 1.0)
    return a.compute(), b


def train_power_transformation(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedPowerTransformation:
    """Fit per-month (a, b) power-transform parameters for every calendar month in `hist`."""
    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedPowerTransformation = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m)
    return trained


def apply_power_transformation(sim: xr.DataArray, trained: TrainedPowerTransformation, cfg: dict) -> xr.DataArray:
    """Apply the trained per-month power transform to `sim`, month by month."""
    corrected_months = []
    for month, (a, b) in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append(a * sim_m**b)

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
