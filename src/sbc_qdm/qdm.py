"""Quantile Delta Mapping training/application.

Why this is hand-rolled instead of xsdba.QuantileDeltaMapping directly:
xsdba's calendar-aware Grouper ("time.dayofyear" / "time.month") crashes inside
`.adjust()`'s interp_on_quantiles step whenever the training/adjustment data
doesn't span a full calendar year -- reproduced even when adjusting the exact
array used for training, on both xsdba 0.5.0 and 0.7.0. A single May-init
seasonal forecast season (183 days) never spans a full year, so this bites
unconditionally here. An *ungrouped* Grouper("time") works fine, so we do the
per-month grouping ourselves in plain xarray and reuse only the pieces of
xsdba confirmed to work: `adapt_freq` (wet-day frequency correction).

Design:
  - One multiplicative QDM fit per calendar month (May..Oct), pooling all
    hindcast years and all 51 ensemble members as training samples for that
    month -- this is the "add_dims" pooling idea, just done manually since
    xsdba's own add_dims path is entangled with the broken Grouper.
  - adapt_freq corrects wet-day frequency mismatch before fitting quantiles.
  - Adjustment factors are interpolated linearly between quantile nodes, with
    constant extrapolation beyond the observed range (numpy.interp's default
    behaviour already does this -- no extra code needed).
  - leave_one_year_out() reproduces train+adjust once per hindcast year,
    excluding that year from training, for unbiased skill validation.
    apply_operational() fits on the full hindcast record and corrects the
    live forecast year.
"""

from __future__ import annotations

import time

import dask
import numpy as np
import xarray as xr
from xsdba import Grouper
from xsdba.processing import adapt_freq

# This machine runs with very little free memory in practice (observed as low
# as ~2.4 GiB available under normal desktop load -- VS Code, browser, etc.),
# and a MemoryError as small as ~37 MiB surfaced even in a single fresh
# subprocess with no other work competing. Capping dask's thread pool bounds
# how many chunks are processed concurrently (and therefore peak memory),
# trading some speed for reliability under that constraint.
dask.config.set(scheduler="threads", num_workers=2)

SAMPLE_DIMS = ("time", "realization")


def equally_spaced_quantiles(n: int) -> np.ndarray:
    """n quantile nodes centered in n equal-width bins of [0, 1] (xclim's convention)."""
    return np.arange(1, 2 * n, 2) / (2 * n)


def _sample_dims(da: xr.DataArray) -> list[str]:
    return [d for d in SAMPLE_DIMS if d in da.dims]


SPATIAL_CHUNK = 5


def _rechunk_for_grouping(da: xr.DataArray) -> xr.DataArray:
    """Single chunk along time/realization (required by xsdba's grouping), tiled over lat/lon.

    Materializing a whole month's slice across the full spatial domain at
    once (48x60 pixels x ~930 samples x 51 members) is ~1 GiB and repeatedly
    hit MemoryError under Windows' allocator once xsdba's internal groupby
    needed a second same-sized scratch array. Keeping lat/lon dask-chunked
    bounds peak memory per block to a few tens of MB regardless of domain size.
    A 37 MiB single-block allocation still failed under this machine's actual
    available memory (as low as ~2.4 GiB free under normal desktop load), so
    this is tiled smaller (5x5) than the minimum that failed.
    """
    chunks = {d: -1 for d in SAMPLE_DIMS if d in da.dims}
    chunks.update({d: SPATIAL_CHUNK for d in ("lat", "lon") if d in da.dims})
    return da.chunk(chunks)


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray, quantiles: np.ndarray, adapt_freq_thresh: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Empirical quantiles of ref/adapted-hist for one month, and the multiplicative adjustment factor.

    adapt_freq injects uniform random noise below its wet-day threshold (see
    its docstring); train_qdm() seeds numpy's global RNG before calling this
    so that repeated runs on the same inputs are reproducible.
    """
    ref_m = _rechunk_for_grouping(ref_m)
    hist_m = _rechunk_for_grouping(hist_m)
    hist_adj, _pth, _dP0 = adapt_freq(ref_m, hist_m, group=Grouper("time"), thresh=adapt_freq_thresh)

    ref_q = ref_m.quantile(quantiles, dim="time", skipna=True).rename({"quantile": "quantiles"})
    hist_q = hist_adj.quantile(quantiles, dim=_sample_dims(hist_adj), skipna=True).rename({"quantile": "quantiles"})
    # Divide by a zero-free stand-in for hist_q (value doesn't matter where
    # hist_q==0, since xr.where discards it there anyway) so numpy never
    # actually computes a 0/0 or x/0 and warns about it.
    safe_hist_q = hist_q.where(hist_q != 0, 1.0)
    af = xr.where(hist_q != 0, ref_q / safe_hist_q, 1.0)
    return af.compute(), hist_q.compute()


def train_qdm(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> dict[int, tuple[xr.DataArray, xr.DataArray]]:
    """Train one multiplicative QDM per calendar month present in `hist`.

    ref: CHIRPS reference, dims (time, lat, lon), restricted to hist's time range.
    hist: ECMWF hindcast, dims (realization, time, lat, lon).
    Returns {month: (af, hist_q)}, each indexed by (quantiles, lat, lon).
    """
    n = cfg["qdm"]["nquantiles"]
    quantiles = equally_spaced_quantiles(n)
    thresh = cfg["qdm"]["adapt_freq_thresh"]
    np.random.seed(cfg["qdm"].get("random_seed", 0))

    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m, quantiles, thresh)
    return trained


def _adjust_one_month(sim_m: xr.DataArray, af: xr.DataArray, hist_q: xr.DataArray, quantiles: np.ndarray) -> xr.DataArray:
    """Map sim through hist's empirical CDF, then through the ref/hist adjustment factor.

    apply_ufunc's dask="parallelized" requires core dims (time/realization
    here) to be a single chunk; regrid.py leaves the source data chunked
    along those dims for memory-bounded regridding, so it's rechunked here
    the same way training does (single chunk on time/realization, tiled over
    lat/lon to keep peak memory bounded regardless of domain size).
    """
    sim_m = _rechunk_for_grouping(sim_m)

    def _adjust_pixel(x, hist_q_1d, af_1d):
        tau = np.interp(x, hist_q_1d, quantiles)
        factor = np.interp(tau, quantiles, af_1d)
        return x * factor

    sample_dims = _sample_dims(sim_m)
    return xr.apply_ufunc(
        _adjust_pixel,
        sim_m, hist_q, af,
        input_core_dims=[sample_dims, ["quantiles"], ["quantiles"]],
        output_core_dims=[sample_dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sim_m.dtype],
    )


def apply_qdm(sim: xr.DataArray, trained: dict[int, tuple[xr.DataArray, xr.DataArray]], cfg: dict) -> xr.DataArray:
    """Apply a trained per-month QDM (see train_qdm) to `sim`, month by month."""
    n = cfg["qdm"]["nquantiles"]
    quantiles = equally_spaced_quantiles(n)

    corrected_months = []
    for month, (af, hist_q) in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append(_adjust_one_month(sim_m, af, hist_q, quantiles))

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected


def leave_one_year_out(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> xr.DataArray:
    """Cross-validated correction: train on all-but-one hindcast year, correct that year, repeat.

    ref and hist must already share the same time axis (e.g. via
    `chirps.sel(time=hist.time)` before calling this).
    """
    years = sorted(set(hist.time.dt.year.values.tolist()))
    corrected_years = []
    for i, year in enumerate(years):
        t0 = time.time()
        is_held_out = hist.time.dt.year == year
        ref_train = ref.sel(time=~is_held_out)
        hist_train = hist.sel(time=~is_held_out)
        sim_holdout = hist.sel(time=is_held_out)

        trained = train_qdm(ref_train, hist_train, cfg)
        # computed per-fold (not deferred) so peak memory stays bounded to one
        # fold's size instead of one huge lazy graph over all 33 years at the
        # final write.
        corrected_years.append(apply_qdm(sim_holdout, trained, cfg).compute())
        print(f"leave_one_year_out: fold {i + 1}/{len(years)} (year {year}) done in {time.time() - t0:.1f}s", flush=True)

    return xr.concat(corrected_years, dim="time").sortby("time")


def apply_operational(ref: xr.DataArray, hist: xr.DataArray, sim: xr.DataArray, cfg: dict) -> xr.DataArray:
    """Train on the full hindcast record and correct the live operational forecast."""
    trained = train_qdm(ref, hist, cfg)
    return apply_qdm(sim, trained, cfg)
