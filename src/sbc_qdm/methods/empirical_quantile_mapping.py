"""Empirical Quantile Mapping (EQM) bias correction.

Plain quantile mapping: map a raw value onto the reference's value at the
same quantile, directly substituting it rather than computing a ratio/delta
and applying that to the raw value (contrast with qdm.py's
Quantile Delta Mapping, which QDM -- Cannon et al. 2015 -- was specifically
designed to improve on). EQM corrects the full shape of the distribution
like QDM does, but discards the raw value's own magnitude beyond its rank:
two raw values at the same quantile always map to the exact same corrected
value, regardless of how the target period's own distribution might differ
from the training period's.

Reuses QDM's quantile-node grid (`qdm.evaluation_quantiles`, including the
tail-concentrated nodes) and `adapt_freq` wet-day-frequency preprocessing, so
the comparison against QDM isn't confounded by different quantile
resolution or different drizzle handling -- the only difference is the final
substitution-vs-ratio mapping step.

    tau = interp(raw, hist_q, quantiles)
    corrected = interp(tau, quantiles, ref_q)

Trained state per month: (ref_q, hist_q), both indexed by (quantiles, lat, lon).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from xsdba import Grouper
from xsdba.processing import adapt_freq

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims
from sbc_qdm.qdm import evaluation_quantiles

TrainedEQM = dict[int, tuple[xr.DataArray, xr.DataArray]]


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray, quantiles: np.ndarray, adapt_freq_thresh: str) -> tuple[xr.DataArray, xr.DataArray]:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)
    hist_adj, _pth, _dP0 = adapt_freq(ref_m, hist_m, group=Grouper("time"), thresh=adapt_freq_thresh)

    ref_q = ref_m.quantile(quantiles, dim="time", skipna=True).rename({"quantile": "quantiles"})
    hist_q = hist_adj.quantile(quantiles, dim=sample_dims(hist_adj), skipna=True).rename({"quantile": "quantiles"})
    return ref_q.compute(), hist_q.compute()


def train_empirical_quantile_mapping(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedEQM:
    """Fit one empirical quantile mapping per calendar month present in `hist`.

    Reads nquantiles/tail_quantiles/adapt_freq_thresh from cfg["qdm"] (not a
    separate config block) so EQM and QDM are compared on identical quantile
    resolution and wet-day handling.
    """
    n = cfg["qdm"]["nquantiles"]
    quantiles = evaluation_quantiles(n, tuple(cfg["qdm"].get("tail_quantiles", ())))
    thresh = cfg["qdm"]["adapt_freq_thresh"]
    np.random.seed(cfg["qdm"].get("random_seed", 0))

    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedEQM = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m, quantiles, thresh)
    return trained


def _adjust_one_month(sim_m: xr.DataArray, ref_q: xr.DataArray, hist_q: xr.DataArray, quantiles: np.ndarray) -> xr.DataArray:
    sim_m = rechunk_for_grouping(sim_m)

    def _adjust_pixel(x, hist_q_1d, ref_q_1d):
        tau = np.interp(x, hist_q_1d, quantiles)
        return np.interp(tau, quantiles, ref_q_1d)

    dims = sample_dims(sim_m)
    return xr.apply_ufunc(
        _adjust_pixel,
        sim_m, hist_q, ref_q,
        input_core_dims=[dims, ["quantiles"], ["quantiles"]],
        output_core_dims=[dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sim_m.dtype],
    )


def apply_empirical_quantile_mapping(sim: xr.DataArray, trained: TrainedEQM, cfg: dict) -> xr.DataArray:
    """Apply a trained per-month EQM (see train_empirical_quantile_mapping) to `sim`, month by month."""
    n = cfg["qdm"]["nquantiles"]
    quantiles = evaluation_quantiles(n, tuple(cfg["qdm"].get("tail_quantiles", ())))

    corrected_months = []
    for month, (ref_q, hist_q) in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append(_adjust_one_month(sim_m, ref_q, hist_q, quantiles))

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
