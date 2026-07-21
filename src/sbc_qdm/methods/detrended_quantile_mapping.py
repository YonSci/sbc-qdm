"""Detrended Quantile Mapping (DQM) bias correction.

Cannon et al. (2015)'s DQM: like Empirical Quantile Mapping, but normalizes
the target period by its own mean before quantile-mapping (using the
training period's ref_q/hist_q), then re-applies the target's own mean-shift
multiplicatively afterward. This preserves the target period's own signal
(e.g. a held-out year that happens to be much wetter or drier than the
training climatology as a whole) instead of discarding it the way EQM's
direct substitution does -- EQM would map two raw values at the same
quantile to the exact same corrected value even if one came from an
unusually wet year and one from an unusually dry one; DQM keeps them
distinguishable by rescaling around the target period's own mean:

    mu_hist = training hist_month mean (per pixel, pooled over training years+members)
    mu_target = target/held-out period's own mean (per pixel, from its own raw
                values only -- uses no reference/observation data, so this is
                not leakage)
    raw_normalized = raw * (mu_hist / mu_target)
    tau = interp(raw_normalized, hist_q, quantiles)
    qm = interp(tau, quantiles, ref_q)
    corrected = qm * (mu_target / mu_hist)

Reuses QDM's quantile-node grid and `adapt_freq` preprocessing (same as
empirical_quantile_mapping.py) so the QDM/EQM/DQM three-way comparison isn't
confounded by different quantile resolution or wet-day handling -- the only
difference between the three is the final mapping step.

Trained state per month: (ref_q, hist_q, mu_hist). `mu_target` is computed
at apply-time directly from whatever `sim` is passed in (the held-out year
during cross-validation, or the live operational forecast in production).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from xsdba import Grouper
from xsdba.processing import adapt_freq

from sbc_qdm.chunking import rechunk_for_grouping, sample_dims
from sbc_qdm.qdm import evaluation_quantiles

TrainedDQM = dict[int, tuple[xr.DataArray, xr.DataArray, xr.DataArray]]


def _train_one_month(ref_m: xr.DataArray, hist_m: xr.DataArray, quantiles: np.ndarray, adapt_freq_thresh: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    ref_m = rechunk_for_grouping(ref_m)
    hist_m = rechunk_for_grouping(hist_m)
    hist_adj, _pth, _dP0 = adapt_freq(ref_m, hist_m, group=Grouper("time"), thresh=adapt_freq_thresh)

    ref_q = ref_m.quantile(quantiles, dim="time", skipna=True).rename({"quantile": "quantiles"})
    hist_q = hist_adj.quantile(quantiles, dim=sample_dims(hist_adj), skipna=True).rename({"quantile": "quantiles"})
    mu_hist = hist_m.mean(sample_dims(hist_m), skipna=True)
    return ref_q.compute(), hist_q.compute(), mu_hist.compute()


def train_detrended_quantile_mapping(ref: xr.DataArray, hist: xr.DataArray, cfg: dict) -> TrainedDQM:
    """Fit one DQM (ref_q, hist_q, mu_hist) per calendar month present in `hist`.

    Reads nquantiles/tail_quantiles/adapt_freq_thresh from cfg["qdm"], same as
    empirical_quantile_mapping.py, for a fair QDM/EQM/DQM comparison.
    """
    n = cfg["qdm"]["nquantiles"]
    quantiles = evaluation_quantiles(n, tuple(cfg["qdm"].get("tail_quantiles", ())))
    thresh = cfg["qdm"]["adapt_freq_thresh"]
    np.random.seed(cfg["qdm"].get("random_seed", 0))

    months = sorted(set(hist.time.dt.month.values.tolist()))
    trained: TrainedDQM = {}
    for month in months:
        ref_m = ref.sel(time=ref.time.dt.month == month)
        hist_m = hist.sel(time=hist.time.dt.month == month)
        trained[month] = _train_one_month(ref_m, hist_m, quantiles, thresh)
    return trained


def _adjust_one_month(sim_m: xr.DataArray, ref_q: xr.DataArray, hist_q: xr.DataArray, mu_hist: xr.DataArray, quantiles: np.ndarray) -> xr.DataArray:
    sim_m = rechunk_for_grouping(sim_m)
    dims = sample_dims(sim_m)

    mu_target = sim_m.mean(dims, skipna=True)
    safe_mu_target = mu_target.where(mu_target != 0, 1.0)
    to_training_scale = xr.where(mu_target != 0, mu_hist / safe_mu_target, 1.0)
    from_training_scale = xr.where(mu_target != 0, mu_target / mu_hist.where(mu_hist != 0, 1.0), 1.0)

    sim_normalized = sim_m * to_training_scale

    def _adjust_pixel(x, hist_q_1d, ref_q_1d):
        tau = np.interp(x, hist_q_1d, quantiles)
        return np.interp(tau, quantiles, ref_q_1d)

    qm = xr.apply_ufunc(
        _adjust_pixel,
        sim_normalized, hist_q, ref_q,
        input_core_dims=[dims, ["quantiles"], ["quantiles"]],
        output_core_dims=[dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sim_m.dtype],
    )
    return qm * from_training_scale


def apply_detrended_quantile_mapping(sim: xr.DataArray, trained: TrainedDQM, cfg: dict) -> xr.DataArray:
    """Apply a trained per-month DQM (see train_detrended_quantile_mapping) to `sim`, month by month."""
    n = cfg["qdm"]["nquantiles"]
    quantiles = evaluation_quantiles(n, tuple(cfg["qdm"].get("tail_quantiles", ())))

    corrected_months = []
    for month, (ref_q, hist_q, mu_hist) in trained.items():
        sim_m = sim.sel(time=sim.time.dt.month == month)
        if sim_m.sizes.get("time", 0) == 0:
            continue
        corrected_months.append(_adjust_one_month(sim_m, ref_q, hist_q, mu_hist, quantiles))

    corrected = xr.concat(corrected_months, dim="time").sortby("time")
    corrected.attrs.update(sim.attrs)
    return corrected
