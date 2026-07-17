"""Probabilistic ensemble metrics: CRPS/CRPSS (generalized to any sample_dim),
RPSS, Brier Score/BSS, ROC area/skill.

RPSS/Brier/ROC all score against the tercile categories from aggregate.py
(below/near/above-normal, defined from CHIRPS' own climatology) -- the
standard WMO/IRI convention for seasonal forecast verification.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import rankdata

from sbc_qdm.validate import crps_ensemble

CATEGORY_CODES = {"below": 0, "near": 1, "above": 2}
N_CATEGORIES = 3


def crps_skill_score(ref: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray, sample_dim: str = "time") -> xr.Dataset:
    """CRPS pre/post correction and the skill score, generalized to any sample_dim
    (e.g. "year" for monthly/JJAS-aggregated ensembles, vs the daily "time" default).
    """
    crps_raw = crps_ensemble(ref, raw).mean(sample_dim)
    crps_corrected = crps_ensemble(ref, corrected).mean(sample_dim)
    crpss = 1 - crps_corrected / crps_raw
    return xr.Dataset({"crps_raw": crps_raw, "crps_corrected": crps_corrected, "crpss": crpss})


def obs_indicator_for_category(obs_category: xr.DataArray, category: str) -> xr.DataArray:
    """1.0 where obs_category equals the named category ('below'/'near'/'above'), else 0.0."""
    code = CATEGORY_CODES[category]
    return (obs_category == code).astype(float).where(obs_category.notnull())


def ranked_probability_score(forecast_probs: xr.DataArray, obs_category: xr.DataArray, sample_dim: str = "year", category_dim: str = "category") -> xr.DataArray:
    """Mean RPS over sample_dim. forecast_probs: (..., category) fractions summing to 1."""
    # apply_ufunc requires the core dim (category) to be a single chunk.
    forecast_probs = forecast_probs.chunk({category_dim: -1})

    def _rps(probs, obs_cat):
        n_cat = probs.shape[-1]
        cum_forecast = np.cumsum(probs, axis=-1)
        k = np.arange(n_cat)
        cum_obs = (obs_cat.astype(int)[..., None] <= k).astype(float)
        return np.sum((cum_forecast - cum_obs) ** 2, axis=-1)

    rps = xr.apply_ufunc(
        _rps,
        forecast_probs,
        obs_category,
        input_core_dims=[[category_dim], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return rps.mean(sample_dim, skipna=True)


def rps_skill_score(forecast_probs: xr.DataArray, obs_category: xr.DataArray, sample_dim: str = "year", category_dim: str = "category") -> xr.DataArray:
    """RPSS = 1 - RPS(model)/RPS(climatology), climatology = equal 1/N_CATEGORIES probability."""
    rps_model = ranked_probability_score(forecast_probs, obs_category, sample_dim, category_dim)
    clim_probs = xr.full_like(forecast_probs, 1.0 / N_CATEGORIES)
    rps_clim = ranked_probability_score(clim_probs, obs_category, sample_dim, category_dim)
    return 1 - rps_model / rps_clim


def brier_score(forecast_prob: xr.DataArray, obs_indicator: xr.DataArray, sample_dim: str = "year") -> xr.DataArray:
    return ((forecast_prob - obs_indicator) ** 2).mean(sample_dim, skipna=True)


def brier_skill_score(forecast_prob: xr.DataArray, obs_indicator: xr.DataArray, sample_dim: str = "year", climatology_prob: float = 1.0 / N_CATEGORIES) -> xr.DataArray:
    """BSS = 1 - BS(model)/BS(climatology), climatology = constant 1/N_CATEGORIES probability."""
    bs_model = brier_score(forecast_prob, obs_indicator, sample_dim)
    bs_clim = brier_score(xr.full_like(forecast_prob, climatology_prob), obs_indicator, sample_dim)
    return 1 - bs_model / bs_clim


def roc_area(forecast_prob: xr.DataArray, obs_indicator: xr.DataArray, sample_dim: str = "year") -> xr.DataArray:
    """Area under the ROC curve via the Mann-Whitney U statistic (rank-based, exact
    for any number of forecast probability levels -- no threshold sweep needed).
    """
    # apply_ufunc requires the core dim (sample_dim) to be a single chunk.
    forecast_prob = forecast_prob.chunk({sample_dim: -1})
    obs_indicator = obs_indicator.chunk({sample_dim: -1})

    def _auc(probs, obs):
        obs = obs.astype(bool)
        n_pos = int(obs.sum())
        n_neg = int((~obs).sum())
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = rankdata(probs)
        return (ranks[obs].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return xr.apply_ufunc(
        _auc,
        forecast_prob,
        obs_indicator,
        input_core_dims=[[sample_dim], [sample_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def roc_skill_score(forecast_prob: xr.DataArray, obs_indicator: xr.DataArray, sample_dim: str = "year") -> xr.DataArray:
    """2*(ROC area - 0.5): 0 = no skill, 1 = perfect discrimination."""
    return 2 * (roc_area(forecast_prob, obs_indicator, sample_dim) - 0.5)
