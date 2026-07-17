"""Ensemble calibration: reliability diagrams, spread-skill ratio.

Rank histograms already live in validate.py (reused as-is by the CLI's daily
diagnostics); this module adds the two calibration diagnostics that weren't
built yet.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sbc_qdm.verify.aggregate import ensemble_mean


def reliability_diagram_data(
    forecast_prob: xr.DataArray, obs_indicator: xr.DataArray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Domain+sample-pooled reliability diagram data for one binary event.

    Bins forecast probabilities into n_bins equal-width bins and computes the
    observed frequency of the event within each bin. Returns
    (bin_center_forecast_prob, observed_frequency, sample_count_per_bin) --
    pooled across every remaining dim (pixels, years), since a per-pixel
    reliability diagram isn't a meaningful single-pixel quantity (each pixel
    only has ~33 samples, far too few to bin).
    """
    probs = forecast_prob.values.ravel()
    obs = obs_indicator.values.ravel()
    valid = ~(np.isnan(probs) | np.isnan(obs))
    probs, obs = probs[valid], obs[valid]

    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)

    forecast_mean = np.full(n_bins, np.nan)
    observed_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            forecast_mean[b] = probs[mask].mean()
            observed_freq[b] = obs[mask].mean()

    return forecast_mean, observed_freq, counts


def spread_skill_ratio(
    model: xr.DataArray, ref: xr.DataArray, sample_dim: str = "time", realization_dim: str = "realization"
) -> xr.DataArray:
    """Ratio of ensemble spread to ensemble-mean RMSE -- ~1 means a well-calibrated ensemble.

    Spread is the ensemble std averaged over sample_dim; skill is the RMSE of
    the ensemble mean against ref, also over sample_dim.
    """
    spread = model.std(realization_dim).mean(sample_dim, skipna=True)
    mean_forecast = ensemble_mean(model, realization_dim)
    rmse = ((mean_forecast - ref) ** 2).mean(sample_dim, skipna=True) ** 0.5
    return spread / rmse
