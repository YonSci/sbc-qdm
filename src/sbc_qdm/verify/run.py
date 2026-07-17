"""Orchestrates the full scientific evaluation suite: daily / monthly / JJAS-seasonal,
spatial, computed for both raw and QDM-corrected against CHIRPS, and written to
output/evaluation/*.nc plus a curated set of figures under output/evaluation/figures/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr

from sbc_qdm.verify import aggregate as agg
from sbc_qdm.verify import calibration as cal
from sbc_qdm.verify import deterministic as det
from sbc_qdm.verify import distributions as dist
from sbc_qdm.verify import probabilistic as prob
from sbc_qdm.verify import skill
from sbc_qdm.verify import spatial as sp
from sbc_qdm.verify import spells
from sbc_qdm.verify import viz


def _log(msg: str) -> None:
    print(f"[evaluate] {msg}", flush=True)


def _prefix_vars(ds: xr.Dataset, prefix: str) -> xr.Dataset:
    return ds.rename({name: f"{prefix}{name}" for name in ds.data_vars})


def _deterministic_suite(model: xr.DataArray, ref: xr.DataArray, sample_dim: str) -> xr.Dataset:
    return xr.Dataset(
        {
            "mbe": det.mean_bias_error(model, ref, sample_dim=sample_dim),
            "mae": det.mean_absolute_error(model, ref, sample_dim=sample_dim),
            "pbias": det.percentage_bias(model, ref, sample_dim=sample_dim),
            "rmse": det.root_mean_square_error(model, ref, sample_dim=sample_dim),
            "sd_ratio": det.sd_ratio(model, ref, sample_dim=sample_dim),
            "cv_ratio": det.cv_ratio(model, ref, sample_dim=sample_dim),
        }
    )


def _skill_suite(model: xr.DataArray, ref: xr.DataArray, sample_dim: str) -> xr.Dataset:
    return xr.Dataset(
        {
            "acc": skill.anomaly_correlation(model, ref, sample_dim=sample_dim),
            "spearman_acc": skill.spearman_anomaly_correlation(model, ref, sample_dim=sample_dim),
            "rmsess": skill.rmse_skill_score(model, ref, sample_dim=sample_dim),
            "interannual_variability_ratio": skill.interannual_variability_ratio(model, ref, sample_dim=sample_dim),
        }
    )


def _tercile_probabilistic_suite(forecast_probs: xr.DataArray, obs_category: xr.DataArray) -> xr.Dataset:
    rpss = prob.rps_skill_score(forecast_probs, obs_category)
    bss = {}
    rocss = {}
    for category in ("below", "near", "above"):
        obs_ind = prob.obs_indicator_for_category(obs_category, category)
        fc_prob = forecast_probs.sel(category=category)
        bss[category] = prob.brier_skill_score(fc_prob, obs_ind)
        rocss[category] = prob.roc_skill_score(fc_prob, obs_ind)

    cat_dim = xr.DataArray(["below", "near", "above"], dims="category", name="category")
    return xr.Dataset(
        {
            "rpss": rpss,
            "bss": xr.concat([bss[c] for c in ("below", "near", "above")], dim=cat_dim),
            "roc_skill_score": xr.concat([rocss[c] for c in ("below", "near", "above")], dim=cat_dim),
        }
    )


def run_full_evaluation(ref_full: xr.DataArray, hist: xr.DataArray, corrected: xr.DataArray, out_dir: Path) -> None:
    """ref_full: full CHIRPS record (any calendar months). hist/corrected: (lat,lon,time,realization)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ref = ref_full.sel(time=hist["time"])

    # ---- daily scale ---------------------------------------------------
    t0 = time.time()
    daily_raw = _prefix_vars(_deterministic_suite(hist, ref, sample_dim="time"), "raw_")
    daily_corrected = _prefix_vars(_deterministic_suite(corrected, ref, sample_dim="time"), "corrected_")
    daily = xr.merge([daily_raw, daily_corrected])

    qbias_raw = dist.quantile_bias(hist, ref)
    qbias_corrected = dist.quantile_bias(corrected, ref)
    daily["quantile_bias_raw"] = qbias_raw
    daily["quantile_bias_corrected"] = qbias_corrected
    daily["wet_day_freq_bias_raw"] = dist.wet_day_frequency_bias(hist, ref)
    daily["wet_day_freq_bias_corrected"] = dist.wet_day_frequency_bias(corrected, ref)
    daily["spread_skill_ratio_raw"] = cal.spread_skill_ratio(hist, ref)
    daily["spread_skill_ratio_corrected"] = cal.spread_skill_ratio(corrected, ref)
    daily = daily.compute()
    daily.to_netcdf(out_dir / "daily_deterministic.nc")
    _log(f"daily deterministic suite done in {time.time() - t0:.1f}s")

    viz.plot_deterministic_map(daily["raw_mbe"], daily["corrected_mbe"], fig_dir / "daily_mbe.png", "Mean Bias Error (daily)", "mm/day")
    viz.plot_deterministic_map(daily["raw_rmse"], daily["corrected_rmse"], fig_dir / "daily_rmse.png", "RMSE (daily)", "mm/day", diverging=False)
    viz.plot_deterministic_map(daily["raw_pbias"], daily["corrected_pbias"], fig_dir / "daily_pbias.png", "Percentage Bias (daily)", "%")
    viz.plot_quantile_bias_grid(qbias_raw.compute(), qbias_corrected.compute(), fig_dir / "quantile_bias.png")
    viz.plot_spread_skill_map(daily["spread_skill_ratio_corrected"], fig_dir / "spread_skill_ratio.png")

    # ---- distributions (domain-pooled, daily) ---------------------------
    t0 = time.time()
    quantiles, raw_q, ref_q_qq = dist.qq_pairs(hist, ref)
    _, corrected_q, _ = dist.qq_pairs(corrected, ref)
    qq_ds = xr.Dataset(
        {"raw": ("quantile", raw_q), "corrected": ("quantile", corrected_q), "ref": ("quantile", ref_q_qq)},
        coords={"quantile": quantiles},
    )
    qq_ds.to_netcdf(out_dir / "qq_pairs.nc")
    viz.plot_qq(ref_q_qq, raw_q, corrected_q, fig_dir / "qq_plot.png")
    viz.plot_ecdf(ref, hist, corrected, fig_dir / "ecdf.png")
    viz.plot_pdf(ref, hist, corrected, fig_dir / "pdf.png")
    _log(f"distributions (Q-Q/ECDF/PDF) done in {time.time() - t0:.1f}s")

    # ---- wet/dry spell distributions (domain-pooled, daily) -------------
    t0 = time.time()
    obs_wet = spells.spell_lengths(ref, spell_type="wet")
    raw_wet = spells.spell_lengths(hist, spell_type="wet")
    corrected_wet = spells.spell_lengths(corrected, spell_type="wet")
    obs_dry = spells.spell_lengths(ref, spell_type="dry")
    raw_dry = spells.spell_lengths(hist, spell_type="dry")
    corrected_dry = spells.spell_lengths(corrected, spell_type="dry")
    viz.plot_spell_distributions(obs_wet, raw_wet, corrected_wet, obs_dry, raw_dry, corrected_dry, fig_dir / "spell_distributions.png")
    np.savez(
        out_dir / "spell_lengths.npz",
        obs_wet=obs_wet, raw_wet=raw_wet, corrected_wet=corrected_wet,
        obs_dry=obs_dry, raw_dry=raw_dry, corrected_dry=corrected_dry,
    )
    _log(f"wet/dry spell distributions done in {time.time() - t0:.1f}s")

    # ---- spatial performance (daily, across pixels for each day) --------
    t0 = time.time()
    hist_clim = agg.ensemble_mean(hist).mean("time", skipna=True)
    corrected_clim = agg.ensemble_mean(corrected).mean("time", skipna=True)
    ref_clim = ref.mean("time", skipna=True)

    spatial_ds = xr.Dataset(
        {
            "spatial_correlation_raw": sp.spatial_correlation(hist, ref),
            "spatial_correlation_corrected": sp.spatial_correlation(corrected, ref),
            "spatial_pattern_correlation_raw": sp.spatial_pattern_correlation(hist, ref, hist_clim, ref_clim),
            "spatial_pattern_correlation_corrected": sp.spatial_pattern_correlation(corrected, ref, corrected_clim, ref_clim),
            "spatial_rmse_raw": sp.spatial_rmse(hist, ref),
            "spatial_rmse_corrected": sp.spatial_rmse(corrected, ref),
        }
    ).compute()
    spatial_ds.to_netcdf(out_dir / "daily_spatial_timeseries.nc")
    viz.plot_spatial_metric_timeseries(
        spatial_ds["spatial_pattern_correlation_raw"], spatial_ds["spatial_pattern_correlation_corrected"],
        fig_dir / "spatial_pattern_correlation.png", "Spatial pattern correlation", "Daily spatial pattern correlation (anomaly-based): raw vs corrected",
    )
    _log(f"spatial performance (daily) done in {time.time() - t0:.1f}s")

    # ---- monthly scale ---------------------------------------------------
    t0 = time.time()
    ref_m = agg.monthly_totals(ref)
    hist_m = agg.monthly_totals(hist)
    corrected_m = agg.monthly_totals(corrected)
    ref_m = ref_m.sel(month=hist_m["month"].values)

    monthly_raw = _prefix_vars(_deterministic_suite(hist_m, ref_m, sample_dim="year"), "raw_")
    monthly_corrected = _prefix_vars(_deterministic_suite(corrected_m, ref_m, sample_dim="year"), "corrected_")
    monthly_skill_raw = _prefix_vars(_skill_suite(hist_m, ref_m, sample_dim="year"), "raw_")
    monthly_skill_corrected = _prefix_vars(_skill_suite(corrected_m, ref_m, sample_dim="year"), "corrected_")
    monthly = xr.merge([monthly_raw, monthly_corrected, monthly_skill_raw, monthly_skill_corrected]).compute()
    monthly.to_netcdf(out_dir / "monthly_deterministic_and_skill.nc")
    _log(f"monthly scale done in {time.time() - t0:.1f}s")

    # ---- JJAS-seasonal scale ----------------------------------------------
    t0 = time.time()
    ref_j = agg.jjas_totals(ref)
    hist_j = agg.jjas_totals(hist)
    corrected_j = agg.jjas_totals(corrected)

    jjas_raw = _prefix_vars(_deterministic_suite(hist_j, ref_j, sample_dim="year"), "raw_")
    jjas_corrected = _prefix_vars(_deterministic_suite(corrected_j, ref_j, sample_dim="year"), "corrected_")
    jjas_skill_raw = _prefix_vars(_skill_suite(hist_j, ref_j, sample_dim="year"), "raw_")
    jjas_skill_corrected = _prefix_vars(_skill_suite(corrected_j, ref_j, sample_dim="year"), "corrected_")
    jjas = xr.merge([jjas_raw, jjas_corrected, jjas_skill_raw, jjas_skill_corrected]).compute()
    jjas.to_netcdf(out_dir / "jjas_deterministic_and_skill.nc")
    viz.plot_skill_maps(
        jjas["corrected_acc"], jjas["corrected_spearman_acc"], jjas["corrected_rmsess"], jjas["corrected_interannual_variability_ratio"],
        fig_dir / "jjas_skill_maps.png",
    )
    _log(f"JJAS deterministic + skill done in {time.time() - t0:.1f}s")

    # ---- JJAS tercile-based probabilistic metrics -------------------------
    t0 = time.time()
    thresholds = agg.tercile_thresholds(ref_j)
    obs_category = agg.tercile_category(ref_j, thresholds)
    raw_probs = agg.ensemble_tercile_probabilities(hist_j, thresholds)
    corrected_probs = agg.ensemble_tercile_probabilities(corrected_j, thresholds)

    jjas_prob_raw = _prefix_vars(_tercile_probabilistic_suite(raw_probs, obs_category), "raw_")
    jjas_prob_corrected = _prefix_vars(_tercile_probabilistic_suite(corrected_probs, obs_category), "corrected_")
    jjas_crps = prob.crps_skill_score(ref_j, hist_j, corrected_j, sample_dim="year")
    jjas_prob = xr.merge([jjas_prob_raw, jjas_prob_corrected, jjas_crps]).compute()
    jjas_prob.to_netcdf(out_dir / "jjas_probabilistic.nc")

    viz.plot_probabilistic_skill_maps(
        jjas_prob["corrected_rpss"], jjas_prob["corrected_bss"].sel(category="above"), jjas_prob["corrected_roc_skill_score"].sel(category="above"),
        fig_dir / "jjas_probabilistic_skill.png",
    )

    obs_ind_above = prob.obs_indicator_for_category(obs_category, "above")
    fmean, ofreq, counts = cal.reliability_diagram_data(corrected_probs.sel(category="above"), obs_ind_above)
    viz.plot_reliability_diagram(fmean, ofreq, counts, fig_dir / "reliability_diagram.png")
    _log(f"JJAS probabilistic (RPSS/BSS/ROC) done in {time.time() - t0:.1f}s")

    _log(f"Full evaluation suite written to {out_dir}")
