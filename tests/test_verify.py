"""Unit tests for the src/sbc_qdm/verify/ scientific evaluation suite.

Unlike test_io_preprocess.py / test_regrid_qdm.py (which exercise the real
pipeline against real project data), these are synthetic: small, hand-
constructed arrays with known expected outputs, so they don't depend on the
large local data/ directory and can run in CI. See conftest.py for the
"requires_data" marker used by the other two test files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sbc_qdm.verify import aggregate as agg
from sbc_qdm.verify import calibration as cal
from sbc_qdm.verify import deterministic as det
from sbc_qdm.verify import distributions as dist
from sbc_qdm.verify import probabilistic as prob
from sbc_qdm.verify import skill
from sbc_qdm.verify import spatial as sp
from sbc_qdm.verify import spells

# ---------------------------------------------------------------------------
# aggregate.py
# ---------------------------------------------------------------------------


def _daily_series(start, end, value=1.0, lat=2, lon=2):
    time = pd.date_range(start, end, freq="D")
    data = np.full((len(time), lat, lon), value)
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time})


def test_ensemble_mean_averages_realization_and_passes_through_without_it():
    da = xr.DataArray([[1.0, 2.0, 3.0]], dims=("x", "realization"))
    assert float(agg.ensemble_mean(da).isel(x=0)) == pytest.approx(2.0)

    no_ens = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    assert agg.ensemble_mean(no_ens).identical(no_ens)


def test_monthly_totals_sums_days_within_each_calendar_month():
    da = _daily_series("2020-01-01", "2020-02-29")  # Jan (31d) + Feb (29d, leap)
    monthly = agg.monthly_totals(da)
    assert set(monthly["month"].values.tolist()) == {1, 2}
    assert float(monthly.sel(year=2020, month=1).isel(lat=0, lon=0)) == pytest.approx(31.0)
    assert float(monthly.sel(year=2020, month=2).isel(lat=0, lon=0)) == pytest.approx(29.0)


def test_jjas_totals_sums_only_june_through_september():
    da = _daily_series("2020-01-01", "2020-12-31")
    jjas = agg.jjas_totals(da)
    assert jjas.sizes["year"] == 1
    # Jun(30)+Jul(31)+Aug(31)+Sep(30) = 122 days, each contributing 1.0
    assert float(jjas.sel(year=2020).isel(lat=0, lon=0)) == pytest.approx(122.0)


def test_climatology_averages_across_years():
    years = xr.DataArray([1, 2, 3], dims="year")
    da = xr.DataArray([[10.0], [20.0], [30.0]], dims=("year", "lat"), coords={"year": years})
    clim = agg.climatology(da)
    assert float(clim.isel(lat=0)) == pytest.approx(20.0)


def test_anomaly_subtracts_climatology():
    da = xr.DataArray([1.0, 2.0, 3.0], dims="year", coords={"year": [1, 2, 3]})
    anom = agg.anomaly(da)
    assert np.allclose(anom.values, [-1.0, 0.0, 1.0])


def test_tercile_thresholds_and_category_split_evenly():
    # 30 evenly-spaced samples -> thresholds should land near the 10th/20th values
    values = xr.DataArray(np.arange(1, 31, dtype=float), dims="year", coords={"year": np.arange(30)})
    thresholds = agg.tercile_thresholds(values)
    category = agg.tercile_category(values, thresholds)
    counts = {c: int((category == c).sum()) for c in (0, 1, 2)}
    assert counts[0] + counts[1] + counts[2] == 30
    # exact tercile split for evenly-spaced data pooled from itself
    assert counts[0] == pytest.approx(10, abs=2)
    assert counts[2] == pytest.approx(10, abs=2)


def test_ensemble_tercile_probabilities_sum_to_one():
    years = np.arange(5)
    ref = xr.DataArray(np.linspace(1, 10, 5), dims="year", coords={"year": years})
    thresholds = agg.tercile_thresholds(ref)

    ensemble = xr.DataArray(
        np.random.default_rng(0).uniform(1, 10, size=(5, 8)),
        dims=("year", "realization"),
        coords={"year": years},
    )
    probs = agg.ensemble_tercile_probabilities(ensemble, thresholds)
    totals = probs.sum("category")
    assert np.allclose(totals.values, 1.0)
    assert bool((probs >= 0).all()) and bool((probs <= 1).all())


def test_iter_spatial_blocks_covers_every_pixel_exactly_once():
    lat, lon = np.arange(7), np.arange(11)  # deliberately not a multiple of block size
    da = xr.DataArray(np.arange(len(lat) * len(lon)).reshape(len(lat), len(lon)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    seen = set()
    for block in agg.iter_spatial_blocks(da, block=4):
        for la in block["lat"].values:
            for lo in block["lon"].values:
                key = (int(la), int(lo))
                assert key not in seen, "pixel visited more than once"
                seen.add(key)
    assert seen == {(int(la), int(lo)) for la in lat for lo in lon}


# ---------------------------------------------------------------------------
# deterministic.py
# ---------------------------------------------------------------------------


def test_deterministic_metrics_on_constant_bias():
    time = np.arange(10)
    ref = xr.DataArray(np.full(10, 5.0), dims="time", coords={"time": time})
    model = ref + 2.0  # constant +2 bias, no timing error

    assert float(det.mean_bias_error(model, ref, sample_dim="time")) == pytest.approx(2.0)
    assert float(det.mean_absolute_error(model, ref, sample_dim="time")) == pytest.approx(2.0)
    assert float(det.root_mean_square_error(model, ref, sample_dim="time")) == pytest.approx(2.0)
    assert float(det.percentage_bias(model, ref, sample_dim="time")) == pytest.approx(100 * 2.0 / 5.0)


def test_sd_ratio_and_cv_ratio_scale_correctly():
    time = np.arange(20)
    ref = xr.DataArray(np.linspace(1, 20, 20), dims="time", coords={"time": time})
    model = ref * 2.0  # doubles both mean and std -> CV unchanged, SD ratio = 2

    assert float(det.sd_ratio(model, ref, sample_dim="time")) == pytest.approx(2.0)
    assert float(det.cv_ratio(model, ref, sample_dim="time")) == pytest.approx(1.0, abs=1e-6)


def test_deterministic_metrics_average_ensemble_first():
    time = np.arange(5)
    ref = xr.DataArray(np.full(5, 5.0), dims="time", coords={"time": time})
    ensemble = xr.DataArray(
        np.stack([np.full(5, 4.0), np.full(5, 6.0)], axis=-1),  # mean = 5.0 -> zero bias
        dims=("time", "realization"),
        coords={"time": time},
    )
    assert float(det.mean_bias_error(ensemble, ref, sample_dim="time")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# distributions.py
# ---------------------------------------------------------------------------


def _grid_series(values, lat=2, lon=2):
    time = np.arange(len(values))
    data = np.broadcast_to(np.asarray(values)[:, None, None], (len(values), lat, lon)).copy()
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time})


def test_qq_pairs_identical_for_matching_distributions():
    values = np.linspace(0, 100, 50)
    da = _grid_series(values)
    quantiles, model_q, ref_q = dist.qq_pairs(da, da, n_quantiles=11)
    assert np.allclose(model_q, ref_q)


def test_ecdf_is_sorted_and_bounded():
    rng = np.random.default_rng(1)
    da = _grid_series(rng.exponential(size=200))
    x, y = dist.ecdf(da, n_points=50)
    assert np.all(np.diff(x) >= 0)
    assert y.min() >= 0 and y.max() <= 1
    assert np.all(np.diff(y) >= -1e-12)


def test_pdf_histogram_integrates_to_one():
    rng = np.random.default_rng(2)
    da = _grid_series(rng.normal(loc=10, scale=2, size=300))
    centers, density = dist.pdf_histogram(da, bins=20)
    bin_width = centers[1] - centers[0]
    assert float(np.sum(density) * bin_width) == pytest.approx(1.0, abs=0.05)


def test_quantile_bias_zero_for_identical_inputs():
    da = _grid_series(np.linspace(0, 50, 40))
    bias = dist.quantile_bias(da, da, quantiles=(0.1, 0.5, 0.9), sample_dims=("time",))
    assert np.allclose(bias.values, 0.0)


def test_wet_day_frequency_bias_matches_manual_fraction():
    model = _grid_series([0, 0, 2, 2, 2])  # 3/5 wet at >1mm
    ref = _grid_series([0, 2, 2, 0, 0])  # 2/5 wet
    bias = dist.wet_day_frequency_bias(model, ref, threshold_mm=1.0)
    assert float(bias.isel(lat=0, lon=0)) == pytest.approx(3 / 5 - 2 / 5)


# ---------------------------------------------------------------------------
# spells.py
# ---------------------------------------------------------------------------


def test_spell_lengths_matches_hand_counted_runs():
    # wet (>1): [2,2,2]->3, [3]->1, [5,5]->2 ; dry (<=1): [0,0]->2, [0]->1, [0,0]->2
    series = [0, 0, 2, 2, 2, 0, 3, 0, 0, 5, 5]
    da = xr.DataArray(np.array(series)[:, None, None], dims=("time", "lat", "lon"))

    wet = sorted(spells.spell_lengths(da, threshold_mm=1.0, spell_type="wet").tolist())
    dry = sorted(spells.spell_lengths(da, threshold_mm=1.0, spell_type="dry").tolist())
    assert wet == [1, 2, 3]
    assert dry == [1, 2, 2]


def test_spell_length_histogram_sums_to_one():
    lengths = np.array([1, 1, 2, 3, 3, 3])
    x, y = spells.spell_length_histogram(lengths)
    assert float(y.sum()) == pytest.approx(1.0)
    assert float(y[x == 3][0]) == pytest.approx(3 / 6)


def test_spell_lengths_pools_across_pixels():
    # two independent pixels, each with one wet run of a different known length
    col_a = [2, 2, 0, 0]  # wet run length 2
    col_b = [0, 2, 2, 2]  # wet run length 3
    data = np.stack([col_a, col_b], axis=-1)[:, :, None]  # (time, lat=2, lon=1)
    da = xr.DataArray(data, dims=("time", "lat", "lon"))
    wet = sorted(spells.spell_lengths(da, threshold_mm=1.0, spell_type="wet").tolist())
    assert wet == [2, 3]


# ---------------------------------------------------------------------------
# skill.py
# ---------------------------------------------------------------------------


def _year_series(values):
    years = np.arange(len(values))
    return xr.DataArray(np.asarray(values, dtype=float), dims="year", coords={"year": years})


def test_anomaly_correlation_perfect_for_identical_series():
    ref = _year_series([1, 5, 2, 8, 3, 9, 4])
    assert float(skill.anomaly_correlation(ref, ref, sample_dim="year")) == pytest.approx(1.0, abs=1e-9)


def test_anomaly_correlation_minus_one_for_inverted_series():
    ref = _year_series([1, 5, 2, 8, 3, 9, 4])
    model = -ref
    assert float(skill.anomaly_correlation(model, ref, sample_dim="year")) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_anomaly_correlation_perfect_for_identical_series():
    ref = _year_series([1, 5, 2, 8, 3, 9, 4])
    assert float(skill.spearman_anomaly_correlation(ref, ref, sample_dim="year")) == pytest.approx(1.0, abs=1e-9)


def test_rmse_skill_score_perfect_and_climatology_baseline():
    ref = _year_series([1, 5, 2, 8, 3, 9, 4])
    perfect = ref
    assert float(skill.rmse_skill_score(perfect, ref, sample_dim="year")) == pytest.approx(1.0, abs=1e-9)

    climatology_forecast = _year_series([float(ref.mean())] * ref.sizes["year"])
    assert float(skill.rmse_skill_score(climatology_forecast, ref, sample_dim="year")) == pytest.approx(0.0, abs=1e-9)


def test_interannual_variability_ratio_unaffected_by_constant_offset():
    ref = _year_series([1, 5, 2, 8, 3, 9, 4])
    model = ref + 100.0
    assert float(skill.interannual_variability_ratio(model, ref, sample_dim="year")) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# probabilistic.py
# ---------------------------------------------------------------------------


def test_obs_indicator_for_category_matches_codes():
    obs_category = xr.DataArray([0, 1, 2, 0, 1, 2], dims="year")
    below = prob.obs_indicator_for_category(obs_category, "below")
    above = prob.obs_indicator_for_category(obs_category, "above")
    assert np.allclose(below.values, [1, 0, 0, 1, 0, 0])
    assert np.allclose(above.values, [0, 0, 1, 0, 0, 1])


def test_ranked_probability_score_matches_hand_calculation():
    # forecast [below=0.5, near=0.3, above=0.2], true category = above (2)
    probs = xr.DataArray([[0.5, 0.3, 0.2]], dims=("year", "category"), coords={"category": ["below", "near", "above"]})
    obs_category = xr.DataArray([2], dims="year")
    rps = prob.ranked_probability_score(probs, obs_category, sample_dim="year")
    expected = (0.5 - 0) ** 2 + (0.8 - 0) ** 2 + (1.0 - 1) ** 2
    assert float(rps) == pytest.approx(expected)


def test_ranked_probability_score_zero_for_perfect_forecast():
    probs = xr.DataArray([[0, 0, 1], [1, 0, 0]], dims=("year", "category"), coords={"category": ["below", "near", "above"]})
    obs_category = xr.DataArray([2, 0], dims="year")
    rps = prob.ranked_probability_score(probs, obs_category, sample_dim="year")
    assert float(rps) == pytest.approx(0.0, abs=1e-9)


def test_rps_skill_score_one_for_perfect_forecast():
    probs = xr.DataArray([[0, 0, 1]] * 4, dims=("year", "category"), coords={"category": ["below", "near", "above"]})
    obs_category = xr.DataArray([2, 2, 2, 2], dims="year")
    rpss = prob.rps_skill_score(probs, obs_category, sample_dim="year")
    assert float(rpss) == pytest.approx(1.0, abs=1e-9)


def test_brier_score_matches_hand_calculation():
    forecast_prob = xr.DataArray([0.7], dims="year")
    obs_indicator = xr.DataArray([1.0], dims="year")
    bs = prob.brier_score(forecast_prob, obs_indicator, sample_dim="year")
    assert float(bs) == pytest.approx((0.7 - 1) ** 2)


def test_brier_skill_score_one_for_perfect_forecast():
    forecast_prob = xr.DataArray([1.0, 1.0, 1.0], dims="year")
    obs_indicator = xr.DataArray([1.0, 1.0, 1.0], dims="year")
    bss = prob.brier_skill_score(forecast_prob, obs_indicator, sample_dim="year")
    assert float(bss) == pytest.approx(1.0, abs=1e-9)


def test_roc_area_perfect_discrimination():
    # all "event" samples score higher than all "no event" samples
    forecast_prob = xr.DataArray([0.1, 0.2, 0.8, 0.9], dims="year")
    obs_indicator = xr.DataArray([0, 0, 1, 1], dims="year")
    auc = prob.roc_area(forecast_prob, obs_indicator, sample_dim="year")
    assert float(auc) == pytest.approx(1.0)


def test_roc_area_no_discrimination_when_prob_identical():
    forecast_prob = xr.DataArray([0.5, 0.5, 0.5, 0.5], dims="year")
    obs_indicator = xr.DataArray([0, 1, 0, 1], dims="year")
    auc = prob.roc_area(forecast_prob, obs_indicator, sample_dim="year")
    assert float(auc) == pytest.approx(0.5)


def test_crps_skill_score_positive_when_corrected_matches_ref():
    time = np.arange(6)
    ref = xr.DataArray(np.full(6, 5.0), dims="time", coords={"time": time})
    raw = xr.DataArray(np.full((6, 4), 9.0), dims=("time", "realization"), coords={"time": time})  # biased ensemble
    corrected = xr.DataArray(np.full((6, 4), 5.0), dims=("time", "realization"), coords={"time": time})  # matches ref exactly

    result = prob.crps_skill_score(ref, raw, corrected, sample_dim="time")
    assert float(result["crps_corrected"]) == pytest.approx(0.0, abs=1e-9)
    assert float(result["crpss"]) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------


def test_reliability_diagram_data_recovers_constant_forecast_frequency():
    rng = np.random.default_rng(3)
    n = 2000
    forecast_prob = xr.DataArray(np.full(n, 0.3))
    # construct obs so that the event occurs in exactly 30% of samples, matching the forecast
    obs_indicator = xr.DataArray((rng.uniform(size=n) < 0.3).astype(float))

    forecast_mean, observed_freq, counts = cal.reliability_diagram_data(forecast_prob, obs_indicator, n_bins=10)
    (nonempty,) = np.where(counts > 0)
    assert len(nonempty) == 1  # every sample falls in the same probability bin
    assert observed_freq[nonempty[0]] == pytest.approx(0.3, abs=0.05)


def test_spread_skill_ratio_matches_hand_calculation():
    time = np.arange(4)
    ref = xr.DataArray(np.full(4, 5.0), dims="time", coords={"time": time})
    # members = ref + 1 + {-1, 0, 1}: ensemble mean = ref + 1 (RMSE == 1), spread = std([-1,0,1])
    offsets = np.array([-1.0, 0.0, 1.0])
    model = ref + 1.0 + xr.DataArray(offsets, dims="realization")

    ratio = cal.spread_skill_ratio(model, ref, sample_dim="time")
    expected_spread = float(np.std(offsets))  # population std, ddof=0
    assert float(ratio) == pytest.approx(expected_spread / 1.0)


# ---------------------------------------------------------------------------
# spatial.py
# ---------------------------------------------------------------------------


def _spatial_field(pattern, n_time=3):
    """pattern: 2D array (lat, lon); repeated identically across n_time steps."""
    data = np.broadcast_to(np.asarray(pattern), (n_time, *np.asarray(pattern).shape)).copy()
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": np.arange(n_time)})


def test_spatial_correlation_perfect_for_identical_fields():
    pattern = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
    da = _spatial_field(pattern)
    corr = sp.spatial_correlation(da, da)
    assert np.allclose(corr.values, 1.0)


def test_spatial_rmse_zero_for_identical_fields():
    pattern = np.array([[1.0, 2.0], [3.0, 4.0]])
    da = _spatial_field(pattern)
    rmse = sp.spatial_rmse(da, da)
    assert np.allclose(rmse.values, 0.0, atol=1e-9)


def test_spatial_pattern_correlation_invariant_to_additive_bias():
    # time-varying (not repeated) patterns, so the anomaly-from-climatology
    # is non-degenerate -- a constant-across-time field would trivially
    # anomaly to all-zeros (undefined correlation) regardless of this metric.
    rng = np.random.default_rng(4)
    ref = xr.DataArray(rng.normal(size=(5, 2, 3)), dims=("time", "lat", "lon"), coords={"time": np.arange(5)})
    model = ref + 5.0  # same spatial/temporal shape, shifted up uniformly

    ref_clim = ref.mean("time")
    model_clim = model.mean("time")  # correctly includes the +5 bias

    corr = sp.spatial_pattern_correlation(model, ref, model_clim, ref_clim)
    assert np.allclose(corr.values, 1.0)


def test_spatial_correlation_averages_ensemble_first():
    ref_pattern = np.array([[1.0, 2.0], [3.0, 4.0]])
    ref = _spatial_field(ref_pattern, n_time=1)
    members = np.stack([ref_pattern - 1, ref_pattern + 1], axis=-1)  # mean == ref_pattern
    model = xr.DataArray(members[None, ...], dims=("time", "lat", "lon", "realization"))

    corr = sp.spatial_correlation(model, ref)
    assert np.allclose(corr.values, 1.0)


# ---------------------------------------------------------------------------
# boundary.py
# ---------------------------------------------------------------------------

geopandas = pytest.importorskip("geopandas")


@pytest.fixture()
def square_shapefile(tmp_path):
    """A simple 2x2-degree square polygon, written as a real shapefile on disk."""
    from shapely.geometry import Polygon

    square = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    gdf = geopandas.GeoDataFrame({"name": ["square"]}, geometry=[square], crs="EPSG:4326")
    path = tmp_path / "square.shp"
    gdf.to_file(path)
    return path


def test_load_country_mask_identifies_pixels_inside_polygon(square_shapefile):
    from sbc_qdm.verify.boundary import load_country_mask

    da = xr.DataArray(
        np.zeros((4, 4)),
        dims=("lat", "lon"),
        coords={"lat": [0.5, 1.5, 2.5, 3.5], "lon": [0.5, 1.5, 2.5, 3.5]},
    )
    mask = load_country_mask(da, square_shapefile)

    # pixels at (0.5,0.5) and (1.5,1.5) fall inside the [0,2]x[0,2] square; (2.5,*) and (3.5,*) don't
    assert bool(mask.sel(lat=0.5, lon=0.5))
    assert bool(mask.sel(lat=1.5, lon=1.5))
    assert not bool(mask.sel(lat=2.5, lon=2.5))
    assert not bool(mask.sel(lat=3.5, lon=3.5))
