"""Unit tests for src/sbc_qdm/methods/ (the 6 alternative bias-correction
methods compared against QDM).

Synthetic, hand-constructed arrays with a known bias -- no dependency on the
large local data/ directory, runs in CI (mirrors tests/test_verify.py's style).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sbc_qdm.methods.delta_change import apply_delta_change, train_delta_change
from sbc_qdm.methods.detrended_quantile_mapping import apply_detrended_quantile_mapping, train_detrended_quantile_mapping
from sbc_qdm.methods.empirical_quantile_mapping import apply_empirical_quantile_mapping, train_empirical_quantile_mapping
from sbc_qdm.methods.linear_scaling import apply_linear_scaling, train_linear_scaling
from sbc_qdm.methods.power_transformation import apply_power_transformation, train_power_transformation
from sbc_qdm.methods.variance_scaling import apply_variance_scaling, train_variance_scaling

CFG = {"qdm": {"nquantiles": 20, "tail_quantiles": (), "adapt_freq_thresh": "0.1 mm/day", "random_seed": 0}}

N_YEARS = 6
N_DAYS = 30  # one calendar month, May
N_LAT, N_LON, N_REALIZATION = 2, 2, 6


def _may_time(n_years: int) -> pd.DatetimeIndex:
    chunks = [pd.date_range(f"{2015 + y}-05-01", periods=N_DAYS, freq="D") for y in range(n_years)]
    return pd.DatetimeIndex(np.concatenate(chunks))


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


def _with_units(da: xr.DataArray) -> xr.DataArray:
    da.attrs["units"] = "mm/day"
    return da


@pytest.fixture(scope="module")
def ref(rng):
    """Reference 'obs': mean 10, std 2, strictly positive, no realization dim."""
    time = _may_time(N_YEARS)
    data = np.clip(rng.normal(loc=10.0, scale=2.0, size=(len(time), N_LAT, N_LON)), 0.1, None)
    return _with_units(xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time}))


@pytest.fixture(scope="module")
def hist(rng):
    """Raw 'model': biased high (mean ~22) and overdispersed (std ~6) vs ref, with a realization dim."""
    time = _may_time(N_YEARS)
    data = np.clip(rng.normal(loc=22.0, scale=6.0, size=(len(time), N_LAT, N_LON, N_REALIZATION)), 0.1, None)
    return _with_units(xr.DataArray(data, dims=("time", "lat", "lon", "realization"), coords={"time": time}))


@pytest.fixture(scope="module")
def target(rng):
    """A fresh 'held-out year' style forecast, same generating distribution as hist but independent draws."""
    time = pd.date_range("2025-05-01", periods=N_DAYS, freq="D")
    data = np.clip(rng.normal(loc=22.0, scale=6.0, size=(len(time), N_LAT, N_LON, N_REALIZATION)), 0.1, None)
    return _with_units(xr.DataArray(data, dims=("time", "lat", "lon", "realization"), coords={"time": time}))


@pytest.fixture(scope="module")
def ref_mild(rng):
    """A milder ref/hist pair (small bias relative to spread) for exact closed-form checks where
    apply_delta_change's/apply_variance_scaling's clip(min=0) must never trigger."""
    time = _may_time(N_YEARS)
    data = np.clip(rng.normal(loc=10.0, scale=1.0, size=(len(time), N_LAT, N_LON)), 0.1, None)
    return _with_units(xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time}))


@pytest.fixture(scope="module")
def hist_mild(rng):
    time = _may_time(N_YEARS)
    data = np.clip(rng.normal(loc=13.0, scale=1.2, size=(len(time), N_LAT, N_LON, N_REALIZATION)), 0.1, None)
    return _with_units(xr.DataArray(data, dims=("time", "lat", "lon", "realization"), coords={"time": time}))


def _shape_preserved(corrected: xr.DataArray, sim: xr.DataArray) -> bool:
    return corrected.sizes == sim.sizes and set(corrected.dims) == set(sim.dims)


def _closer_to_ref(corrected: xr.DataArray, raw: xr.DataArray, ref_: xr.DataArray) -> bool:
    ref_mean = float(ref_.mean())
    raw_mean = float(raw.mean())
    corrected_mean = float(corrected.mean())
    return abs(corrected_mean - ref_mean) < abs(raw_mean - ref_mean)


# ---------------------------------------------------------------------------
# Linear Scaling
# ---------------------------------------------------------------------------


def test_linear_scaling_shape_and_generalization(ref, hist, target):
    trained = train_linear_scaling(ref, hist, CFG)
    corrected = apply_linear_scaling(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


def test_linear_scaling_matches_ref_mean_exactly_on_training_data(ref, hist):
    """Closed-form guarantee: applying the trained factor back onto the same
    hist used for training reproduces ref's mean exactly (factor = ref_mean/hist_mean,
    a per-pixel scalar, so mean(hist * factor) == factor * mean(hist) == ref_mean)."""
    trained = train_linear_scaling(ref, hist, CFG)
    corrected = apply_linear_scaling(hist, trained, CFG)
    ref_mean = float(ref.mean())
    corrected_mean = float(corrected.mean())
    assert corrected_mean == pytest.approx(ref_mean, rel=1e-6)


# ---------------------------------------------------------------------------
# Delta Change
# ---------------------------------------------------------------------------


def test_delta_change_shape_and_generalization(ref, hist, target):
    trained = train_delta_change(ref, hist, CFG)
    corrected = apply_delta_change(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


def test_delta_change_matches_ref_mean_exactly_on_training_data(ref_mild, hist_mild):
    trained = train_delta_change(ref_mild, hist_mild, CFG)
    corrected = apply_delta_change(hist_mild, trained, CFG)
    ref_mean = float(ref_mild.mean())
    corrected_mean = float(corrected.mean())
    assert corrected_mean == pytest.approx(ref_mean, rel=1e-6)


def test_delta_change_clips_negative_values():
    time = pd.date_range("2020-05-01", periods=5, freq="D")
    ref_small = xr.DataArray(np.full((5, 1, 1), 0.5), dims=("time", "lat", "lon"), coords={"time": time})
    hist_small = xr.DataArray(np.full((5, 1, 1, 2), 10.0), dims=("time", "lat", "lon", "realization"), coords={"time": time})
    trained = train_delta_change(ref_small, hist_small, CFG)
    corrected = apply_delta_change(hist_small, trained, CFG)
    assert float(corrected.min()) >= 0.0


# ---------------------------------------------------------------------------
# Variance Scaling
# ---------------------------------------------------------------------------


def test_variance_scaling_shape_and_generalization(ref, hist, target):
    trained = train_variance_scaling(ref, hist, CFG)
    corrected = apply_variance_scaling(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


def test_variance_scaling_matches_ref_mean_and_std_on_training_data(ref_mild, hist_mild):
    trained = train_variance_scaling(ref_mild, hist_mild, CFG)
    corrected = apply_variance_scaling(hist_mild, trained, CFG)
    assert float(corrected.mean()) == pytest.approx(float(ref_mild.mean()), rel=1e-6)
    assert float(corrected.std()) == pytest.approx(float(ref_mild.std()), rel=1e-2)


# ---------------------------------------------------------------------------
# Power Transformation
# ---------------------------------------------------------------------------


def test_power_transformation_shape_and_generalization(ref, hist, target):
    trained = train_power_transformation(ref, hist, CFG)
    corrected = apply_power_transformation(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


def test_power_transformation_matches_ref_mean_on_training_data(ref, hist):
    trained = train_power_transformation(ref, hist, CFG)
    corrected = apply_power_transformation(hist, trained, CFG)
    assert float(corrected.mean()) == pytest.approx(float(ref.mean()), rel=1e-2)


# ---------------------------------------------------------------------------
# Empirical Quantile Mapping
# ---------------------------------------------------------------------------


def test_empirical_quantile_mapping_shape_and_generalization(ref, hist, target):
    trained = train_empirical_quantile_mapping(ref, hist, CFG)
    corrected = apply_empirical_quantile_mapping(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


# ---------------------------------------------------------------------------
# Detrended Quantile Mapping
# ---------------------------------------------------------------------------


def test_detrended_quantile_mapping_shape_and_generalization(ref, hist, target):
    trained = train_detrended_quantile_mapping(ref, hist, CFG)
    corrected = apply_detrended_quantile_mapping(target, trained, CFG)
    assert _shape_preserved(corrected, target)
    assert _closer_to_ref(corrected, target, ref)


def test_detrended_quantile_mapping_reduces_to_eqm_when_target_mean_equals_training_mean(ref, hist):
    """DQM's normalization step (mu_hist/mu_target) is a no-op when the target
    IS the training data (mu_target == mu_hist), so DQM should collapse to
    plain EQM's result in that special case."""
    dqm_trained = train_detrended_quantile_mapping(ref, hist, CFG)
    eqm_trained = train_empirical_quantile_mapping(ref, hist, CFG)

    dqm_corrected = apply_detrended_quantile_mapping(hist, dqm_trained, CFG)
    eqm_corrected = apply_empirical_quantile_mapping(hist, eqm_trained, CFG)

    np.testing.assert_allclose(dqm_corrected.values, eqm_corrected.values, rtol=1e-6)
