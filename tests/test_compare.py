"""Unit tests for src/sbc_qdm/verify/compare.py (the cross-method comparison
summary/figures built on top of each method's already-computed `sbc-qdm
evaluate` output).

Synthetic, hand-constructed per-method evaluation/ directories with known
expected domain-mean values -- no dependency on the large local data/
directory, runs in CI (mirrors tests/test_verify.py's style). This module
had zero test coverage before: the xarray `.sel(method=...)` reserved-kwarg
collision and a couple of hand-transcription errors in the README were only
ever caught by manually running `sbc-qdm compare-methods` and eyeballing
the output, not by an automated check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sbc_qdm.verify.compare import (
    HEADLINE_METRICS,
    SKILL_METRICS,
    comparison_summary,
    mbe_maps,
    plot_method_comparison,
    plot_method_comparison_maps,
    roc_skill_maps,
    wet_day_freq_bias_maps,
)

LAT = xr.DataArray([1.0, 2.0], dims="lat", name="lat")
LON = xr.DataArray([1.0, 2.0], dims="lon", name="lon")


def _map(value: float) -> xr.DataArray:
    return xr.DataArray(np.full((2, 2), value), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})


def _write_method_eval_dir(tmp_path: Path, name: str, *, raw_mbe: float, corrected_mbe: float, corrected_pbias: float, corrected_rmse: float, corrected_wet_day_freq_bias: float, corrected_acc: float, corrected_roc_skill: float, corrected_wet_spell_mean: float, corrected_dry_spell_mean: float) -> Path:
    """Builds one method's `evaluation/` dir with the exact variable names/dims
    `comparison_summary()` and the map-loader functions expect, per the real
    schema in daily_deterministic.nc / jjas_deterministic_and_skill.nc /
    jjas_probabilistic.nc / spell_lengths.npz (see verify/run.py's output).
    """
    eval_dir = tmp_path / name / "evaluation"
    eval_dir.mkdir(parents=True)

    daily = xr.Dataset(
        {
            "raw_mbe": _map(raw_mbe),
            "raw_pbias": _map(20.0),
            "raw_rmse": _map(5.0),
            "wet_day_freq_bias_raw": _map(-0.01),
            "corrected_mbe": _map(corrected_mbe),
            "corrected_pbias": _map(corrected_pbias),
            "corrected_rmse": _map(corrected_rmse),
            "wet_day_freq_bias_corrected": _map(corrected_wet_day_freq_bias),
        }
    )
    daily.to_netcdf(eval_dir / "daily_deterministic.nc")

    jjas = xr.Dataset(
        {
            "raw_rmse": _map(100.0),
            "corrected_rmse": _map(corrected_rmse * 10),
            "raw_acc": _map(0.25),
            "corrected_acc": _map(corrected_acc),
        }
    )
    jjas.to_netcdf(eval_dir / "jjas_deterministic_and_skill.nc")

    category = xr.DataArray(["below", "near", "above"], dims="category", name="category")
    jjas_prob = xr.Dataset(
        {
            "raw_roc_skill_score": xr.concat([_map(0.2), _map(0.2), _map(0.25)], dim=category),
            "corrected_roc_skill_score": xr.concat([_map(0.2), _map(0.2), _map(corrected_roc_skill)], dim=category),
            "crpss": _map(0.3),
        }
    )
    jjas_prob.to_netcdf(eval_dir / "jjas_probabilistic.nc")

    # spell_lengths.npz: obs mean fixed at 2.0 (wet) / 6.0 (dry) so the *_bias
    # helper's known offset is easy to hand-verify against corrected_*_spell_mean.
    np.savez(
        eval_dir / "spell_lengths.npz",
        obs_wet=np.array([2.0, 2.0]),
        raw_wet=np.array([5.0, 5.0]),
        corrected_wet=np.array([corrected_wet_spell_mean, corrected_wet_spell_mean]),
        obs_dry=np.array([6.0, 6.0]),
        raw_dry=np.array([13.0, 13.0]),
        corrected_dry=np.array([corrected_dry_spell_mean, corrected_dry_spell_mean]),
    )

    return eval_dir


@pytest.fixture
def method_eval_dirs(tmp_path) -> dict[str, Path]:
    qdm_dir = _write_method_eval_dir(
        tmp_path, "qdm",
        raw_mbe=1.0, corrected_mbe=0.1, corrected_pbias=2.0, corrected_rmse=4.5,
        corrected_wet_day_freq_bias=-0.1, corrected_acc=0.20, corrected_roc_skill=0.24,
        corrected_wet_spell_mean=2.5, corrected_dry_spell_mean=7.0,
    )
    other_dir = _write_method_eval_dir(
        tmp_path, "linear_scaling",
        raw_mbe=1.0, corrected_mbe=0.2, corrected_pbias=3.0, corrected_rmse=4.8,
        corrected_wet_day_freq_bias=-0.02, corrected_acc=0.19, corrected_roc_skill=0.23,
        corrected_wet_spell_mean=5.0, corrected_dry_spell_mean=9.0,
    )
    return {"qdm": qdm_dir, "linear_scaling": other_dir}


# ---------------------------------------------------------------------------
# comparison_summary()
# ---------------------------------------------------------------------------


def test_comparison_summary_has_raw_plus_one_row_per_method(method_eval_dirs):
    summary = comparison_summary(method_eval_dirs)
    assert set(summary["method"].values.tolist()) == {"raw", "qdm", "linear_scaling"}


def test_comparison_summary_domain_means_match_hand_computed_values(method_eval_dirs):
    summary = comparison_summary(method_eval_dirs)

    assert float(summary["daily_mbe"].sel({"method": "qdm"})) == pytest.approx(0.1)
    assert float(summary["daily_mbe"].sel({"method": "raw"})) == pytest.approx(1.0)
    assert float(summary["jjas_acc"].sel({"method": "qdm"})) == pytest.approx(0.20)
    assert float(summary["jjas_acc"].sel({"method": "raw"})) == pytest.approx(0.25)
    # above-normal category specifically, not below/near
    assert float(summary["jjas_roc_skill"].sel({"method": "qdm"})) == pytest.approx(0.24)


def test_comparison_summary_crpss_raw_baseline_is_zero_by_definition(method_eval_dirs):
    summary = comparison_summary(method_eval_dirs)
    assert float(summary["jjas_crpss"].sel({"method": "raw"})) == pytest.approx(0.0)
    assert float(summary["jjas_crpss"].sel({"method": "qdm"})) == pytest.approx(0.3)


def test_comparison_summary_spell_bias_is_offset_from_obs_mean(method_eval_dirs):
    """corrected_wet_spell_mean=2.5 vs obs_wet mean=2.0 -> bias = +0.5.
    raw_wet mean=5.0 vs obs_wet mean=2.0 -> raw bias = +3.0.
    """
    summary = comparison_summary(method_eval_dirs)
    assert float(summary["wet_spell_bias"].sel({"method": "qdm"})) == pytest.approx(0.5)
    assert float(summary["wet_spell_bias"].sel({"method": "raw"})) == pytest.approx(3.0)
    assert float(summary["dry_spell_bias"].sel({"method": "qdm"})) == pytest.approx(1.0)


def test_comparison_summary_method_dim_select_does_not_collide_with_xarray_reserved_kwarg(method_eval_dirs):
    """Regression test for a real bug: summary[var].sel(method=m) silently
    no-ops (returns the full unsliced array instead of a scalar) because
    xarray's own .sel() reserves `method=` for nearest-neighbor lookup, and
    this project's dimension happens to also be named "method". Only the
    dict-form .sel({"method": m}) actually slices correctly -- this test
    would fail if compare.py's internals ever regressed back to the
    keyword-argument form.
    """
    summary = comparison_summary(method_eval_dirs)
    sliced = summary["daily_mbe"].sel({"method": "qdm"})
    assert sliced.ndim == 0
    assert float(sliced) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Map-loader functions (mbe_maps / wet_day_freq_bias_maps / roc_skill_maps)
# ---------------------------------------------------------------------------


def test_mbe_maps_returns_per_method_corrected_field_and_shared_raw_field(method_eval_dirs):
    fields, raw_field = mbe_maps(method_eval_dirs)
    assert set(fields.keys()) == {"qdm", "linear_scaling"}
    assert float(fields["qdm"].isel(lat=0, lon=0)) == pytest.approx(0.1)
    assert float(fields["linear_scaling"].isel(lat=0, lon=0)) == pytest.approx(0.2)
    assert float(raw_field.isel(lat=0, lon=0)) == pytest.approx(1.0)


def test_wet_day_freq_bias_maps_uses_reversed_variable_naming(method_eval_dirs):
    """wet_day_freq_bias_raw / wet_day_freq_bias_corrected -- prefix comes
    *after* the metric name here, unlike mbe/pbias/rmse's raw_X/corrected_X.
    Regression test for exactly that naming inconsistency being handled.
    """
    fields, raw_field = wet_day_freq_bias_maps(method_eval_dirs)
    assert float(fields["qdm"].isel(lat=0, lon=0)) == pytest.approx(-0.1)
    assert float(raw_field.isel(lat=0, lon=0)) == pytest.approx(-0.01)


def test_roc_skill_maps_selects_above_normal_category_by_default(method_eval_dirs):
    fields = roc_skill_maps(method_eval_dirs)
    assert float(fields["qdm"].isel(lat=0, lon=0)) == pytest.approx(0.24)
    # not the below/near category's 0.2 placeholder value
    assert float(fields["qdm"].isel(lat=0, lon=0)) != pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Plotting functions -- smoke tests (produce a non-empty file, no exception)
# ---------------------------------------------------------------------------


def test_plot_method_comparison_writes_a_file(method_eval_dirs, tmp_path):
    summary = comparison_summary(method_eval_dirs)
    out_path = tmp_path / "comparison.png"
    plot_method_comparison(summary, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_method_comparison_covers_all_headline_and_skill_metrics(method_eval_dirs):
    """Every metric plot_method_comparison() iterates over must actually be
    present in comparison_summary()'s output, or the bar chart silently
    drops a panel instead of erroring.
    """
    summary = comparison_summary(method_eval_dirs)
    for var, _title in HEADLINE_METRICS + SKILL_METRICS:
        assert var in summary.data_vars, f"{var} missing from comparison_summary() output"


def test_plot_method_comparison_maps_writes_a_file_with_raw_panel(method_eval_dirs, tmp_path):
    fields, raw_field = mbe_maps(method_eval_dirs)
    out_path = tmp_path / "maps.png"
    plot_method_comparison_maps(fields, out_path, "Mean Bias Error", "mm/day", raw_field=raw_field)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_method_comparison_maps_writes_a_file_without_raw_panel(method_eval_dirs, tmp_path):
    """roc_skill_maps() has no raw baseline (see its docstring) -- the
    function must support that instead of requiring a raw_field.
    """
    fields = roc_skill_maps(method_eval_dirs)
    out_path = tmp_path / "roc_maps.png"
    plot_method_comparison_maps(fields, out_path, "ROC skill", "")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
