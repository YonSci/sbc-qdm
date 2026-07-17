"""Smoke tests for regrid.py / qdm.py against real project data.

Restricted to a small spatial subset and a handful of hindcast years to keep
runtime reasonable -- these are integration checks that the real pipeline
wires together correctly, not exhaustive numerical validation.
"""

from __future__ import annotations

import pytest

from sbc_qdm.config import load_config
from sbc_qdm.io import load_chirps_reference, load_ecmwf_hindcast, load_ecmwf_year
from sbc_qdm.preprocess import build_land_mask, apply_mask, ecmwf_precip_to_mm, rename_ecmwf_grid
from sbc_qdm.regrid import regrid_to_chirps
from sbc_qdm.qdm import train_qdm, apply_qdm, leave_one_year_out, apply_operational

pytestmark = pytest.mark.requires_data

SUB_LAT = slice(0, 6)
SUB_LON = slice(0, 6)
HINDCAST_YEARS = (1993, 1997)  # small 5-year window to keep tests fast
TARGET_YEAR = 1999


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _prep_ecmwf(ds, chirps, mask, cfg):
    pr = ecmwf_precip_to_mm(ds, cfg)
    pr = rename_ecmwf_grid(pr, cfg)
    pr = regrid_to_chirps(pr, chirps)
    pr = apply_mask(pr, mask)
    pr.attrs["units"] = "mm/day"
    return pr


@pytest.fixture(scope="module")
def hindcast_subset(cfg):
    chirps = load_chirps_reference(cfg)
    mask = build_land_mask(chirps)

    hind = load_ecmwf_hindcast(cfg, *HINDCAST_YEARS)
    hist = _prep_ecmwf(hind, chirps, mask, cfg)

    ref = chirps.sel(time=hist.time)
    ref = apply_mask(ref, mask)
    ref.attrs["units"] = "mm/day"

    return {
        "ref": ref.isel(lat=SUB_LAT, lon=SUB_LON),
        "hist": hist.isel(lat=SUB_LAT, lon=SUB_LON),
        "chirps": chirps,
        "mask": mask,
    }


@pytest.fixture(scope="module")
def target_subset(cfg, hindcast_subset):
    ds = load_ecmwf_year(cfg, TARGET_YEAR)
    pr = _prep_ecmwf(ds, hindcast_subset["chirps"], hindcast_subset["mask"], cfg)
    return pr.isel(lat=SUB_LAT, lon=SUB_LON)


def test_regrid_to_chirps_matches_grid(cfg, hindcast_subset):
    hist = hindcast_subset["hist"]
    ref = hindcast_subset["ref"]
    assert hist.sizes["lat"] == ref.sizes["lat"]
    assert hist.sizes["lon"] == ref.sizes["lon"]
    assert bool((hist.lat == ref.lat).all())
    assert bool((hist.lon == ref.lon).all())


def test_train_qdm_produces_one_fit_per_month(hindcast_subset, cfg):
    trained = train_qdm(hindcast_subset["ref"], hindcast_subset["hist"], cfg)
    assert set(trained.keys()) == {5, 6, 7, 8, 9, 10}
    for af, hist_q in trained.values():
        assert "quantiles" in af.dims
        assert af.sizes["quantiles"] == cfg["qdm"]["nquantiles"]
        assert "realization" not in af.dims  # members pooled into the fit, not broadcast


def test_apply_qdm_preserves_shape_and_shifts_mean_toward_ref(hindcast_subset, target_subset, cfg):
    trained = train_qdm(hindcast_subset["ref"], hindcast_subset["hist"], cfg)
    corrected = apply_qdm(target_subset, trained, cfg).compute()

    assert corrected.sizes == target_subset.sizes
    assert float(corrected.min()) >= 0.0

    ref_mean = float(hindcast_subset["ref"].mean())
    raw_mean = float(target_subset.mean())
    corrected_mean = float(corrected.mean())
    # not a strict guarantee for any single held-out year, but corrected should
    # generally land closer to the reference climatology than the raw forecast
    assert abs(corrected_mean - ref_mean) <= abs(raw_mean - ref_mean) + 1.0


def test_leave_one_year_out_covers_all_hindcast_years(hindcast_subset, cfg):
    loyo = leave_one_year_out(hindcast_subset["ref"], hindcast_subset["hist"], cfg).compute()
    assert loyo.sizes["time"] == hindcast_subset["hist"].sizes["time"]
    assert set(loyo.time.dt.year.values.tolist()) == set(range(HINDCAST_YEARS[0], HINDCAST_YEARS[1] + 1))


def test_apply_operational_matches_manual_train_and_apply(hindcast_subset, target_subset, cfg):
    trained = train_qdm(hindcast_subset["ref"], hindcast_subset["hist"], cfg)
    manual = apply_qdm(target_subset, trained, cfg).compute()
    operational = apply_operational(hindcast_subset["ref"], hindcast_subset["hist"], target_subset, cfg).compute()
    assert bool((manual.fillna(-1) == operational.fillna(-1)).all())
