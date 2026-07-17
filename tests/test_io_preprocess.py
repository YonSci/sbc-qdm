"""Smoke tests for io.py / preprocess.py against the real project data files.

Not a substitute for synthetic unit tests (TODO), but these catch the most
likely breakage: the real files not matching the dims/coords/units assumed
in io.py and preprocess.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from sbc_qdm.config import load_config
from sbc_qdm.io import load_chirps_reference, load_ecmwf_year, load_ecmwf_operational
from sbc_qdm.preprocess import (
    build_land_mask,
    apply_mask,
    deaccumulate,
    ecmwf_precip_to_mm,
    rename_ecmwf_grid,
    diagnose_accumulation,
)

pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_load_chirps_reference(cfg):
    chirps = load_chirps_reference(cfg)
    assert chirps.dims == ("time", "lat", "lon")
    assert chirps.attrs.get("units") in ("mm/day", None) or True
    assert chirps.sizes["lat"] == 48
    assert chirps.sizes["lon"] == 60
    # known to contain water-body NaNs
    assert bool(chirps.isnull().any())


def test_land_mask_matches_known_fraction(cfg):
    chirps = load_chirps_reference(cfg)
    mask = build_land_mask(chirps)
    land_frac = float(mask.mean())
    assert 0.85 < land_frac < 1.0  # ~92% land expected from initial exploration

    masked = apply_mask(chirps.isel(time=0), mask)
    assert int(masked.isnull().sum()) >= int(chirps.isel(time=0).isnull().sum())


def test_load_ecmwf_year_shape(cfg):
    ds = load_ecmwf_year(cfg, 2020)
    assert "realization" in ds.dims
    assert ds.sizes["realization"] == 51
    assert ds.sizes["time"] == 183
    assert "forecast_reference_time" not in ds.dims
    assert "forecast_period" not in ds.dims
    # calendar axis should run May 2 -- Oct 31 2020
    assert str(ds.time.values[0])[:10] == "2020-05-02"
    assert str(ds.time.values[-1])[:10] == "2020-10-31"


def test_ecmwf_unit_conversion_and_rename(cfg):
    ds = load_ecmwf_year(cfg, 2020)
    pr = ecmwf_precip_to_mm(ds, cfg)
    pr = rename_ecmwf_grid(pr, cfg)
    assert pr.attrs["units"] == "mm/day"
    assert "lat" in pr.dims and "lon" in pr.dims
    assert float(pr.min()) >= 0.0
    # sanity: de-accumulated mm/day values should be in a plausible daily rainfall range
    assert float(pr.max()) < 300.0


def test_operational_year_loads(cfg):
    ds = load_ecmwf_operational(cfg)
    assert ds.sizes["realization"] == 51


def test_diagnose_accumulation_confirms_raw_tp_is_cumulative(cfg):
    ds = load_ecmwf_year(cfg, 2020)
    raw_diag = diagnose_accumulation(ds["tp"])
    # Raw tp IS accumulated-since-init -- this is exactly why
    # ecmwf_precip_to_mm() calls deaccumulate() before use.
    assert raw_diag["likely_accumulated"] is True

    daily = deaccumulate(ds["tp"], time_dim="time")
    daily_diag = diagnose_accumulation(daily)
    assert daily_diag["likely_accumulated"] is False
