"""Composition of io -> preprocess -> regrid into the two data products the
CLI (and cross-validation) needs: the full hindcast (ref, hist), and a single
target year's harmonized forecast.
"""

from __future__ import annotations

import xarray as xr

from sbc_qdm.io import load_chirps_reference, load_ecmwf_hindcast, load_ecmwf_year
from sbc_qdm.preprocess import (
    apply_mask,
    build_land_mask,
    ecmwf_precip_to_mm,
    rename_ecmwf_grid,
)
from sbc_qdm.regrid import regrid_to_chirps


def harmonize_ecmwf(ds: xr.Dataset, chirps: xr.DataArray, mask: xr.DataArray, cfg: dict) -> xr.DataArray:
    """De-accumulate + convert units, rename grid, regrid to CHIRPS' grid, mask."""
    pr = ecmwf_precip_to_mm(ds, cfg)
    pr = rename_ecmwf_grid(pr, cfg)
    pr = regrid_to_chirps(pr, chirps)
    pr = apply_mask(pr, mask)
    pr.attrs["units"] = "mm/day"
    return pr


def prepare_hindcast(cfg: dict) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Returns (chirps_full, land_mask, ref, hist) for the configured hindcast period.

    CHIRPS' merged reference is missing 2017 entirely (an incomplete source
    download -- see data/chirps_pr_et/chirps-v2.0.2017.days_p25.nc.part),
    even though ECMWF has a hindcast for that year. Rather than assume full
    date coverage (chirps.sel(time=hist.time) crashes on the gap), intersect
    the two time indices so any such gap just drops those dates from both
    sides instead of failing the whole pipeline.
    """
    chirps = load_chirps_reference(cfg)
    mask = build_land_mask(chirps)

    hind = load_ecmwf_hindcast(cfg)
    hist = harmonize_ecmwf(hind, chirps, mask, cfg)

    common_times = hist.time.to_index().intersection(chirps.time.to_index())
    hist = hist.sel(time=common_times)
    ref = chirps.sel(time=common_times)
    ref = apply_mask(ref, mask)
    ref.attrs["units"] = "mm/day"

    return chirps, mask, ref, hist


def prepare_target_year(cfg: dict, chirps: xr.DataArray, mask: xr.DataArray, year: int | None = None) -> xr.DataArray:
    """Harmonized ECMWF forecast for a single year (operational or held-out).

    2017+ forecasts carry 51 members vs 25 for the 1993-2016 hindcasts (see
    config/domain.yaml). Clipped down to the first `hindcast_n_members` here
    so the corrected output has a consistent ensemble size across years
    instead of jumping 25->51 at the operational boundary.
    """
    ds = load_ecmwf_year(cfg, year or cfg["time"]["operational_year"])
    pr = harmonize_ecmwf(ds, chirps, mask, cfg)

    n = cfg["ensemble"]["hindcast_n_members"]
    if pr.sizes["realization"] > n:
        pr = pr.isel(realization=slice(0, n))
    return pr
