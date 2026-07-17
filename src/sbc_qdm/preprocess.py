"""Unit harmonization, coordinate renaming, masking and sanity checks.

Handles the concrete quirks found in this project's source files:
  - ECMWF "tp" is in metres AND accumulated since forecast start (confirmed
    by diagnose_accumulation() -- despite GRIB_stepType being labelled
    "instant", raw values grow monotonically from ~0.0004 m at lead day 1 to
    ~1.18 m by lead day 183). It must be de-accumulated (diff along time)
    before use.
  - CHIRPS "precip" is already per-day mm/day; ECMWF needs de-accumulation
    then *1000 to go from metres/day to mm/day.
  - ECMWF uses "latitude"/"longitude"; CHIRPS uses "lat"/"lon".
  - CHIRPS has ~8% NaN pixels within the domain bbox (water bodies), which
    must be masked out of both datasets consistently before quantile fitting.
"""

from __future__ import annotations

import xarray as xr


def deaccumulate(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """Convert a forecast-start-accumulated field into per-timestep values.

    lead 1 is kept as-is (accumulation began at 0 at forecast init); every
    subsequent step is the first difference of the cumulative series.
    """
    first_step = da.isel({time_dim: slice(0, 1)})
    increments = da.diff(time_dim)
    return xr.concat([first_step, increments], dim=time_dim)


def ecmwf_precip_to_mm(ds: xr.Dataset, cfg: dict) -> xr.DataArray:
    """De-accumulate ECMWF tp (metres, cumulative) into a "pr" DataArray in mm/day.

    Negative values (floating-point noise from differencing) are clipped to 0.
    """
    var = cfg["variables"]["ecmwf_var"]
    daily_m = deaccumulate(ds[var], time_dim="time")
    pr = daily_m * 1000.0
    pr = pr.clip(min=0.0)
    pr.name = cfg["variables"]["corrected_var"]
    pr.attrs.update(units="mm/day", long_name="Total precipitation")
    return pr


def rename_ecmwf_grid(da: xr.DataArray, cfg: dict) -> xr.DataArray:
    """Rename ECMWF's latitude/longitude to lat/lon (matches CHIRPS)."""
    lat_name = cfg["grids"]["ecmwf"]["lat_name"]
    lon_name = cfg["grids"]["ecmwf"]["lon_name"]
    return da.rename({lat_name: "lat", lon_name: "lon"})


def build_land_mask(chirps_da: xr.DataArray) -> xr.DataArray:
    """A pixel is land iff it has at least one non-null observation.

    CHIRPS NaNs over water bodies are static (every day is NaN at that
    pixel), so `.notnull().any("time")` cleanly separates land from water/
    ocean pixels even though some land pixels have sporadic missing days.
    """
    return chirps_da.notnull().any("time")


def apply_mask(da: xr.DataArray, land_mask: xr.DataArray) -> xr.DataArray:
    """Set non-land pixels to NaN, broadcasting the mask over any extra dims."""
    return da.where(land_mask)


def diagnose_accumulation(tp_da: xr.DataArray, time_dim: str = "time") -> dict:
    """Heuristic check for whether a precip field is accumulated-since-init.

    Computes, per grid cell/member, the fraction of consecutive time steps
    where value[t+1] >= value[t]. True daily rainfall should show a fraction
    well below 1 (rain days are interspersed with drier ones); a value very
    close to 1 suggests the field is still a running accumulation and needs
    de-accumulation (np.diff along time_dim) before use.
    """
    diffs = tp_da.diff(time_dim)
    nondecreasing_frac = (diffs >= 0).mean().item()
    return {
        "nondecreasing_fraction": nondecreasing_frac,
        "likely_accumulated": nondecreasing_frac > 0.95,
    }
