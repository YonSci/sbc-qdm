"""Loaders for the CHIRPS reference and ECMWF seasonal reforecast/forecast.

ECMWF files (data/ecmwf/ecmwf_{year}05_d01.nc) are single-May-initialization
seasonal forecasts: dims (number=51, forecast_reference_time=1,
forecast_period=183, latitude, longitude), variable "tp" in metres. Each
year's forecast_period coordinate carries a "valid_time" auxiliary coordinate
(actual calendar date, May 2 -- Oct 31 of that year). We swap onto that real
calendar axis before concatenating years, so that a dayofyear-based Grouper
pools same-season samples across hindcast years and ensemble members without
ever needing an arbitrary shared "lead index" axis.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def load_chirps_reference(cfg: dict) -> xr.DataArray:
    """Load the merged CHIRPS daily precip record (mm/day), 1993-2025."""
    ds = xr.open_dataset(cfg["paths"]["chirps_reference"])
    return ds[cfg["variables"]["chirps_var"]]


def _harmonize_ecmwf_year(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Reshape one year's ECMWF file onto a real calendar time axis.

    Drops the singleton forecast_reference_time dim, swaps forecast_period
    for the actual valid_time calendar axis, and renames the ensemble
    dimension to "realization". Leaves units/variable naming untouched --
    that is preprocess.py's job.
    """
    ens_dim = cfg["ensemble"]["realization_dim"]

    ds = ds.isel(forecast_reference_time=0, drop=True)
    ds = ds.assign_coords(time=("forecast_period", ds["valid_time"].values))
    ds = ds.swap_dims({"forecast_period": "time"})
    ds = ds.drop_vars(["forecast_period", "valid_time"], errors="ignore")
    ds = ds.rename({ens_dim: "realization"})
    return ds


def load_ecmwf_year(cfg: dict, year: int) -> xr.Dataset:
    """Load and harmonize a single year's ECMWF May-init forecast."""
    ecmwf_dir = Path(cfg["paths"]["ecmwf_dir"])
    fname = cfg["paths"]["ecmwf_pattern"].format(year=year)
    ds = xr.open_dataset(ecmwf_dir / fname)
    return _harmonize_ecmwf_year(ds, cfg)


def load_ecmwf_hindcast(cfg: dict, start_year: int | None = None, end_year: int | None = None) -> xr.Dataset:
    """Load and concatenate all hindcast years onto one calendar time axis.

    Each year's May-Oct window is disjoint from every other year's, so
    concatenating along "time" produces a single axis spanning
    start_year-05-02 .. end_year-10-31 with real (non-overlapping) gaps
    Nov-Apr -- exactly what a "time.dayofyear" Grouper expects to pool
    same-season samples across years.

    ECMWF SEAS5 hindcasts (1993-2016) carry 25 ensemble members vs 51 for
    2017+; xr.concat pads the 25-member years up to 51 with NaN along
    "realization". This is harmless for training (xarray's .quantile and
    xsdba's adapt_freq both skip NaN correctly), so no de-duplication or
    special-casing is done here.
    """
    start_year = start_year or cfg["time"]["hindcast_years"][0]
    end_year = end_year or cfg["time"]["hindcast_years"][1]

    years = range(start_year, end_year + 1)
    datasets = [load_ecmwf_year(cfg, year) for year in years]
    return xr.concat(datasets, dim="time")


def load_ecmwf_operational(cfg: dict, year: int | None = None) -> xr.Dataset:
    """Load the live forecast year to be bias-corrected (default: config's operational_year)."""
    year = year or cfg["time"]["operational_year"]
    return load_ecmwf_year(cfg, year)
