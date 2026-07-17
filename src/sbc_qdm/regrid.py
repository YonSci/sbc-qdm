"""Regrid ECMWF's 1 deg forecast onto CHIRPS' 0.25 deg grid.

Bilinear interpolation via xarray/scipy (no ESMF/xesmf dependency, chosen for
Windows portability -- see project decision log). Precipitation is smooth
enough at this domain size that bilinear is an acceptable approximation;
revisit with conservative (area-weighted) remapping via xesmf if mass
conservation becomes a concern.
"""

from __future__ import annotations

import xarray as xr


def regrid_to_chirps(da: xr.DataArray, chirps_da: xr.DataArray) -> xr.DataArray:
    """Bilinearly interpolate `da` (lat/lon on the ECMWF grid) onto CHIRPS' grid.

    CHIRPS' domain extends ~0.375 deg beyond ECMWF's grid on every edge (it
    was clipped independently), so edge cells need extrapolation rather than
    falling back to NaN -- `fill_value="extrapolate"` handles that. This
    still leaves those edge cells wholly dependent on ECMWF's outermost row/
    column; downstream masking against CHIRPS' land mask (preprocess.
    build_land_mask) is what actually bounds the usable domain, not this
    interpolation step.

    Linear extrapolation at those edges can overshoot below zero (observed
    up to ~-30 mm/day at domain edges even though the input was already
    clipped to >=0) -- clipped again here since negative precipitation is
    never physical.

    xarray/scipy interpolate the whole array in one shot unless it's
    dask-backed: doing that eagerly over the full 33-year hindcast (51
    members x ~6000 days) tried to allocate ~6.6 GiB and OOM'd. Chunking
    along time/realization first makes xarray apply the interpolation
    blockwise via dask instead. Chunk size of 10 (down from an initial 20)
    after repeated MemoryErrors on this machine's fluctuating available RAM
    (as low as under 1 GiB free under normal desktop load) even at this
    stage, before qdm.py's own (already tighter) rechunking applies.
    """
    chunk_sizes = {d: 10 for d in ("time", "realization") if d in da.dims}
    if chunk_sizes:
        da = da.chunk(chunk_sizes)

    interpolated = da.interp(
        lat=chirps_da["lat"],
        lon=chirps_da["lon"],
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return interpolated.clip(min=0.0)
