"""Memory-bounded dask chunking helpers shared by qdm.py and methods/.

Extracted from qdm.py so every bias-correction method (not just QDM) tiles
spatially the same way -- this machine has run with as little as ~0.5 GiB
free under normal desktop load, and a single-block allocation across the full
spatial domain repeatedly hit MemoryError even well below that. See qdm.py's
_rechunk_for_grouping docstring history for the full story; the tuning here
is unchanged, just no longer duplicated per method.
"""

from __future__ import annotations

import xarray as xr

SAMPLE_DIMS = ("time", "realization")
SPATIAL_CHUNK = 5


def sample_dims(da: xr.DataArray) -> list[str]:
    return [d for d in SAMPLE_DIMS if d in da.dims]


def rechunk_for_grouping(da: xr.DataArray) -> xr.DataArray:
    """Single chunk along time/realization (required by xsdba's grouping and by
    apply_ufunc's core dims), tiled over lat/lon to keep peak memory bounded
    regardless of domain size.
    """
    chunks = {d: -1 for d in SAMPLE_DIMS if d in da.dims}
    chunks.update({d: SPATIAL_CHUNK for d in ("lat", "lon") if d in da.dims})
    return da.chunk(chunks)
