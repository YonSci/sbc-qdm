"""Country-boundary masking from a shapefile, for a spatially-clipped variant
of the evaluation suite (e.g. Ethiopia only, vs the full ECMWF/CHIRPS domain
bbox which also covers slivers of Somalia, Kenya, Eritrea, Djibouti, and
South Sudan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_country_mask(da: xr.DataArray, shapefile_path: str | Path) -> xr.DataArray:
    """Boolean (lat, lon) mask, True where the pixel center falls inside the
    shapefile's polygon(s). Assumes a single-country shapefile (all rows'
    geometries are unioned) in EPSG:4326, matching CHIRPS/ECMWF's lat/lon grid.
    """
    import geopandas as gpd
    from shapely.vectorized import contains

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326", "WGS84"):
        gdf = gdf.to_crs4326() if hasattr(gdf, "to_crs4326") else gdf.to_crs(epsg=4326)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union

    lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values)
    mask = contains(geom, lon2d, lat2d)
    return xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]}, name="country_mask")
