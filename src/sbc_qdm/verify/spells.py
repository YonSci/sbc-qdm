"""Wet/dry spell length distributions.

Run-length (spell) statistics are inherently ragged -- different pixels/
members/years produce different numbers of spells of different lengths, so
there's no clean way to keep them as a rectangular per-pixel DataArray.
Spell distributions are therefore computed domain-pooled (all pixels, all
ensemble members, all years flattened together) as a single comparable
distribution per dataset (raw / corrected / CHIRPS) -- consistent with how
Q-Q/ECDF/PDF are handled in distributions.py, and standard practice for
checking whether QDM (a purely marginal correction) preserves day-to-day
persistence, which it has no explicit mechanism to guarantee.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from sbc_qdm.verify.aggregate import iter_spatial_blocks


def _spell_lengths_1d(is_event: np.ndarray) -> np.ndarray:
    """Run lengths of consecutive True values in a 1D boolean array."""
    if is_event.size == 0:
        return np.array([], dtype=int)
    change = np.diff(is_event.astype(int))
    starts = np.where(change == 1)[0] + 1
    ends = np.where(change == -1)[0] + 1
    if is_event[0]:
        starts = np.r_[0, starts]
    if is_event[-1]:
        ends = np.r_[ends, is_event.size]
    return ends - starts


def spell_lengths(da: xr.DataArray, threshold_mm: float = 1.0, spell_type: str = "wet", time_dim: str = "time") -> np.ndarray:
    """Domain-pooled spell lengths (in days) for wet (>threshold) or dry (<=threshold) runs.

    Iterates 1D time series per (lat, lon[, realization]) combination -- a
    plain Python loop, since run-length extraction doesn't vectorize into a
    fixed-shape array. Streamed in (lat, lon) tiles via iter_spatial_blocks
    (see its docstring) rather than pulling the full 33-year x 51-member x
    full-domain array into memory at once -- the per-series inputs are small,
    but calling .values on everything simultaneously isn't.
    """
    if spell_type not in ("wet", "dry"):
        raise ValueError(f"spell_type must be 'wet' or 'dry', got {spell_type!r}")

    other_dims = [d for d in da.dims if d != time_dim]

    all_lengths = []
    for block in iter_spatial_blocks(da):
        stacked = block.stack(_series=other_dims) if other_dims else block.expand_dims("_series", axis=-1)
        stacked = stacked.transpose(time_dim, "_series")

        values = stacked.values
        for i in range(values.shape[1]):
            series = values[:, i]
            if np.all(np.isnan(series)):
                continue
            is_wet = series > threshold_mm
            is_event = is_wet if spell_type == "wet" else ~is_wet
            all_lengths.append(_spell_lengths_1d(is_event))

    return np.concatenate(all_lengths) if all_lengths else np.array([], dtype=int)


def spell_length_histogram(lengths: np.ndarray, max_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probability mass function of spell lengths: (length, fraction_of_spells)."""
    if lengths.size == 0:
        return np.array([]), np.array([])
    max_length = max_length or int(lengths.max())
    bins = np.arange(1, max_length + 2)
    counts, edges = np.histogram(lengths, bins=bins, density=False)
    return edges[:-1], counts / counts.sum()
