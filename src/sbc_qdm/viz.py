"""Spatial and distributional diagnostic plots for the QDM correction.

Colors follow a fixed project palette rather than matplotlib defaults:
a single-hue sequential blue ramp for magnitude fields (wet-day frequency,
CRPS), a blue<->red diverging ramp with a neutral gray midpoint for polarity
fields (bias, CRPSS), and two fixed categorical hues (blue for raw, aqua for
corrected) for the raw-vs-corrected comparison in the rank histogram -- never
cycled, never a rainbow colormap.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("sbc_qdm_sequential_blue", _SEQUENTIAL_BLUE_STEPS)

_DIVERGING_STEPS = [
    "#0d366b",  # dark blue: low / dry
    "#2a78d6",  # blue
    "#f0efec",  # neutral gray midpoint: zero
    "#e34948",  # red
    "#8f2c2b",  # dark red: high / wet
]
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list("sbc_qdm_diverging", _DIVERGING_STEPS)

CAT_RAW = "#2a78d6"  # categorical slot 1 -- blue
CAT_CORRECTED = "#1baf7a"  # categorical slot 2 -- aqua

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def spatial_panel(ax, da: xr.DataArray, cmap, norm=None, vmin=None, vmax=None, title: str = ""):
    cmap = copy.copy(cmap)
    cmap.set_bad(color=SURFACE, alpha=0)
    mesh = ax.pcolormesh(da["lon"], da["lat"], da.values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
    style_axes(ax)
    return mesh


def plot_bias_maps(diagnostics: xr.Dataset, out_path: Path) -> None:
    raw, corrected = diagnostics["raw_bias"], diagnostics["corrected_bias"]
    vmax = float(max(abs(raw).max(), abs(corrected).max()))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    mesh = spatial_panel(axes[0], raw, DIVERGING_BLUE_RED, norm=norm, title="Raw ECMWF bias")
    spatial_panel(axes[1], corrected, DIVERGING_BLUE_RED, norm=norm, title="QDM-corrected bias")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.14, aspect=40)
    cbar.set_label("Mean bias vs CHIRPS (mm/day) -- blue = too dry, red = too wet", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle("Precipitation bias: raw vs QDM-corrected (33-yr leave-one-year-out)", color=INK_PRIMARY, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_wet_day_frequency(diagnostics: xr.Dataset, out_path: Path) -> None:
    raw, corrected = diagnostics["wet_day_freq_raw"], diagnostics["wet_day_freq_corrected"]
    vmax = float(max(raw.max(), corrected.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    mesh = spatial_panel(axes[0], raw, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="Raw ECMWF wet-day frequency")
    spatial_panel(axes[1], corrected, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="QDM-corrected wet-day frequency")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.14, aspect=40)
    cbar.set_label("Fraction of days > 1 mm/day", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle("Wet-day frequency: raw vs QDM-corrected", color=INK_PRIMARY, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_crps(diagnostics: xr.Dataset, out_path: Path) -> None:
    raw, corrected = diagnostics["crps_raw"], diagnostics["crps_corrected"]
    vmax = float(max(raw.max(), corrected.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    mesh = spatial_panel(axes[0], raw, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="Raw ECMWF CRPS")
    spatial_panel(axes[1], corrected, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="QDM-corrected CRPS")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.14, aspect=40)
    cbar.set_label("CRPS (mm/day, lower is better)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle("Ensemble CRPS: raw vs QDM-corrected", color=INK_PRIMARY, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_crpss(diagnostics: xr.Dataset, out_path: Path) -> None:
    crpss = diagnostics["crpss"]
    vmax = float(np.nanmax(np.abs(crpss.values)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.6), facecolor=SURFACE)
    mesh = spatial_panel(ax, crpss, DIVERGING_BLUE_RED, norm=norm, title="CRPS skill score (vs raw)")
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    cbar.set_label("CRPSS -- red = correction helps, blue = correction hurts", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_rank_histogram(diagnostics: xr.Dataset, out_path: Path) -> None:
    raw = diagnostics["rank_hist_raw"].values
    corrected = diagnostics["rank_hist_corrected"].values
    ranks = diagnostics["rank"].values
    width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor=SURFACE)
    ax.bar(ranks - width / 2, raw, width=width, color=CAT_RAW, label="Raw ECMWF")
    ax.bar(ranks + width / 2, corrected, width=width, color=CAT_CORRECTED, label="QDM-corrected")
    ax.set_yscale("log")
    ax.set_xlabel("Rank of CHIRPS observation within ensemble")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(
        "Rank histogram, pooled over time/lat/lon -- flat is well-calibrated",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
    )
    style_axes(ax)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_all(diagnostics_path: Path, out_dir: Path) -> list[Path]:
    """Render all diagnostic figures from a saved loyo_diagnostics.nc into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(diagnostics_path)

    plotters = {
        "bias_maps.png": plot_bias_maps,
        "wet_day_frequency.png": plot_wet_day_frequency,
        "crps.png": plot_crps,
        "crpss.png": plot_crpss,
        "rank_histogram.png": plot_rank_histogram,
    }
    paths = []
    for fname, plotter in plotters.items():
        path = out_dir / fname
        plotter(ds, path)
        paths.append(path)
    return paths
