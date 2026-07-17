"""Figures for the scientific evaluation suite (verify/*.py).

Reuses the palette and pcolormesh/axis-styling helpers from the top-level
sbc_qdm.viz (same sequential-blue/diverging-blue-red/fixed-categorical
scheme), adding a third fixed categorical color (yellow) for "observed"
wherever a plot needs three simultaneous series (raw / corrected / CHIRPS)
rather than the two (raw / corrected) the original diagnostics plots use.

This module intentionally plots a *curated* subset of everything computed in
verify/*.py, not one figure per metric per temporal scale -- see the
module-level docstrings in each verify/*.py file and the evaluation output
netCDF files for the full set of daily/monthly/JJAS breakdowns that aren't
separately plotted here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from sbc_qdm.viz import (
    BASELINE,
    CAT_CORRECTED,
    CAT_RAW,
    DIVERGING_BLUE_RED,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SURFACE,
    spatial_panel,
    style_axes,
)

CAT_OBS = "#eda100"  # categorical slot 3 -- yellow, reserved for "Observed (CHIRPS)"


def _diverging_map_row(fig, axes, raw: xr.DataArray, corrected: xr.DataArray, cbar_label: str, suptitle: str) -> None:
    vmax = float(max(np.nanmax(np.abs(raw.values)), np.nanmax(np.abs(corrected.values))))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    mesh = spatial_panel(axes[0], raw, DIVERGING_BLUE_RED, norm=norm, title="Raw ECMWF")
    spatial_panel(axes[1], corrected, DIVERGING_BLUE_RED, norm=norm, title="QDM-corrected")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.14, aspect=40)
    cbar.set_label(cbar_label, color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(suptitle, color=INK_PRIMARY, fontsize=12, y=1.02)


def plot_deterministic_map(raw: xr.DataArray, corrected: xr.DataArray, out_path: Path, metric_name: str, units: str, diverging: bool = True) -> None:
    """Generic raw-vs-corrected spatial map for one deterministic metric (any scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    if diverging:
        _diverging_map_row(fig, axes, raw, corrected, units, f"{metric_name}: raw vs QDM-corrected")
    else:
        vmax = float(max(np.nanmax(raw.values), np.nanmax(corrected.values)))
        mesh = spatial_panel(axes[0], raw, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="Raw ECMWF")
        spatial_panel(axes[1], corrected, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title="QDM-corrected")
        cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.14, aspect=40)
        cbar.set_label(units, color=INK_SECONDARY, fontsize=9)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
        fig.suptitle(f"{metric_name}: raw vs QDM-corrected", color=INK_PRIMARY, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_quantile_bias_grid(qbias_raw: xr.DataArray, qbias_corrected: xr.DataArray, out_path: Path) -> None:
    """Grid of quantile-bias maps: rows = quantiles (e.g. Q10/Q50/Q90/Q95), cols = raw | corrected."""
    quantiles = qbias_raw["quantile"].values
    n = len(quantiles)
    vmax = float(max(np.nanmax(np.abs(qbias_raw.values)), np.nanmax(np.abs(qbias_corrected.values))))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(n, 2, figsize=(8, 2.6 * n), facecolor=SURFACE)
    mesh = None
    for i, q in enumerate(quantiles):
        mesh = spatial_panel(axes[i, 0], qbias_raw.sel(quantile=q), DIVERGING_BLUE_RED, norm=norm, title=f"Q{int(q * 100)} raw" if i == 0 else f"Q{int(q * 100)}")
        spatial_panel(axes[i, 1], qbias_corrected.sel(quantile=q), DIVERGING_BLUE_RED, norm=norm, title=f"Q{int(q * 100)} corrected" if i == 0 else "")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.03, pad=0.1, aspect=50)
    cbar.set_label("Quantile bias, model - CHIRPS (mm/day)", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle("Quantile bias: raw vs QDM-corrected", color=INK_PRIMARY, fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_qq(obs_q: np.ndarray, raw_q: np.ndarray, corrected_q: np.ndarray, out_path: Path) -> None:
    """Q-Q plot: model quantiles (y) vs CHIRPS quantiles (x), with a 1:1 reference line."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), facecolor=SURFACE)
    lims = [0, max(obs_q.max(), raw_q.max(), corrected_q.max()) * 1.05]
    ax.plot(lims, lims, color=BASELINE, linestyle="--", linewidth=1, label="1:1")
    ax.plot(obs_q, raw_q, color=CAT_RAW, linewidth=2, label="Raw ECMWF")
    ax.plot(obs_q, corrected_q, color=CAT_CORRECTED, linewidth=2, label="QDM-corrected")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("CHIRPS quantile (mm/day)")
    ax.set_ylabel("Model quantile (mm/day)")
    ax.set_title("Q-Q plot (domain-pooled)", color=INK_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_ecdf(obs: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray, out_path: Path) -> None:
    from sbc_qdm.verify.distributions import ecdf

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), facecolor=SURFACE)
    for da, color, label in [(obs, CAT_OBS, "CHIRPS"), (raw, CAT_RAW, "Raw ECMWF"), (corrected, CAT_CORRECTED, "QDM-corrected")]:
        x, y = ecdf(da)
        ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.set_xlabel("Daily precipitation (mm/day)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical CDF (domain-pooled)", color=INK_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_pdf(obs: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray, out_path: Path, wet_day_only: bool = True) -> None:
    from sbc_qdm.verify.distributions import pdf_histogram

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), facecolor=SURFACE)
    for da, color, label in [(obs, CAT_OBS, "CHIRPS"), (raw, CAT_RAW, "Raw ECMWF"), (corrected, CAT_CORRECTED, "QDM-corrected")]:
        series = da.where(da > 1.0) if wet_day_only else da
        x, y = pdf_histogram(series)
        ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.set_xlabel("Daily precipitation (mm/day)" + (", wet days only (>1mm)" if wet_day_only else ""))
    ax.set_ylabel("Density")
    ax.set_title("Probability density (domain-pooled)", color=INK_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_spell_distributions(
    obs_wet: np.ndarray, raw_wet: np.ndarray, corrected_wet: np.ndarray, obs_dry: np.ndarray, raw_dry: np.ndarray, corrected_dry: np.ndarray, out_path: Path
) -> None:
    from sbc_qdm.verify.spells import spell_length_histogram

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    for ax, (obs_l, raw_l, corr_l), title in [
        (axes[0], (obs_wet, raw_wet, corrected_wet), "Wet-spell length distribution"),
        (axes[1], (obs_dry, raw_dry, corrected_dry), "Dry-spell length distribution"),
    ]:
        for lengths, color, label in [(obs_l, CAT_OBS, "CHIRPS"), (raw_l, CAT_RAW, "Raw ECMWF"), (corr_l, CAT_CORRECTED, "QDM-corrected")]:
            x, y = spell_length_histogram(lengths, max_length=20)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3, label=label)
        ax.set_xlabel("Spell length (days)")
        ax.set_ylabel("Fraction of spells")
        ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
        style_axes(ax)
    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_skill_maps(acc: xr.DataArray, spearman_acc: xr.DataArray, rmsess: xr.DataArray, ivr: xr.DataArray, out_path: Path) -> None:
    """2x2 grid: ACC, Spearman ACC, RMSESS (all diverging, centered at 0), interannual
    variability ratio (diverging, centered at 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), facecolor=SURFACE)

    for ax, da, title in [(axes[0, 0], acc, "Anomaly Correlation Coefficient"), (axes[0, 1], spearman_acc, "Spearman Anomaly Correlation")]:
        vmax = float(np.nanmax(np.abs(da.values)))
        norm = TwoSlopeNorm(vcenter=0, vmin=-max(vmax, 0.01), vmax=max(vmax, 0.01))
        mesh = spatial_panel(ax, da, DIVERGING_BLUE_RED, norm=norm, title=title)
        fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    vmax = float(np.nanmax(np.abs(rmsess.values)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-max(vmax, 0.01), vmax=max(vmax, 0.01))
    mesh = spatial_panel(axes[1, 0], rmsess, DIVERGING_BLUE_RED, norm=norm, title="RMSE Skill Score (vs climatology)")
    fig.colorbar(mesh, ax=axes[1, 0], orientation="vertical", fraction=0.046, pad=0.04)

    ivr_dev = float(np.nanmax(np.abs(ivr.values - 1.0)))
    norm_ivr = TwoSlopeNorm(vcenter=1.0, vmin=1 - max(ivr_dev, 0.01), vmax=1 + max(ivr_dev, 0.01))
    mesh = spatial_panel(axes[1, 1], ivr, DIVERGING_BLUE_RED, norm=norm_ivr, title="Interannual Variability Ratio (model/obs)")
    fig.colorbar(mesh, ax=axes[1, 1], orientation="vertical", fraction=0.046, pad=0.04)

    fig.suptitle("Deterministic monthly/JJAS skill vs climatology (QDM-corrected)", color=INK_PRIMARY, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_probabilistic_skill_maps(rpss: xr.DataArray, bss: xr.DataArray, rocss: xr.DataArray, out_path: Path) -> None:
    """1x3 grid: RPSS, BSS (above-normal), ROC skill score (above-normal) -- all diverging at 0."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, da, title in [(axes[0], rpss, "RPSS"), (axes[1], bss, "BSS (above-normal)"), (axes[2], rocss, "ROC skill score (above-normal)")]:
        vmax = float(np.nanmax(np.abs(da.values)))
        norm = TwoSlopeNorm(vcenter=0, vmin=-max(vmax, 0.01), vmax=max(vmax, 0.01))
        mesh = spatial_panel(ax, da, DIVERGING_BLUE_RED, norm=norm, title=title)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.1)
    fig.suptitle("Probabilistic tercile-category skill (QDM-corrected, JJAS)", color=INK_PRIMARY, fontsize=12, y=1.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_reliability_diagram(forecast_mean: np.ndarray, observed_freq: np.ndarray, counts: np.ndarray, out_path: Path, event_label: str = "above-normal") -> None:
    fig, (ax_main, ax_hist) = plt.subplots(2, 1, figsize=(5.5, 6.5), facecolor=SURFACE, height_ratios=[3, 1], sharex=True)

    ax_main.plot([0, 1], [0, 1], color=BASELINE, linestyle="--", linewidth=1, label="Perfectly reliable")
    valid = ~np.isnan(forecast_mean)
    ax_main.plot(forecast_mean[valid], observed_freq[valid], color=CAT_CORRECTED, marker="o", markersize=5, linewidth=2, label="QDM-corrected")
    ax_main.axhline(1 / 3, color=INK_MUTED, linewidth=0.8, linestyle=":")
    ax_main.set_ylabel("Observed frequency")
    ax_main.set_title(f"Reliability diagram: {event_label} event", color=INK_PRIMARY, fontsize=11, loc="left")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    style_axes(ax_main)
    ax_main.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    bin_width = 1.0 / len(counts)
    centers = np.arange(len(counts)) * bin_width + bin_width / 2
    ax_hist.bar(centers, counts, width=bin_width * 0.9, color=CAT_CORRECTED)
    ax_hist.set_xlabel("Forecast probability")
    ax_hist.set_ylabel("Count")
    style_axes(ax_hist)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_spread_skill_map(ssr: xr.DataArray, out_path: Path) -> None:
    dev = float(np.nanmax(np.abs(ssr.values - 1.0)))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1 - max(dev, 0.01), vmax=1 + max(dev, 0.01))
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.6), facecolor=SURFACE)
    mesh = spatial_panel(ax, ssr, DIVERGING_BLUE_RED, norm=norm, title="Spread-skill ratio (QDM-corrected)")
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.05, pad=0.04)
    cbar.set_label("Spread/RMSE -- 1.0 = well calibrated", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_spatial_metric_timeseries(raw: xr.DataArray, corrected: xr.DataArray, out_path: Path, ylabel: str, title: str, time_dim: str = "time") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor=SURFACE)
    ax.plot(raw[time_dim].values, raw.values, color=CAT_RAW, linewidth=1, alpha=0.8, label="Raw ECMWF")
    ax.plot(corrected[time_dim].values, corrected.values, color=CAT_CORRECTED, linewidth=1, alpha=0.8, label="QDM-corrected")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def plot_monthly_comparison_grid(raw: xr.DataArray, corrected: xr.DataArray, climatology: xr.DataArray, out_path: Path) -> None:
    """Grid of monthly-total maps: rows = calendar month, cols = raw | corrected | climatology.

    One colorbar per row rather than a single shared scale -- a shoulder
    month (May, Oct) and a peak-season month (Jul, Aug) can differ by a
    factor of several in total rainfall, so a single global scale would
    wash out the shoulder months entirely.
    """
    months = sorted(int(m) for m in raw["month"].values)
    n = len(months)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.8 * n), facecolor=SURFACE)
    col_labels = ["Raw ECMWF", "QDM-corrected", "CHIRPS climatology"]

    for i, m in enumerate(months):
        panels = [raw.sel(month=m), corrected.sel(month=m), climatology.sel(month=m)]
        vmax = float(np.nanmax([np.nanmax(p.values) for p in panels]))
        mesh = None
        for j, (panel, label) in enumerate(zip(panels, col_labels)):
            mesh = spatial_panel(axes[i, j], panel, SEQUENTIAL_BLUE, vmin=0, vmax=vmax, title=label if i == 0 else "")
        axes[i, 0].set_ylabel(MONTH_NAMES.get(m, str(m)), color=INK_SECONDARY, fontsize=10)
        cbar = fig.colorbar(mesh, ax=axes[i, :], orientation="vertical", fraction=0.02, pad=0.015)
        cbar.set_label("mm", color=INK_SECONDARY, fontsize=8)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle("Monthly total precipitation: raw vs QDM-corrected vs CHIRPS climatology", color=INK_PRIMARY, fontsize=13, y=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_monthly_anomaly_grid(raw: xr.DataArray, corrected: xr.DataArray, climatology: xr.DataArray, out_path: Path) -> None:
    """Grid of monthly anomaly-from-climatology maps: rows = calendar month, cols = raw | corrected.

    Shows whether the forecast expects a wetter/drier-than-normal month
    (red/blue) at each pixel, and whether the correction changes that
    wetter/drier call rather than just its magnitude.
    """
    months = sorted(int(m) for m in raw["month"].values)
    n = len(months)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.8 * n), facecolor=SURFACE)
    col_labels = ["Raw ECMWF anomaly", "QDM-corrected anomaly"]

    for i, m in enumerate(months):
        raw_anom = raw.sel(month=m) - climatology.sel(month=m)
        corr_anom = corrected.sel(month=m) - climatology.sel(month=m)
        vmax = float(max(np.nanmax(np.abs(raw_anom.values)), np.nanmax(np.abs(corr_anom.values))))
        norm = TwoSlopeNorm(vcenter=0, vmin=-max(vmax, 1e-6), vmax=max(vmax, 1e-6))
        mesh = spatial_panel(axes[i, 0], raw_anom, DIVERGING_BLUE_RED, norm=norm, title=col_labels[0] if i == 0 else "")
        spatial_panel(axes[i, 1], corr_anom, DIVERGING_BLUE_RED, norm=norm, title=col_labels[1] if i == 0 else "")
        axes[i, 0].set_ylabel(MONTH_NAMES.get(m, str(m)), color=INK_SECONDARY, fontsize=10)
        cbar = fig.colorbar(mesh, ax=axes[i, :], orientation="vertical", fraction=0.02, pad=0.015)
        cbar.set_label("mm vs normal", color=INK_SECONDARY, fontsize=8)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=7)

    fig.suptitle("Monthly anomaly vs 1993-2025 climatology: raw vs QDM-corrected", color=INK_PRIMARY, fontsize=13, y=1.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
