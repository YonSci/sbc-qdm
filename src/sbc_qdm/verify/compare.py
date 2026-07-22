"""Cross-method comparison summary, built on top of each method's already-computed
`sbc-qdm evaluate` output (daily_deterministic.nc / jjas_deterministic_and_skill.nc /
jjas_probabilistic.nc) -- no recomputation, just reads + domain-means + stacks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from sbc_qdm.viz import (
    CAT_CORRECTED,
    CAT_RAW,
    DIVERGING_BLUE_RED,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SURFACE,
    spatial_panel,
)

HEADLINE_METRICS = [
    ("daily_mbe", "Daily MBE (mm/day)"),
    ("daily_pbias", "Daily PBIAS (%)"),
    ("daily_rmse", "Daily RMSE (mm/day)"),
    ("wet_day_freq_bias", "Wet-day frequency bias"),
    ("jjas_rmse", "JJAS-total RMSE (mm)"),
    ("jjas_crpss", "JJAS-total CRPSS"),
]

# Metrics added specifically to check whether the *other* 6 methods share
# QDM's known weaknesses (see README "Known limitations"), rather than only
# reporting the bias-flavored metrics every method predictably does fine on.
SKILL_METRICS = [
    ("jjas_acc", "JJAS anomaly correlation (ACC)"),
    ("jjas_roc_skill", "JJAS ROC skill, above-normal"),
    ("wet_spell_bias", "Wet-spell length bias (days)"),
    ("dry_spell_bias", "Dry-spell length bias (days)"),
]


def _spell_length_bias(eval_dir: Path) -> tuple[float, float, float, float]:
    """(raw_wet_bias, corrected_wet_bias, raw_dry_bias, corrected_dry_bias),
    each the mean spell length (days) vs CHIRPS' own mean spell length --
    spell_lengths.npz holds raw per-spell-length samples (obs/raw/corrected,
    domain-pooled, see spells.py), not a reducible DataArray, so this reads
    the npz directly instead of going through xarray.
    """
    d = np.load(eval_dir / "spell_lengths.npz")
    obs_wet, obs_dry = d["obs_wet"].mean(), d["obs_dry"].mean()
    return (
        float(d["raw_wet"].mean() - obs_wet),
        float(d["corrected_wet"].mean() - obs_wet),
        float(d["raw_dry"].mean() - obs_dry),
        float(d["corrected_dry"].mean() - obs_dry),
    )


def _masked_mean(da: xr.DataArray, mask: xr.DataArray | None) -> xr.DataArray:
    return da.where(mask).mean() if mask is not None else da.mean()


def comparison_summary(method_eval_dirs: dict[str, Path], mask: xr.DataArray | None = None) -> xr.Dataset:
    """method_eval_dirs: {method_name: path to that method's evaluation/ dir},
    in the order methods should appear along the resulting "method" dimension.

    `mask`: optional (lat, lon) boolean DataArray (e.g. from
    verify.boundary.load_country_mask) to restrict every *pixel-level*
    metric's domain mean to, e.g. Ethiopia only. The two spell-length
    metrics can't be restricted this way -- spell_lengths.npz already holds
    domain-pooled samples from `sbc-qdm evaluate` (see spells.py), so masking
    here would need re-scanning the full 33-year record, not just re-averaging
    already-computed output -- they stay domain-wide regardless of `mask`.
    """
    rows = []
    method_names = []
    raw_row = None

    for method, eval_dir in method_eval_dirs.items():
        daily = xr.open_dataset(eval_dir / "daily_deterministic.nc")
        jjas = xr.open_dataset(eval_dir / "jjas_deterministic_and_skill.nc")
        jjas_prob = xr.open_dataset(eval_dir / "jjas_probabilistic.nc")
        raw_wet_spell, corrected_wet_spell, raw_dry_spell, corrected_dry_spell = _spell_length_bias(eval_dir)

        if raw_row is None:
            raw_row = xr.Dataset(
                {
                    "daily_mbe": _masked_mean(daily["raw_mbe"], mask),
                    "daily_pbias": _masked_mean(daily["raw_pbias"], mask),
                    "daily_rmse": _masked_mean(daily["raw_rmse"], mask),
                    "wet_day_freq_bias": _masked_mean(daily["wet_day_freq_bias_raw"], mask),
                    "jjas_rmse": _masked_mean(jjas["raw_rmse"], mask),
                    "jjas_crpss": xr.zeros_like(_masked_mean(jjas_prob["crpss"], mask)),  # CRPSS is skill-vs-raw by definition
                    "jjas_acc": _masked_mean(jjas["raw_acc"], mask),
                    "jjas_roc_skill": _masked_mean(jjas_prob["raw_roc_skill_score"].sel(category="above"), mask),
                    "wet_spell_bias": xr.DataArray(raw_wet_spell),
                    "dry_spell_bias": xr.DataArray(raw_dry_spell),
                }
            )

        row = xr.Dataset(
            {
                "daily_mbe": _masked_mean(daily["corrected_mbe"], mask),
                "daily_pbias": _masked_mean(daily["corrected_pbias"], mask),
                "daily_rmse": _masked_mean(daily["corrected_rmse"], mask),
                "wet_day_freq_bias": _masked_mean(daily["wet_day_freq_bias_corrected"], mask),
                "jjas_rmse": _masked_mean(jjas["corrected_rmse"], mask),
                "jjas_crpss": _masked_mean(jjas_prob["crpss"], mask),
                "jjas_acc": _masked_mean(jjas["corrected_acc"], mask),
                "jjas_roc_skill": _masked_mean(jjas_prob["corrected_roc_skill_score"].sel(category="above"), mask),
                "wet_spell_bias": xr.DataArray(corrected_wet_spell),
                "dry_spell_bias": xr.DataArray(corrected_dry_spell),
            }
        )
        rows.append(row)
        method_names.append(method)

    method_dim = xr.DataArray(["raw", *method_names], dims="method", name="method")
    return xr.concat([raw_row, *rows], dim=method_dim).compute()


def plot_method_comparison(summary: xr.Dataset, out_path: Path) -> None:
    """One bar-chart panel per metric (top row: bias-flavored HEADLINE_METRICS,
    every method predictably improves on these; bottom row: SKILL_METRICS,
    checking whether the other 6 methods share QDM's known weaknesses --
    ACC drop, flat ROC skill, untouched spell persistence -- rather than only
    reporting where every method looks fine), methods along the x-axis. QDM
    is highlighted (aqua, matching how it's shown as "corrected" elsewhere in
    this project); "raw" is a muted reference bar; the other 5 alternative
    methods get a shared muted blue so the eye isn't drawn to any one of them
    over another -- only QDM is meant to stand out.
    """
    methods = [m for m in summary["method"].values.tolist() if m != "raw"]
    all_metrics = HEADLINE_METRICS + SKILL_METRICS
    ncols = len(HEADLINE_METRICS)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 8.0), facecolor=SURFACE)
    axes = axes.flatten()
    no_raw_baseline = {"jjas_crpss"}  # CRPSS is skill-vs-raw by definition, so raw's own value is always 0

    for ax, (var, title) in zip(axes, all_metrics):
        # NB: must use dict-form sel here -- summary[var].sel(method=m) silently
        # collides with xarray's own reserved `method=` kwarg (nearest-neighbor
        # lookup), since our dimension happens to also be named "method". That
        # collision makes .sel() a no-op (returns the full array, not a scalar).
        values = [float(summary[var].sel({"method": m})) for m in methods]
        colors = [CAT_CORRECTED if m == "qdm" else "#9ec5f4" for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors)

        if var not in no_raw_baseline:
            raw_val = float(summary[var].sel({"method": "raw"}))
            ax.axhline(raw_val, color=CAT_RAW, linestyle="--", linewidth=1.2, label="Raw ECMWF")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7.5, color=INK_SECONDARY)
        ax.set_title(title, color=INK_PRIMARY, fontsize=9.5, loc="left")
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        del bars

    for ax in axes[len(all_metrics):]:
        ax.axis("off")

    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=7.5, loc="upper left")
    fig.text(0.01, 0.5, "Skill metrics ->", color=INK_MUTED, fontsize=8, rotation=90, va="center")
    fig.suptitle("Bias-correction method comparison (33-year leave-one-year-out cross-validation)", color=INK_PRIMARY, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def mbe_maps(method_eval_dirs: dict[str, Path]) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """{method: corrected_mbe map}, raw_mbe map (identical raw input across methods)."""
    fields, raw_field = {}, None
    for method, eval_dir in method_eval_dirs.items():
        daily = xr.open_dataset(eval_dir / "daily_deterministic.nc")
        if raw_field is None:
            raw_field = daily["raw_mbe"]
        fields[method] = daily["corrected_mbe"]
    return fields, raw_field


def wet_day_freq_bias_maps(method_eval_dirs: dict[str, Path]) -> tuple[dict[str, xr.DataArray], xr.DataArray]:
    """{method: corrected wet-day-frequency-bias map}, raw's own map.

    Variable naming here is reversed vs mbe/pbias/rmse (`wet_day_freq_bias_raw`/
    `_corrected`, not `raw_wet_day_freq_bias`) -- an inconsistency already
    present in verify/distributions.py's output, not introduced here.
    """
    fields, raw_field = {}, None
    for method, eval_dir in method_eval_dirs.items():
        daily = xr.open_dataset(eval_dir / "daily_deterministic.nc")
        if raw_field is None:
            raw_field = daily["wet_day_freq_bias_raw"]
        fields[method] = daily["wet_day_freq_bias_corrected"]
    return fields, raw_field


def roc_skill_maps(method_eval_dirs: dict[str, Path], category: str = "above") -> dict[str, xr.DataArray]:
    """{method: corrected JJAS ROC skill score map}. No raw-baseline panel:
    unlike CRPSS, raw's own ROC skill score is a real (non-zero) number, but
    showing it here would suggest a raw-vs-corrected bias comparison, when
    what's actually informative is *where* each method's discrimination
    skill holds up spatially -- see README "Known limitations": ROC skill
    is domain-mean-unchanged by QDM, but that could still hide method-to-
    method or region-to-region differences a spatial map would reveal.
    """
    fields = {}
    for method, eval_dir in method_eval_dirs.items():
        jjas_prob = xr.open_dataset(eval_dir / "jjas_probabilistic.nc")
        fields[method] = jjas_prob["corrected_roc_skill_score"].sel(category=category)
    return fields


def plot_method_comparison_maps(
    fields: dict[str, xr.DataArray],
    out_path: Path,
    metric_name: str,
    units: str,
    raw_field: xr.DataArray | None = None,
    diverging: bool = True,
) -> None:
    """Small-multiples spatial map grid: one panel per method (plus an optional
    leading "Raw ECMWF" reference panel), sharing a single colorbar so the
    *spatial pattern* is directly comparable method-to-method. This is the
    complement to plot_method_comparison()'s domain-mean bar chart, which
    can't distinguish two methods landing on the same domain-mean number via
    very different spatial patterns (e.g. one overcorrecting the wet
    northwest while undercorrecting the dry southeast, versus a spatially
    uniform residual).
    """
    panels = ([("Raw ECMWF", raw_field)] if raw_field is not None else []) + list(fields.items())
    n = len(panels)
    ncols = 4
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.1 * nrows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).flatten()

    if diverging:
        vmax = float(max(np.nanmax(np.abs(field.values)) for _, field in panels))
        norm, vmin_seq, vmax_seq = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), None, None
        cmap = DIVERGING_BLUE_RED
    else:
        norm = None
        vmin_seq, vmax_seq = 0.0, float(max(np.nanmax(field.values) for _, field in panels))
        cmap = SEQUENTIAL_BLUE

    mesh = None
    for ax, (label, field) in zip(axes, panels):
        mesh = spatial_panel(ax, field, cmap, norm=norm, vmin=vmin_seq, vmax=vmax_seq, title=label)
        if label == "qdm":
            for spine in ax.spines.values():
                spine.set_edgecolor(CAT_CORRECTED)
                spine.set_linewidth(2)

    for ax in axes[n:]:
        ax.axis("off")

    cbar = fig.colorbar(mesh, ax=axes[:n].tolist(), orientation="horizontal", fraction=0.04, pad=0.1, aspect=50)
    cbar.set_label(f"{metric_name} ({units})" if units else metric_name, color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(f"{metric_name}: spatial pattern by correction method", color=INK_PRIMARY, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
