"""Cross-method comparison summary, built on top of each method's already-computed
`sbc-qdm evaluate` output (daily_deterministic.nc / jjas_deterministic_and_skill.nc /
jjas_probabilistic.nc) -- no recomputation, just reads + domain-means + stacks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from sbc_qdm.viz import CAT_CORRECTED, CAT_RAW, INK_PRIMARY, INK_SECONDARY, SURFACE

HEADLINE_METRICS = [
    ("daily_mbe", "Daily MBE (mm/day)"),
    ("daily_pbias", "Daily PBIAS (%)"),
    ("daily_rmse", "Daily RMSE (mm/day)"),
    ("wet_day_freq_bias", "Wet-day frequency bias"),
    ("jjas_rmse", "JJAS-total RMSE (mm)"),
    ("jjas_crpss", "JJAS-total CRPSS"),
]


def comparison_summary(method_eval_dirs: dict[str, Path]) -> xr.Dataset:
    """method_eval_dirs: {method_name: path to that method's evaluation/ dir},
    in the order methods should appear along the resulting "method" dimension.
    """
    rows = []
    method_names = []
    raw_row = None

    for method, eval_dir in method_eval_dirs.items():
        daily = xr.open_dataset(eval_dir / "daily_deterministic.nc")
        jjas = xr.open_dataset(eval_dir / "jjas_deterministic_and_skill.nc")
        jjas_prob = xr.open_dataset(eval_dir / "jjas_probabilistic.nc")

        if raw_row is None:
            raw_row = xr.Dataset(
                {
                    "daily_mbe": daily["raw_mbe"].mean(),
                    "daily_pbias": daily["raw_pbias"].mean(),
                    "daily_rmse": daily["raw_rmse"].mean(),
                    "wet_day_freq_bias": daily["wet_day_freq_bias_raw"].mean(),
                    "jjas_rmse": jjas["raw_rmse"].mean(),
                    "jjas_crpss": xr.zeros_like(jjas_prob["crpss"].mean()),  # CRPSS is skill-vs-raw by definition
                }
            )

        row = xr.Dataset(
            {
                "daily_mbe": daily["corrected_mbe"].mean(),
                "daily_pbias": daily["corrected_pbias"].mean(),
                "daily_rmse": daily["corrected_rmse"].mean(),
                "wet_day_freq_bias": daily["wet_day_freq_bias_corrected"].mean(),
                "jjas_rmse": jjas["corrected_rmse"].mean(),
                "jjas_crpss": jjas_prob["crpss"].mean(),
            }
        )
        rows.append(row)
        method_names.append(method)

    method_dim = xr.DataArray(["raw", *method_names], dims="method", name="method")
    return xr.concat([raw_row, *rows], dim=method_dim).compute()


def plot_method_comparison(summary: xr.Dataset, out_path: Path) -> None:
    """One bar-chart panel per headline metric, methods along the x-axis.
    QDM is highlighted (aqua, matching how it's shown as "corrected"
    elsewhere in this project); "raw" is a muted reference bar; the other
    5 alternative methods get a shared muted blue so the eye isn't drawn to
    any one of them over another -- only QDM is meant to stand out.
    """
    methods = [m for m in summary["method"].values.tolist() if m != "raw"]
    n = len(HEADLINE_METRICS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2), facecolor=SURFACE)

    for ax, (var, title) in zip(axes, HEADLINE_METRICS):
        # NB: must use dict-form sel here -- summary[var].sel(method=m) silently
        # collides with xarray's own reserved `method=` kwarg (nearest-neighbor
        # lookup), since our dimension happens to also be named "method". That
        # collision makes .sel() a no-op (returns the full array, not a scalar).
        values = [float(summary[var].sel({"method": m})) for m in methods]
        colors = [CAT_CORRECTED if m == "qdm" else "#9ec5f4" for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors)

        if var != "jjas_crpss":  # CRPSS has no "raw" baseline (raw's own CRPSS is 0 by definition)
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

    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=7.5, loc="upper left")
    fig.suptitle("Bias-correction method comparison (33-year leave-one-year-out cross-validation)", color=INK_PRIMARY, fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
