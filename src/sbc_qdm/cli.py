"""Command-line entrypoints: sbc-qdm train|cross-validate|cross-validate-fold|apply|validate|plot-diagnostics|evaluate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import dask
import typer
import xarray as xr

from sbc_qdm.config import DEFAULT_CONFIG_PATH, load_config
from sbc_qdm.io import load_chirps_reference
from sbc_qdm.pipeline import prepare_hindcast, prepare_target_year
from sbc_qdm.preprocess import build_land_mask
from sbc_qdm.qdm import apply_operational, apply_qdm, train_qdm
from sbc_qdm.validate import bias_maps, crps_skill_score, rank_histogram, wet_day_frequency
from sbc_qdm.verify.run import run_full_evaluation
from sbc_qdm.viz import plot_all

app = typer.Typer(help="QDM bias correction of ECMWF seasonal forecasts against CHIRPS.")


def _trained_to_dataset(trained: dict[int, tuple[xr.DataArray, xr.DataArray]]) -> xr.Dataset:
    months = sorted(trained.keys())
    month_index = xr.DataArray(months, dims="month", name="month")
    af = xr.concat([trained[m][0] for m in months], dim=month_index)
    hist_q = xr.concat([trained[m][1] for m in months], dim=month_index)
    return xr.Dataset({"af": af, "hist_q": hist_q})


def _dataset_to_trained(ds: xr.Dataset) -> dict[int, tuple[xr.DataArray, xr.DataArray]]:
    return {int(m): (ds["af"].sel(month=m), ds["hist_q"].sel(month=m)) for m in ds.month.values}


@app.command()
def train(config: str = str(DEFAULT_CONFIG_PATH)):
    """Fit QDM on the full hindcast record and save the trained adjustment."""
    cfg = load_config(config)
    _, _, ref, hist = prepare_hindcast(cfg)

    trained = train_qdm(ref, hist, cfg)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qdm_trained.nc"
    _trained_to_dataset(trained).to_netcdf(out_path)
    typer.echo(f"Trained QDM saved to {out_path}")


@app.command("cross-validate-fold")
def cross_validate_fold(config: str = str(DEFAULT_CONFIG_PATH), year: int = typer.Option(...)):
    """Run a single leave-one-year-out fold and save it.

    Internal command used by `cross-validate` -- it shells out to this, one
    subprocess per year, rather than looping in-process. A single long-lived
    process training+adjusting 33 full-domain/full-ensemble folds back to
    back started hitting MemoryError on allocations as small as ~37 MiB after
    5 successful folds, on a machine with several GB still free -- classic
    heap fragmentation from a long-running numpy/dask-heavy loop (exacerbated
    by ambient memory pressure from other processes on the machine, not
    necessarily a leak in this code). A fresh process per fold gives the OS a
    clean slate every time instead of accumulating fragmentation for hours.
    """
    cfg = load_config(config)
    _, _, ref, hist = prepare_hindcast(cfg)

    is_held_out = hist.time.dt.year == year
    ref_train = ref.sel(time=~is_held_out)
    hist_train = hist.sel(time=~is_held_out)
    sim_holdout = hist.sel(time=is_held_out)

    trained = train_qdm(ref_train, hist_train, cfg)
    corrected = apply_qdm(sim_holdout, trained, cfg).compute()

    fold_dir = Path(cfg["paths"]["output_dir"]) / "loyo_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_netcdf(fold_dir / f"{year}.nc")
    typer.echo(f"fold {year} done")


@app.command("cross-validate")
def cross_validate(config: str = str(DEFAULT_CONFIG_PATH)):
    """Run leave-one-year-out cross-validation over the hindcast record.

    Drives cross-validate-fold as one subprocess per year (see its docstring
    for why) and is resumable: existing loyo_folds/{year}.nc files are
    skipped, so a killed/crashed run can just be re-invoked.
    """
    cfg = load_config(config)
    chirps, mask, ref, hist = prepare_hindcast(cfg)
    years = sorted(set(hist.time.dt.year.values.tolist()))
    del ref, hist  # driver only needs the year list; drop the lazy arrays before spawning subprocesses

    out_dir = Path(cfg["paths"]["output_dir"])
    fold_dir = out_dir / "loyo_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for i, year in enumerate(years):
        fold_path = fold_dir / f"{year}.nc"
        if fold_path.exists():
            typer.echo(f"[{i + 1}/{len(years)}] fold {year} already done, skipping")
            continue

        typer.echo(f"[{i + 1}/{len(years)}] running fold {year} in a subprocess...")
        result = subprocess.run(
            [sys.executable, "-m", "sbc_qdm.cli", "cross-validate-fold", "--config", config, "--year", str(year)],
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)

    _, _, ref, hist = prepare_hindcast(cfg)
    # chunks=... keeps each ~107 MB fold file dask-backed on read; without it
    # xr.open_dataarray loads eagerly, and concatenating 32+ of those plus the
    # full hist record for diagnostics forced a single ~3.2 GiB allocation
    # that OOM'd on this machine.
    fold_arrays = [xr.open_dataarray(fold_dir / f"{y}.nc", chunks={"lat": 10, "lon": 10}) for y in years]
    corrected = xr.concat(fold_arrays, dim="time").sortby("time")

    # qdm.py sets the process-wide dask scheduler to threaded (for the
    # per-fold training/adjustment work). netCDF4/HDF5 isn't reliably
    # thread-safe for writes, and threaded writes of this concatenated,
    # multi-chunk array previously deadlocked silently (the process sat for
    # 18+ hours using ~20s of CPU time, having written only the file header).
    # Forcing "synchronous" here processes one chunk at a time -- slower, but
    # correct -- for the write and the CRPS apply_ufunc call in diagnostics.
    with dask.config.set(scheduler="synchronous"):
        corrected.to_netcdf(out_dir / "loyo_corrected.nc")
        typer.echo(f"Cross-validated corrected hindcast saved to {out_dir / 'loyo_corrected.nc'}")

        # .compute() once here: diagnostics is still a lazy dask graph after
        # _compute_diagnostics(), and without this, to_netcdf() below
        # computes it once for the write, then _print_diagnostics_summary()
        # recomputes the entire thing again (including the CRPS ensemble
        # calculation) from scratch just to print a few means -- this cost
        # ~27 minutes of redundant work before it was caught and fixed.
        diagnostics = _compute_diagnostics(ref, hist, corrected).compute()
        diagnostics.to_netcdf(out_dir / "loyo_diagnostics.nc")
    typer.echo(f"Diagnostics saved to {out_dir / 'loyo_diagnostics.nc'}")
    _print_diagnostics_summary(diagnostics)


@app.command()
def apply(config: str = str(DEFAULT_CONFIG_PATH), year: Optional[int] = None):
    """Apply the trained QDM adjustment to the operational forecast year."""
    cfg = load_config(config)
    chirps, mask, ref, hist = prepare_hindcast(cfg)
    target = prepare_target_year(cfg, chirps, mask, year)

    corrected = apply_operational(ref, hist, target, cfg)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target_year = year or cfg["time"]["operational_year"]
    out_path = out_dir / f"corrected_{target_year}.nc"
    # Write with the synchronous scheduler, not the threaded one qdm.py sets
    # process-wide -- netCDF4/HDF5 writes under a threaded dask scheduler
    # deadlocked silently during cross-validate's final combine step (see
    # its docstring), and apply_operational() likewise returns a lazy,
    # dask-chunked array straight into to_netcdf() here.
    with dask.config.set(scheduler="synchronous"):
        corrected.to_netcdf(out_path)
    typer.echo(f"Corrected forecast for {target_year} saved to {out_path}")


@app.command()
def validate(config: str = str(DEFAULT_CONFIG_PATH)):
    """Compute pre/post-correction skill diagnostics from the cross-validated hindcast."""
    cfg = load_config(config)
    chirps, mask, ref, hist = prepare_hindcast(cfg)

    out_dir = Path(cfg["paths"]["output_dir"])
    loyo_path = out_dir / "loyo_corrected.nc"
    if loyo_path.exists():
        corrected = xr.open_dataarray(loyo_path, chunks={"lat": 10, "lon": 10})
    else:
        typer.echo("No cached loyo_corrected.nc found -- running cross-validation first.")
        cross_validate(config)
        corrected = xr.open_dataarray(loyo_path, chunks={"lat": 10, "lon": 10})

    # synchronous scheduler: netCDF4/HDF5 writes under qdm.py's process-wide
    # threaded scheduler deadlocked silently elsewhere (see cross_validate()'s
    # comment) -- this command never hit it only because it wasn't exercised
    # standalone at full scale after that fix went in, not because it's safe.
    # .compute() once, for the same reason as cross_validate(): without it,
    # to_netcdf() and _print_diagnostics_summary() each force their own full
    # recomputation of the lazy diagnostics graph.
    with dask.config.set(scheduler="synchronous"):
        diagnostics = _compute_diagnostics(ref, hist, corrected).compute()
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.to_netcdf(out_dir / "loyo_diagnostics.nc")
    _print_diagnostics_summary(diagnostics)


@app.command("plot-diagnostics")
def plot_diagnostics(config: str = str(DEFAULT_CONFIG_PATH)):
    """Render spatial/distributional figures from loyo_diagnostics.nc (running validate first if needed)."""
    cfg = load_config(config)
    out_dir = Path(cfg["paths"]["output_dir"])
    diagnostics_path = out_dir / "loyo_diagnostics.nc"
    if not diagnostics_path.exists():
        typer.echo("No cached loyo_diagnostics.nc found -- running validate first.")
        validate(config)

    fig_dir = out_dir / "figures"
    paths = plot_all(diagnostics_path, fig_dir)
    for path in paths:
        typer.echo(f"Saved {path}")


@app.command()
def evaluate(config: str = str(DEFAULT_CONFIG_PATH)):
    """Full scientific evaluation: daily/monthly/JJAS-seasonal, spatial, raw vs QDM-corrected.

    Requires output/loyo_corrected.nc (run cross-validate first). Computes
    deterministic (MBE/MAE/PBIAS/RMSE/SD & CV ratio), distributional (Q-Q/
    ECDF/PDF, quantile bias, wet-day frequency bias), wet/dry spell
    distributions, deterministic skill vs climatology (ACC/Spearman ACC/
    RMSESS/interannual variability ratio), probabilistic tercile-category
    skill (RPSS/BSS/ROC) and CRPS/CRPSS, ensemble calibration (spread-skill
    ratio, reliability diagram), and spatial performance (pattern
    correlation, spatial RMSE) -- see src/sbc_qdm/verify/ for definitions.
    """
    cfg = load_config(config)
    out_dir = Path(cfg["paths"]["output_dir"])
    loyo_path = out_dir / "loyo_corrected.nc"
    if not loyo_path.exists():
        typer.echo("No cached loyo_corrected.nc found -- running cross-validation first.")
        cross_validate(config)

    chirps = load_chirps_reference(cfg)
    mask = build_land_mask(chirps)
    chirps = chirps.where(mask)  # match the land mask already applied to hist/corrected
    _, _, ref, hist = prepare_hindcast(cfg)
    corrected = xr.open_dataarray(loyo_path, chunks={"lat": 10, "lon": 10})

    eval_dir = out_dir / "evaluation"
    # Same reasoning as cross-validate's final write: force synchronous so no
    # netCDF write or apply_ufunc call runs under qdm.py's process-wide
    # threaded scheduler (see qdm.py's dask.config.set docstring).
    with dask.config.set(scheduler="synchronous"):
        run_full_evaluation(chirps, hist, corrected, eval_dir)
    typer.echo(f"Full evaluation suite written to {eval_dir}")


def _compute_diagnostics(ref: xr.DataArray, raw: xr.DataArray, corrected: xr.DataArray) -> xr.Dataset:
    bm = bias_maps(ref, raw, corrected)
    wf_raw = wet_day_frequency(raw).rename("wet_day_freq_raw")
    wf_corrected = wet_day_frequency(corrected).rename("wet_day_freq_corrected")
    crpss = crps_skill_score(ref, raw, corrected)
    rh_raw = rank_histogram(ref, raw).rename("rank_hist_raw")
    rh_corrected = rank_histogram(ref, corrected).rename("rank_hist_corrected")

    return xr.merge([bm, wf_raw, wf_corrected, crpss, rh_raw, rh_corrected])


def _print_diagnostics_summary(diagnostics: xr.Dataset) -> None:
    typer.echo("--- Diagnostics summary (domain mean) ---")
    typer.echo(f"Raw bias:        {float(diagnostics['raw_bias'].mean()):.3f} mm/day")
    typer.echo(f"Corrected bias:  {float(diagnostics['corrected_bias'].mean()):.3f} mm/day")
    typer.echo(f"Raw wet-day freq:       {float(diagnostics['wet_day_freq_raw'].mean()):.3f}")
    typer.echo(f"Corrected wet-day freq: {float(diagnostics['wet_day_freq_corrected'].mean()):.3f}")
    typer.echo(f"CRPS raw:        {float(diagnostics['crps_raw'].mean()):.3f}")
    typer.echo(f"CRPS corrected:  {float(diagnostics['crps_corrected'].mean()):.3f}")
    typer.echo(f"CRPSS (skill vs raw): {float(diagnostics['crpss'].mean()):.3f}")


if __name__ == "__main__":
    app()
