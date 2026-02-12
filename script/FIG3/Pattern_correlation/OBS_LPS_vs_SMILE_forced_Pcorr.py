#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centered Pattern correlations (Santer et al. 1993) between SMILE realizations / observations and SMILE ensemble-mean
forced trend patterns for sliding windows 1950–2022...2013–2022 (10–73 years).

For a given model:

1) For each run and each sliding period:
   - Compute area-weighted pattern correlation with the SMILE ensemble-mean
     forced trend pattern for that period.

2) For each period:
   - Compute area-weighted pattern correlation between observed forced trend
     pattern and the SMILE ensemble-mean forced trend pattern.

Outputs (one NetCDF per model):

- pattern_corr_run_vs_mmem(run, period)
- pattern_corr_obs_vs_mmem(period)

Usage:
    mpirun -np N python -u Setp4_pattern_correlations_MPI.py <MODEL_NAME> <OBS_FORCED_FILE>

Example:
    mpirun -np 8 python -u Setp4_pattern_correlations_MPI.py CESM2 \
        /work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS/OBS_forced_MK_trend_1950-2022_sliding.nc
"""
# %%
import os
import sys
import numpy as np
import xarray as xr
from mpi4py import MPI

import src.SAT_function_Obs_Fingerprint as data_process
# %%
# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()

print(f"[rank {rank}] MPI world size = {npro}", flush=True)
# %%
# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
if len(sys.argv) < 3:
    if rank == 0:
        print(
            "Usage: python Setp4_pattern_correlations_MPI.py <MODEL_NAME> <OBS_FORCED_FILE>",
            flush=True,
        )
    sys.exit(1)

model = sys.argv[1]
obs_forced_path = sys.argv[2]

print(f"[rank {rank}] Model: {model}", flush=True)
if rank == 0:
    print(f"[rank 0] Observed forced trend file: {obs_forced_path}", flush=True)
# %%
# ---------------------------------------------------------------------
# Paths (edit if needed)
# ---------------------------------------------------------------------
dir_members = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}"      # OBS-LPS framework obtained forced trend path
)

dir_forced = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}/SMILE_forced"
)

dir_out = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Pattern_correlation/{model}/"
)

if rank == 0:
    os.makedirs(dir_out, exist_ok=True)
comm.Barrier()
# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
start_year = 1950
end_year = 2022
min_length = 10  # 10–73 years

# Sliding windows: 2013–2022, 2012–2022, ..., 1950–2022
period_starts = list(range(end_year - min_length + 1, start_year - 1, -1))
period_labels = [f"{by}-{end_year}" for by in period_starts]
n_period = len(period_starts)
# %%
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pattern_corr_da(x: xr.DataArray,
                    y: xr.DataArray,
                    lat: xr.DataArray,
                    use_weights: bool = True) -> float:
    """
    Area-weighted pattern correlation between two 2D fields x(lat, lon) and y(lat, lon).

    - Uses cos(lat) weights if use_weights=True.
    - Handles NaNs consistently (drops grid points where x or y is NaN).
    """
    ds = xr.Dataset({"x": x, "y": y})

    # Drop points where either field is NaN
    ds = ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]), drop=True)

    if ds["x"].size == 0:
        return np.nan

    x_ = ds["x"]
    y_ = ds["y"]

    if use_weights:
        w = np.cos(np.deg2rad(lat))
        w = w / w.mean()
        w2 = w.broadcast_like(x_)
    else:
        w2 = xr.ones_like(x_)

    w_sum = w2.sum(dim=("lat", "lon"))

    x_mean = (w2 * x_).sum(dim=("lat", "lon")) / w_sum
    y_mean = (w2 * y_).sum(dim=("lat", "lon")) / w_sum

    x_dev = x_ - x_mean
    y_dev = y_ - y_mean

    cov = (w2 * x_dev * y_dev).sum(dim=("lat", "lon")) / w_sum
    var_x = (w2 * x_dev**2).sum(dim=("lat", "lon")) / w_sum
    var_y = (w2 * y_dev**2).sum(dim=("lat", "lon")) / w_sum

    # Avoid division by zero
    var_x = var_x.where(var_x > 0.0)
    var_y = var_y.where(var_y > 0.0)
    cov = cov.where(np.isfinite(var_x) & np.isfinite(var_y))

    corr = cov / np.sqrt(var_x * var_y)
    return float(corr)
# ---------------------------------------------------------------------
# Load SMILE members and SMILE ensemble-mean forced trend
# ---------------------------------------------------------------------
def load_members_and_forced(model_name: str):
    # Members
    mem_file = os.path.join(dir_members, f"forced_{model_name}_MK_trend_1950-2022_sliding.nc")
    if not os.path.exists(mem_file):
        raise FileNotFoundError(f"[rank {rank}] Members file not found: {mem_file}")

    ds_mem = xr.open_dataset(mem_file)
    tas_mem = ds_mem["trend"].sel(period=period_labels)  # (run, period, lat, lon)
    
    # SMILE ensemble-mean forced MK trends
    forced_file = os.path.join(
        dir_forced, f"{model_name}_ENSmean_forced_MK_trend_1950-2022_sliding.nc"
    )
    if not os.path.exists(forced_file):
        raise FileNotFoundError(f"[rank {rank}] Forced trend file not found: {forced_file}")

    ds_forced = xr.open_dataset(forced_file)
    if "trend" not in ds_forced.data_vars:
        raise KeyError(f"[rank {rank}] 'trend' variable not found in {forced_file}")

    forced_trend = ds_forced["trend"]  # (period, lat, lon)
    lat = forced_trend["lat"]
    lon = forced_trend["lon"]

    # Align periods with our period_labels if needed
    if "period" not in forced_trend.dims:
        raise ValueError(f"[rank {rank}] Forced trend has no 'period' dimension.")

    # Reorder / subset to match our period_labels
    forced_trend = forced_trend.sel(period=period_labels)

    ds_mem.close()
    # Keep ds_forced open only for forced_trend
    return tas_mem, forced_trend, lat, lon
# %%
# ---------------------------------------------------------------------
# Load observed forced trend patterns
# Observed forced trend file available from: FigS5_S6
# /work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/trend_forced_HadCRUT5_annual/forced_HadCRUT5_MMLE_MK_trend_1950-2022_sliding.nc
# ---------------------------------------------------------------------
def load_observed_forced(obs_path: str, forced_periods: xr.DataArray):
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"[rank {rank}] Observed forced file not found: {obs_path}")

    ds_obs = xr.open_dataset(obs_path)
    if "trend" not in ds_obs.data_vars:
        raise KeyError(f"[rank {rank}] 'trend' variable not found in {obs_path}")

    obs_trend = ds_obs["trend"]

    if "period" not in obs_trend.dims:
        raise ValueError(f"[rank {rank}] Observed trend has no 'period' dimension.")

    # Align periods with the SMILE forced periods
    obs_trend = obs_trend.sel(period=forced_periods)
    return obs_trend
# %%
# ---------------------------------------------------------------------
# 1) Pattern correlation: each run vs SMILE MMEM forced trend
# ---------------------------------------------------------------------
def compute_run_vs_mmem_correlations(tas_mem: xr.DataArray,
                                     forced_trend: xr.DataArray,
                                     lat: xr.DataArray) -> np.ndarray:
    """
    Compute pattern correlation between each run's MK trend map and the SMILE
    ensemble-mean forced trend map for each sliding period.

    tas_mem: (run, year, lat, lon) annual anomalies
    forced_trend: (period, lat, lon) MK trend (K/decade)
    lat: 1D latitude coordinate

    Returns (on rank 0): corr_global[run, period]
    """
    runs = tas_mem["run"].values
    n_runs_total = runs.size

    # Each rank handles a subset of run indices
    local_indices = list(range(rank, n_runs_total, npro))

    # Local storage (full size, but filled only for local_indices)
    corr_local = np.full((n_runs_total, n_period), np.nan, dtype=np.float64)

    for ir in local_indices:
        r_val = runs[ir]
        print(f"[rank {rank}] Processing run index {ir} (run={r_val})", flush=True)

        tas_run = tas_mem.sel(run=r_val)  # (year, lat, lon)

        for ip in range(n_period):
            LE_trend_map = tas_run.sel(period=period_labels[ip])
            # Forced MMEM trend for this period
            forced_map = forced_trend.sel(period=period_labels[ip])

            # Pattern correlation
            corr_val = pattern_corr_da(LE_trend_map, forced_map, lat)
            corr_local[ir, ip] = corr_val

    # Gather all local pieces to rank 0
    corr_list = comm.gather(corr_local, root=0)

    if rank == 0:
        # Combine by taking nanmax across ranks (only one non-NaN per entry)
        corr_stack = np.stack(corr_list, axis=0)  # (npro, n_runs, n_period)
        corr_global = np.nanmax(corr_stack, axis=0)  # (n_runs, n_period)
        return corr_global
    else:
        return None
# %%
# ---------------------------------------------------------------------
# 2) Pattern correlation: observed forced vs SMILE MMEM forced
# ---------------------------------------------------------------------
def compute_obs_vs_mmem_correlations(obs_trend: xr.DataArray,
                                     forced_trend: xr.DataArray,
                                     lat: xr.DataArray) -> np.ndarray:
    """
    Compute pattern correlation between observed forced trend and SMILE
    ensemble-mean forced trend, for each period.

    obs_trend: (period, lat, lon)
    forced_trend: (period, lat, lon)
    lat: 1D latitude coordinate

    Returns: corr_obs[period]
    """
    corr_obs = np.full((n_period,), np.nan, dtype=np.float64)

    for ip, period_label in enumerate(period_labels):
        obs_map = obs_trend.sel(period=period_label)
        forced_map = forced_trend.sel(period=period_label)
        corr_obs[ip] = pattern_corr_da(obs_map, forced_map, lat)

    return corr_obs
# %%
# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
# Load SMILE members & MMEM forced trends
tas_mem, forced_trend, lat, lon = load_members_and_forced(model)

if rank == 0:
    print(
        f"[rank 0] tas_mem dims: {tas_mem.dims}, "
        f"forced_trend dims: {forced_trend.dims}",
        flush=True,
    )

# Compute pattern correlation: run vs MMEM
corr_global = compute_run_vs_mmem_correlations(tas_mem, forced_trend, lat)

# Only rank 0 proceeds with obs
if rank == 0:
    # Load observed forced trends
    obs_trend = load_observed_forced(obs_forced_path, forced_trend["period"])

    # Compute obs vs MMEM correlations
    corr_obs = compute_obs_vs_mmem_correlations(obs_trend, forced_trend, lat)

    # Build output dataset
    runs = tas_mem["run"].values
    corr_run_da = xr.DataArray(
        corr_global,
        dims=("run", "period"),
        coords={"run": runs, "period": period_labels},
        name="pattern_corr_run_vs_mmem",
    )
    corr_run_da.attrs["description"] = (
        "Area-weighted pattern correlation between each run's MK trend "
        "and the SMILE ensemble-mean forced MK trend, for sliding windows "
        "1950–2022...2013–2022 (10–73 years)."
    )

    corr_obs_da = xr.DataArray(
        corr_obs,
        dims=("period",),
        coords={"period": period_labels},
        name="pattern_corr_obs_vs_mmem",
    )
    corr_obs_da.attrs["description"] = (
        "Area-weighted pattern correlation between observed forced MK trend "
        "and the SMILE ensemble-mean forced MK trend, for sliding windows "
        "1950–2022...2013–2022 (10–73 years)."
    )

    ds_out = xr.Dataset(
        {
            "pattern_corr_run_vs_mmem": corr_run_da,
            "pattern_corr_obs_vs_mmem": corr_obs_da,
        }
    )

    out_file = os.path.join(
        dir_out, f"{model}_forced_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc"
    )
    print(f"[rank 0] Writing output: {out_file}", flush=True)
    ds_out.to_netcdf(out_file)
    
    print(
        f"[rank 0] Done. Output dims: "
        f"run={ds_out.dims['run']}, "
        f"period={ds_out.dims['period']}",
        flush=True,
    )
# %%
comm.Barrier()
if rank != 0:
    print(f"[rank {rank}] Finished.", flush=True)
# %%