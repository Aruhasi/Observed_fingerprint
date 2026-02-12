#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple (unweighted) Pearson pattern correlations between SMILE realizations /
observations and SMILE ensemble-mean emergence timescale patterns

For a given model:

1) For each run:
     - Compute Pearson pattern correlation between each individual run and the OBS emergence timescale.

2) For each LE ensemble-mean:
     - Compute Pearson pattern correlation between observed emergence timescale
         and the SMILE ensemble-mean emergence timescale.

Outputs (one NetCDF per model):

- pattern_corr_run_vs_OBS(run, OBS)
- pattern_corr_ENS_vs_OBS(ENS, OBS)

Usage:
        mpirun -np N python -u Emergence_timescale_Pcorr_SMILE_vs_OBS.py <MODEL_NAME> <OBS_EMERGENCE_TIMESCALE_FILE>

Example:
        mpirun -np 8 python -u Emergence_timescale_Pcorr_SMILE_vs_OBS.py CESM2 \
                /work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/OBS_Emergence_time_scale.nc
"""
# %%
import os
import sys
import numpy as np
import xarray as xr
from mpi4py import MPI
from scipy import stats
from typing import List

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
            "Usage: python Emergence_timescale_Pcorr_SMILE_vs_OBS.py <MODEL_NAME> <OBS_EMERGENCE_TIMESCALE_FILE>",
            flush=True,
        )
    sys.exit(1)

model = sys.argv[1]
obs_emergence_timescale_path = sys.argv[2]

print(f"[rank {rank}] Model: {model}", flush=True)
if rank == 0:
    print(f"[rank 0] Observed emergence timescale file: {obs_emergence_timescale_path}", flush=True)
# %%
# ---------------------------------------------------------------------
# Paths (edit if needed)
# ---------------------------------------------------------------------
dir_members = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/mean"      # OBS-LPS framework obtained forced trend path
)

dir_MMLE = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/"
)

dir_out = (
    f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/Pattern_correlation_pearson/"
)

if rank == 0:
    os.makedirs(dir_out, exist_ok=True)
comm.Barrier()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pattern_corr_da(x: xr.DataArray, y: xr.DataArray) -> float:
    """Pearson correlation between two 2D patterns with on-the-fly alignment.

    - Aligns coordinates (lat/lon/period) on the intersection so grid sizes match.
    - Drops any grid point where either field is NaN after alignment.
    """
    x_aligned, y_aligned = xr.align(x, y, join="inner")

    valid = np.isfinite(x_aligned) & np.isfinite(y_aligned)
    if valid.sum().item() < 2:
        return np.nan

    x_flat = x_aligned.where(valid).values.ravel()
    y_flat = y_aligned.where(valid).values.ravel()

    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr_val = corr_matrix[0, 1]
    return float(corr_val) if np.isfinite(corr_val) else np.nan


def get_missing_run_ids(model_name: str) -> List[int]:
    """Return run ids to drop for known models."""
    kick_map = {
        "MIROC6": [19],
        "MPI_ESM": [2, 18, 19, 21],
        "CESM2": [2],
        "EC_Earth3": [19, 20],
        "IPSL_CM6A": [17, 22, 30],
    }
    return kick_map.get(model_name, [])


def drop_missing_runs(tas_mem: xr.DataArray, model_name: str) -> xr.DataArray:
    """Remove runs flagged as missing, if they exist in the run coordinate."""
    to_drop = get_missing_run_ids(model_name)
    if not to_drop or "run" not in tas_mem.coords:
        return tas_mem
    existing = [r for r in to_drop if r in tas_mem["run"].values]
    if not existing:
        return tas_mem
    return tas_mem.drop_sel(run=existing)


def resolve_period_labels(*arrays: xr.DataArray) -> list:
    """Return the common period labels across provided arrays (or [None])."""
    period_axes = []
    for arr in arrays:
        if "period" in arr.coords:
            period_axes.append(arr["period"].values)

    if not period_axes:
        return [None]

    common = period_axes[0]
    for axis in period_axes[1:]:
        common = np.intersect1d(common, axis)

    return list(common)


def make_period_coord(period_labels: list) -> np.ndarray:
    """Return a coordinate array for xarray outputs."""
    if len(period_labels) == 1 and period_labels[0] is None:
        return np.array(["full"])
    return np.array(period_labels)
# ---------------------------------------------------------------------
# Load SMILE members and SMILE ensemble-mean forced trend
# ---------------------------------------------------------------------
def load_members_and_forced(model_name: str):
    # Members
    mem_file = os.path.join(dir_members, f"{model_name}_emergence_timescale.nc")
    if not os.path.exists(mem_file):
        raise FileNotFoundError(f"[rank {rank}] Members file not found: {mem_file}")

    ds_mem = xr.open_dataset(mem_file)
    tas_mem = ds_mem["emergence_timescale"]  # (run, lat, lon)
    
    # SMILE ensemble-mean emergence timescale
    forced_file = os.path.join(
        dir_MMLE, f"MMEM_emergence_timescale_mean.nc"
    )
    if not os.path.exists(forced_file):
        raise FileNotFoundError(f"[rank {rank}] Forced trend file not found: {forced_file}")

    ds_forced = xr.open_dataset(forced_file)
    if "emergence_timescale_median" not in ds_forced.data_vars:
        raise KeyError(f"[rank {rank}] 'emergence_timescale' variable not found in {forced_file}")

    forced_trend = ds_forced["emergence_timescale_median"]  # (run, lat, lon)

    ds_mem.close()
    # Keep ds_forced open only for forced_trend
    return tas_mem, forced_trend
# %%
# ---------------------------------------------------------------------
# Load observed emergence timescale patterns
# ---------------------------------------------------------------------
def load_observed_emergence(obs_path: str) -> xr.DataArray:
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"[rank {rank}] Observed emergence file not found: {obs_path}")

    ds_obs = xr.open_dataset(obs_path).rename({"__xarray_dataarray_variable__": "emergence_timescale"})
    if "emergence_timescale" not in ds_obs.data_vars:
        raise KeyError(f"[rank {rank}] 'emergence_timescale' variable not found in {obs_path}")

    return ds_obs["emergence_timescale"]

# ---------------------------------------------------------------------
# 1) Pattern correlation: each run vs OBS emergence timescale
#    Grid alignment handled inside pattern_corr_da
# ---------------------------------------------------------------------
def compute_run_vs_obs_correlations(
    tas_mem: xr.DataArray, obs_trend: xr.DataArray, period_labels: list
) -> np.ndarray:
    runs = tas_mem["run"].values
    n_runs_total = runs.size
    n_period = len(period_labels)

    local_indices = list(range(rank, n_runs_total, npro))
    corr_local = np.full((n_runs_total, n_period), np.nan, dtype=np.float64)

    for ir in local_indices:
        r_val = runs[ir]
        print(f"[rank {rank}] Processing run index {ir} (run={r_val})", flush=True)

        tas_run = tas_mem.sel(run=r_val)

        for ip, period_label in enumerate(period_labels):
            run_map = tas_run if period_label is None else tas_run.sel(period=period_label)
            obs_map = obs_trend if period_label is None else obs_trend.sel(period=period_label)
            corr_local[ir, ip] = pattern_corr_da(run_map, obs_map)

    corr_list = comm.gather(corr_local, root=0)

    if rank == 0:
        corr_stack = np.stack(corr_list, axis=0)
        corr_global = np.nanmax(corr_stack, axis=0)
        return corr_global
    return None

# ---------------------------------------------------------------------
# 2) Pattern correlation: observed forced vs SMILE MMEM forced
# ---------------------------------------------------------------------
def compute_obs_vs_mmem_correlations(
    obs_trend: xr.DataArray, forced_trend: xr.DataArray, period_labels: list
) -> np.ndarray:
    n_period = len(period_labels)
    corr_obs = np.full((n_period,), np.nan, dtype=np.float64)

    for ip, period_label in enumerate(period_labels):
        obs_map = obs_trend if period_label is None else obs_trend.sel(period=period_label)
        forced_map = forced_trend if period_label is None else forced_trend.sel(period=period_label)
        corr_obs[ip] = pattern_corr_da(obs_map, forced_map)

    return corr_obs
# %%
# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
# Load SMILE members & MMEM forced trends
tas_mem_raw, forced_trend = load_members_and_forced(model)
tas_mem = drop_missing_runs(tas_mem_raw, model)

if rank == 0:
    dropped = set(tas_mem_raw["run"].values) - set(tas_mem["run"].values)
    drop_msg = f"; dropped runs={sorted(dropped)}" if dropped else ""
    print(
        f"[rank 0] tas_mem dims: {tas_mem.dims}{drop_msg}, "
        f"forced_trend dims: {forced_trend.dims}",
        flush=True,
    )

# Load observed emergence timescale
obs_trend = load_observed_emergence(obs_emergence_timescale_path)

# Resolve period coordinates consistently
period_labels_run = resolve_period_labels(tas_mem, obs_trend)
period_labels_obs_forced = resolve_period_labels(obs_trend, forced_trend)

# Compute pattern correlation: run vs OBS
corr_global = compute_run_vs_obs_correlations(tas_mem, obs_trend, period_labels_run)

# Only rank 0 proceeds with writing
if rank == 0:
    # Compute obs vs MMEM correlations on rank 0 (lightweight)
    corr_obs = compute_obs_vs_mmem_correlations(obs_trend, forced_trend, period_labels_obs_forced)

    runs = tas_mem["run"].values
    period_coord_run = make_period_coord(period_labels_run)
    period_coord_obs = make_period_coord(period_labels_obs_forced)

    corr_run_da = xr.DataArray(
        corr_global,
        dims=("run", "period"),
        coords={"run": runs, "period": period_coord_run},
        name="pattern_corr_run_vs_obs",
    )
    corr_run_da.attrs["description"] = (
        "Pearson pattern correlation between each SMILE run and the observed "
        "emergence timescale pattern (grids aligned, missing values removed)."
    )

    corr_obs_da = xr.DataArray(
        corr_obs,
        dims=("period",),
        coords={"period": period_coord_obs},
        name="pattern_corr_obs_vs_mmem",
    )
    corr_obs_da.attrs["description"] = (
        "Pearson pattern correlation between the observed emergence timescale "
        "pattern and the SMILE ensemble-mean emergence timescale pattern."
    )

    ds_out = xr.Dataset(
        {
            "pattern_corr_run_vs_obs": corr_run_da,
            "pattern_corr_obs_vs_mmem": corr_obs_da,
        }
    )

    out_file = os.path.join(
        dir_out, f"{model}_pattern_correlation_run_and_obs_vs_MMEM.nc"
    )
    print(f"[rank 0] Writing output: {out_file}", flush=True)
    ds_out.to_netcdf(out_file)

    print(
        f"[rank 0] Done. Output dims: "
        f"run={corr_run_da.sizes['run']}, "
        f"period (run vs obs)={corr_run_da.sizes['period']}, "
        f"period (obs vs mmem)={corr_obs_da.sizes['period']}",
        flush=True,
    )
# %%
comm.Barrier()
if rank != 0:
    print(f"[rank {rank}] Finished.", flush=True)