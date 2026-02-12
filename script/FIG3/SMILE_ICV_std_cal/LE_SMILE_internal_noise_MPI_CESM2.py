#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SMILE internal-variability (noise) trend stddev patterns
for a single model, using MPI over the run dimension.

- Internal variability = member anomalies minus ENS-mean forced signal.
- Sliding windows: 2013–2022, 2012–2022, ..., 1950–2022
  (window lengths 10–73 years).
- For each window:
    * compute MK trend per member (run, lat, lon)
    * convert to K/decade
    * compute std over run -> noise pattern (K/decade)

Usage:
    mpirun -np N python -u LE_SMILE_internal_noise_sliding_MPI.py ACCESS
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

# ---------------------------------------------------------------------
# Get model name from command line
# ---------------------------------------------------------------------
if len(sys.argv) < 2:
    if rank == 0:
        print("Usage: python LE_SMILE_internal_noise_sliding_MPI.py <MODEL_NAME>", flush=True)
    sys.exit(1)

model = sys.argv[1]
print(f"[rank {rank}] Running SMILE internal variability for model: {model}", flush=True)

# ---------------------------------------------------------------------
# Paths & parameters  >>>> EDIT PATHS IF NEEDED <<<<
# ---------------------------------------------------------------------
# Members (full LE):
#   typical file: tas_<MODEL>_annual_ano_1850_2022.nc  with dims (run, year, lat, lon)
dir_members = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data"

# ENS mean (forced signal):
#   file: <MODEL>_annual_ano_ensemble_mean_1950_2022.nc with dims (year, lat, lon)
dir_ens = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data/ENS"

# Output for noise patterns
dir_out = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}/SMILE_internal"
if rank == 0:
    os.makedirs(dir_out, exist_ok=True)
comm.Barrier()

start_year = 1950
end_year   = 2022
min_length = 10  # 10–73 years

# sliding windows: 2013–2022, 2012–2022, ... 1950–2022
period_starts = list(range(end_year - min_length + 1, start_year - 1, -1))
# This is [2013, 2012, ..., 1950]
period_labels = [f"{by}-{end_year}" for by in period_starts]
n_period = len(period_starts)

# ---------------------------------------------------------------------
# Helper: apply MK along "year"
# ---------------------------------------------------------------------
def mk_func(x):
    """
    Wrapper for Mann–Kendall test from your SAT_function_Obs_Fingerprint.

    Input: 1D array over 'year'
    Output: slope, p-value
    """
    slope, p_val = data_process.apply_mannkendall(x)
    return slope, p_val

# ---------------------------------------------------------------------
# CESM2 variable-renaming helper
# ---------------------------------------------------------------------
def get_tas_var(ds, model_name, context="members"):
    """
    Return a DataArray for surface air temperature, named 'tas'.

    - If 'tas' exists, use it.
    - For CESM2, try common names and rename to 'tas'.
    """
    if "tas" in ds.data_vars:
        return ds["tas"]

    if model_name == "CESM2":
        # try common CESM2 SAT variables
        for cand in ("TREFHT", "TS", "TEMP", "TAS"):
            if cand in ds.data_vars:
                da = ds[cand]
                da = da.rename("tas")
                # rename CESM2 "member" dimension to "run"
                if "member" in da.dims:
                    da = da.rename({"member": "run"})
                print(f"[rank {rank}] CESM2 {context}: using variable '{cand}' as 'tas'", flush=True)
                return da

    raise KeyError(
        f"[rank {rank}] Could not find 'tas' in {context} dataset for model '{model_name}'. "
        "Please check variable names."
    )
# %%
# ---------------------------------------------------------------------
# 1. Load data and build residual (internal variability)
# ---------------------------------------------------------------------
def load_data_for_model(model_name):
    # members
    mem_file = os.path.join(dir_members, f"tas_{model_name}_CMIP6_SMBB_annual_ano_1850_2022.nc")
    if not os.path.exists(mem_file):
        raise FileNotFoundError(f"[rank {rank}] Members file not found: {mem_file}")

    ds_mem = xr.open_mfdataset(mem_file, combine="by_coords")
    tas_mem = get_tas_var(ds_mem, model_name, context="members")

    # ensure 'year' and subset
    if "year" not in tas_mem.dims:
        if "time" in tas_mem.dims:
            tas_mem = tas_mem.rename({"time": "year"})
        else:
            raise ValueError("[rank {rank}] tas_mem has no 'year' or 'time' dimension.")
    tas_mem = tas_mem.sel(year=slice(str(start_year), str(end_year)))

    # ENS mean
    ens_file = os.path.join(dir_ens, f"{model_name}_annual_ano_ensemble_mean_1950_2022_cmip6+smbb.nc")
    if not os.path.exists(ens_file):
        raise FileNotFoundError(f"[rank {rank}] ENS-mean file not found: {ens_file}")

    ds_ens = xr.open_dataset(ens_file)
    tas_ens = get_tas_var(ds_ens, model_name, context="ENS mean")

    if "year" not in tas_ens.dims:
        if "time" in tas_ens.dims:
            tas_ens = tas_ens.rename({"time": "year"})
        else:
            raise ValueError("[rank {rank}] tas_ens has no 'year' or 'time' dimension.")
    tas_ens = tas_ens.sel(year=slice(str(start_year), str(end_year)))

    # broadcast ENS mean across run -> residual internal variability
    tas_resid = tas_mem - tas_ens

    lat = tas_resid["lat"]
    lon = tas_resid["lon"]

    ds_mem.close()
    ds_ens.close()

    return tas_resid, lat, lon
# %%
# ---------------------------------------------------------------------
# 2. Compute noise std patterns for all sliding windows using MPI over run
# ---------------------------------------------------------------------
# def compute_noise_std_sliding(tas_resid):
#     """
#     For each sliding window [begin_year, end_year=2022], compute:
#         std_over_run( MK_trend( residual ) )  in K/decade

#     tas_resid dims: (run, year, lat, lon)
#     Returns (rank 0): DataArray noise_all(period, lat, lon)
#     """
#     da = tas_resid  # (run, year, lat, lon)

#     all_runs = da["run"].values
#     n_runs_total = all_runs.size

#     # distribute runs across ranks (round-robin)
#     local_runs = all_runs[rank::npro]
#     n_local = local_runs.size

#     lats = da["lat"].values
#     lons = da["lon"].values
#     n_lat = lats.size
#     n_lon = lons.size

#     # partial sums over runs for *all* periods:
#     # shape: (period, lat, lon)
#     local_sum   = np.zeros((n_period, n_lat, n_lon), dtype=np.float64)
#     local_sumsq = np.zeros((n_period, n_lat, n_lon), dtype=np.float64)

#     if n_local > 0:
#         da_local = da.sel(run=local_runs)  # (local_run, year, lat, lon)

#         # loop over sliding windows
#         for ip, begin_year in enumerate(period_starts):
#             # subset in time
#             time_slice = da_local.sel(year=slice(str(begin_year), str(end_year)))
#             # dims: (run, year, lat, lon)

#             # MK along year, vectorized over (run, lat, lon)
#             slope_local, _ = xr.apply_ufunc(
#                 data_process.apply_mannkendall,
#                 time_slice,
#                 input_core_dims=[["year"]],
#                 output_core_dims=[[], []],
#                 vectorize=True,
#                 dask="parallelized",
#                 output_dtypes=[float, float],
#                 dask_gufunc_kwargs={"allow_rechunk": True},
#             )
#             # slope_local dims: (run, lat, lon)

#             # Convert to K/decade
#             slope_local = slope_local * 10.0

#             # accumulate partial sums over local runs
#             # sum over run dimension -> (lat, lon)
#             sum_run   = slope_local.sum(dim="run").values
#             sumsq_run = (slope_local ** 2).sum(dim="run").values

#             local_sum[ip, :, :]   += sum_run
#             local_sumsq[ip, :, :] += sumsq_run

#     # global reduction over ranks (sum and sumsq for each period, lat, lon)
#     global_sum   = np.empty_like(local_sum)
#     global_sumsq = np.empty_like(local_sumsq)

#     comm.Allreduce(local_sum,   global_sum,   op=MPI.SUM)
#     comm.Allreduce(local_sumsq, global_sumsq, op=MPI.SUM)

#     if rank == 0:
#         # unbiased variance across all runs
#         N = float(n_runs_total)
#         mean = global_sum / N
#         var = global_sumsq / N - mean ** 2
#         if N > 1:
#             var = var * N / (N - 1.0)
#         var = np.maximum(var, 0.0)
#         std = np.sqrt(var)

#         noise_all = xr.DataArray(
#             std,
#             dims=("period", "lat", "lon"),
#             coords={
#                 "period": ("period", period_labels),
#                 "lat": lats,
#                 "lon": lons,
#             },
#             name="noise_trend_std",
#         )
#         noise_all.attrs["units"] = "K/decade"
#         noise_all.attrs["description"] = (
#             "Std over run of MK trend of internal SAT anomalies "
#             "for sliding windows 2013–2022 ... 1950–2022 "
#             "(window length 10–73 years)."
#         )
#         return noise_all
#     else:
#         return None
def compute_noise_std_sliding(tas_resid):
    """Optimized version using NumPy vectorization instead of dask."""
    da = tas_resid  # (run, year, lat, lon)

    all_runs = da["run"].values
    n_runs_total = all_runs.size

    local_runs = all_runs[rank::npro]
    n_local = local_runs.size

    lats = da["lat"].values
    lons = da["lon"].values
    n_lat = lats.size
    n_lon = lons.size

    local_sum   = np.zeros((n_period, n_lat, n_lon), dtype=np.float64)
    local_sumsq = np.zeros((n_period, n_lat, n_lon), dtype=np.float64)

    if n_local > 0:
        da_local = da.sel(run=local_runs).values  # Convert to NumPy: (local_run, year, lat, lon)

        for ip, begin_year in enumerate(period_starts):
            # Find indices for time slice
            year_vals = da["year"].values
            year_mask = (year_vals >= begin_year) & (year_vals <= end_year)
            time_idx = np.where(year_mask)[0]
            
            data_slice = da_local[:, time_idx, :, :]  # (local_run, n_years, lat, lon)

            # Vectorized MK trend computation over (local_run, lat, lon)
            n_years = data_slice.shape[1]
            slope_all = np.zeros((n_local, n_lat, n_lon), dtype=np.float64)
            
            for ir in range(n_local):
                for ilat in range(n_lat):
                    for ilon in range(n_lon):
                        ts = data_slice[ir, :, ilat, ilon]
                        slope, _ = data_process.apply_mannkendall(ts)
                        slope_all[ir, ilat, ilon] = slope * 10.0  # K/decade

            local_sum[ip, :, :]   += slope_all.sum(axis=0)
            local_sumsq[ip, :, :] += (slope_all ** 2).sum(axis=0)

    global_sum   = np.empty_like(local_sum)
    global_sumsq = np.empty_like(local_sumsq)

    comm.Allreduce(local_sum,   global_sum,   op=MPI.SUM)
    comm.Allreduce(local_sumsq, global_sumsq, op=MPI.SUM)

    if rank == 0:
        N = float(n_runs_total)
        mean = global_sum / N
        var = global_sumsq / N - mean ** 2
        if N > 1:
            var = var * N / (N - 1.0)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        noise_all = xr.DataArray(
            std,
            dims=("period", "lat", "lon"),
            coords={
                "period": ("period", period_labels),
                "lat": lats,
                "lon": lons,
            },
            name="noise_trend_std",
        )
        noise_all.attrs["units"] = "K/decade"
        noise_all.attrs["description"] = (
            "Std over run of MK trend of internal SAT anomalies "
            "for sliding windows 2013–2022 ... 1950–2022 "
            "(window length 10–73 years)."
        )
        return noise_all
    else:
        return None
# %%
# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
tas_resid, lat, lon = load_data_for_model(model)

if rank == 0:
    print(
        f"[rank 0] Loaded residuals for model {model} with dims: "
        f"run={tas_resid.sizes['run']}, "
        f"year={tas_resid.sizes['year']}, "
        f"lat={tas_resid.sizes['lat']}, "
        f"lon={tas_resid.sizes['lon']}",
        flush=True,
    )

noise_all = compute_noise_std_sliding(tas_resid)

# rank 0: write output
if rank == 0:
    ds_out = xr.Dataset({"noise_trend_std": noise_all})
    out_file = os.path.join(
        dir_out, f"{model}_SMILE_noise_trend_std_sliding_1950_2022.nc"
    )
    print(f"[rank 0] Writing output: {out_file}", flush=True)
    ds_out.to_netcdf(out_file)

    print(
        f"[rank 0] Done. Output dims: "
        f"period={ds_out.sizes['period']}, "
        f"lat={ds_out.sizes['lat']}, "
        f"lon={ds_out.sizes['lon']}",
        flush=True,
    )
# %%
comm.Barrier()
if rank != 0:
    print(f"[rank {rank}] Finished.", flush=True)
# %%