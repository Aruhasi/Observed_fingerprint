# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:23:00 2025
"""
import numpy as np
import xarray as xr
import os
import sys
import src.SAT_function_Obs_Fingerprint as data_process

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()

print(f"[rank {rank}] MPI world size = {npro}", flush=True)

model = sys.argv[1]

print(" This node is running model: ", model)
print(f"This node is running {npro} processes")

start_year = 1950
end_year   = 2022
min_length = 10

dir_in  = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}"
dir_out = dir_in  # same dir for output

# Detect number of runs from the input file (dynamic, not hardcoded)
in_file = f"{dir_in}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"
ds_probe = xr.open_dataset(in_file)
if "run" in ds_probe.dims:
    num_runs = ds_probe.dims["run"]
elif "run" in ds_probe.coords:
    num_runs = len(ds_probe.coords["run"])
else:
    raise ValueError(f"{in_file} has no 'run' dimension or coordinate")
ds_probe.close()

print(f"[rank {rank}] Model {model} has {num_runs} runs", flush=True)

runs       = np.arange(1, num_runs + 1)  # Dynamic: 1..num_runs
run_single = np.array_split(runs, npro)[rank]

def func_mk(x):
    results = data_process.mk_test(x)
    slope, p_val = results[0], results[1]
    return slope, p_val
# %%
def process_realization(run):
    """
    Compute MK trends for a single realization.
    """
    in_file = f"{dir_in}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"
    print(f"[rank {rank}] Processing run {run} from {in_file}", flush=True)

    ds_in = xr.open_dataset(in_file, chunks={"year": 50, "lat": 45, "lon": 45})

    if "run" not in ds_in.dims and "run" not in ds_in.coords:
        raise ValueError(f"{in_file} has no 'run' dimension / coordinate")

    run_coord = ds_in["run"]

    # --- robust selection of the realization ---
    if np.issubdtype(run_coord.dtype, np.number):
        # numeric labels (e.g. 1..50)
        if run in run_coord.values:
            forced = ds_in["forced_signal"].sel(run=run)
        else:
            # fall back to positional index: run 1 -> index 0
            forced = ds_in["forced_signal"].isel(run=run - 1)
    else:
        # non-numeric labels (e.g. 'r1i1p1f1'); use index position
        forced = ds_in["forced_signal"].isel(run=run - 1)

    # Ensure time dimension is 'year'
    if "year" not in forced.dims:
        if "time" in forced.dims:
            forced = forced.rename({"time": "year"})
        else:
            raise ValueError(f"{in_file} has no 'year' or 'time' dimension.")

    forced = forced.sel(year=slice(start_year, end_year))
    lat = forced["lat"]
    lon = forced["lon"]

    trend_list, pvalue_list, period_names = [], [], []

    for begin_year in range(start_year, end_year - min_length + 2):
        time_slice = forced.sel(year=slice(begin_year, end_year))

        trend, p_values = xr.apply_ufunc(
            func_mk,
            time_slice,
            input_core_dims=[["year"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        period_key = f"{begin_year}-{end_year}"
        period_names.append(period_key)
        trend_list.append(trend)
        pvalue_list.append(p_values)

    trend_all = xr.concat([t * 10.0 for t in trend_list], dim="period")
    pval_all  = xr.concat(pvalue_list, dim="period")

    trend_all = trend_all.assign_coords(
        period=("period", period_names),
        lat=lat,
        lon=lon,
    )
    pval_all = pval_all.assign_coords(
        period=("period", period_names),
        lat=lat,
        lon=lon,
    )

    trend_all.name = "trend"
    pval_all.name  = "p_value"

    ds_out = xr.Dataset({"trend": trend_all, "p_value": pval_all})
    ds_in.close()
    return ds_out
# %%
# ----------------- per-rank work -----------------
ds_list = []
for run in run_single:
    print(f"[rank {rank}] Processing run {run}", flush=True)
    ds_run = process_realization(run).expand_dims(run=[run])
    ds_list.append(ds_run)

if ds_list:
    ds_rank = xr.concat(ds_list, dim="run")
    print(f"[rank {rank}] Processed runs: {list(run_single)}", flush=True)

    os.makedirs(dir_out, exist_ok=True)
    temp_file = f"{dir_out}/forced_{model}_MK_trend_{start_year}-{end_year}_sliding_rank{rank}.nc"
    print(f"[rank {rank}] Writing {temp_file}", flush=True)
    ds_rank.to_netcdf(temp_file)
else:
    print(f"[rank {rank}] No runs to process", flush=True)

comm.Barrier()
# %%
# ----------------- merge on rank 0 -----------------
if rank == 0:
    print("Merging all rank files into single output...", flush=True)
    temp_files = [
        f"{dir_out}/forced_{model}_MK_trend_{start_year}-{end_year}_sliding_rank{r}.nc"
        for r in range(npro)
        if os.path.exists(f"{dir_out}/forced_{model}_MK_trend_{start_year}-{end_year}_sliding_rank{r}.nc")
    ]

    if not temp_files:
        raise RuntimeError("No temporary rank files found to merge.")

    out_file = f"{dir_out}/forced_{model}_MK_trend_{start_year}-{end_year}_sliding.nc"
    
    # Merge incrementally to avoid loading all rank files at once
    ds_all = None
    for i, tf in enumerate(temp_files):
        print(f"Merging rank file {i+1}/{len(temp_files)}: {tf}", flush=True)
        ds_rank = xr.open_dataset(tf)
        if ds_all is None:
            ds_all = ds_rank
        else:
            ds_all = xr.concat([ds_all, ds_rank], dim="run")
        ds_rank.close()
    
    print(f"Writing final output: {out_file}", flush=True)
    ds_all.to_netcdf(out_file)

    # keep dimensions info before closing
    n_run    = ds_all.dims["run"]
    n_period = ds_all.dims["period"]
    n_lat    = ds_all.dims["lat"]
    n_lon    = ds_all.dims["lon"]
    ds_all.close()

    for tf in temp_files:
        os.remove(tf)
        print(f"Removed {tf}", flush=True)

    print(f"All ranks finished. Final file: {out_file}", flush=True)
    print(f"Output dimensions: run={n_run}, period={n_period}, lat={n_lat}, lon={n_lon}", flush=True)
# %%