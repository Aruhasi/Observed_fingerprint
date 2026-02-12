#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute sliding-window Mann–Kendall trends for the *ensemble-mean*
forced SAT anomalies for a single LE model, and save to one NetCDF file.

Usage (example):
    mpirun -n 4 python compute_MK_ensmean_sliding.py CESM2
"""

import numpy as np
import xarray as xr
import os
import sys
import src.SAT_function_Obs_Fingerprint as data_process

from mpi4py import MPI

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()

print(f"[rank {rank}] MPI world size = {npro}", flush=True)

# ---------------------------------------------------------------------
# Model name from command line
# ---------------------------------------------------------------------
if len(sys.argv) < 2:
    if rank == 0:
        raise SystemExit("Usage: python script.py <MODEL_NAME>")
    else:
        sys.exit(0)

model = sys.argv[1]
print(f"[rank {rank}] This node is running model: {model}", flush=True)

# ---------------------------------------------------------------------
# Parameters & paths
# ---------------------------------------------------------------------
start_year = 1950
end_year   = 2013
min_length = 10

dir_in  = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data/ENS/"
dir_out = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_forced_2013"

os.makedirs(dir_out, exist_ok=True)

def func_mk(x):
    """
    Wrapper for the Mann–Kendall test.
    Expects a 1D array over time and returns slope, p-value.
    """
    results = data_process.mk_test(x)
    slope, p_val = results[0], results[1]
    return slope, p_val

# ---------------------------------------------------------------------
# Core processing: ensemble-mean sliding trends for one model
# ---------------------------------------------------------------------
def process_ensmean_forced(model):
    """
    Compute MK trends for the ensemble-mean forced SAT anomalies
    for a given model. Returns an xarray.Dataset with dimensions:

        period, lat, lon

    and variables:
        trend   (K / decade)
        p_value (MK p-value)
    """
    in_file = f"{dir_in}/{model}_annual_ano_ensemble_mean_1950_2022_smbbCESM2_complement.nc"
    print(f"[rank {rank}] Processing ensemble mean from {in_file}", flush=True)

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file not found: {in_file}")

    ds_in = xr.open_dataset(in_file)
    if "tas" not in ds_in.data_vars:
        # rename the var to "tas" if needed
        var_name = list(ds_in.data_vars)[0]
        ds_in = ds_in.rename({var_name: "tas"})
    # --- get ensemble-mean SAT anomalies ---
    forced = ds_in["tas"]

    # Ensure time dimension is named 'year'
    if "year" not in forced.dims:
        if "time" in forced.dims:
            forced = forced.rename({"time": "year"})
        else:
            raise ValueError(f"{in_file} has no 'year' or 'time' dimension.")

    # Subset to desired period
    forced = forced.sel(year=slice(start_year, end_year))

    lat = forced["lat"]
    lon = forced["lon"]

    trend_list = []
    pvalue_list = []
    period_names = []

    # Sliding windows: [begin_year, end_year] with length >= min_length
    # Example for 1950–2022, min_length=10 -> 1950–2022 ... 2013–2022
    for begin_year in range(start_year, end_year - min_length + 2):
        time_slice = forced.sel(year=slice(begin_year, end_year))

        # Apply MK along 'year', vectorized over lat, lon
        trend, p_values = xr.apply_ufunc(
            data_process.apply_mannkendall,
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

    # Concatenate over the new "period" dimension
    trend_all = xr.concat([t * 10.0 for t in trend_list], dim="period")  # K/yr -> K/decade
    pval_all  = xr.concat(pvalue_list, dim="period")

    # Assign coordinates
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

# ---------------------------------------------------------------------
# Run only on rank 0 and write one file
# ---------------------------------------------------------------------
if rank == 0:
    ds_out = process_ensmean_forced(model)

    out_file = (
        f"{dir_out}/{model}_ENSmean_forced_MK_trend_"
        f"{start_year}-{end_year}_sliding.nc"
    )
    print(f"[rank 0] Writing final output: {out_file}", flush=True)
    ds_out.to_netcdf(out_file)

    # print dimension info
    print(
        f"[rank 0] Output dimensions: "
        f"period={ds_out.dims['period']}, "
        f"lat={ds_out.dims['lat']}, "
        f"lon={ds_out.dims['lon']}",
        flush=True,
    )

# Make sure all ranks finish cleanly
comm.Barrier()
if rank != 0:
    print(f"[rank {rank}] Finished (no I/O).", flush=True)
