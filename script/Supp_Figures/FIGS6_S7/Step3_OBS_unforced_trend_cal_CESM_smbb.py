#!/usr/bin/env python3
"""
Compute sliding-window Mann-Kendall trends for ICV HadCRUT5 SAT
for each model's ICV signal (OBS-LPS partition), using MPI.
"""
# %%
import xarray as xr
import os
from mpi4py import MPI
import numpy as np

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess
import src.plot_func as plot_func

dir_in = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6"
dir_out = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/trend_ICV_HadCRUT5_annual"
os.makedirs(dir_out, exist_ok=True)
start_year = 1950
end_year = 2022
min_length = 10

MODEL_LIST = ["CESM2-smbb"]

def func_mk(x):
    """
    Safe Mann-Kendall test for trend.
    Drops NaNs and skips too-short / constant series to avoid
    var_s <= 0 issues inside pymannkendall.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]

    # Not enough data -> undefined trend
    if arr.size < min_length:
        return np.nan, np.nan

    # Constant series -> MK variance zero
    if np.allclose(arr, arr[0]):
        return 0.0, 1.0

    slope, p_val = data_process.mk_test(arr)

    # Extra safety
    if not np.isfinite(slope) or not np.isfinite(p_val):
        return np.nan, np.nan

    return slope, p_val
# %%
def process_model(model):
    """
    For a given model:
      - Read OBS_SAT_anomaly_partition_wrt_{model}_ENS.nc
      - Take 'internal_variability'
      - Compute MK trend & p-value for all windows [begin_year..end_year]
      - Save a single NetCDF file with dims: period, lat, lon
    """
    in_file = f"{dir_in}/OBS_SAT_anomaly_partition_wrt_{model}_ENS.nc"
    print(f"[rank {rank}] Opening {in_file}", flush=True)

    ds_in = xr.open_dataset(in_file)  # <<< no need for mfdataset here
    ICV = ds_in["internal_variability"]

    # Ensure time dimension is called 'year'
    if "year" not in ICV.dims:
        if "time" in ICV.dims:
            ICV = ICV.rename({"time": "year"})
        else:
            raise ValueError(f"{in_file} has no 'year' or 'time' dimension.")

    # Restrict to analysis period
    ICV = ICV.sel(year=slice(start_year, end_year))

    lat = ICV["lat"]
    lon = ICV["lon"]

    trend_list   = []
    pvalue_list  = []
    period_names = []

    for begin_year in range(start_year, end_year - min_length + 2):
        time_slice = ICV.sel(year=slice(begin_year, end_year))

        trend, p_values = xr.apply_ufunc(
            func_mk,
            time_slice,
            input_core_dims=[["year"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="forbidden",              # <<< you can also use "parallelized", but forbidden is safer here
            output_dtypes=[float, float],
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

    out_file = f"{dir_out}/ICV_HadCRUT5_{model}_MK_trend_{start_year}-{end_year}_sliding.nc"
    print(f"[rank {rank}] Writing {out_file}", flush=True)
    ds_out.to_netcdf(out_file)
    ds_in.close()
    print(f"[rank {rank}] Done model {model}", flush=True)

# %%
for i, model in enumerate(MODEL_LIST):
    if i % size == rank:
        process_model(model)

comm.Barrier()
if rank == 0:
    print("All ranks finished MK trend calculation.", flush=True)
# %%