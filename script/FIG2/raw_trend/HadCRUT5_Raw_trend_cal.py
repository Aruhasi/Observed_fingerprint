#!/usr/bin/env python3
# %%
"""
Compute sliding-window Mann-Kendall trends for forced HadCRUT5 SAT
for each model's forced signal (OBS-LPS partition), using MPI.

Each MPI rank processes a subset of models in MODEL_LIST and writes
one NetCDF file per model with trend and p-value fields for all
windows from start_year..end_year (min_length threshold).
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
# %%
# define function
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess
import src.plot_func as plot_func
# %%
def func_mk(x):
    """
    Mann-Kendall test for trend
    """
    results = data_process.mk_test(x)
    slope = results[0]
    p_val = results[1]
    return slope, p_val
# %%
dir_in = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG1/"
dir_out = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG2/"
os.makedirs(dir_out, exist_ok=True)
start_year = 1950
end_year = 2022
min_length = 10
# %%
# reversed trend calculation based on Mann-Kendall test from 1950 to 2022
# ---------------------------------------------------------------------
# Per-model computation
# ---------------------------------------------------------------------
def main():
    """
    For a given model:
      - Read OBS_SAT_anomaly_partition_wrt_{model}_ENS.nc
      - Take 'forced_signal'
      - Compute MK trend & p-value for all windows [begin_year..end_year]
      - Save a single NetCDF file with dims: period, lat, lon
    """
    in_file = f"{dir_in}/tas_HadCRUT5_annual_anomalies_processed.nc"

    ds_in = xr.open_dataset(in_file).rename({'__xarray_dataarray_variable__': 'tas'})  # forced_signal, internal_variability, dims: year, lat, lon
    Raw = ds_in["tas"]

    # Ensure time dimension is called 'year' (rename if necessary)
    if "year" not in Raw.dims:
        if "time" in Raw.dims:
            Raw = Raw.rename({"time": "year"})
        else:
            raise ValueError(f"{in_file} has no 'year' or 'time' dimension.")

    # Restrict to the analysis period
    Raw = Raw.sel(year=slice(start_year, end_year))

    lat = Raw["lat"]
    lon = Raw["lon"]

    trend_list   = []
    pvalue_list  = []
    period_names = []

    # Sliding windows: begin_year from start_year .. (end_year - min_length + 1)
    for begin_year in range(start_year, end_year - min_length + 2):
        time_slice = Raw.sel(year=slice(begin_year, end_year))

        # Apply MK test along 'year' dimension at each (lat, lon)
        trend, p_values = xr.apply_ufunc(
            func_mk,
            time_slice,
            input_core_dims=[["year"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="forbidden",          # no dask here; each rank is small enough
            output_dtypes=[float, float],
        )

        period_key = f"{begin_year}-{end_year}"
        period_names.append(period_key)
        trend_list.append(trend)
        pvalue_list.append(p_values)

    # Stack into a single DataArray with 'period' dimension
    trend_all = xr.concat([t*10.0 for t in trend_list], dim="period")
    pval_all  = xr.concat(pvalue_list,  dim="period")

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

    ds_out = xr.Dataset(
        {
            "trend": trend_all,
            "p_value": pval_all,
        }
    )

    out_file = f"{dir_out}/Raw_HadCRUT5_MK_trend_{start_year}-{end_year}_sliding.nc"
    print(f"Writing {out_file}", flush=True)
    ds_out.to_netcdf(out_file)
    ds_in.close()
# %%
if __name__ == "__main__":
    main()
# %%