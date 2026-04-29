#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute end-fixed (2022) MK trend PATTERNS (lat,lon) of LE residual anomalies
for ONE run (Slurm job-array friendly).

Residual = member(run) - ENSmean(forced)
Windows: 2013–2022, 2012–2022, ..., 1950–2022  (tau = 10..73)

Outputs one file per run:
  trend(period, lat, lon) in K/decade
  p_value(period, lat, lon)

Usage:
  python Segment_multi_run_globalpattern.py <RUN_ID> <MODEL>
Example:
  python Segment_multi_run_globalpattern.py 17 CESM2
"""
# %%
import os
import sys
import numpy as np
import xarray as xr
import src.SAT_function_Obs_Fingerprint as data_process

# -----------------------
# ARGS
# -----------------------
if len(sys.argv) < 3:
    raise SystemExit("Usage: python Segment_multi_run_globalpattern.py <RUN_ID> <MODEL>")

run_id = int(sys.argv[1])   # Slurm array 1..N
model  = sys.argv[2]

# -----------------------
# CONFIG
# -----------------------
# DIR_MEM = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data"
DIR_IN  = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}"
DIR_OUT_BASE = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_actual_residual_ICV"
CHUNKS = {"lat": 64, "lon": 64}

START_YEAR = 1950
END_YEAR   = 2022
MIN_LEN    = 10

# period starts: 2013..1950 (end fixed)
period_starts = list(range(END_YEAR - MIN_LEN + 1, START_YEAR - 1, -1))  # 2013..1950
period_labels = [f"{s}-{END_YEAR}" for s in period_starts]

DIR_OUT = os.path.join(DIR_OUT_BASE, model, "per_run_patterns")
os.makedirs(DIR_OUT, exist_ok=True)
# %%
# -----------------------
# HELPERS
# -----------------------
# def get_tas_var(ds, model_name):
#     if "tas" in ds.data_vars:
#         return ds["tas"]
#     if model_name == "CESM2":
#         for cand in ("TREFHT", "TS", "TEMP", "TAS"):
#             if cand in ds.data_vars:
#                 return ds[cand].rename("tas")
#     raise KeyError(f"Cannot find tas variable for model={model_name}")

def ensure_year_dim(da):
    if "year" in da.dims:
        return da
    if "time" in da.dims:
        return da.rename({"time": "year"})
    raise ValueError("No 'year' or 'time' dimension found.")

def mk_all_windows(x):
    """Return (slope, pvalue) arrays for all end-fixed windows for one grid point."""
    slopes = np.empty(len(period_starts), dtype=float)
    pvals = np.empty(len(period_starts), dtype=float)
    for idx, begin_year in enumerate(period_starts):
        start_idx = begin_year - START_YEAR
        seg = x[start_idx:]
        slopes[idx], pvals[idx] = data_process.apply_mannkendall(seg)
    return slopes * 10.0, pvals  # convert to K/decade once
# %%
# -----------------------
# LOAD + RESIDUAL (internal variability)
# -----------------------
# mem_file = os.path.join(DIR_MEM, f"tas_{model}_annual_ano_1850_2022.nc")
in_file = f"{DIR_IN}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"
# print(f"[rank {rank}] Processing run {run} from {in_file}", flush=True)

ds_in = xr.open_dataset(in_file, chunks=CHUNKS)  # no chunks
# ens_file = os.path.join(, f"SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc")
# if not os.path.exists(ens_file):
#     raise FileNotFoundError(ens_file)

# ds_mem = xr.open_dataset(ds_in, )
tas_mem = ensure_year_dim(ds_in["internal_variability"]).sel(year=slice(START_YEAR, END_YEAR))
# select ONE run by index (robust even if run coordinate labels differ)
run_index = run_id - 1
tas_run = tas_mem.isel(run=run_index)  # (year, lat, lon)

# ds_ens = xr.open_dataset(ens_file, chunks=CHUNKS)
# tas_ens = ensure_year_dim(get_tas_var(ds_ens, model)).sel(year=slice(START_YEAR, END_YEAR))
# # residual field
# resid = tas_run - tas_ens  # (year, lat, lon)
# resid = resid.chunk(CHUNKS)

lat = tas_run["lat"]
lon = tas_run["lon"]

# -----------------------
# COMPUTE MK TREND PATTERNS FOR EACH END-FIXED WINDOW
# -----------------------
trend_all, pval_all = xr.apply_ufunc(
    mk_all_windows,
    tas_run,
    input_core_dims=[["year"]],
    output_core_dims=[["period"], ["period"]],
    output_sizes={"period": len(period_starts)},
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float, float],
)

trend_all = trend_all.assign_coords(period=("period", period_labels), lat=lat, lon=lon)
pval_all = pval_all.assign_coords(period=("period", period_labels), lat=lat, lon=lon)

trend_all.name = "trend"
pval_all.name  = "p_value"
trend_all.attrs["units"] = "K/decade"

ds_out = xr.Dataset({"trend": trend_all, "p_value": pval_all})
ds_out.attrs["description"] = (
    "End-fixed (2022) MK trend patterns of LE internal variability anomalies "
    "(member(run)) for windows 2013–2022 ... 1950–2022."
)
ds_out.attrs["model"] = model
ds_out.attrs["run_id"] = run_id

out_file = os.path.join(DIR_OUT, f"{model}_run{run_id:03d}_OBS_LPS_ICV_MKtrend_patterns_1950_2022_sliding.nc")
print(f"Writing {out_file}", flush=True)
ds_out.to_netcdf(out_file)

# ds_mem.close()
# ds_ens.close()
print("Done.", flush=True)
# %%