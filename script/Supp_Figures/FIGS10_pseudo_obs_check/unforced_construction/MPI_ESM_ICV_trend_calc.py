# In[1]:
print("Starting internal variability trend calculation script")
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import logging
import sys
# In[2]:
# define function
import src.SAT_function_Obs_Fingerprint as data_process
# %%
try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
  npro = comm.Get_size()  # 10
except:
  print('::: Warning: Proceeding without mpi4py! :::')
  rank = 0
  npro = 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s')
# %%
model = sys.argv[1]
# %%
segment_lengths = [30]
trend_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/pseudo_obs_check"
input_file = os.path.join(trend_dir, f"MPI_ESM_tas_unforced_ano_wrt_{model}_1850_2022.nc")
output_dir = os.path.join(trend_dir, "trend")
os.makedirs(output_dir, exist_ok=True)
# %%
def func_mk(x):
    """
    Mann-Kendall test for trend
    """
    results = data_process.apply_mannkendall(x)
    slope = results[0]
    # slope_x = x.isel(time = 0).copy(slope)
    return slope * 10
# %%
def separate_interval(data, segment_length):
    return [data.isel(year=slice(i, i + segment_length)) for i in range(data.sizes['year'] - segment_length + 1)]
# %%
def compute_trend_for_segments(data, segment_lengths):
    trend_ds = xr.Dataset()
    for seg_len in segment_lengths:
        segments = separate_interval(data, seg_len)
        trend_list = []
        for seg in segments:
            slope = xr.apply_ufunc(
                func_mk,
                seg,
                input_core_dims=[["year"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float]
            )
            # Ensure it's a DataArray
            if isinstance(slope, xr.Dataset):
                slope = slope.to_array().squeeze()
            trend_list.append(slope)

        trend_da = xr.concat(trend_list, dim="segment")
        trend_da = trend_da.assign_coords(segment=np.arange(len(trend_list)))
        trend_ds[f"trend_{seg_len}"] = trend_da
    return trend_ds
# %%
# === load data ===
ds = xr.open_dataset(input_file)['tas']
all_runs = ds['run'].values
assigned_runs = np.array_split(all_runs, npro)[rank]

# === Process assigned runs
local_results = []
for i, run in enumerate(assigned_runs):
    intermediate_out = os.path.join(output_dir, f"MPI_ESM_tas_ICV_trend_wrt_{model}_run_{run}.nc")

    if os.path.exists(intermediate_out):
        print(f"[Rank {rank}] Skipping run {run}, file already exists.")
        continue

    print(f"[Rank {rank}] Processing run {run} ({i+1}/{len(assigned_runs)})")
    ds_run = ds.sel(run=run)
    trend_result = compute_trend_for_segments(ds_run, segment_lengths)
    trend_result = trend_result.expand_dims(run=[run])
    local_results.append(trend_result)
    trend_result.to_netcdf(intermediate_out)
    print(f"[Rank {rank}] Finished processing and saved run {run}")

# === Gather results from all ranks
local_combined = xr.concat(local_results, dim="run") if local_results else None
all_results = comm.gather(local_combined, root=0)

# === Only rank 0 merges and saves
if rank == 0:
    final_result = xr.concat([r for r in all_results if r is not None], dim="run")
    trend_out = os.path.join(output_dir, f"MPI_ESM_tas_ICV_trend_wrt_{model}.nc")
    final_result.to_netcdf(trend_out)
    print(f"[Rank 0] Saved final result to: {trend_out}")
