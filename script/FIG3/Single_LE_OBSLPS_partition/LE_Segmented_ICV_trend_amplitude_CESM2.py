# In[1]:
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
# In[2]:
# define function
# NOTE: apply_mannkendall lives in SAT_function_Obs_Fingerprint.py
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprosess
# In[3]:
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()
# In[4]:
start_year = 1850
end_year   = 2022
print(f"[rank {rank}] MPI world size = {npro}", flush=True)

model = sys.argv[1]

# Number of realizations per LE
N_RUNS_DICT = {
    "MPI_ESM"  : 50,
    "MIROC6"   : 50,
    "CESM2"    : 50,
    "CanESM5"  : 50,
    "ACCESS"   : 40,
    "EC_Earth3": 21,
    "IPSL_CM6A": 32,
}

if model not in N_RUNS_DICT:
    raise ValueError(f"Unknown model {model!r} – please add its run count to N_RUNS_DICT")

N_RUNS = N_RUNS_DICT[model]
runs   = np.arange(1, N_RUNS + 1)   # 1..N_RUNS
run_single = np.array_split(runs, npro)[rank]

dir_in  = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}"
dir_out = dir_in  # same dir for output
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
def generate_segments(data, segment_length):
    """
    Generate time segments for each segment length lazily (using slices).
    """
    # Calculate start years for each segment
    num_segments = data.sizes["year"] - segment_length + 1
    segments = [data.isel(year=slice(i, i + segment_length)) for i in range(num_segments)]
    return xr.concat(segments, dim="segment")
# %%
def compute_trend_for_segment_length(data, seg_len, trend_function):
    """
    Compute MK trend for a single segment length in a memory-safe way.

    For each segment (sliding window along 'year'):
      - slice data[year=i:i+seg_len]
      - compute MK slope over 'year' (vectorized over lat, lon)
    Returns:
      trend_da: DataArray (segment, lat, lon)
      num_segments: int
    """
    n_year = data.sizes["year"]
    num_segments = n_year - seg_len + 1
    if num_segments <= 0:
        raise ValueError(f"segment length {seg_len} is longer than available years {n_year}")

    trend_list = []

    for i in range(num_segments):
        seg = data.isel(year=slice(i, i + seg_len))

        # MK over 'year', vectorized over lat/lon
        trend = xr.apply_ufunc(
            trend_function,
            seg,
            input_core_dims=[["year"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="forbidden",      # no Dask, we work with in-memory arrays
            output_dtypes=[float],
        )

        # add segment dimension
        trend = trend.expand_dims(segment=[i])
        trend_list.append(trend)

    # (segment, lat, lon)
    trend_da = xr.concat(trend_list, dim="segment")
    trend_da = trend_da.assign_coords(segment=np.arange(num_segments))

    return trend_da, num_segments
# %%
def compute_trend(data, segment_lengths, trend_function):
    """Compute trends for multiple segment lengths, memory-safe."""
    if isinstance(segment_lengths, int):
        segment_lengths = [segment_lengths]

    ICV_segments_ds = xr.Dataset()
    max_segments = 0

    for seg_len in segment_lengths:
        print(f"[rank {rank}] Computing trend for segment_length={seg_len}", flush=True)
        trend_da, num_segments = compute_trend_for_segment_length(data, seg_len, trend_function)
        max_segments = max(max_segments, num_segments)

        # Store as float32 to reduce memory / disk
        ICV_segments_ds[f"trend_{seg_len}"] = trend_da.astype("float32")

    # Pad all variables to the same 'segment' length
    padded_ds = xr.Dataset()
    for var_name, da in ICV_segments_ds.data_vars.items():
        num_seg = da.sizes["segment"]
        if num_seg < max_segments:
            padding = xr.DataArray(
                np.full(
                    (max_segments - num_seg, *da.shape[1:]),
                    np.nan,
                    dtype=da.dtype,
                ),
                dims=["segment", *da.dims[1:]],
                coords={**da.coords, "segment": np.arange(num_seg, max_segments)},
            )
            padded_ds[var_name] = xr.concat([da, padding], dim="segment")
        else:
            padded_ds[var_name] = da

    return padded_ds
# %%
# def process_realization(run):
#     """
#     Compute MK trends for a single realization.
#     """
#     in_file = f"{dir_in}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"
#     print(f"[rank {rank}] Processing run {run} from {in_file}", flush=True)

#     ds_in = xr.open_dataset(in_file, chunks={"year": 50, "lat": 45, "lon": 45})

#     if "run" not in ds_in.dims and "run" not in ds_in.coords:
#         raise ValueError(f"{in_file} has no 'run' dimension / coordinate")

#     run_coord = ds_in["run"]

#     # --- robust selection of the realization ---
#     if np.issubdtype(run_coord.dtype, np.number):
#         # numeric labels (e.g. 1..50)
#         if run in run_coord.values:
#             ds_ICV = ds_in["internal_variability"].sel(run=run)
#         else:
#             # fall back to positional index: run 1 -> index 0
#             ds_ICV = ds_in["internal_variability"].isel(run=run - 1)
#     else:
#         # non-numeric labels (e.g. 'r1i1p1f1'); use index position
#         ds_ICV = ds_in["internal_variability"].isel(run=run - 1)
#     # run = runs
#     # Define segment lengths and compute trends
#     segment_lengths = range(10, 74, 1)
#     combined_results = compute_trend(ds_ICV, segment_lengths, func_mk)
#     combined_results = combined_results.assign_coords(run=run)
#     return combined_results
def process_realization(run):
    """
    Compute MK trends for a single realization.
    """
    in_file = f"{dir_in}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"
    print(f"[rank {rank}] Processing run {run} from {in_file}", flush=True)

    ds_in = xr.open_dataset(in_file)  # no chunks
    # rename the dimension 'member' to 'run'
    if "member" in ds_in.dims:
        ds_in = ds_in.rename({"member": "run"})
    if "run" not in ds_in.dims and "run" not in ds_in.coords:
        raise ValueError(f"{in_file} has no 'run' dimension / coordinate")

    run_coord = ds_in["run"]

    # robust selection
    if np.issubdtype(run_coord.dtype, np.number):
        if run in run_coord.values:
            ds_ICV = ds_in["internal_variability"].sel(run=run)
        else:
            ds_ICV = ds_in["internal_variability"].isel(run=run - 1)
    else:
        ds_ICV = ds_in["internal_variability"].isel(run=run - 1)

    # Load just this run into memory (time x lat x lon)
    ds_ICV = ds_ICV.load()
    ds_in.close()

    segment_lengths = range(10, 74, 1)
    combined_results = compute_trend(ds_ICV, segment_lengths, func_mk)

    combined_results = combined_results.assign_coords(run=run)
    return combined_results
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
    temp_file = f"{dir_out}/ICV_{model}_MK_trend_{start_year}-{end_year}_segments_rank{rank}.nc"
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
        f"{dir_out}/ICV_{model}_MK_trend_{start_year}-{end_year}_segments_rank{r}.nc"
        for r in range(npro)
        if os.path.exists(f"{dir_out}/ICV_{model}_MK_trend_{start_year}-{end_year}_segments_rank{r}.nc")
    ]

    if not temp_files:
        raise RuntimeError("No temporary rank files found to merge.")

    out_file = f"{dir_out}/ICV_{model}_MK_trend_{start_year}-{end_year}_segments.nc"
    
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
    n_run     = ds_all.dims["run"]
    n_segment = ds_all.dims["segment"]
    n_lat     = ds_all.dims["lat"]
    n_lon     = ds_all.dims["lon"]
    ds_all.close()

    for tf in temp_files:
        os.remove(tf)
        print(f"Removed {tf}", flush=True)

    print(f"All ranks finished. Final file: {out_file}", flush=True)
    print(f"Output dimensions: run={n_run}, segment={n_segment}, lat={n_lat}, lon={n_lon}", flush=True)
# %%