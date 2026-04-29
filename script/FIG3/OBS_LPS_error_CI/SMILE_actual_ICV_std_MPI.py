#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ICV standard deviation using sliding windows for each SMILE realization.

For each model realization, this script:
1. Extracts the internal variability (residual: member - ensemble mean)
2. For each trend length (10-73 years):
   - Generates all possible overlapping windows
   - Computes Mann-Kendall trend for each window
   - Calculates standard deviation across all windows
3. Result: One STD per (run, trend_length, lat, lon)

Output: {MODEL}_SMILE_noise_trend_std_sliding_1950_2022.nc
  - Variable: icv_std(run, period, lat, lon) in K/decade
  - 64 periods (trend lengths 10-73 years)
  
MPI parallelization: Models distributed across ranks (7 models → 7 ranks)
"""
# %%
import sys
import os
import numpy as np
import xarray as xr
from mpi4py import MPI
import warnings

# Add source directory to path
sys.path.append("/work/mh0033/m301036/OBS_LPS_revision/src")
import Data_Preprocess as data_process

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()
print(f"[rank {rank}] MPI world size = {npro}", flush=True)

# ---------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------
# Input: SMILE member files and ensemble means
DIR_MEMBERS = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data"
DIR_ENS = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data/ENS"

# Output directory
OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_actual_ICV_std"
if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
comm.Barrier()

# Get model name from command line argument
if len(sys.argv) < 2:
    if rank == 0:
        print("ERROR: Model name required as command-line argument")
        print("Usage: python SMILE_actual_ICV_std_MPI.py <MODEL_NAME>")
        print("Available models: CanESM5, CESM2, IPSL_CM6A, EC_Earth3, ACCESS, MPI_ESM, MIROC6")
    sys.exit(1)

MODEL = sys.argv[1]

AVAILABLE_MODELS = [
    "CanESM5",
    "CESM2",
    "IPSL_CM6A",
    "EC_Earth3",
    "ACCESS",
    "MPI_ESM",
    "MIROC6",
]

if MODEL not in AVAILABLE_MODELS:
    if rank == 0:
        print(f"ERROR: Unknown model '{MODEL}'")
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    sys.exit(1)

START_YEAR = 1950
END_YEAR = 2022
MIN_LENGTH = 10
MAX_LENGTH = END_YEAR - START_YEAR + 1  # 73 years

# Trend lengths: Only compute for specific lengths to speed up calculation
trend_lengths = [10, 30, 60]  # years
n_lengths = len(trend_lengths)

# Period labels for output (matching the convention)
period_labels = [f"{END_YEAR - L + 1}-{END_YEAR}" for L in trend_lengths]

if rank == 0:
    print(f"Processing model: {MODEL}")
    print(f"Computing STD for {n_lengths} trend lengths: {trend_lengths}", flush=True)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def get_tas_var(ds, model, context=""):
    """Extract temperature variable with robust name matching."""
    var_candidates = ['tas', 'SAT', 'temp', 'temperature', 'T']
    for vname in var_candidates:
        if vname in ds.data_vars:
            return ds[vname]
    # Fallback: first variable
    first_var = list(ds.data_vars)[0]
    print(f"[rank {rank}] WARNING ({model} {context}): Using first variable '{first_var}'", flush=True)
    return ds[first_var]

def load_residual_data(model):
    """Load members and ensemble mean, compute residuals (internal variability)."""
    # Load members - CESM2 has different file naming convention
    if model == "CESM2":
        mem_file = os.path.join(DIR_MEMBERS, f"tas_{model}_CMIP6_SMBB_annual_ano_1850_2022.nc")
    else:
        mem_file = os.path.join(DIR_MEMBERS, f"tas_{model}_annual_ano_1850_2022.nc")
    
    if not os.path.exists(mem_file):
        raise FileNotFoundError(f"Members file not found: {mem_file}")
    
    ds_mem = xr.open_dataset(mem_file, chunks={'year': 50, 'lat': 45, 'lon': 45})
    tas_mem = get_tas_var(ds_mem, model, context="members")
    
    # Ensure year dimension
    if "year" not in tas_mem.dims:
        if "time" in tas_mem.dims:
            tas_mem = tas_mem.rename({"time": "year"})
        else:
            raise ValueError(f"No 'year' or 'time' dimension in members for {model}")
    
    # Subset to analysis period
    tas_mem = tas_mem.sel(year=slice(START_YEAR, END_YEAR))
    
    # Load ensemble mean - CESM2 has different file naming convention
    if model == "CESM2":
        ens_file = os.path.join(DIR_ENS, f"{model}_annual_ano_ensemble_mean_1950_2022_cmip6+smbb.nc")
    else:
        ens_file = os.path.join(DIR_ENS, f"{model}_annual_ano_ensemble_mean_1950_2022.nc")
    
    if not os.path.exists(ens_file):
        raise FileNotFoundError(f"Ensemble mean file not found: {ens_file}")
    
    ds_ens = xr.open_dataset(ens_file)
    tas_ens = get_tas_var(ds_ens, model, context="ensemble mean")
    
    if "year" not in tas_ens.dims:
        if "time" in tas_ens.dims:
            tas_ens = tas_ens.rename({"time": "year"})
        else:
            raise ValueError(f"No 'year' or 'time' dimension in ensemble mean for {model}")
    
    tas_ens = tas_ens.sel(year=slice(START_YEAR, END_YEAR))
    
    # Compute residuals (internal variability)
    # Broadcast ensemble mean across runs
    tas_resid = tas_mem - tas_ens
    
    ds_mem.close()
    ds_ens.close()
    
    print(f"[rank {rank}] {model}: Loaded residuals with shape {tas_resid.shape}", flush=True)
    return tas_resid

def generate_sliding_windows(data, window_length):
    """
    Generate all overlapping windows of given length from time series.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data with dimensions (run, year, lat, lon)
    window_length : int
        Length of each window in years
    
    Returns:
    --------
    windows : xarray.DataArray
        Windows with dimensions (run, window, year_in_window, lat, lon)
    """
    n_years = data.sizes['year']
    n_windows = n_years - window_length + 1
    
    if n_windows <= 0:
        raise ValueError(f"Window length {window_length} too long for {n_years} years")
    
    # Create list of windows - reset year coordinate to avoid alignment issues
    windows_list = []
    for i in range(n_windows):
        window = data.isel(year=slice(i, i + window_length))
        # Drop year coordinate and create new integer index to avoid concat alignment issues
        window = window.drop_vars('year', errors='ignore').assign_coords(window=i)
        windows_list.append(window)
    
    # Stack windows - use combine='nested' to avoid alignment
    windows = xr.concat(windows_list, dim='window', coords='minimal', compat='override')
    
    return windows

def compute_mk_trend_batch(data_3d):
    """
    Compute Mann-Kendall trend for 3D array (time, lat, lon).
    Returns trend in original units per year.
    """
    from scipy import stats
    
    n_time = data_3d.shape[0]
    years = np.arange(n_time)
    
    # Flatten spatial dimensions
    n_lat, n_lon = data_3d.shape[1], data_3d.shape[2]
    data_2d = data_3d.reshape(n_time, -1)  # (time, space)
    
    trends = np.full(n_lat * n_lon, np.nan)
    
    for i in range(n_lat * n_lon):
        ts = data_2d[:, i]
        if np.all(np.isnan(ts)):
            continue
        # Mann-Kendall trend (Theil-Sen slope)
        result = stats.theilslopes(ts, years)
        trends[i] = result[0]  # slope in units per year
    
    return trends.reshape(n_lat, n_lon)

def process_one_run_one_length(run_data, trend_length):
    """
    Process one realization for one trend length.
    
    Parameters:
    -----------
    run_data : xarray.DataArray
        Residual data for one run, dimensions (year, lat, lon)
    trend_length : int
        Length of trend windows in years
    
    Returns:
    --------
    std_pattern : xarray.DataArray
        Standard deviation of trends across windows, dimensions (lat, lon)
    """
    # Generate sliding windows
    windows = generate_sliding_windows(run_data.expand_dims('run'), trend_length)
    # windows shape: (run=1, window, year, lat, lon)
    
    n_windows = windows.sizes['window']
    n_lat = windows.sizes['lat']
    n_lon = windows.sizes['lon']
    
    # Compute trend for each window
    trends = np.full((n_windows, n_lat, n_lon), np.nan)
    
    for iw in range(n_windows):
        window_data = windows.isel(run=0, window=iw).values  # (year, lat, lon)
        trends[iw, :, :] = compute_mk_trend_batch(window_data)
    
    # Convert to K/decade
    trends = trends * 10.0
    
    # Compute standard deviation across windows
    # Handle case where n_windows <= 1 (e.g., 73-year trend has only 1 window)
    if n_windows <= 1:
        # STD undefined for single value - set to NaN
        std_pattern = np.full((n_lat, n_lon), np.nan)
    else:
        # Suppress expected warnings for grid cells with <2 valid values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice')
            std_pattern = np.nanstd(trends, axis=0, ddof=1)
    
    # Convert to DataArray
    std_da = xr.DataArray(
        std_pattern,
        dims=('lat', 'lon'),
        coords={
            'lat': run_data.lat,
            'lon': run_data.lon
        }
    )
    
    return std_da
# %%
def process_runs_parallel(model, run_indices, tas_resid):
    """
    Process assigned realizations on this rank.
    
    Parameters:
    -----------
    model : str
        Model name
    run_indices : list
        Indices of runs to process on this rank
    tas_resid : xarray.DataArray
        Full residual data for all runs
    
    Returns:
    --------
    results : dict
        Dictionary mapping run index to computed STD patterns
    """
    results = {}
    
    n_lat = tas_resid.sizes['lat']
    n_lon = tas_resid.sizes['lon']
    
    for irun in run_indices:
        print(f"[rank {rank}] {model}: Processing run {irun+1}", flush=True)
        
        run_data = tas_resid.isel(run=irun)
        
        # Storage for this run's results
        std_run = np.full((n_lengths, n_lat, n_lon), np.nan)
        
        # Process each trend length
        for ilen, trend_length in enumerate(trend_lengths):
            print(f"[rank {rank}]   Run {irun+1}: Trend length {trend_length} ({ilen+1}/{n_lengths})", flush=True)
            
            std_pattern = process_one_run_one_length(run_data, trend_length)
            std_run[ilen, :, :] = std_pattern.values
        
        results[irun] = std_run
    
    return results
# %%
# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
if rank == 0:
    print("="*80)
    print(f"Computing SMILE ICV standard deviation for {MODEL}")
    print(f"Trend lengths: {trend_lengths}")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"MPI ranks: {npro}")
    print("="*80)

comm.Barrier()

# Load residual data for this model
if rank == 0:
    print(f"\nLoading data for {MODEL}...", flush=True)

tas_resid = load_residual_data(MODEL)
n_runs = tas_resid.sizes['run']
n_lat = tas_resid.sizes['lat']
n_lon = tas_resid.sizes['lon']

if rank == 0:
    print(f"Total realizations: {n_runs}")
    print(f"Grid size: {n_lat} × {n_lon}")
    print(f"Distributing {n_runs} runs across {npro} MPI ranks...\n", flush=True)

# Distribute runs across ranks (round-robin)
local_runs = [i for i in range(n_runs) if i % npro == rank]

print(f"[rank {rank}] Assigned {len(local_runs)} runs: {local_runs[:5]}{'...' if len(local_runs) > 5 else ''}", flush=True)
comm.Barrier()

# Process assigned runs on this rank
try:
    local_results = process_runs_parallel(MODEL, local_runs, tas_resid)
    print(f"[rank {rank}] Completed {len(local_results)} runs", flush=True)
except Exception as e:
    print(f"[rank {rank}] ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    local_results = {}

comm.Barrier()

# Gather results on rank 0
if rank == 0:
    print("\nGathering results from all ranks...", flush=True)

all_results = comm.gather(local_results, root=0)

# Rank 0 assembles and writes output
if rank == 0:
    print("Assembling final dataset...", flush=True)
    
    # Initialize full array
    std_all = np.full((n_runs, n_lengths, n_lat, n_lon), np.nan)
    
    # Combine results from all ranks
    for rank_results in all_results:
        for irun, std_data in rank_results.items():
            std_all[irun, :, :, :] = std_data
    
    # Create output dataset
    ds_out = xr.Dataset({
        'icv_std': xr.DataArray(
            std_all,
            dims=('run', 'period', 'lat', 'lon'),
            coords={
                'run': tas_resid.run,
                'period': period_labels,
                'lat': tas_resid.lat,
                'lon': tas_resid.lon
            },
            attrs={
                'units': 'K/decade',
                'description': (
                    f'Standard deviation of ICV trends computed from sliding windows. '
                    f'For each realization and trend length, all overlapping windows are used. '
                    f'Trend lengths: {trend_lengths} years.'
                ),
                'model': MODEL,
                'n_runs': n_runs
            }
        )
    })
    
    # Write output
    out_file = f"{OUT_DIR}/{MODEL}_SMILE_noise_trend_std_sliding_{START_YEAR}_{END_YEAR}.nc"
    
    print(f"Writing {out_file}...", flush=True)
    encoding = {'icv_std': {'zlib': True, 'complevel': 4}}
    ds_out.to_netcdf(out_file, encoding=encoding)
    
    size_mb = os.path.getsize(out_file) / (1024**2)
    
    print("\n" + "="*80)
    print(f"COMPLETE: {MODEL}")
    print(f"Output: {out_file}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Shape: {std_all.shape}")
    print("="*80)
    
    ds_out.close()

print(f"[rank {rank}] Finished.", flush=True)
# %%