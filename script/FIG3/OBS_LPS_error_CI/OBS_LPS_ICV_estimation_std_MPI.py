#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Store all individual OBS-LPS ICV trend estimates per model.

Perfect-model framework: Collect all OBS-LPS ICV estimates from each model
to preserve the full distribution for later uncertainty analysis.

- Input: Per-run OBS-LPS ICV spatial patterns from all models
- For each model:
    * Load ICV trend patterns from all runs for that model
    * Store them with run dimension preserved
    * Output one file per model
- Output: ICV estimation patterns (K/decade) with dimensions (run, period, lat, lon)

Example: CanESM5 with 50 OBS-LPS estimates → output has 50 × 64 periods × lat × lon

MPI parallelization: Models distributed across ranks (7 models → 7 ranks)
"""
# %%
import sys
import os
import glob
import shutil
import numpy as np
import xarray as xr
from mpi4py import MPI

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
# OBS-LPS ICV estimates (per run, per model)
OBS_LPS_ICV_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_actual_residual_ICV"

# Output directory
OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/OBS_LPS_ICV_std"
if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
comm.Barrier()

MODELS = [
    "CanESM5",
    "CESM2",
    "IPSL_CM6A",
    "EC_Earth3",
    "ACCESS",
    "MPI_ESM",
    "MIROC6",
]

START_YEAR = 1950
END_YEAR = 2022
MIN_LENGTH = 10

# Sliding windows: 2013-2022, 2012-2022, ..., 1950-2022
period_starts = list(range(END_YEAR - MIN_LENGTH + 1, START_YEAR - 1, -1))
period_labels = [f"{by}-{END_YEAR}" for by in period_starts]
n_period = len(period_labels)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _norm_str(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _sel_like(da, dim, key):
    """Robust selection along a dimension."""
    if dim not in da.dims and dim not in da.coords:
        return da
    vals = da[dim].values
    keyn = _norm_str(key)
    # exact match
    if key in vals:
        return da.sel({dim: key})
    # normalized match
    for v in vals:
        if _norm_str(v) == keyn:
            return da.sel({dim: v})
    # contains match
    for v in vals:
        if keyn in _norm_str(v):
            return da.sel({dim: v})
    raise KeyError(f"Could not match {key} on dim '{dim}'. Available: {list(vals)[:10]}")

def extract_trend_pattern(ds, period_str):
    """Extract spatial trend pattern for a given period."""
    # Find trend variable
    var_candidates = ['trend', 'MK_trend', 'icv_trend', 'slope']
    var_name = None
    for v in var_candidates:
        if v in ds.data_vars:
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    
    da = ds[var_name]
    
    # Select period
    if 'period' in da.dims or 'period' in da.coords:
        da = _sel_like(da, 'period', period_str)
    elif 'tau' in da.dims:
        # Convert period to tau
        tau = int(period_str.split('-')[1]) - int(period_str.split('-')[0]) + 1
        da = da.sel(tau=tau, method='nearest')
    
    return da
# %%
# ---------------------------------------------------------------------
# Load OBS-LPS ICV estimates for a specific model
# ---------------------------------------------------------------------
def load_model_obs_lps_icv(model):
    """Load all OBS-LPS ICV pattern runs for a specific model."""
    dir_in = f"{OBS_LPS_ICV_DIR}/{model}/per_run_patterns/"
    pattern = f"{dir_in}/{model}_run*_OBS_LPS_ICV_MKtrend_patterns_1950_2022_sliding.nc"
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No OBS-LPS ICV files found for {model} at {pattern}")
    
    print(f"[rank {rank}] Found {len(files)} OBS-LPS ICV runs for {model}", flush=True)
    return files
# %%
# ---------------------------------------------------------------------
# Collect all ICV estimates for a specific model
# ---------------------------------------------------------------------
def process_model_obs_lps_icv(model, model_files):
    """
    Collect OBS-LPS ICV trends for all runs of a specific model.
    Returns Dataset with icv_trend(run, period, lat, lon).
    """
    print(f"[rank {rank}] Processing {model}: {len(model_files)} runs", flush=True)
    
    # Storage: one dataset per run
    run_datasets = []
    
    for irun, fpath in enumerate(model_files):
        run_id = irun + 1  # 1-based indexing
        
        try:
            ds = xr.open_dataset(fpath)
            
            # Extract all periods for this run
            period_patterns = []
            
            for period_str in period_labels:
                try:
                    pattern = extract_trend_pattern(ds, period_str)
                    
                    # Drop unnecessary coordinates
                    coords_to_drop = [c for c in pattern.coords if c not in ['lat', 'lon', 'latitude', 'longitude']]
                    if coords_to_drop:
                        pattern = pattern.drop_vars(coords_to_drop, errors='ignore')
                    
                    # Add period dimension
                    pattern = pattern.expand_dims(period=[period_str])
                    period_patterns.append(pattern)
                    
                except Exception as e:
                    print(f"[rank {rank}] WARNING: Could not load {period_str} from {os.path.basename(fpath)}: {e}", flush=True)
                    continue
            
            if not period_patterns:
                print(f"[rank {rank}] WARNING: No periods loaded for run {run_id}", flush=True)
                ds.close()
                continue
            
            # Concatenate all periods for this run: (period, lat, lon)
            da_run = xr.concat(period_patterns, dim='period', coords='minimal', compat='override')
            
            # Add run dimension
            da_run = da_run.expand_dims(run=[run_id])
            
            run_datasets.append(da_run)
            ds.close()
            
        except Exception as e:
            print(f"[rank {rank}] ERROR processing {os.path.basename(fpath)}: {e}", flush=True)
            continue
    
    if not run_datasets:
        raise RuntimeError(f"[rank {rank}] No data loaded for {model}")
    
    # Concatenate all runs: (run, period, lat, lon)
    all_runs = xr.concat(run_datasets, dim='run', coords='minimal', compat='override')
    all_runs.name = 'icv_trend'
    all_runs.attrs['units'] = 'K/decade'
    all_runs.attrs['description'] = (
        f"OBS-LPS ICV trend estimates from {model} for sliding windows "
        f"{START_YEAR}-{END_YEAR} (all runs preserved)"
    )
    all_runs.attrs['n_runs'] = len(model_files)
    all_runs.attrs['model'] = model
    
    ds_model = xr.Dataset({'icv_trend': all_runs})
    print(f"[rank {rank}] {model} complete: shape = {ds_model['icv_trend'].shape}", flush=True)
    
    return ds_model

# %%
# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
if rank == 0:
    print("="*80)
    print("Loading OBS-LPS ICV estimates per model (preserving all runs)")
    print("Output: icv_trend(run, period, lat, lon) for each model")
    print("="*80)

comm.Barrier()

# Distribute models across ranks (round-robin)
local_models = [MODELS[i] for i in range(len(MODELS)) if i % npro == rank]

# All ranks report their assignments
print(f"[rank {rank}] Assigned models: {local_models}", flush=True)
comm.Barrier()

if not local_models:
    print(f"[rank {rank}] No models assigned.", flush=True)
else:
    for model in local_models:
        print(f"\n[rank {rank}] ===== Processing {model} =====", flush=True)
        
        try:
            # Load all OBS-LPS ICV files for this model
            model_files = load_model_obs_lps_icv(model)
            
            # Process all runs and periods
            ds_model = process_model_obs_lps_icv(model, model_files)
            
            # Write to UNIQUE temporary file first (avoid file locking conflicts)
            tmp_file = f"{OUT_DIR}/{model}_OBS_LPS_ICV_trends_{START_YEAR}_{END_YEAR}_len10-73_rank{rank}.tmp.nc"
            out_file = f"{OUT_DIR}/{model}_OBS_LPS_ICV_trends_{START_YEAR}_{END_YEAR}_len10-73.nc"
            
            print(f"[rank {rank}] Writing to temp file: {tmp_file}", flush=True)
            print(f"[rank {rank}]   Shape: {ds_model['icv_trend'].shape}", flush=True)
            
            try:
                # Write to temporary file
                encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_model.data_vars}
                ds_model.to_netcdf(tmp_file, encoding=encoding)
                ds_model.close()
                
                # Atomically rename to final filename
                import shutil
                if os.path.exists(out_file):
                    os.remove(out_file)
                shutil.move(tmp_file, out_file)
                
                # Get file size
                size_mb = os.path.getsize(out_file) / (1024**2)
                print(f"[rank {rank}] {model} complete! Size: {size_mb:.1f} MB", flush=True)
                
            except Exception as write_error:
                print(f"[rank {rank}] ERROR writing {model}: {write_error}", flush=True)
                import traceback
                traceback.print_exc()
                # Clean up temp file if it exists
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
            finally:
                if 'ds_model' in locals():
                    try:
                        ds_model.close()
                    except:
                        pass
            
        except Exception as e:
            print(f"[rank {rank}] ERROR processing {model}: {e}", flush=True)
            import traceback
            traceback.print_exc()

# Wait for all ranks to finish writing
comm.Barrier()

if rank == 0:
    print("\n" + "="*80)
    print("OBS-LPS ICV TRENDS (PER MODEL) COMPLETE!")
    print(f"Output directory: {OUT_DIR}")
    print("="*80)
# %%
