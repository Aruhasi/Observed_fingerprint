# %%
"""
Calculate the OBS-LPS estimation error distributions for each grid point: 
Get the confidence intervals of the errors by pooling across all ensemble members and models for each period over each grid point.

Step A: Read in all the 343 realization trends from the OBS-LPS forced and ICV estimates (per-run) for each period, 
along with the corresponding SMILE predicted forced and ICV values (scalar per model/region/period).

This provides a plausible "significance test" on the OBS-LPS observation paratitioning and the MMLE simulated forced/ICV.
Step B: For each region/period, compute the error = OBS-LPS estimate - SMILE predicted value for each run, pool across all runs and models, 
and plot the error distribution as a PDF (kernel density estimate).
"""
# %%
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# %%
# -----------------------------
# Define directories for spatial map analysis
# For OBS-LPS estimated (per-run) spatial patterns
"""
For internal variability (ICV): we are using the standard deviation as the "actual" ICV pattern, and the OBS-LPS estimated ICV patterns (per run) to compute the error distribution.
- SMILE actual ICV patterns (per run, per model):
{MODEL_ICV_PATTERN_DIR}/{model}/per_run_patterns/{model}_run{run_id:03d}_ICV_MKtrend_patterns_1950_2022_sliding.nc
- OBS-LPS ICV patterns (per run, per model):
{OBS_LPS_ICV_PATTERN_DIR}/{model}/per_run_patterns/{model}_run{run_id:03d}_OBS_LPS_ICV_MKtrend_patterns_1950_2022_sliding.nc
"""
# Key periods for spatial map analysis
KEY_PERIODS = {
    "10-year": "2013-2022",
    "30-year": "1993-2022",
    "60-year": "1963-2022",
}

file_id_to_label = {
    "CanESM5":   "CanESM5(50)",
    "CESM2":     "CESM2(100)",
    "IPSL_CM6A": "IPSL-CM6A-LR(32)",
    "EC_Earth3": "EC-Earth3(21)",
    "ACCESS":    "ACCESS-ESM1.5(40)",
    "MIROC6":    "MIROC6(50)",
    "MPI_ESM":   "MPI-ESM1.2-LR(50)",
}

# RGB_dict = {
#     "CanESM5(50)": "#A60E16",
#     "CESM2(100)": "#EE3B2A",
#     "IPSL-CM6A-LR(32)": "#FC9171",
#     "EC-Earth3(21)": "#FDDFCF",
#     "ACCESS-ESM1.5(40)": "#5FB7B5",
#     "MPI-ESM1.2-LR(50)": "#7AB0DF",
#     "MIROC6(50)": "#0F55C5",
#     "MMLE": "black",
# }
RGB_dict = {'CanESM5(50)':'#A60E16', 
            'CESM2(100)':'#EE3B2A',
            'IPSL-CM6A-LR(32)':'#FC9171', 
            'EC-Earth3(21)':"#F5BFA2", 
            'ACCESS-ESM1.5(40)':"#5FB7B5",
            'MPI-ESM1.2-LR(50)':"#246BA1", 
            'MIROC6(50)':"#073278", 
            'MMLE':'black'}

category_style = {
    "member":         dict(marker="o", mfc="none", mec="k", mew=0.9, ms=30, alpha=0.85, linestyle=""),
    "HadCRUT5":       dict(marker="s", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
    "NOAAGlobalTemp": dict(marker="8", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
    "BEST":           dict(marker="^", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
}
MEMBER_SCATTER_STYLE = category_style["member"]
MODELS = [
    "CanESM5",
    "CESM2",
    "IPSL_CM6A",
    "EC_Earth3",
    "ACCESS",
    "MPI_ESM",
    "MIROC6",
]

END_YEAR = 2022
START_MIN = 1950
TAU_MIN = 10
TAU_MAX = END_YEAR - START_MIN + 1  # 73
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)

PERIODS_TO_PLOT = {
    "10-year": "2013-2022",
    "30-year": "1993-2022",
    "60-year": "1963-2022",
}
FIG_OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/OBS_LPS_error_CI/"
os.makedirs(FIG_OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)

OBS_LPS_ICV_PATTERN_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_actual_residual_ICV"
OBS_LPS_FORCED_PATTERN_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/"

# For model-simulated "true" spatial patterns (forced and ICV)
MODEL_FORCED_PATTERN_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/"
MODEL_ICV_PATTERN_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/"
# Example file patterns for loading
# OBS-LPS ICV (per run, per model)
# {OBS_LPS_ICV_PATTERN_DIR}/{model}/per_run_patterns/{model}_run{run_id:03d}_OBS_LPS_ICV_MKtrend_patterns_1950_2022_sliding.nc
# OBS-LPS FORCED (per model)
# {OBS_LPS_FORCED_PATTERN_DIR}/{model}/forced_{model}_MK_trend_1950-2022_sliding.nc
# Model "true" FORCED (per model)
# {MODEL_FORCED_PATTERN_DIR}/{model}_ENSforced_global_trend_1950-2022_sliding.nc
# -----------------------------
# Data loaders for SPATIAL PATTERNS (gridded data)
# -----------------------------
def load_spatial_actual_forced():
    """Load model ensemble mean forced patterns (spatial maps)."""
    combined = {}
    base = MODEL_FORCED_PATTERN_DIR
    
    for model in MODELS:
        # Try to find the ensemble mean forced spatial pattern
        pattern = f"{base}/{model}/SMILE_forced/{model}_ENSmean_forced_MK_trend_1950-2022_sliding.nc"
        if os.path.exists(pattern):
            combined[model] = xr.open_dataset(pattern)
            print(f"Loaded actual forced pattern for {model}")
        else:
            print(f"WARNING: No actual forced pattern found for {model} at {pattern}")
            combined[model] = None
    return combined

def load_spatial_actual_icv():
    """Load model actual ICV patterns (per-run spatial maps)."""
    combined = {}
    base = MODEL_ICV_PATTERN_DIR
    
    for model in MODELS:
        dir_in = f"{base}/{model}/per_run_patterns/"
        files = sorted(glob.glob(f"{dir_in}/{model}_run*_resid_MKtrend_patterns_1950_2022_sliding.nc"))
        if files:
            ds_list = [xr.open_dataset(f) for f in files]
            combined[model] = xr.concat(ds_list, dim="run")
            print(f"Loaded {len(files)} actual ICV pattern runs for {model}")
        else:
            print(f"WARNING: No actual ICV patterns found for {model}")
            combined[model] = None
    return combined
# %%
def load_spatial_estimated_forced():
    """Load OBS-LPS estimated forced patterns (PER-RUN spatial maps)."""
    combined = {}
    base = OBS_LPS_FORCED_PATTERN_DIR
    
    for model in MODELS:
        # The forced pattern file contains PER-RUN forced estimates (not ensemble mean)
        pattern = f"{base}/{model}/forced_{model}_MK_trend_1950-2022_sliding.nc"
        if os.path.exists(pattern):
            combined[model] = xr.open_dataset(pattern)
            n_runs = combined[model].dims.get('run', 1)
            print(f"Loaded OBS-LPS forced estimate for {model} with {n_runs} runs")
        else:
            print(f"WARNING: No OBS-LPS forced estimate found for {model} at {pattern}")
            combined[model] = None
    return combined

def load_spatial_estimated_icv():
    """Load OBS-LPS estimated ICV patterns (per-run spatial maps)."""
    combined = {}
    base = OBS_LPS_ICV_PATTERN_DIR
    
    for model in MODELS:
        dir_in = f"{base}/{model}/per_run_patterns/"
        files = sorted(glob.glob(f"{dir_in}/{model}_run*_OBS_LPS_ICV_MKtrend_patterns_1950_2022_sliding.nc"))
        if files:
            ds_list = [xr.open_dataset(f) for f in files]
            combined[model] = xr.concat(ds_list, dim="run")
            print(f"Loaded {len(files)} OBS-LPS ICV pattern runs for {model}")
        else:
            print(f"WARNING: No OBS-LPS ICV patterns found for {model}")
            combined[model] = None
    return combined

# Load spatial pattern data
print("\n" + "="*60)
print("Loading spatial pattern data for gridded analysis...")
print("="*60)
spatial_SMILE_forced_pred = load_spatial_actual_forced()
spatial_OBS_LPS_forced_est = load_spatial_estimated_forced()
spatial_SMILE_ICV_actual = load_spatial_actual_icv()
spatial_OBS_LPS_ICV_est = load_spatial_estimated_icv()
print("="*60 + "\n")
# %%
# -----------------------------
# Helpers (robust selection for spatial analysis)
# -----------------------------
def _first_data_var(ds):
    return list(ds.data_vars)[0]

def _pick_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return _first_data_var(ds)

def _norm_str(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _sel_like(da, dim, key):
    """Select along a coord dim with robust matching."""
    if dim not in da.dims and dim not in da.coords:
        return da
    vals = da[dim].values
    keyn = _norm_str(key)
    # exact match first
    if key in vals:
        return da.sel({dim: key})
    # normalized match
    for v in vals:
        if _norm_str(v) == keyn:
            return da.sel({dim: v})
    # try contains match
    for v in vals:
        if keyn in _norm_str(v):
            return da.sel({dim: v})
    raise KeyError(f"Could not match {key} on dim/coord '{dim}'. Available: {list(vals)[:10]}...")

def _to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)
# %%
# -----------------------------
# SPATIAL ERROR ANALYSIS
# Pool all 343 ensemble errors at each grid point
# -----------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_common_grid(ds_list):
    """Extract common lat/lon grid from a list of datasets."""
    # Try common lat/lon coordinate names
    for lat_name in ['lat', 'latitude', 'y']:
        for lon_name in ['lon', 'longitude', 'x']:
            if lat_name in ds_list[0].coords and lon_name in ds_list[0].coords:
                return ds_list[0][lat_name].values, ds_list[0][lon_name].values, lat_name, lon_name
    raise ValueError("Could not find lat/lon coordinates in datasets")

def extract_trend_pattern(ds, period_str, lat_name='lat', lon_name='lon'):
    """Extract spatial trend pattern for a given period."""
    # Find the trend variable
    var_candidates = ['trend', 'MK_trend', 'forced', 'slope', 'tau']
    var_name = None
    for v in var_candidates:
        if v in ds.data_vars:
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]
    
    da = ds[var_name]
    
    # Select period if it exists as a dimension
    if 'period' in da.dims or 'period' in da.coords:
        da = _sel_like(da, 'period', period_str)
    elif 'tau' in da.dims:
        # Convert period string to tau value
        tau = int(period_str.split('-')[0]) - int(period_str.split('-')[1]) + 1
        da = da.sel(tau=abs(tau), method='nearest')
    
    return da
# %%
# Analyze each period
periods_to_analyze = {
    "10-year": "2013-2022",
    "30-year": "1993-2022",
    "60-year": "1963-2022"
}

for period_tag, period_str in periods_to_analyze.items():
    print(f"\n{'='*80}")
    print(f"Processing spatial pattern errors for: {period_tag} ({period_str})")
    print(f"{'='*80}")
    
    # -------------------------
    # FORCED ERROR PATTERNS (per run)
    # -------------------------
    print("\nComputing FORCED errors at each grid point (per run)...")
    forced_error_list = []
    total_forced_runs = 0
    
    for model in MODELS:
        # Get model truth (ensemble mean forced pattern) - single pattern per model
        ds_true = spatial_SMILE_forced_pred.get(model)
        # Get OBS-LPS forced estimates - PER RUN
        ds_est = spatial_OBS_LPS_forced_est.get(model)
        
        if ds_true is None or ds_est is None:
            print(f"  Skipping {model} - missing forced data")
            continue
        
        try:
            # Extract true forced pattern (ensemble mean) for this period
            true_pattern = extract_trend_pattern(ds_true, period_str)
            
            # Get number of runs in OBS-LPS estimates
            n_runs = ds_est.dims.get('run', 0)
            if n_runs == 0:
                print(f"  Skipping {model} - no run dimension in OBS-LPS forced")
                continue
            
            # Process each run
            for irun in range(n_runs):
                # Extract OBS-LPS forced estimate for this run
                est_run = extract_trend_pattern(ds_est.isel(run=irun), period_str)
                
                # Align grids if needed
                if true_pattern.shape != est_run.shape:
                    est_run = est_run.interp_like(true_pattern)
                
                # Compute error = OBS-LPS estimate - Truth (ensemble mean)
                error_pattern = est_run - true_pattern
                forced_error_list.append(error_pattern.expand_dims(run=[total_forced_runs]))
                total_forced_runs += 1
            
            print(f"  {model}: {n_runs} forced error patterns computed")
            
        except Exception as e:
            print(f"  ERROR processing {model} forced: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal forced runs collected: {total_forced_runs}")
    
    # -------------------------
    # ICV ERROR PATTERNS (per run)
    # -------------------------
    print("\nComputing ICV errors at each grid point (per run)...")
    icv_error_list = []
    total_runs = 0
    
    for model in MODELS:
        ds_true = spatial_SMILE_ICV_actual.get(model)
        ds_est = spatial_OBS_LPS_ICV_est.get(model)
        
        if ds_true is None or ds_est is None:
            print(f"  Skipping {model} - missing ICV data")
            continue
        
        try:
            # Get number of runs
            n_runs = ds_true.dims.get('run', 0)
            if n_runs == 0:
                print(f"  Skipping {model} - no run dimension")
                continue
            
            # Process each run
            for irun in range(n_runs):
                true_run = extract_trend_pattern(ds_true.isel(run=irun), period_str)
                est_run = extract_trend_pattern(ds_est.isel(run=irun), period_str)
                
                # Align grids if needed
                if true_run.shape != est_run.shape:
                    est_run = est_run.interp_like(true_run)
                
                # Error = estimate - truth
                error_run = est_run - true_run
                icv_error_list.append(error_run.expand_dims(run=[total_runs]))
                total_runs += 1
            
            print(f"  {model}: {n_runs} ICV runs processed")
            
        except Exception as e:
            print(f"  ERROR processing {model} ICV: {e}")
            continue
    
    print(f"\nTotal runs collected: {total_runs}")
    
    # -------------------------
    # POOL ALL ERRORS
    # -------------------------
    if len(forced_error_list) > 0:
        print(f"\nPooling forced errors across all {total_forced_runs} runs...")
        all_forced_errors = xr.concat(forced_error_list, dim='run', coords='minimal', compat='override')
        
        # Calculate statistics at each grid point
        forced_mean = all_forced_errors.mean(dim='run')
        forced_std = all_forced_errors.std(dim='run')
        forced_rmse = np.sqrt((all_forced_errors**2).mean(dim='run'))
        forced_ci_lower = all_forced_errors.quantile(0.025, dim='run')
        forced_ci_upper = all_forced_errors.quantile(0.975, dim='run')
        
        print(f"  Forced errors pooled: {total_forced_runs} runs from {len(MODELS)} models")
        print(f"  Mean error range: [{float(forced_mean.min()):.4f}, {float(forced_mean.max()):.4f}]")
        print(f"  RMSE range: [{float(forced_rmse.min()):.4f}, {float(forced_rmse.max()):.4f}]")
    
    if len(icv_error_list) > 0:
        print(f"\nPooling ICV errors across all {total_runs} runs...")
        all_icv_errors = xr.concat(icv_error_list, dim='run', coords='minimal', compat='override')
        
        # Calculate statistics at each grid point
        icv_mean = all_icv_errors.mean(dim='run')
        icv_std = all_icv_errors.std(dim='run')
        icv_rmse = np.sqrt((all_icv_errors**2).mean(dim='run'))
        icv_ci_lower = all_icv_errors.quantile(0.025, dim='run')
        icv_ci_upper = all_icv_errors.quantile(0.975, dim='run')
        
        print(f"  ICV errors pooled: {total_runs} runs")
        print(f"  Mean error range: [{float(icv_mean.min()):.4f}, {float(icv_mean.max()):.4f}]")
        print(f"  RMSE range: [{float(icv_rmse.min()):.4f}, {float(icv_rmse.max()):.4f}]")
    
    # -------------------------
    # SAVE RESULTS
    # -------------------------
    print("\nSaving spatial error statistics...")
    out_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/spatial_error_analysis/"
    os.makedirs(out_dir, exist_ok=True)
    
    if len(forced_error_list) > 0:
        # Save forced error statistics
        # Drop 'quantile' coordinate from CI bounds to avoid conflicts
        ds_forced_stats = xr.Dataset({
            'error_mean': forced_mean,
            'error_std': forced_std,
            'error_rmse': forced_rmse,
            'error_ci_lower': forced_ci_lower.drop_vars('quantile', errors='ignore'),
            'error_ci_upper': forced_ci_upper.drop_vars('quantile', errors='ignore'),
        })
        out_file = f"{out_dir}/forced_error_stats_{period_tag}_{period_str}.nc"
        ds_forced_stats.to_netcdf(out_file)
        print(f"  Saved: {out_file}")
    
    if len(icv_error_list) > 0:
        # Save ICV error statistics
        # Drop 'quantile' coordinate from CI bounds to avoid conflicts
        ds_icv_stats = xr.Dataset({
            'error_mean': icv_mean,
            'error_std': icv_std,
            'error_rmse': icv_rmse,
            'error_ci_lower': icv_ci_lower.drop_vars('quantile', errors='ignore'),
            'error_ci_upper': icv_ci_upper.drop_vars('quantile', errors='ignore'),
        })
        out_file = f"{out_dir}/icv_error_stats_{period_tag}_{period_str}.nc"
        ds_icv_stats.to_netcdf(out_file)
        print(f"  Saved: {out_file}")
    
    # -------------------------
    # CREATE SPATIAL MAPS
    # -------------------------
    print("\nCreating spatial error maps...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Define common colormap settings
    vmin_error, vmax_error = -0.3, 0.3
    vmin_rmse, vmax_rmse = 0, 0.4
    
    # Panel 1: Forced error mean
    if len(forced_error_list) > 0:
        ax1 = plt.subplot(2, 3, 1, projection=ccrs.Robinson())
        im1 = forced_mean.plot(ax=ax1, transform=ccrs.PlateCarree(),
                               cmap='RdBu_r', vmin=vmin_error, vmax=vmax_error,
                               add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax1.coastlines()
        ax1.set_title(f'(A) Forced Error Mean\n{period_str}', fontsize=14, fontweight='bold')
        
        # Panel 2: Forced error RMSE
        ax2 = plt.subplot(2, 3, 2, projection=ccrs.Robinson())
        forced_rmse.plot(ax=ax2, transform=ccrs.PlateCarree(),
                         cmap='YlOrRd', vmin=vmin_rmse, vmax=vmax_rmse,
                         add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax2.coastlines()
        ax2.set_title(f'(B) Forced Error RMSE\n{period_str}', fontsize=14, fontweight='bold')
        
        # Panel 3: Forced 95% CI width
        ax3 = plt.subplot(2, 3, 3, projection=ccrs.Robinson())
        ci_width = forced_ci_upper - forced_ci_lower
        ci_width.plot(ax=ax3, transform=ccrs.PlateCarree(),
                      cmap='viridis', vmin=0, vmax=0.8,
                      add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax3.coastlines()
        ax3.set_title(f'(C) Forced 95% CI Width\n{period_str}', fontsize=14, fontweight='bold')
    
    # Panel 4: ICV error mean
    if len(icv_error_list) > 0:
        ax4 = plt.subplot(2, 3, 4, projection=ccrs.Robinson())
        icv_mean.plot(ax=ax4, transform=ccrs.PlateCarree(),
                      cmap='RdBu_r', vmin=vmin_error, vmax=vmax_error,
                      add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax4.coastlines()
        ax4.set_title(f'(D) ICV Error Mean\n{period_str}', fontsize=14, fontweight='bold')
        
        # Panel 5: ICV error RMSE
        ax5 = plt.subplot(2, 3, 5, projection=ccrs.Robinson())
        icv_rmse.plot(ax=ax5, transform=ccrs.PlateCarree(),
                      cmap='YlOrRd', vmin=vmin_rmse, vmax=vmax_rmse,
                      add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax5.coastlines()
        ax5.set_title(f'(E) ICV Error RMSE\n{period_str}', fontsize=14, fontweight='bold')
        
        # Panel 6: ICV 95% CI width
        ax6 = plt.subplot(2, 3, 6, projection=ccrs.Robinson())
        ci_width_icv = icv_ci_upper - icv_ci_lower
        ci_width_icv.plot(ax=ax6, transform=ccrs.PlateCarree(),
                          cmap='viridis', vmin=0, vmax=0.8,
                          add_colorbar=True, cbar_kwargs={'label': '°C/decade', 'shrink': 0.8})
        ax6.coastlines()
        ax6.set_title(f'(F) ICV 95% CI Width\n{period_str}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/Revision_check/FIG5_spatial_error_maps/"
    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = f"{fig_dir}/Spatial_error_maps_{period_tag}_{period_str}.pdf"
    out_png = f"{fig_dir}/Spatial_error_maps_{period_tag}_{period_str}.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_pdf}")

print("\n" + "="*80)
print("SPATIAL ERROR ANALYSIS COMPLETE!")
print("="*80)
# %%


