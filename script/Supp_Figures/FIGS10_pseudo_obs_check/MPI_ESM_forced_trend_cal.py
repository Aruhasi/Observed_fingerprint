# %%
print("starting scripts")
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
import numpy as np
import xarray as xr
import os
import logging
import sys

from mpi4py import MPI
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.mpl.ticker as cticker
import cartopy.util as cutil
from matplotlib.colors import ListedColormap
sys.path.append("/work/mh0033/m301036/Land_surf_temp")
import src.SAT_function_Obs_Fingerprint as data_process
# %%
import pymannkendall as mk
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

# %%
model = sys.argv[1]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s')

# model_names = ["MIROC6", "ACCESS", "EC_Earth", "IPSL_CM6A", "CanESM5"]
interval_name = "30yr"
time_interval = {interval_name: (1993, 2022)}

trend_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/pseudo_obs_check"
out_dir = os.path.join(trend_dir, "trend")
os.makedirs(out_dir, exist_ok=True)

# === Functions ===
def separate_interval(data, interval_key, time_interval):
    start, end = time_interval[interval_key]
    return data.sel(year=slice(str(start), str(end)))

def func_mk(x):
    results = data_process.mk_test(x)
    return results[0], results[1]

def plot_trend_with_significance(trend_data, lats, lons,
                                 levels=None, extend=None, cmap=None, title="", ax=None,
                                 show_xticks=False, show_yticks=False):
    # insignificance_mask = p_values >= 0.10
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()

    contour_obj = ax.contourf(lons, lats, trend_data, levels=levels, extend=extend,
                              cmap=cmap, transform=ccrs.PlateCarree())
    # ax.contourf(lons, lats, insignificance_mask, levels=[0.0, 0.10, 1.5],
    #             hatches=[None, '///'], colors='none', transform=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.5, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    ax.set_title(title, loc='center', fontsize=14, pad=3.0)
    return contour_obj
# %%
forced_path = f"{trend_dir}/MPI_ESM_tas_forced_ano_wrt_{model}_1850_2022.nc"
if not os.path.exists(forced_path):
    print(f"[Rank {rank}] File not found: {forced_path}")
    sys.exit(1)

ds = xr.open_dataset(forced_path)['tas']
run_values = ds['run'].values
run_single = np.array_split(run_values, npro)[rank]
# %%
local_results = []
for i, run in enumerate(run_single):
    print(f"[Rank {rank}] Processing run {run} ({i+1}/{len(run_single)})")
    ds_run = ds.sel(run=run)
    forced_30yr = separate_interval(ds_run, interval_name, time_interval)

    slope, _ = xr.apply_ufunc(
        func_mk, forced_30yr,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    slope = slope.expand_dims(run=[run])
    local_results.append(slope)

# Gather results from all ranks
local_concat = xr.concat(local_results, dim="run") if local_results else None
all_gathered = comm.gather(local_concat, root=0)

# Only rank 0 merges and saves
if rank == 0:
    final_result = xr.concat([r for r in all_gathered if r is not None], dim="run")
    trend_out = os.path.join(out_dir, f"MPI_ESM_tas_forced_trend_{interval_name}_wrt_{model}_1993_2022.nc")
    final_result.name = f"{interval_name}_forced"
    final_result.to_netcdf(trend_out)
    print(f"[Rank 0] Saved final result to: {trend_out}")
    # === Plot ===
    plt.rcParams.update({
        'figure.figsize': (8, 10),
        'font.size': 16,
        'axes.labelsize': 16,
        'ytick.direction': 'out',
        'ytick.minor.visible': True,
        'ytick.major.right': True,
        'ytick.right': True,
        'xtick.bottom': True,
        'savefig.transparent': True
    })

    trend_scaled = final_result * 10.0  # °C/decade

    lat = ds['lat']
    lon = ds['lon']

    levels = np.arange(-0.6, 0.65, 0.05)
    extend = 'both'

    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(10, 5, wspace=0.05, hspace=0.05)

    for i, run in enumerate(run_values):
        ax = fig.add_subplot(gs[i // 5, i % 5], projection=ccrs.Robinson(180))
        trend_data = trend_scaled.sel(run=run)
        trend_cyclic, lon_cyclic = cutil.add_cyclic_point(trend_data, coord=lon)

        contour_obj = plot_trend_with_significance(trend_cyclic, lat, lon_cyclic, 
                                                   levels=levels, extend=extend,
                                                   cmap='twilight_shifted', title="", ax=ax)
        ax.text(0.5, 1.1, f"{run}", ha='center', va='center', transform=ax.transAxes)

        if i == 0:
            contour_obj_for_cbar = contour_obj

    cbar_ax = fig.add_axes([0.25, 0.06, 0.6, 0.012])
    cbar = plt.colorbar(contour_obj_for_cbar, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('SAT Trend (°C/decade)', fontsize=16)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{model}_MPI_ESM_forced_30yr_trend_panel_MMEM_Based.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Rank 0] Saved plot: {plot_path}")
    plt.show()
