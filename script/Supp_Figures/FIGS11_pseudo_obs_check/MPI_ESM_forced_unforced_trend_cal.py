# %%
# In[1]:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import logging
from mpi4py import MPI
# %%
# define function
import src.SAT_function as data_process
import src.Data_Preprocess as preprocess
import sys
# %%
# scluster
import src.slurm_cluster as slurm_cluster
client, scluster = slurm_cluster.init_dask_slurm_cluster()
# %%
dir_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Revision_check/pseudo_obs_check/data'

model_name = ["MMEM"]
# model_name = sys.arguv[1] # get modle name from command line

MPI_ESM_forced = xr.open_mfdataset(dir_in + '/MPI_ESM_tas_forced_ano_wrt_MMEM_1850_2022.nc', chunks={'run': 1})
# MPI_ESM_unforced = xr.open_mfdataset(dir_in + '/MPI_ESM_tas_unforced_ano_wrt_MIROC6_1850_2022.nc', chunks={'run': 1})
# %%
# variable_name = ['10yr', '30yr', '60yr']
variable_name = ['30yr']
# define the function
# Define time interval
start_year, end_year = 1993, 2022
# Select the data slice directly
MPI_ESM_forced_ano = MPI_ESM_forced.sel(year=slice(str(start_year), str(end_year)))

# %%
# calculate the trend
def func_mk(x):
    """
    Mann-Kendall test for trend
    """
    results = data_process.mk_test(x)
    slope = results[0]
    p_val = results[1]
    return slope, p_val
# Calculate the trend and p-value for each time interval of each realization
data_var = MPI_ESM_forced_ano['tas']
slope, p_values = xr.apply_ufunc(
        func_mk,
        data_var,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

# %%
# Save the trend and p-value data
dir_out = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Revision_check/pseudo_obs_check/data/trend'
dir_out = os.path.join(dir_in, "trend")
os.makedirs(dir_out, exist_ok=True)
slope.to_netcdf(f"{dir_out}/MPI_ESM_tas_forced_trend_30yr_ano_wrt_MMEM_1993_2022.nc")
p_values.to_netcdf(f"{dir_out}/MPI_ESM_tas_forced_pvalue_30yr_ano_wrt_MMEM_1993_2022.nc")
# %%
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True # save the figure with a transparent background
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
import cartopy.util as cutil
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap

def plot_trend_with_significance(trend_data, lats, lons, p_values, GMST_p_values=None, levels=None, extend=None, cmap=None, title="", ax=None, show_xticks=False, show_yticks=False):
    """
    Plot the trend spatial pattern using Robinson projection with significance overlaid.

    Parameters:
    - trend_data: 2D numpy array with the trend values.
    - lats, lons: 1D arrays of latitudes and longitudes.
    - p_values: 2D array with p-values for each grid point.
    - GMST_p_values: 2D array with GMST p-values for each grid point.
    - title: Title for the plot.
    - ax: Existing axis to plot on. If None, a new axis will be created.
    - show_xticks, show_yticks: Boolean flags to show x and y axis ticks.
    
    Returns:
    - contour_obj: The contour object from the plot.
    """
    # Create a new figure/axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()

    # Determine significance mask (where p-values are less than 0.05)
    insignificance_mask = p_values >= 0.10
    # Plotting
    # contour_obj = ax.pcolormesh(lons, lats, trend_data,  cmap='RdBu_r',vmin=-5.0, vmax=5.0, transform=ccrs.PlateCarree(central_longitude=180), shading='auto')
    contour_obj = ax.contourf(lons, lats, trend_data, levels=levels, extend=extend, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=0))

    # Plot significance masks with different hatches
    ax.contourf(lons, lats, insignificance_mask, levels=[0.0, 0.10, 1.5],hatches=[None,'///'], colors='none', transform=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      colors='gray', alpha=0.5, linestyle='--', linewidth=0.5)

    # Disable labels on the top and right of the plot
    gl.top_labels = False
    gl.right_labels = False

    # Enable labels on the bottom and left of the plot
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    
    if show_xticks:
        gl.bottom_labels = True
    if show_yticks:
        gl.left_labels = True
    
    ax.set_title(title, loc='center', fontsize=18, pad=5.0)

    return contour_obj
# %%
lat = MPI_ESM_forced_ano['lat']
lon = MPI_ESM_forced_ano['lon']

trend_annual_data = slope * 10.0  # Convert to °C/decade
pvalue_annual_data = p_values

levels = np.arange(-0.6, 0.65, 0.05)
extend = 'both'
# Plot the trend patterns 
def plot_data(ax, run_number):
    
    trend_data = trend_annual_data.sel({'run': run_number}) 
    p_values = pvalue_annual_data.sel({'run': run_number})
    
    trend_data_with_cyclic, lon_cyclic = cutil.add_cyclic_point(trend_data, coord=lon)
    p_values_with_cyclic, _ = cutil.add_cyclic_point(p_values, coord=lon)
    
    ax.set_global()
    
    # Assuming plot_trend_with_significance is defined correctly and returns a contour object
    contour_obj = plot_trend_with_significance(trend_data_with_cyclic, lat, lon_cyclic,
                                               p_values_with_cyclic,
                                               levels=levels, extend=extend,
                                               cmap='twilight_shifted', title=" ",
                                               ax=ax, 
                                               show_xticks=False, 
                                               show_yticks=False)
    ax.text(0.5, 1.15, run_number, ha='center', va='center', transform=ax.transAxes)  # Adjusted to use transform
    ax.set_xticks([])
    ax.set_yticks([])
    return contour_obj

# Prepare for the loop
fig = plt.figure(figsize=(12, 15))
gs = gridspec.GridSpec(10, 5, wspace=0.05, hspace=0.05)

# Loop through and create subplots
for i, run in enumerate(MPI_ESM_forced_ano['run'].values):
    ax = fig.add_subplot(gs[i // 5, i % 5], projection=ccrs.Robinson(180))
    contour_obj=plot_data(ax, run)
    if i == 0:  # Arbitrarily choose the first plot's contour_obj for the colorbar
        contour_obj_for_cbar = contour_obj
        
# Assuming contour_obj is defined and consistent across plots
# Add horizontal colorbar
cbar_ax = fig.add_axes([0.25, 0.06, 0.6, 0.012])  # Adjust these values as needed
cbar = plt.colorbar(contour_obj, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.ax.tick_params(labelsize=14)
cbar.set_label('SAT Trend (°C/decade)', fontsize=16)

plt.tight_layout()
fig.savefig('MPI_ESM-MMEM-1993-2022-30year-forced-trendPatterns-sig90%.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
# close the scluster
client.close()
scluster.close()
# %%