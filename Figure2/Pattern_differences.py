# In[0]:
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

# In[1]:
# input the 30-year forced trend patterns of OBS and MMEM
dir_forced_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure1/data_revision/'

HadCRUT5_trend = xr.open_dataset(dir_forced_input + 
                                 'HadCRUT5_annual_forced_30yr_trend.nc')
# In[2]:
dir_model_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure2_Forced/data/Smiles_ensemble/'
MMEM_annual_trend = xr.open_dataset(dir_model_in + 
                                    'MMEM_annual_forced_30yr_trend.nc')
# rename the variable name
MMEM_annual_trend = MMEM_annual_trend.rename({'__xarray_dataarray_variable__': 'trend'})
HadCRUT5_trend  = HadCRUT5_trend.rename({'__xarray_dataarray_variable__': 'trend'})
# %%
# Obs_trend_da = HadCRUT5_trend.trend*10.0
# Model_trend_da = MMEM_annual_trend.trend*10.0

# In[3]:
# calculate the pattern difference between OBS and MMEM
pattern_diff = HadCRUT5_trend.trend - MMEM_annual_trend.trend
# pattern_diff = pattern_diff.rename({'__xarray_dataarray_variable__': 'trend'})
# In[4]:
def cal_ratio(data,pattern_diff):
    data = pattern_diff/data
    return data
# %%
pattern_ratio = cal_ratio(HadCRUT5_trend.trend, pattern_diff)
print(pattern_ratio)
# %%
print(pattern_ratio.min().values)
# calculate the global mean values
mean_ratio = pattern_ratio.mean().values*100
# In[5]:
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
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap

"""
subplot1: plot the pattern difference
subplot2: plot the ratio of pattern difference

"""
# Create the figure
fig = plt.figure(figsize=(20, 15))
gs = plt.GridSpec(1, 2)

# Plot OBS data
ax = plt.subplot(gs[0], projection=ccrs.Robinson(central_longitude=180))
ax.coastlines(resolution='110m')
# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, linestyle='--',
                  color='gray', alpha=0.35)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = cticker.LongitudeFormatter()  # Longitude formatter
gl.yformatter = cticker.LatitudeFormatter()   # Latitude formatter
gl.xlabel_style = {'size': 18}
gl.ylabel_style = {'size': 18}
gl.bottom_labels = True
gl.left_labels = True

# Define levels and colors
levels_obs = np.arange(-0.6, 0.7, 0.1)
cmap_obs = plt.get_cmap("RdBu_r")#"twilight_shifted"
colors_obs = cmap_obs(np.linspace(0, 1, len(levels_obs)+1))
# colors_obs = np.vstack([colors_obs, [1, 1, 1, 1]])  # Add white for the 'over' bin
custom_cmap_obs = ListedColormap(colors_obs)
# Update BoundaryNorm
norm_obs = BoundaryNorm(levels_obs, ncolors=len(levels_obs)+1, extend="both")

# Plot data
p_obs = pattern_diff.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap_obs,
    norm=norm_obs,
    levels=levels_obs,
    add_colorbar=False
)
ax.set_title("c", loc='left', fontsize=28, pad=10,fontweight='bold')

# Add colorbar
# Add colorbar for OBS
cbar_ax_obs = fig.add_axes([0.16, 0.3, 0.3, 0.025])  # Adjusted position
cbar_obs = plt.colorbar(
    p_obs,
    cax=cbar_ax_obs,
    orientation="horizontal",
    ticks=levels_obs,
    extend="max"
)
cbar_obs = plt.colorbar(p_obs, cax=cbar_ax_obs, orientation='horizontal', ticks=levels_obs)
cbar_obs.set_label('Emergence timescale (years) with S/N > 1', fontsize=22)
cbar_obs.ax.tick_params(labelsize=18)

# cbar_ax_obs = fig.add_axes([0.16, 0.3, 0.3, 0.025])  # Adjusted position
# cbar_obs = plt.colorbar(p_obs, cax=cbar_ax_obs, orientation='horizontal', ticks=levels_obs)
# cbar_obs.set_label('Emergence timescale (years) with S/N > 1', fontsize=22)
# cbar_obs.ax.tick_params(labelsize=18)
# cbar_obs.cmap.set_over('white')  # Set the largest value to white

# Plot DIFF data
ax1 = plt.subplot(gs[1], projection=ccrs.Robinson(central_longitude=180))
ax1.coastlines(resolution='110m')
# Add gridlines to ax1 (MPI-ESM)
gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, linestyle='--',
                    color='gray', alpha=0.35)
gl1.top_labels = False
gl1.right_labels = False
gl1.xformatter = cticker.LongitudeFormatter()  # Longitude formatter
gl1.yformatter = cticker.LatitudeFormatter()   # Latitude formatter
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.bottom_labels = True
gl1.left_labels = True

# Define levels and normalization for DIFF
levels_diff = np.arange(-25, 27.5, 2.5)  # Extend to 40 for the upper bound
cmap_diff = plt.get_cmap("RdBu_r")
colors_diff = cmap_diff(np.linspace(0, 1, len(levels_diff)+1))
# Add extra colors for extensions
# colors_diff = np.vstack([colors_diff, [1, 1, 1, 1]])  # Add gray (min) and white (max)
custom_cmap_diff = ListedColormap(colors_diff)
# Ensure BoundaryNorm matches the extended levels
norm_diff = BoundaryNorm(levels_diff, ncolors=len(colors_diff) + 1, extend="both")

p_diff = pattern_diff.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=custom_cmap_diff,
                   norm=norm_diff, levels=levels_diff, add_colorbar=False)
# Add title
ax1.set_title("d", loc='left', fontsize=28, pad=10, color='black',
              fontweight='bold')

# Add colorbar for DIFF
cbar_ax_diff = fig.add_axes([0.58, 0.3, 0.3, 0.025])
cbar_diff = plt.colorbar(p_diff, cax=cbar_ax_diff, orientation='horizontal', ticks=levels_diff)
cbar_diff.set_label('Explainable ratio in OBS', fontsize=22)
cbar_diff.ax.tick_params(labelsize=18)

# add regional lines and labels
# Arctic box
arctic_lon_mid = (0 + 360) / 2
arctic_lat_mid = (66.5 + 90) / 2
ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for Arctic
ax1.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax1.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for Arctic

# WH box
wh_lon_mid = (310 + 350) / 2
wh_lat_mid = (42 + 60) / 2
box_lons = np.array([310, 350, 350, 310, 310])
box_lats = np.array([42, 42, 60, 60, 42])
ax.plot(box_lons, box_lats, color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for WH
ax1.plot(box_lons, box_lats, color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax1.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for WH

# Southeast Pacific box
sep_lon_mid = (200 + 320) / 2
sep_lat_mid = (0 + -25) / 2
ax.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SEP
ax1.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax1.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SEP

# Extratropical South Pacific box
sop_lon_mid = (180 + 260) / 2
sop_lat_mid = (-70 + -60) / 2
ax.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SOP
ax1.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax1.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SOP
# Save the figure
fig.savefig('OBS_minus_MMEM_human_forced_pattern_Ratio.png', dpi=300, bbox_inches='tight')
fig.savefig('OBS_minus_MMEM_human_forced_pattern_Ratio.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
