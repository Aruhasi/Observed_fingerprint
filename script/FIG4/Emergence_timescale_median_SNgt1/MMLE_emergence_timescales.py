#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
# %%
# dir_in = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/"
# data_obs = xr.open_dataset(dir_in + "OBS_Emergence_time_scale.nc")
# data_obs = data_obs.fillna(75)
data_obs = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure3/data/fig3_final/fig3_final.nc")
data_obs = data_obs.fillna(75)
# %%
# # rename the variable name
data_obs = data_obs.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})
# %%
dir_in = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/median_SNgt1/"
MIROC6_ENS = xr.open_dataset(dir_in + "MIROC6_emergence_timescale_median.nc")
CESM2_ENS = xr.open_dataset(dir_in + "CESM2_emergence_timescale_median.nc")
MPI_ENS = xr.open_dataset(dir_in + "MPI_ESM_emergence_timescale_median.nc")
ACCESS_ENS = xr.open_dataset(dir_in + "ACCESS_emergence_timescale_median.nc")
EC_Earth3 = xr.open_dataset(dir_in + "EC_Earth3_emergence_timescale_median.nc")
IPSL_ENS = xr.open_dataset(dir_in + "IPSL_CM6A_emergence_timescale_median.nc")
CanESM5_ENS = xr.open_dataset(dir_in + "CanESM5_emergence_timescale_median.nc")
# %%
# Calculate the MMEM emergence timescale (mean of all models)
# assume each ds has a variable 'emergence_timescale_median'
models = {
    "MIROC6": MIROC6_ENS["emergence_timescale_median"],
    "CESM2": CESM2_ENS["emergence_timescale_median"],
    "MPI-ESM": MPI_ENS["emergence_timescale_median"],
    "ACCESS": ACCESS_ENS["emergence_timescale_median"],
    "EC-Earth3": EC_Earth3["emergence_timescale_median"],
    "IPSL": IPSL_ENS["emergence_timescale_median"],
    "CanESM5": CanESM5_ENS["emergence_timescale_median"],
}

# stack into one DataArray with a 'model' dimension
stack = xr.concat(list(models.values()), dim="model")
stack = stack.assign_coords(model=list(models.keys()))

# (A) multi-model **median** (AR6-like for ToE)
MMEM_median = stack.median("model")
# %%
dir_out = dir_in
MMEM_median.to_dataset(name="emergence_timescale_median").to_netcdf(dir_out + "MMEM_emergence_timescale_median.nc")
# %%
# check the NaN values
print(data_obs.emergence_timescale_mean.isnull().sum())
print(MPI_ENS.emergence_timescale_mean.isnull().sum())
# %%
# def skip_NaN_minus(data, data_minus):
#   data_minus = data_minus.where(data != -1)
#   return data_minus
# %%
# data_obs = skip_NaN_minus(data_obs.emergence_timescale_mean, data_obs.emergence_timescale_mean)

# %%
# print(data_obs.dims, data_obs.shape)
# print(diff.dims, diff.shape)
# %%
# check the data values
# print(diff.emergence_timescale_mean)
# save the difference
# diff.to_netcdf("diff_emergence_timescale_obs_MPIESM.nc")
# %%
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

# %%
# two subplots, 1 row, 2 columns
# Replace invalid values with NaN for consistent masking
data_obs = data_obs.where(data_obs != -1)
MMEM = MMEM_median.where(MMEM_median != -1)
diff = data_obs - MMEM
diff = diff.where(diff.notnull())  # Mask invalid data

# # Define levels and normalization for OBS
# levels_obs = np.arange(10, 85, 5)  # Extend to 85 to allow the upper bound
# cmap_obs = plt.get_cmap("Spectral")
# colors_obs = cmap_obs(np.linspace(0, 1, len(levels_obs) - 1))
# custom_cmap_obs = ListedColormap(colors_obs)
# norm_obs = BoundaryNorm(levels_obs, ncolors=len(colors_obs), extend="max")

# Define levels and normalization for DIFF
# levels_diff = np.arange(-30, 40, 5)  # Extend to 40 for the upper bound
# cmap_diff = plt.get_cmap("RdBu")
# colors_diff = cmap_diff(np.linspace(0, 1, len(levels_diff) - 1))
# custom_cmap_diff = ListedColormap(colors_diff)
# norm_diff = BoundaryNorm(levels_diff, ncolors=len(colors_diff), extend="both")
# %%
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True

import matplotlib.colors as mcolors
import palettable
cmap = mcolors.ListedColormap(palettable.cmocean.sequential.Solar_17.mpl_colors)
# cmap = mcolors.ListedColormap(palettable.matplotlib.Plasma_17.mpl_colors)
# cmap = mcolors.ListedColormap(palettable.scientific.sequential.LaJolla_20.mpl_colors[::-1])
cmap_diff=mcolors.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors[::-1])
# reverse the color map
cmap_r = mcolors.ListedColormap(palettable.cmocean.sequential.Amp_17.mpl_colors[::-1])
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
levels_obs = np.arange(10, 80, 5)
cmap_obs = plt.get_cmap("Spectral")
colors_obs = cmap_obs(np.linspace(0, 1, len(levels_obs) - 1))
colors_obs = np.vstack([colors_obs, [1, 1, 1, 1]])  # Add grey for the 'over' bin
custom_cmap_obs = ListedColormap(colors_obs)

# Update BoundaryNorm
norm_obs = BoundaryNorm(levels_obs, ncolors=len(levels_obs), extend="max")

# Plot data
p_obs = data_obs['emergence_timescale_mean'].plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap_obs,
    norm=norm_obs,
    levels=levels_obs,
    add_colorbar=False
)
ax.set_title("a", loc='left', fontsize=28, pad=10,fontweight='bold')

# Add colorbar
# Add colorbar for OBS
cbar_ax_obs = fig.add_axes([0.16, 0.32, 0.3, 0.025])  # Adjusted position
cbar_obs = plt.colorbar(
    p_obs,
    cax=cbar_ax_obs,
    orientation="horizontal",
    ticks=levels_obs,
    extend="max"
)
cbar_obs = plt.colorbar(p_obs, cax=cbar_ax_obs, orientation='horizontal', ticks=levels_obs)
cbar_obs.set_label('Emergence timescale with S/N > 1\n(year)', fontsize=20, loc='center')
cbar_obs.ax.tick_params(labelsize=18)
cbar_obs.ax.tick_params(direction='out', length=8, width=2)
# add the unit to the colorbar, at the end of the colorbar
# cbar_obs.ax.text(1.05, -0.35, 
#                  'units:year', va='top', ha='left', fontsize=18, transform=cbar_obs.ax.transAxes)
# cbar_ax_obs = fig.add_axes([0.16, 0.3, 0.3, 0.025])  # Adjusted position
# cbar_obs = plt.colorbar(p_obs, cax=cbar_ax_obs, orientation='horizontal', ticks=levels_obs)
# cbar_obs.set_label('Emergence timescale (years) with S/N > 1', fontsize=22)
# cbar_obs.ax.tick_params(labelsize=18)
# cbar_obs.cmap.set_over('grey')  # Set the largest value to grey

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
levels_diff = np.arange(-25, 30, 5)  # Extend to 40 for the upper bound
cmap_diff = plt.get_cmap(cmap_diff)
colors_diff = cmap_diff(np.linspace(0, 1, len(levels_diff)+1))

# Add extra colors for extensions
# colors_diff = np.vstack([colors_diff, [1, 1, 1, 1]])  # Add gray (min) and grey (max)
custom_cmap_diff = ListedColormap(colors_diff)

# Ensure BoundaryNorm matches the extended levels
norm_diff = BoundaryNorm(levels_diff, ncolors=len(colors_diff), extend="both")

p_diff = diff['emergence_timescale_mean'].plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=custom_cmap_diff,
                   norm=norm_diff, levels=levels_diff, add_colorbar=False)
# Add title
ax1.set_title("b", loc='left', fontsize=28, pad=10, color='black',
              fontweight='bold')

# Add colorbar for DIFF
cbar_ax_diff = fig.add_axes([0.58, 0.32, 0.3, 0.025])
cbar_diff = plt.colorbar(p_diff, cax=cbar_ax_diff, orientation='horizontal', ticks=levels_diff)
cbar_diff.set_label('Emergence timescale. Obs minus MMEM\n(year)', fontsize=20, loc='center')
cbar_diff.ax.tick_params(labelsize=18)
cbar_diff.ax.tick_params(direction='out', length=8, width=2)  
# add the unit to the colorbar, at the end of the colorbar
# cbar_diff.ax.text(1.05, -0.35, 'units:year', 
#                   va='top', ha='left', fontsize=18, transform=cbar_diff.ax.transAxes)

# add regional lines and labels
# Arctic box
arctic_lon_mid = (0 + 360) / 2
arctic_lat_mid = (66.5 + 90) / 2
ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
ax.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='white', fontsize=22,transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for Arctic
ax1.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
# ax1.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='grey', fontsize=22, fontweight='bold',transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for Arctic

# WH box
wh_lon_mid = (310 + 350) / 2
wh_lat_mid = (42 + 60) / 2
box_lons = np.array([310, 350, 350, 310, 310])
box_lats = np.array([42, 42, 60, 60, 42])
ax.plot(box_lons, box_lats, color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
ax.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='Black', fontsize=22,  transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for WH
ax1.plot(box_lons, box_lats, color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
# ax1.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='grey', fontsize=22, fontweight='bold',transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for WH

# Southeast Pacific box
sep_lon_mid = (200 + 320) / 2
sep_lat_mid = (0 + -25) / 2
ax.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
ax.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SEP
ax1.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
# ax1.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, fontweight='bold', transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SEP

# Extratropical South Pacific box
sop_lon_mid = (180 + 260) / 2
sop_lat_mid = (-70 + -60) / 2
ax.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SOP
ax1.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
        color='grey', linewidth=2.5, transform=ccrs.PlateCarree())
# ax1.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, fontweight='bold', transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SOP
# Save the figure
fig.savefig('OBS_MMLE_SN_gt_1.png', dpi=300, bbox_inches='tight')
fig.savefig('OBS_MMLE_SN_gt_1.pdf', dpi=300, bbox_inches='tight')
fig.savefig('OBS_MMLE_SN_gt_1.eps', dpi=300, bbox_inches='tight')
plt.show()
# %%