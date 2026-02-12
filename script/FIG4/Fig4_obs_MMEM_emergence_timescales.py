# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
# import all the plotting packages
from src.plot_func import *
# %%
# import importlib
# importlib.reload(src.plot_func)
# call once at the start of your script / notebook:
set_science_advances_style(column='double')  # or 'single'
# %%
dir_in = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/"
data_obs = xr.open_dataset(dir_in + "OBS_Emergence_time_scale.nc")
data_obs = data_obs.fillna(75)
# %%
# input monotonicity 
dir_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/monotonicity/'
monotonic_test = xr.open_dataset(dir_in+'obs_emergence_monotonicity.nc')
# %%
# rename the variable name
data_obs = data_obs.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})
# %%
# %%
dir_MMLE = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/"
MMEM = xr.open_dataset(dir_MMLE + "MMEM_emergence_timescale_mean.nc")
# %%
# check the NaN values
print(data_obs.emergence_timescale_mean.isnull().sum())
# print(MPI_ENS.emergence_timescale_mean.isnull().sum())
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
MMEM = MMEM.where(MMEM != -1)
diff = MMEM - data_obs
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
import matplotlib.colors as mcolors
import palettable
import cartopy.util as cutil

# Define levels and colors
# levels_obs = np.arange(10, 80, 5)
# cmap_obs = plt.get_cmap("Spectral")
# colors_obs = cmap_obs(np.linspace(0, 1, len(levels_obs) - 1))
# colors_obs = np.vstack([colors_obs, [1, 1, 1, 1]])  # Add grey for the 'over' bin
# custom_cmap_obs = ListedColormap(colors_obs)

# # Update BoundaryNorm
# norm_obs = BoundaryNorm(levels_obs, ncolors=len(levels_obs), extend="max")

# levels_obs = np.append(np.arange(10, 75, 5), 75)  # [10, 15, ..., 75, 80]
# cmap_base = plt.get_cmap('Spectral') #'OrRd_r'
# colors = cmap_base(np.linspace(0, 1, len(levels_obs) - 1))  # One less than number of edges
# colors = np.vstack([colors, [1, 1, 1, 1]])  # Add white at the end for >75 years
# custom_cmap = ListedColormap(colors)
# norm = BoundaryNorm(levels_obs, ncolors=len(levels_obs), extend='max')

# levels_obs = np.arange(10, 80, 5)  # 10 to 75OrRd_r
levels_obs = np.array([10, 15, 20, 25, 30, 35,
                       40, 45, 50, 55, 60, 65, 70, 73])

# colours for the *in-range* bins (10–15, …, 70–73)
base_cmap = plt.get_cmap('Spectral')
colors = base_cmap(np.linspace(0, 1, len(levels_obs)))
custom_cmap = ListedColormap(colors)

# set the "over" colour (values > 73) to white
custom_cmap.set_over('white')

# BoundaryNorm with an over-bin
norm = BoundaryNorm(levels_obs, ncolors=custom_cmap.N, extend='max')

# cmap = mcolors.ListedColormap(palettable.cmocean.sequential.Solar_17.mpl_colors)
# cmap = mcolors.ListedColormap(palettable.matplotlib.Plasma_17.mpl_colors)
# cmap = mcolors.ListedColormap(palettable.scientific.sequential.LaJolla_20.mpl_colors[::-1])
# cmap_diff=mcolors.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors[::-1])
# reverse the color map
# cmap_r = mcolors.ListedColormap(palettable.cmocean.sequential.Amp_17.mpl_colors[::-1])
# Create the figure
fig = plt.figure(figsize=(20, 15))
gs = plt.GridSpec(1, 2)

# Plot OBS data
ax = plt.subplot(gs[0], projection=ccrs.Robinson(central_longitude=180))
ax.coastlines(resolution='110m')
# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.25, linestyle='--',
                  color='gray', alpha=0.15)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = cticker.LongitudeFormatter()  # Longitude formatter
gl.yformatter = cticker.LatitudeFormatter()   # Latitude formatter
gl.xlabel_style = {'size': 18}
gl.ylabel_style = {'size': 18}
gl.bottom_labels = True
gl.left_labels = True

# Plot data
p_obs = data_obs['emergence_timescale_mean'].plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap,
    norm=norm,
    levels=levels_obs,
    add_colorbar=False
)
ax.set_title("A", loc='left', fontsize=28, pad=10,fontweight='bold')
ax.text(0.5, 1.05, 'OBS', transform=ax.transAxes, fontsize=22, ha='center', va='bottom')
# overlay the monotonicity check
# Add cyclic point to monotonic_map for plotting
monotonic_with_cyclic, lon_with_cyclic = cutil.add_cyclic_point(monotonic_test['monotonicity'].values, coord=monotonic_test.lon)

# Overlay the monotonicity mask as hatching (False = not monotonic)
ax.contourf(
    lon_with_cyclic, data_obs['emergence_timescale_mean'].lat, ~monotonic_with_cyclic,
    levels=[0.5, 1.5], hatches=['///'], colors='none', transform=ccrs.PlateCarree(), alpha=0
)
# Add colorbar
ticks_obs = levels_obs  # or a subset if you want fewer labels

cbar_ax_obs = fig.add_axes([0.16, 0.32, 0.3, 0.025])
cbar_obs = plt.colorbar(
    p_obs,
    cax=cbar_ax_obs,
    orientation='horizontal',
    ticks=ticks_obs,
    extend='max'         # <<< important: show the white ">" wedge
)
cbar_obs = plt.colorbar(p_obs, cax=cbar_ax_obs, orientation='horizontal', ticks=levels_obs)
cbar_obs.set_label('Emergence timescale with S/N > 2\n(year)', fontsize=20, loc='center')
cbar_obs.ax.tick_params(labelsize=18)   #, labelrotation=45
cbar_obs.ax.tick_params(direction='out', length=8, width=2)
cbar_obs.ax.tick_params(which='minor', bottom=False, top=False, length=0)
# add double legends to the colorbar: below the upper one denote the start year of the signal segments
# Adjust secondary axis slightly lower and flatter
cbar_ax_2 = fig.add_axes([0.16, 0.22, 0.3, 0.012])  # Adjusted position & height

# Set the ticks and labels
cbar_ax_2.set_xlim(cbar_obs.ax.get_xlim())
cbar_ax_2.set_xticks(levels_obs)
cbar_ax_2.set_xticklabels([f"{2022 - tl + 1}" for tl in levels_obs], 
                          fontsize=16, rotation=45)

# Clean up y-axis and spines
cbar_ax_2.spines['top'].set_visible(False)
cbar_ax_2.spines['right'].set_visible(False)
cbar_ax_2.spines['left'].set_visible(False)
# cbar_ax_2.tick_params(axis='y', left=False, labelleft=False)
cbar_ax_2.tick_params(which='minor', bottom=False, top=False, length=0)
cbar_ax_2.tick_params(which='major', bottom=True, top=False, length=0)
cbar_ax_2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
cbar_ax_2.tick_params(axis='x', direction='out', length=6, width=1.5)
# Label aligned with main colorbar
cbar_ax_2.set_xlabel("Start year of signal segment", fontsize=20, labelpad=2)  # Lower labelpad

# Plot DIFF data
ax1 = plt.subplot(gs[1], projection=ccrs.Robinson(central_longitude=180))
ax1.coastlines(resolution='110m')
# Add gridlines to ax1 (MPI-ESM)
gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.25, linestyle='--',
                    color='gray', alpha=0.15)
gl1.top_labels = False
gl1.right_labels = False
gl1.xformatter = cticker.LongitudeFormatter()  # Longitude formatter
gl1.yformatter = cticker.LatitudeFormatter()   # Latitude formatter
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.bottom_labels = True
gl1.left_labels = True

# Define asymmetric levels for the difference map
# levels_diff = np.concatenate([np.arange(-20, 0, 5), [0], np.arange(5, 55, 10)])  # Finer resolution on negative side
cmap_diff = "RdBu_r"
# # Use enough colors to cover bins + extensions when extend='both'
# norm_diff = BoundaryNorm(levels_diff, ncolors=len(levels_diff) + 1, extend='both')

# p_diff = diff['emergence_timescale_mean'].plot(
#         ax=ax1, 
#         transform=ccrs.PlateCarree(), 
#         cmap=cmap_diff, 
#         norm=norm_diff,
#         levels=levels_diff,
#         add_colorbar=False
# )
# Define levels and calculate number of color bins (12 bins from 13 boundaries)
# cmap_diff = mcolors.ListedColormap(palettable.cmocean.diverging.Curl_20.mpl_colors)  # Use a diverging colormap for differences
# cmap_diff = palettable.cmocean.diverging.Balance_20.mpl_colormap
levels_diff = np.arange(-35, 36, 5)  # Symmetric around 0
n_bins = len(levels_diff)-2         # = 10 bins

p_diff = diff['emergence_timescale_mean'].plot(ax=ax1, transform=ccrs.PlateCarree(), cmap=cmap_diff, levels=levels_diff, add_colorbar=False)
# Add title
ax1.set_title("B", loc='left', fontsize=28, pad=10, color='black',
              fontweight='bold')
ax1.text(0.5, 1.05, 'MMLE - OBS Difference', transform=ax1.transAxes, fontsize=22, ha='center', va='bottom')

# Add colorbar for DIFF
cbar_ax_diff = fig.add_axes([0.58, 0.32, 0.3, 0.025])
cbar_diff = plt.colorbar(p_diff, cax=cbar_ax_diff, orientation='horizontal', 
                         ticks=levels_diff, extend='min')
cbar_diff.set_label('Emergence timescale difference\n(year)', fontsize=20, loc='center')
# the label shows every 2 other tick
cbar_diff.ax.tick_params(labelsize=18, labelrotation=45)
cbar_diff.ax.tick_params(direction='out', length=8, width=2)  
cbar_diff.ax.tick_params(which='minor', bottom=False, top=False, length=0)

# add the unit to the colorbar, at the end of the colorbar
# cbar_diff.ax.text(1.05, -0.35, 'units:year', 
#                   va='top', ha='left', fontsize=18, transform=cbar_diff.ax.transAxes)
# --- secondary axis with "Speedy" / "Tardy" arrows ----------------
cbar_ax_diff2 = fig.add_axes([0.58, 0.22, 0.3, 0.012])
cbar_ax_diff2.set_xlim(cbar_diff.ax.get_xlim())

# no tick labels, just a clean line
cbar_ax_diff2.set_xticks([])
cbar_ax_diff2.set_yticks([])
for spine in ["top", "right", "left", "bottom"]:
    cbar_ax_diff2.spines[spine].set_visible(False)

# small baseline
cbar_ax_diff2.spines["bottom"].set_linewidth(1.0)

# draw arrows in axes coordinates (0–1)
# left arrow: speedy (negative difference → earlier emergence in MMLE)
cbar_ax_diff2.annotate(
        "", xy=(0.45, -1.25), xytext=(0.05, -1.25),
        xycoords="axes fraction",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="<|-", lw=2.0, mutation_scale=20)
)
cbar_ax_diff2.text(
        0.25, -1.22, "MMLE Speedy",
    transform=cbar_ax_diff2.transAxes,
    ha="center", va="bottom", fontsize=18, color='#3F60A9'
)

# right arrow: tardy (positive difference → later emergence in MMLE)
cbar_ax_diff2.annotate(
        "", xy=(0.95, -1.25), xytext=(0.55, -1.25),
        xycoords="axes fraction",
        annotation_clip=False,
        arrowprops=dict(arrowstyle="-|>", lw=2.5, mutation_scale=20, capstyle='round')
)
cbar_ax_diff2.text(
        0.75, -1.22, "MMLE Tardy",
    transform=cbar_ax_diff2.transAxes,
    ha="center", va="bottom", fontsize=18, color='#CE2826'
)

# add regional lines and labels
# Arctic box
arctic_lon_mid = (0 + 360) / 2
arctic_lat_mid = (66.5 + 90) / 2
ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
ax.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='white', fontsize=22,transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for Arctic
ax1.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
# ax1.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='gray', fontsize=22, fontweight='bold',transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for Arctic

# WH box
wh_lon_mid = (310 + 350) / 2
wh_lat_mid = (42 + 60) / 2
box_lons = np.array([310, 350, 350, 310, 310])
box_lats = np.array([42, 42, 60, 60, 42])
ax.plot(box_lons, box_lats, color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
ax.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='Black', fontsize=22,  transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for WH
ax1.plot(box_lons, box_lats, color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
# ax1.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='gray', fontsize=22, fontweight='bold',transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for WH

# Southeast Pacific box
sep_lon_mid = (200 + 320) / 2
sep_lat_mid = (0 + -25) / 2
ax.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
ax.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SEP
ax1.plot([200%360, 280%360, 280%360, 250%360, 200%360], [0, 0, -25, -25, 0],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
# ax1.text(sep_lon_mid, sep_lat_mid, 'SEP', color='black', fontsize=22, fontweight='bold', transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SEP
# # SO box 
# so_lon_mid = (0 + 360) / 2
# so_lat_mid = (-65 + -50) / 2
# ax.plot([0, 360, 360, 0, 0], [-70, -70, -50, -50, -70],
#         color='gray', linewidth=2.0, transform=ccrs.PlateCarree())
# ax.text(so_lon_mid, so_lat_mid, 'SO', color='black', fontsize=18, transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SO
# Extratropical South Pacific box
sop_lon_mid = (230 + 280) / 2
sop_lat_mid = (-40 + -70) / 2
ax.plot([230%360, 280%360, 280%360, 230%360, 230%360], [-40, -40, -70, -70, -40],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SOP
ax1.plot([230%360, 280%360, 280%360, 230%360, 230%360], [-40, -40, -70, -70, -40],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
# North Pacific box
npac_lon_mid = (175 + 220) / 2
npac_lat_mid = (30 + 50) / 2
ax.plot([175%360, 220%360, 220%360, 175%360, 175%360], [30, 30, 50, 50, 30],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
ax.text(npac_lon_mid, npac_lat_mid, 'NPM', color='black', fontsize=22, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for NPM 
ax1.plot([175%360, 220%360, 220%360, 175%360, 175%360], [30, 30, 50, 50, 30],
        color='gray', linewidth=2.8, transform=ccrs.PlateCarree())
# # Extratropical South Pacific box [depercated for SA_round1_revision]
# sop_lon_mid = (180 + 260) / 2
# sop_lat_mid = (-70 + -60) / 2
# ax.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
#         color='gray', linewidth=2.5, transform=ccrs.PlateCarree())
# ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SOP
# ax1.plot([180%360, 260%360, 260%360, 180%360, 180%360], [-55, -55, -70, -70, -55],
#         color='gray', linewidth=2.5, transform=ccrs.PlateCarree())
# ax1.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=22, fontweight='bold', transform=ccrs.PlateCarree(),
#         ha='center', va='center')  # Label for SOP
# Save the figure
figure_output = '/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG4/'
for ext in ("png", "pdf"):
    fig.savefig(figure_output+f"FIG4_SNgt2_mean_emergence_timescales_update_defined_regions.{ext}", dpi=300, bbox_inches='tight')

plt.show()
# %%
