#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# %%
MIROC6_ENS = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_MIROC6/output/MIROC6_emergence_timescale_mean.nc")
MIROC6_ENS = MIROC6_ENS.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})
# %%
MPI_ENS = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_MPI/output/MPI_emergence_timescale_mean.nc")
MPI_ENS = MPI_ENS.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})
# %%
ACCESS_ENS = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_ACCESS/output/ACCESS_emergence_timescale_mean.nc")
ACCESS_ENS = ACCESS_ENS.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})

EC_Earth3 = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_EC_Earth/output/EC_Earth_emergence_timescale_mean.nc")
EC_Earth3 = EC_Earth3.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})

IPSL_ENS = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_IPSL/output/IPSL_emergence_timescale_mean.nc")
IPSL_ENS = IPSL_ENS.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})

CanESM5_ENS = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_CanESM5/output/CanESM5_emergence_timescale_mean.nc")
CanESM5_ENS = CanESM5_ENS.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})

# %%
dir_out = "/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_model_evaluation/"
MMEM =  xr.open_dataset(dir_out + "MMEM_emergence_timescale_mean.nc")

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
# 7 subplots, 4 row, 2 columns; fisrt row is for MMEM only 
"""
subplot 1: MMEM in the first row, put in the center
subplot 2: MIROC6
subplot 3: MPI
subplot 4: ACCESS
subplot 5: EC_Earth3
subplot 6: IPSL
subplot 7: CanESM5
shared colorbar
# add gridlines to all subplots
"""
# Create the figure and subplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Define the figure and GridSpec
fig = plt.figure(figsize=(20, 30))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1])
gs.update(hspace=0.05, wspace=0.1)  # Adjust spacing

# Define levels and colors
levels = np.arange(10, 80, 5)
cmap = plt.get_cmap("Spectral")
colors = cmap(np.linspace(0, 1, len(levels) - 1))
# colors = np.vstack([colors, [1, 1, 1, 1]])  # Add white for the 'over' bin
custom_cmap = ListedColormap(colors)

# Centralize MMEM in the first row (spans across two columns)
ax_mmem = fig.add_subplot(gs[0, :], projection=ccrs.Robinson(central_longitude=180))
ax_mmem.coastlines(resolution='110m')
ax_mmem.text(0.5, 1.05, "MMEM", transform=ax_mmem.transAxes, fontsize=24, ha='center')
ax_mmem.text(-0.05, 1.05, "a", transform=ax_mmem.transAxes, fontsize=28, fontweight='bold')  # Add "a" to top-left
im = MMEM.emergence_timescale_mean.plot(ax=ax_mmem, transform=ccrs.PlateCarree(),
                                        cmap=custom_cmap, levels=levels,
                                        add_colorbar=False)

# Add individual model plots
model_axes = [
    fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180)),  # MIROC6
    fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180)),  # MPI
    fig.add_subplot(gs[2, 0], projection=ccrs.Robinson(central_longitude=180)),  # ACCESS
    fig.add_subplot(gs[2, 1], projection=ccrs.Robinson(central_longitude=180)),  # EC-Earth3
    fig.add_subplot(gs[3, 0], projection=ccrs.Robinson(central_longitude=180)),  # IPSL
    fig.add_subplot(gs[3, 1], projection=ccrs.Robinson(central_longitude=180))   # CanESM5
]

model_titles = [
    "MIROC6", "MPI-ESM1.2-LR", "ACCESS-ESM1.5",
    "EC-Earth3", "IPSL-CM6A-LR", "CanESM5"
]

model_data = [
    MIROC6_ENS, MPI_ENS, ACCESS_ENS,
    EC_Earth3, IPSL_ENS, CanESM5_ENS
]

# Labels for top-left corners
subplot_labels = ["b", "c", "d", "e", "f", "g"]

# Add plots, titles, and labels for models
for ax, title, data, label in zip(model_axes, model_titles, model_data, subplot_labels):
    ax.coastlines(resolution='110m')
    # Add top-left corner labels (b-g)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=28, fontweight='bold')  
    # Use ax.text for title placement at the top-center
    ax.text(0.5, 1.05, title, transform=ax.transAxes, fontsize=24, ha='center')
    data.emergence_timescale_mean.plot(ax=ax, transform=ccrs.PlateCarree(),
                                       cmap=custom_cmap, levels=levels,
                                       add_colorbar=False)

# Add gridlines to all axes
for ax in model_axes + [ax_mmem]:
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                      color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 20}
    gl.ylabel_style = {'size': 20}

# Add the regional boxes
arctic_lon_mid = (0 + 360) / 2
arctic_lat_mid = (66.5 + 90) / 2
wh_lon_mid = (310 + 350) / 2
wh_lat_mid = (42 + 60) / 2
box_lons = np.array([310, 350, 350, 310, 310])
box_lats = np.array([42, 42, 60, 60, 42])
sep_lon_mid = (200 + 320) / 2
sep_lat_mid = (0 + -25) / 2
sop_lon_mid = (180 + 260) / 2
sop_lat_mid = (-70 + -60) / 2

for ax in model_axes + [ax_mmem]:
    ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
            color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
    ax.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='white', fontsize=24,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # Arctic
    ax.plot(box_lons, box_lats, color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
    ax.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='grey', fontsize=24,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # WH
    ax.plot([200 % 360, 280 % 360, 280 % 360, 250 % 360, 200 % 360],
            [0, 0, -25, -25, 0], color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
    ax.text(sep_lon_mid, sep_lat_mid, 'SEP', color='white', fontsize=24,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SEP
    ax.plot([180 % 360, 260 % 360, 260 % 360, 180 % 360, 180 % 360],
            [-55, -55, -70, -70, -55], color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
    ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='grey', fontsize=24,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SOP

# Add colorbar
cbar_ax = fig.add_axes([0.25, 0.08, 0.55, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Emergence timescale with S/N > 1\n(year)", fontsize=24, loc='center')
cbar.set_ticks(np.arange(10, 80, 5))
cbar.ax.tick_params(labelsize=20)
cbar.ax.tick_params(direction='out', length=8, width=2)

# Save the figure
fig.savefig('Extended-Fig-9-simulated-emergence-timescales.png', dpi=300, bbox_inches='tight')
fig.savefig('Extended-Fig-9-simulated-emergence-timescales.pdf', dpi=300, bbox_inches='tight')
fig.savefig('Extended-Fig-9-simulated-emergence-timescales.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# %%