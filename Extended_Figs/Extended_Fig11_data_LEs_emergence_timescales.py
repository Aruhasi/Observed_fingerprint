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
# input the large ensemble mean emergence timescale monotonicity 
dir_monotonicity = "/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Revised_SI_figures/monotonicity_check_LEs/data/"
MPI_ESM_monotonicity = xr.open_dataset(dir_monotonicity + "MPI_ESM_emergence_monotonicity.nc")
MIROC6_monotonicity = xr.open_dataset(dir_monotonicity + "MIROC6_emergence_monotonicity.nc")
ACCESS_monotonicity = xr.open_dataset(dir_monotonicity + "ACCESS_emergence_monotonicity.nc")
EC_Earth3_monotonicity = xr.open_dataset(dir_monotonicity + "EC_Earth_emergence_monotonicity.nc")
IPSL_monotonicity = xr.open_dataset(dir_monotonicity + "IPSL_emergence_monotonicity.nc")
CanESM5_monotonicity = xr.open_dataset(dir_monotonicity + "CanESM5_emergence_monotonicity.nc")
# %%
# define the function to detect the large ensemble mean emergence timescale monotonicity
# at each grid point less than 85% of the ensemble members agree on the monotonicity is considered non-monotonic
# this function returns a Boolean mask of shape (lat, lon) where True indicates that at least `threshold` fraction of members agree on monotonicity
# the mask is True if at least `threshold` fraction of members agree on monotonicity
# the mask is False if less than `threshold` fraction of members agree on monotonicity
def detect_monotonicity(ensemble_bool, threshold=0.85):
    """
    Given a Boolean DataArray ensemble_bool(member, lat, lon) where True indicates
    that member shows a monotonic emergence signal at that grid cell, return a 
    Boolean mask(monotonic_mask) of shape (lat, lon) where

      monotonic_mask[y,x] == True   if  fraction_of_members_that_are_True  >= threshold  
      monotonic_mask[y,x] == False  otherwise

    Parameters
    ----------
    ensemble_bool : xarray.DataArray
        dtype=bool, dims=('member','lat','lon').
    threshold : float
        Fraction (0–1) of members that must agree for a cell to count as monotonic.

    Returns
    -------
    monotonic_mask : xarray.DataArray (bool, dims=('lat','lon'))
    """
    # convert True/False → 1/0, then average over members  
    frac_agree = ensemble_bool.astype(float).mean(dim='run')
    # cells with at least `threshold` agreement
    monotonic_mask = frac_agree >= threshold

    print(f"Monotonicity mask: {monotonic_mask}")
    return monotonic_mask
# %%
# suppose `emergence_monotonic` is (member, lat, lon) Boolean
MIROC6_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(MIROC6_monotonicity['monotonicity'], threshold=0.8)
MPI_ESM_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(MPI_ESM_monotonicity['monotonicity'], threshold=0.85)
ACCESS_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(ACCESS_monotonicity['monotonicity'], threshold=0.8)
EC_Earth3_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(EC_Earth3_monotonicity['monotonicity'], threshold=0.85)
IPSL_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(IPSL_monotonicity['monotonicity'], threshold=0.8)
CanESM5_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(CanESM5_monotonicity['monotonicity'], threshold=0.85)
# %%
# if at least 85% of the six models agree on the monotonicity then the mask is True
MMEM_monotonicity = xr.concat([
    MIROC6_monotonicity['emergence_monotonicity_mask'],
    MPI_ESM_monotonicity['emergence_monotonicity_mask'],
    ACCESS_monotonicity['emergence_monotonicity_mask'],
    EC_Earth3_monotonicity['emergence_monotonicity_mask'],
    IPSL_monotonicity['emergence_monotonicity_mask'],
    CanESM5_monotonicity['emergence_monotonicity_mask']
], dim='run')
# %%
MMEM_monotonicity['emergence_monotonicity_mask'] = detect_monotonicity(MMEM_monotonicity, threshold=4/6)

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
# Change GridSpec to 5 rows instead of 4
gs = gridspec.GridSpec(5, 2, width_ratios=[1, 1])
gs.update(hspace=0.05, wspace=0.1)

# Define levels and colors
# levels = np.arange(10, 80, 5)
# cmap = plt.get_cmap("Spectral")
# colors = cmap(np.linspace(0, 1, len(levels) - 1))
# # colors = np.vstack([colors, [1, 1, 1, 1]])  # Add white for the 'over' bin
# custom_cmap = ListedColormap(colors)
levels = np.arange(10, 80, 5)  # 10 to 75
colors = plt.get_cmap('OrRd_r')(np.linspace(0, 1, len(levels) - 1))
custom_cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, ncolors=len(colors), extend='neither')

# MMLE subplot spans the first two rows
ax_mmem = fig.add_subplot(gs[0:2, :], projection=ccrs.Robinson(central_longitude=180))

ax_mmem.coastlines(resolution='110m')
ax_mmem.text(0.5, 1.05, "MMLE", transform=ax_mmem.transAxes, fontsize=35, ha='center')
ax_mmem.text(-0.05, 1.05, "a", transform=ax_mmem.transAxes, fontsize=32, fontweight='bold')  # Add "a" to top-left
im = MMEM.emergence_timescale_mean.plot(ax=ax_mmem, transform=ccrs.PlateCarree(),
                                        cmap=custom_cmap, levels=levels,
                                        add_colorbar=False)
MMEM_monotonicity_mask = MMEM_monotonicity['emergence_monotonicity_mask']
# Overlay the monotonicity mask as hatching (False = not monotonic) according to the ensemble member agreement
cf = ax_mmem.contourf(
        MMEM_monotonicity_mask.lon, MMEM_monotonicity_mask.lat,     # lon, lat coordinates
        ~MMEM_monotonicity_mask,               # invert mask: True where <85% agree
        levels=[0.5, 1.5],
        colors='none',
        hatches=['//'],
        transform=ccrs.PlateCarree(),
    )
for coll in cf.collections:
    coll.set_edgecolor('turquoise')
    coll.set_linewidth(0)
    coll.set_facecolor('none')  
# # Add individual model plots
# model_axes = [
#     fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180)),  # MIROC6
#     fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180)),  # MPI
#     fig.add_subplot(gs[2, 0], projection=ccrs.Robinson(central_longitude=180)),  # ACCESS
#     fig.add_subplot(gs[2, 1], projection=ccrs.Robinson(central_longitude=180)),  # EC-Earth3
#     fig.add_subplot(gs[3, 0], projection=ccrs.Robinson(central_longitude=180)),  # IPSL
#     fig.add_subplot(gs[3, 1], projection=ccrs.Robinson(central_longitude=180))   # CanESM5
# ]
# Update model subplots to occupy rows 2–5
model_axes = [
    fig.add_subplot(gs[2, 0], projection=ccrs.Robinson(central_longitude=180)),  # MIROC6
    fig.add_subplot(gs[2, 1], projection=ccrs.Robinson(central_longitude=180)),  # MPI
    fig.add_subplot(gs[3, 0], projection=ccrs.Robinson(central_longitude=180)),  # ACCESS
    fig.add_subplot(gs[3, 1], projection=ccrs.Robinson(central_longitude=180)),  # EC-Earth3
    fig.add_subplot(gs[4, 0], projection=ccrs.Robinson(central_longitude=180)),  # IPSL
    fig.add_subplot(gs[4, 1], projection=ccrs.Robinson(central_longitude=180))   # CanESM5
]

model_titles = [
    "MIROC6", "MPI-ESM1.2-LR", "ACCESS-ESM1.5",
    "EC-Earth3", "IPSL-CM6A-LR", "CanESM5"
]

model_data = [
    MIROC6_ENS, MPI_ENS, ACCESS_ENS,
    EC_Earth3, IPSL_ENS, CanESM5_ENS
]
monotonic_masks = {
    "MIROC6":    MIROC6_monotonicity['emergence_monotonicity_mask'],
    "MPI-ESM1.2-LR": MPI_ESM_monotonicity['emergence_monotonicity_mask'],
    "ACCESS-ESM1.5": ACCESS_monotonicity['emergence_monotonicity_mask'],
    "EC-Earth3": EC_Earth3_monotonicity['emergence_monotonicity_mask'],
    "IPSL-CM6A-LR": IPSL_monotonicity['emergence_monotonicity_mask'],
    "CanESM5":   CanESM5_monotonicity['emergence_monotonicity_mask'],
}

# Labels for top-left corners
subplot_labels = ["b", "c", "d", "e", "f", "g"]

# Add plots, titles, and labels for models
for ax, title, data, label in zip(model_axes, model_titles, model_data, subplot_labels):
    ax.coastlines(resolution='110m')
    # Add top-left corner labels (b-g)
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=28, fontweight='bold')  
    # Use ax.text for title placement at the top-center
    ax.text(0.5, 1.05, title, transform=ax.transAxes, fontsize=32, ha='center')
    data.emergence_timescale_mean.plot(ax=ax, transform=ccrs.PlateCarree(),
                                       cmap=custom_cmap, levels=levels,
                                       add_colorbar=False)
    # Overlay the monotonicity mask as hatching (False = not monotonic) according to the ensemble member aggreement
    # 95% of the ensemble members agree on the monotonicity then the mask is True
    mask = monotonic_masks[title]
    # Use hatching to indicate monotonicity
    # mask_cyc, lon_cyc = cutil.add_cyclic_point(mask, coord=mask.lon)
    cf_LENS = ax.contourf(
        mask.lon, mask.lat,     # lon, lat coordinates
        ~mask,               # invert mask: True where <85% agree
        levels=[0.5, 1.5],
        colors='none',
        hatches=['//'],
    transform=ccrs.PlateCarree(),
    )
    for coll in cf_LENS.collections:
        coll.set_edgecolor('turquoise')
        coll.set_linewidth(0)
        coll.set_facecolor('none') 
# set for the MMEM
for ax in [ax_mmem]:
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                      color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 26}
    gl.ylabel_style = {'size': 26}
# overlaid the hatching for MMEM according to the six model agreement:

# Add gridlines except for MMEM
model_axes = [ax for ax in model_axes if ax != ax_mmem]  # Exclude MMEM from gridlines
for ax in model_axes:
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

ax_mmem.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
            color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
ax_mmem.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='white', fontsize=32,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # Arctic
ax_mmem.plot(box_lons, box_lats, color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
ax_mmem.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='white', fontsize=32,
    transform=ccrs.PlateCarree(), ha='center', va='center')  # WH
ax_mmem.plot([200 % 360, 280 % 360, 280 % 360, 250 % 360, 200 % 360],
            [0, 0, -25, -25, 0], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
ax_mmem.text(sep_lon_mid, sep_lat_mid, 'SEP', color='white', fontsize=32,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SEP
ax_mmem.plot([180 % 360, 260 % 360, 260 % 360, 180 % 360, 180 % 360],
            [-55, -55, -70, -70, -55], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
ax_mmem.text(sop_lon_mid, sop_lat_mid, 'SOP', color='white', fontsize=32,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SOP
# exclude MMEM from the regional boxes
model_axes = [ax for ax in model_axes if ax != ax_mmem]  # Exclude MMEM from regional boxes
for ax in model_axes:
    ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
            color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(arctic_lon_mid, arctic_lat_mid, 'ARC', color='white', fontsize=28,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # Arctic
    ax.plot(box_lons, box_lats, color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(wh_lon_mid, wh_lat_mid, 'NAWH', color='white', fontsize=28,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # WH
    ax.plot([200 % 360, 280 % 360, 280 % 360, 250 % 360, 200 % 360],
            [0, 0, -25, -25, 0], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(sep_lon_mid, sep_lat_mid, 'SEP', color='white', fontsize=28,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SEP
    ax.plot([180 % 360, 260 % 360, 260 % 360, 180 % 360, 180 % 360],
            [-55, -55, -70, -70, -55], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='white', fontsize=28,
            transform=ccrs.PlateCarree(), ha='center', va='center')  # SOP

# Add colorbar
cbar_ax = fig.add_axes([0.25, 0.08, 0.55, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Emergence timescale with S/N > 1 (year)", fontsize=24, loc='center')
cbar.set_ticks(np.arange(10, 80, 5))
cbar.ax.tick_params(labelsize=20)
cbar.ax.tick_params(direction='out', length=8, width=2)

# add double legends to the colorbar: below the upper one denote the start year of the signal segments
# Adjust secondary axis slightly lower and flatter
cbar_ax_2 = fig.add_axes([0.25, 0.04, 0.55, 0.001])  # Adjusted position & height

# Set the ticks and labels 
cbar_ax_2.set_xlim(cbar.ax.get_xlim())
cbar_ax_2.set_xticks(levels)
cbar_ax_2.set_xticklabels([f"{2022 - tl + 1}" for tl in levels],
                          fontsize=20, rotation=45)

# Clean up y-axis and spines
cbar_ax_2.spines['top'].set_visible(False)
cbar_ax_2.spines['right'].set_visible(False)
cbar_ax_2.spines['left'].set_visible(False)
# cbar_ax_2.tick_params(axis='y', left=False, labelleft=False)
cbar_ax_2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
cbar_ax_2.tick_params(axis='x', direction='out', length=8, width=1.5)

# Label aligned with main colorbar
cbar_ax_2.set_xlabel("Start year of signal segment", fontsize=24, labelpad=2)  # Lower labelpad

# Save the figure
fig.savefig('Extended_Fig10_with_hatching.png', dpi=300, bbox_inches='tight')
fig.savefig('Extended_Fig10_with_hatching.pdf', dpi=300, bbox_inches='tight')
fig.savefig('Extended_Fig10_with_hatching.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()
# %%
