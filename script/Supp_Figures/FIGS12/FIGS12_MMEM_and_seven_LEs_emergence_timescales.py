#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
# %%
# ============================================================
# 0. USER OPTIONS
# ============================================================

case = "median"          # "mean" or "median"
scen = "CESM2_CMIP6"   # just used in figure title / filenames

# output directory
FIG_OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG_SI_LEs_emergence_SNgt1/"

# ============================================================
# 1. HELPERS
# ============================================================
def load_LE_emergence(case="mean"):
    """
    Load emergence timescale fields for all 7 LEs and build an MMLE field
    as the multi-model median of 'emergence_timescale_mean'.

    Returns
    -------
    models : dict
        {model_name: xr.Dataset(...)} each with variable 'emergence_timescale_mean'
    MMEM  : xr.Dataset
        dataset with variable 'emergence_timescale_mean' giving MMLE median pattern
    """
    assert case in ("mean", "median")
    suffix = "mean" if case == "mean" else "median"
    base = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/{suffix}_SNgt1/"

    # --- open raw datasets ---
    MIROC6_ENS    = xr.open_dataset(base + f"MIROC6_emergence_timescale_{suffix}.nc")
    MPI_ESM_ENS   = xr.open_dataset(base + f"MPI_ESM_emergence_timescale_{suffix}.nc")
    ACCESS_ENS    = xr.open_dataset(base + f"ACCESS_emergence_timescale_{suffix}.nc")
    EC_Earth3_ENS = xr.open_dataset(base + f"EC_Earth3_emergence_timescale_{suffix}.nc")
    IPSL_CM6A_ENS = xr.open_dataset(base + f"IPSL_CM6A_emergence_timescale_{suffix}.nc")
    CESM2_ENS     = xr.open_dataset(base + f"CESM2_emergence_timescale_{suffix}.nc")
    CanESM5_ENS   = xr.open_dataset(base + f"CanESM5_emergence_timescale_{suffix}.nc")

    models = {
        "MIROC6":        MIROC6_ENS,
        "MPI-ESM1.2-LR": MPI_ESM_ENS,
        "ACCESS-ESM1.5": ACCESS_ENS,
        "EC-Earth3":     EC_Earth3_ENS,
        "IPSL-CM6A-LR":  IPSL_CM6A_ENS,
        "CESM2":         CESM2_ENS,
        "CanESM5":       CanESM5_ENS,
    }

    # --- ensure each dataset has a variable named 'emergence_timescale_mean' ---
    for model_name in list(models.keys()):
        ds = models[model_name]

        if "emergence_timescale_median" in ds.data_vars:
            # already fine
            models[model_name] = ds
            continue

        # if the NetCDF came from xarray.to_netcdf on a DataArray, the var name
        # is often '__xarray_dataarray_variable__'
        if "__xarray_dataarray_variable__" in ds.data_vars:
            ds = ds.rename({"__xarray_dataarray_variable__": "emergence_timescale_median"})
            print(f"[load_LE_emergence] Renamed '__xarray_dataarray_variable__' -> 'emergence_timescale_median' in {model_name}")
        else:
            # fall back: if there is exactly one data variable, rename that
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1:
                old_name = data_vars[0]
                ds = ds.rename({old_name: "emergence_timescale_median"})
                print(f"[load_LE_emergence] Renamed '{old_name}' -> 'emergence_timescale_median' in {model_name}")
            else:
                raise KeyError(
                    f"In model {model_name}, could not find 'emergence_timescale_median' "
                    f"and there are multiple data variables: {list(ds.data_vars)}"
                )

        # *** IMPORTANT: store the renamed dataset back into the dict ***
        models[model_name] = ds

    # --- MMLE = multi-model median of the per-model fields ------------
    stack = xr.concat(
        [ds["emergence_timescale_median"] for ds in models.values()],
        dim="model",
    )
    stack = stack.assign_coords(model=list(models.keys()))
    MMEM_field = stack.median("model")   # median across models
    MMEM = MMEM_field.to_dataset(name="emergence_timescale_median")

    return models, MMEM
# %%
def detect_monotonicity(ensemble_bool, threshold=0.85):
    """
    Given Boolean DataArray ensemble_bool(run, lat, lon) where True indicates
    that member shows a monotonic emergence signal at that grid point,
    return Boolean mask(lat, lon) where
        True  if fraction_of_members_that_are_True >= threshold
        False otherwise
    """
    frac_agree = ensemble_bool.astype(float).mean(dim="run")
    monotonic_mask = frac_agree >= threshold
    return monotonic_mask
# %%
# ============================================================
# 2. LOAD EMERGENCE TIMESCALES (MEAN OR MEDIAN) + MMLE
# ============================================================
models_dict, MMEM = load_LE_emergence(case=case)
MIROC6_ENS    = models_dict["MIROC6"]
MPI_ESM_ENS   = models_dict["MPI-ESM1.2-LR"]
ACCESS_ENS    = models_dict["ACCESS-ESM1.5"]
EC_Earth3     = models_dict["EC-Earth3"]
IPSL_CM6A_ENS = models_dict["IPSL-CM6A-LR"]
CESM2_ENS     = models_dict["CESM2"]
CanESM5_ENS   = models_dict["CanESM5"]
# %%
# ============================================================
# 3. LOAD MONOTONICITY FIELDS (ALWAYS FROM MEAN DIR HERE)
# ============================================================
dir_monot = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG4/monotonic_SNgt1/"

MPI_ESM_monotonicity   = xr.open_dataset(dir_monot + "MPI_ESM_emergence_timescale_monotonicity.nc")
MIROC6_monotonicity    = xr.open_dataset(dir_monot + "MIROC6_emergence_timescale_monotonicity.nc")
ACCESS_monotonicity    = xr.open_dataset(dir_monot + "ACCESS_emergence_timescale_monotonicity.nc")
EC_Earth3_monotonicity = xr.open_dataset(dir_monot + "EC_Earth3_emergence_timescale_monotonicity.nc")
IPSL_CM6A_monotonicity = xr.open_dataset(dir_monot + "IPSL_CM6A_emergence_timescale_monotonicity.nc")
CESM2_monotonicity     = xr.open_dataset(dir_monot + "CESM2_emergence_timescale_monotonicity.nc")
CanESM5_monotonicity   = xr.open_dataset(dir_monot + "CanESM5_emergence_timescale_monotonicity.nc")

# per-model masks (note: you used slightly different thresholds per model, kept here)
MIROC6_monotonicity["emergence_monotonicity_mask"]    = detect_monotonicity(MIROC6_monotonicity["monotonicity"],    threshold=0.80)
MPI_ESM_monotonicity["emergence_monotonicity_mask"]   = detect_monotonicity(MPI_ESM_monotonicity["monotonicity"],   threshold=0.85)
ACCESS_monotonicity["emergence_monotonicity_mask"]    = detect_monotonicity(ACCESS_monotonicity["monotonicity"],    threshold=0.80)
EC_Earth3_monotonicity["emergence_monotonicity_mask"] = detect_monotonicity(EC_Earth3_monotonicity["monotonicity"], threshold=0.85)
IPSL_CM6A_monotonicity["emergence_monotonicity_mask"] = detect_monotonicity(IPSL_CM6A_monotonicity["monotonicity"], threshold=0.80)
CESM2_monotonicity["emergence_monotonicity_mask"]     = detect_monotonicity(CESM2_monotonicity["monotonicity"],     threshold=0.85)
CanESM5_monotonicity["emergence_monotonicity_mask"]   = detect_monotonicity(CanESM5_monotonicity["monotonicity"],   threshold=0.85)

# ============================================================
# 4. MMLE MONOTONICITY (AGREEMENT ACROSS MODELS)
# ============================================================

MMEM_monotonicity_stack = xr.concat(
    [
        MIROC6_monotonicity["emergence_monotonicity_mask"].squeeze(),
        MPI_ESM_monotonicity["emergence_monotonicity_mask"].squeeze(),
        ACCESS_monotonicity["emergence_monotonicity_mask"].squeeze(),
        EC_Earth3_monotonicity["emergence_monotonicity_mask"].squeeze(),
        IPSL_CM6A_monotonicity["emergence_monotonicity_mask"].squeeze(),
        CESM2_monotonicity["emergence_monotonicity_mask"].squeeze(),
        CanESM5_monotonicity["emergence_monotonicity_mask"].squeeze(),
    ],
    dim="run",
    coords="minimal",
)

MMEM_monotonicity_mask = detect_monotonicity(MMEM_monotonicity_stack, threshold=4/7)
# %%
# Calculate the pattern correlation between OBS and MMEM
data_obs = xr.open_dataset("/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure3/data/fig3_final/fig3_final.nc")
data_obs = data_obs.fillna(75)  # fill NaN with 75 for consistent masking
# input monotonicity 
dir_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Revised_main_figures/Figure4_Emergence_timescale/data/'
monotonic_test = xr.open_dataset(dir_in+'obs_emergence_monotonicity.nc')
# rename the variable name
data_obs = data_obs.rename_vars({'__xarray_dataarray_variable__': 'emergence_timescale_mean'})

# P_correlation function: stats.pearsonr cannot handle NaN values
def pattern_correlation(da1, da2):
    """
    Calculate the pattern correlation between two xarray DataArrays,
    ignoring NaN values.

    Parameters
    ----------
    da1 : xarray.DataArray
        First data array.
    da2 : xarray.DataArray
        Second data array.

    Returns
    -------
    float
        Pattern correlation coefficient.
    """
    # Flatten the DataArrays and drop NaN values
    valid_mask = da1.notnull() & da2.notnull()
    flat_da1 = da1.where(valid_mask).values.flatten()
    flat_da2 = da2.where(valid_mask).values.flatten()

    # Remove NaN values
    flat_da1 = flat_da1[~np.isnan(flat_da1)]
    flat_da2 = flat_da2[~np.isnan(flat_da2)]

    # Calculate the Pearson correlation coefficient
    if len(flat_da1) == 0 or len(flat_da2) == 0:
        return np.nan  # Return NaN if there are no valid data points

    correlation_matrix = np.corrcoef(flat_da1, flat_da2)
    return correlation_matrix[0, 1]
# calculate pattern correlation between OBS and each model/MMEM
model_pattern_data = {
    "MMEM": MMEM["emergence_timescale_median"],
    "MIROC6": MIROC6_ENS["emergence_timescale_median"],
    "MPI-ESM1.2-LR": MPI_ESM_ENS["emergence_timescale_median"],
    "ACCESS-ESM1.5": ACCESS_ENS["emergence_timescale_median"],
    "EC-Earth3": EC_Earth3["emergence_timescale_median"],
    "IPSL-CM6A-LR": IPSL_CM6A_ENS["emergence_timescale_median"],
    "CESM2": CESM2_ENS["emergence_timescale_median"],
    "CanESM5": CanESM5_ENS["emergence_timescale_median"],
}

from src.Statistic_cal import pattern_corr_da

corr_da = {}
for model_name, model_data in model_pattern_data.items():
    # align obs and model data
    aligned_obs, aligned_model = xr.align(data_obs["emergence_timescale_mean"], model_data, join="inner")
    
    corr = pattern_corr_da(aligned_model, aligned_obs)
    print(f"Pattern correlation between OBS and {model_name}: {corr:.3f}")
    corr_da[model_name] = corr
# %%
# ============================================================
# 5. PLOTTING: MMLE + 7 LES WITH DOUBLE COLORBAR
# ============================================================
# color levels for emergence timescale
# levels_obs = np.arange(10, 80, 5)  # 10 to 75OrRd_r
levels = np.array([10, 15, 20, 25, 30, 35,
                       40, 45, 50, 55, 60, 65, 70, 73])

# colours for the *in-range* bins (10–15, …, 70–73)
base_cmap = plt.get_cmap('Spectral')
colors = base_cmap(np.linspace(0, 1, len(levels)))
custom_cmap = ListedColormap(colors)
# levels = np.arange(10, 80, 5)   # 10–75
# colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(levels) - 1))
# custom_cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, ncolors=len(colors), extend="neither")

# figure + layout
fig = plt.figure(figsize=(28, 20))
gs = gridspec.GridSpec(4, 3, wspace=0.125, hspace=0.01, width_ratios=[1, 1, 1])

# --- MMLE panel centered in first row --------------------------------
ax_mmem = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax_empty1 = fig.add_subplot(gs[0, 0]); ax_empty1.axis("off")
ax_empty2 = fig.add_subplot(gs[0, 2]); ax_empty2.axis("off")

ax_mmem.coastlines(resolution="110m")
ax_mmem.text(0.25, 1.0, "MMLE", transform=ax_mmem.transAxes,
             fontsize=26, ha="center", va="bottom")
ax_mmem.text(-0.05, 1.0, "A", transform=ax_mmem.transAxes,
             fontsize=26, fontweight="bold", ha="right", va="bottom")

im = MMEM["emergence_timescale_median"].plot(
    ax=ax_mmem,
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap,
    levels=levels,
    add_colorbar=False,
)

# overlay MMLE monotonicity hatch (False = non-monotonic)
cf = ax_mmem.contourf(
    MMEM_monotonicity_mask.lon,
    MMEM_monotonicity_mask.lat,
    ~MMEM_monotonicity_mask,
    levels=[0.5, 1.5],
    colors="none",
    hatches=["//"],
    transform=ccrs.PlateCarree(),
)
for coll in cf.collections:
    coll.set_edgecolor("gray")
    coll.set_linewidth(0)
    coll.set_facecolor("none")

# MMLE gridlines
gl = ax_mmem.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
    color="gray", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = False
gl.left_labels = True
gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()
gl.xlabel_style = {"size": 18}
gl.ylabel_style = {"size": 18}

# ============================================================
# 5.1 individual model panels
# ============================================================
model_axes = [
    fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180)),  # MIROC6
    fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180)),  # MPI
    fig.add_subplot(gs[1, 2], projection=ccrs.Robinson(central_longitude=180)),  # ACCESS
    fig.add_subplot(gs[2, 0], projection=ccrs.Robinson(central_longitude=180)),  # EC-Earth3
    fig.add_subplot(gs[2, 1], projection=ccrs.Robinson(central_longitude=180)),  # IPSL
    fig.add_subplot(gs[2, 2], projection=ccrs.Robinson(central_longitude=180)),  # CESM2
    fig.add_subplot(gs[3, 1], projection=ccrs.Robinson(central_longitude=180)),  # CanESM5
]

ax_empty3 = fig.add_subplot(gs[3, 0]); ax_empty3.axis("off")
ax_empty4 = fig.add_subplot(gs[3, 2]); ax_empty4.axis("off")

model_titles = [
    "MIROC6", "MPI-ESM1.2-LR", "ACCESS-ESM1.5",
    "EC-Earth3", "IPSL-CM6A-LR", "CESM2", "CanESM5",
]
model_data = [
    MIROC6_ENS, MPI_ESM_ENS, ACCESS_ENS,
    EC_Earth3, IPSL_CM6A_ENS, CESM2_ENS, CanESM5_ENS,
]
monotonic_masks = {
    "MIROC6":        MIROC6_monotonicity["emergence_monotonicity_mask"],
    "MPI-ESM1.2-LR": MPI_ESM_monotonicity["emergence_monotonicity_mask"],
    "ACCESS-ESM1.5": ACCESS_monotonicity["emergence_monotonicity_mask"],
    "EC-Earth3":     EC_Earth3_monotonicity["emergence_monotonicity_mask"],
    "IPSL-CM6A-LR":  IPSL_CM6A_monotonicity["emergence_monotonicity_mask"],
    "CESM2":         CESM2_monotonicity["emergence_monotonicity_mask"],
    "CanESM5":       CanESM5_monotonicity["emergence_monotonicity_mask"],
}
subplot_labels = ["B", "C", "D", "E", "F", "G", "H"]

for idx, (ax, title, data, label) in enumerate(zip(model_axes, model_titles, model_data, subplot_labels)):
    ax.coastlines(resolution="110m")
    ax.text(-0.05, 1.0, label, transform=ax.transAxes,
            fontsize=28, fontweight="bold", ha="right", va="bottom")
    ax.text(0.25, 1.0, title, transform=ax.transAxes,
            fontsize=28, ha="center", va="bottom")

    data["emergence_timescale_median"].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        levels=levels,
        add_colorbar=False,
    )

    mask = monotonic_masks[title]
    cf_LENS = ax.contourf(
        mask.lon, mask.lat,
        ~mask,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["//"],
        transform=ccrs.PlateCarree(),
    )
    for coll in cf_LENS.collections:
        coll.set_edgecolor("gray")
        coll.set_linewidth(0)
        coll.set_facecolor("none")

    # Determine if this is a rightmost or bottom panel
    is_rightmost = (idx in [1, 2, 4, 5])  # panels at columns 2 and last row center (index 6)
    is_bottom = (idx in [3, 5, 6])  # bottom row panels
    
    glm = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
        color="gray", alpha=0.5, linestyle="--"
    )
    glm.top_labels = False
    glm.right_labels = False
    glm.bottom_labels = is_bottom
    glm.left_labels = not is_rightmost
    glm.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    glm.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    glm.xformatter = cticker.LongitudeFormatter()
    glm.yformatter = cticker.LatitudeFormatter()
    glm.xlabel_style = {"size": 20}
    glm.ylabel_style = {"size": 20}

# ============================================================
# 5.2 regional boxes (ARC, NAWH, SEP, SOP)
# ============================================================
arctic_lon_mid = (0 + 360) / 2
arctic_lat_mid = (66.5 + 90) / 2
wh_lon_mid = (310 + 350) / 2
wh_lat_mid = (42 + 60) / 2
box_lons = np.array([310, 350, 350, 310, 310])
box_lats = np.array([42, 42, 60, 60, 42])
sep_lon_mid = (200 + 320) / 2
sep_lat_mid = (0 - 25) / 2
# sop_lon_mid = (180 + 260) / 2
# sop_lat_mid = (-70 - 60) / 2
sop_lon_mid = (180 + 260) / 2
sop_lat_mid = (-55 + -70) / 2
npac_lon_mid = (175 + 220) / 2
npac_lat_mid = (30 + 50) / 2

def draw_boxes(ax, text_size):
    ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
            color="gray", linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(arctic_lon_mid, arctic_lat_mid, "ARC", color="black", fontsize=text_size,
            transform=ccrs.PlateCarree(), ha="center", va="center")
    ax.plot(box_lons, box_lats, color="gray", linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(wh_lon_mid, wh_lat_mid, "NAWH", color="black", fontsize=text_size,
            transform=ccrs.PlateCarree(), ha="center", va="center")
    ax.plot([200 % 360, 280 % 360, 280 % 360, 250 % 360, 200 % 360],
            [0, 0, -25, -25, 0], color="gray", linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(sep_lon_mid, sep_lat_mid, "SEP", color="black", fontsize=text_size,
            transform=ccrs.PlateCarree(), ha="center", va="center")
    # SOP box
    # update: 230%360, 280%360, 280%360, 230%360, 230%360/-40, -40, -70, -70, -40
    
    ax.plot([180%360, 260%360, 260%360, 180%360, 180%360], 
            [-55, -55, -70, -70, -55], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    ax.text(sop_lon_mid, sop_lat_mid, "SOP", color="black", fontsize=text_size,
            transform=ccrs.PlateCarree(), ha="center", va="center")
    # ax.plot([175%360, 220%360, 220%360, 175%360, 175%360], [30, 30, 50, 50, 30],
    #     color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    # ax.text(npac_lon_mid, npac_lat_mid, 'NPM', color='black', fontsize=22, transform=ccrs.PlateCarree(),
    #     ha='center', va='center')
def draw_boxes_without_text(ax):
    ax.plot([0, 360, 360, 0, 0], [66.5, 66.5, 90, 90, 66.5],
            color="gray", linewidth=3.0, transform=ccrs.PlateCarree())

    ax.plot(box_lons, box_lats, color="gray", linewidth=3.0, transform=ccrs.PlateCarree())
    
    ax.plot([200 % 360, 280 % 360, 280 % 360, 250 % 360, 200 % 360],
            [0, 0, -25, -25, 0], color="gray", linewidth=3.0, transform=ccrs.PlateCarree())
   
    # SOP box
    ax.plot([180%360, 260%360, 260%360, 180%360, 180%360], 
            [-55, -55, -70, -70, -55], color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
    
    # ax.plot([175%360, 220%360, 220%360, 175%360, 175%360], [30, 30, 50, 50, 30],
    #     color='gray', linewidth=3.0, transform=ccrs.PlateCarree())
   
draw_boxes(ax_mmem, 28)
for ax in model_axes:
    draw_boxes_without_text(ax)

# ============================================================
# 5.3 shared double colorbar (timescale & start year)
# ============================================================
cbar_ax = fig.add_axes([0.25, 0.08, 0.55, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Emergence timescale with S/N > 1 (year)", fontsize=24, loc="center")
cbar.set_ticks(levels)
cbar.ax.tick_params(labelsize=20, direction="out", length=8, width=2)

# second axis: start year = 2022 - timescale + 1
cbar_ax2 = fig.add_axes([0.25, 0.02, 0.55, 0.001])
cbar_ax2.set_xlim(cbar.ax.get_xlim())
cbar_ax2.set_xticks(levels)
cbar_ax2.set_xticklabels([f"{2022 - L + 1}" for L in levels],
                         fontsize=20, rotation=45)
for spine in ("top", "right", "left"):
    cbar_ax2.spines[spine].set_visible(False)
cbar_ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
cbar_ax2.tick_params(axis="x", direction="out", length=8, width=1.5)
cbar_ax2.set_xlabel("Start year of signal segment", fontsize=24, labelpad=2)

# add correlation text box to each panel
textstr = f"P_corr wrt OBS: {corr_da['MMEM']:.2f}"
# props = dict(boxstyle='round', facecolor='none', alpha=0.8)
ax_mmem.text(0.6, 1.08, textstr, transform=ax_mmem.transAxes, fontsize=18,
             verticalalignment='top', ) #bbox=props
for ax, title in zip(model_axes, model_titles):
    corr_value = corr_da[title]
    textstr = f"P_corr wrt OBS: {corr_value:.2f}"
    ax.text(0.6, 1.08, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', ) #bbox=props

# ============================================================
# 6. TITLE + SAVE
# ============================================================
case_label = "ensemble mean" if case == "mean" else "ensemble median"
# fig.suptitle(f"Emergence timescales ({case_label}, scenario: {scen})",
#              y=0.995, fontsize=22)

os.makedirs(FIG_OUT_DIR, exist_ok=True)
tag = f"{scen}_{case}"
for ext in ("png", "pdf"):
    fname = f"FIGURE_S11_emergence_timescale_LEs_SNgt1_{tag}.{ext}"
    fig.savefig(os.path.join(FIG_OUT_DIR, fname),
                dpi=300, bbox_inches="tight")

plt.show()
# %%