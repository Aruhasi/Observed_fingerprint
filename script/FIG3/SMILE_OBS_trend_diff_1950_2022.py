# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# In[1]:
# input the 30-year forced trend patterns of OBS and MMEM
dir_forced_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/cesm2_100/trend_forced_HadCRUT5_annual/'

HadCRUT5_trend = xr.open_dataset(dir_forced_input + 
                                 'forced_HadCRUT5_MMLE_MK_trend_1950-2022_sliding.nc')
# In[2]:
# input each LE trend patterns
MODELS = ['CanESM5', 'CESM2', 'IPSL_CM6A', 'EC_Earth3', 'ACCESS', 'MPI_ESM', 'MIROC6']

# pretty labels for plotting
MODEL_LABELS = {
    'CanESM5':        'CanESM5',
    'CESM2':          'CESM2',
    'IPSL_CM6A':      'IPSL-CM6A-LR',
    'EC_Earth3':      'EC-Earth3',
    'ACCESS':         'ACCESS-ESM1.5',
    'MPI_ESM':        'MPI-ESM1.2-LR',
    'MIROC6':         'MIROC6',
}
dir_LE_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}/SMILE_forced/'

LE_annual_trend = {}
for model in MODELS:
    LE_annual_trend[model] = xr.open_dataset(
        dir_LE_in.format(model=model) +
        f'{model}_ENSmean_forced_MK_trend_1950-2022_sliding.nc'
    )
# In[3]:
def cal_ratio(data,pattern_diff):
    data = pattern_diff/data
    return data
dir_output = "/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Pattern_diff/"
os.makedirs(dir_output, exist_ok=True)
# %%
dir_model_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_forced/'
# '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure6_Forced/data/Smiles_ensemble/'
MMLE_annual_trend = xr.open_dataset(dir_model_in + 
                                    'MMLE_ENSmean_forced_MK_trend_1950-2022_sliding.nc')
# In[6]:
# calculate the pattern difference between LE_ens and OBS
pattern_diff_LE = {}
for model in MODELS:
    pattern_diff_LE[model] = LE_annual_trend[model].trend - HadCRUT5_trend.trend
    print(f"{model} pattern diff min: {pattern_diff_LE[model].min().values}, "
          f"max: {pattern_diff_LE[model].max().values}")
    # save the data into netcdf file
    dir_output_LE = f"/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Pattern_diff/{model}/"
    os.makedirs(dir_output_LE, exist_ok=True)
    pattern_diff_LE[model].to_dataset(name='trend_diff').to_netcdf(
        dir_output_LE + f'{model}_OBS_forced_pattern_diff_1950_2022.nc'
    )
    # calculate the ratio
    pattern_ratio_LE = cal_ratio(HadCRUT5_trend.trend, pattern_diff_LE[model])
    pattern_ratio_LE.to_dataset(name='trend_ratio').to_netcdf(
        dir_output_LE + f'{model}_OBS_forced_pattern_ratio_1950_2022.nc'
    )
    mean_ratio_LE = pattern_ratio_LE.sel(period="1979-2022").mean().values*100
    print(f"{model} mean ratio (%): {mean_ratio_LE}")
# %%

# In[4]:
# calculate the pattern difference between OBS and MMLE
pattern_diff = MMLE_annual_trend.trend - HadCRUT5_trend.trend
print(pattern_diff)
# %%
pattern_ratio = cal_ratio(HadCRUT5_trend.trend, pattern_diff)
print(pattern_ratio)
# %%
pattern_diff.to_dataset(name='trend_diff').to_netcdf(dir_output + 'MMLE_OBS_forced_pattern_diff_1950_2022.nc')
pattern_ratio.to_dataset(name='trend_ratio').to_netcdf(dir_output + 'MMLE_OBS_forced_pattern_ratio_1950_2022.nc')
# %%
print(pattern_ratio.min().values)
# calculate the global mean values
mean_ratio = pattern_ratio.sel(period="1979-2022").mean().values*100
# In[5]:
# Input the Observational internal trend (wrt MMEM GSAT)
dir_internal_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std/'
HadCRUT5_internal_trend = xr.open_dataset(dir_internal_input +'OBS_ICV_MK_trend_STD_L44.nc')['icv_trend_std'].squeeze()
# In[6]:
dir_model_internal = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_internal/'
MMEM_internal_trend = xr.open_dataset(dir_model_internal + 'MMLE_internal_trend_std_1950-2022_sliding.nc')['trend']
# %%
ICV_diff = MMEM_internal_trend - HadCRUT5_internal_trend
print(ICV_diff)
# %%
Ratio_ICV = cal_ratio(HadCRUT5_internal_trend, ICV_diff)
print(Ratio_ICV)
# %%
# save the data into netcdf file
# dir_output = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Plot_output/"
ICV_diff.to_dataset(name='ICV_trend_std_diff').to_netcdf(dir_output + 'MMLE_OBS_ICV_pattern_diff_1950_2022.nc')
Ratio_ICV.to_dataset(name='ICV_trend_std_ratio').to_netcdf(dir_output + 'MMLE_OBS_ICV_pattern_ratio_1950_2022.nc')
# %%
# calculate the global mean values
mean_ratio_ICV = Ratio_ICV.sel(period="1979-2022").mean().values*100
# %%
print(f"Forced_diff min: {pattern_diff.min()}, max: {pattern_diff.max()}")
print(f"ICV_diff min: {ICV_diff.min()}, max: {ICV_diff.max()}")
# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.colors as mcolors
import palettable
#  cmap = mcolors.ListedColormap(palettable.scientific.diverging.Vik_20.mpl_colors)
cmap=mcolors.ListedColormap(palettable.cmocean.diverging.Curl_20.mpl_colors)  # Reverse the colormap
# %%
print("Cmap base color count:", len(palettable.cmocean.diverging.Curl_20.mpl_colors))
# cmap = 'seismic_r'
# from palettable.colorbrewer.cmocean import Balance_20
# from palettable.colorbrewer.diverging import 
# from palettable.colorbrewer.cartocolors.diverging import Geyser_7
# %%
# Set up the figure with a 2x2 layout
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from matplotlib.colors import BoundaryNorm
# %%
# Create figure with two map panels (1 row × 2 cols)
fig, (ax_forced_map, ax_icv_map) = plt.subplots(
    1, 2,
    figsize=(15, 10),
    subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180))
)

# -------------------- Panel c: Forced Minus MMEM Map --------------------
# Coastlines and gridlines
ax_forced_map.coastlines(resolution='110m')
gl = ax_forced_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])

# Colormap levels    
levels_forced = np.arange(-0.5, 0.55, 0.05)
norm_forced = BoundaryNorm(boundaries=levels_forced, ncolors=len(levels_forced)-1)

# Plot
p_forced = pattern_diff.sel(period="1979-2022").plot(
    ax=ax_forced_map,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=norm_forced,
    levels=levels_forced,
    add_colorbar=False
)

# Annotations
ax_forced_map.text(-0.02, 1.2, "a", transform=ax_forced_map.transAxes,
                   fontsize=24, fontweight='bold', va='top')
ax_forced_map.set_title("MMLE - OBS Difference pattern\n(1979-2022)",
                        fontsize=20, pad=10, loc='center')
ax_forced_map.text(0.95, 1.05, f"{mean_ratio:.0f}%",
                   transform=ax_forced_map.transAxes,
                   fontsize=20, ha='center', va='center')

# Colorbar
cbar_ax = fig.add_axes([0.12, 0.25, 0.3, 0.02])
cbar = plt.colorbar(
    p_forced, cax=cbar_ax, orientation='horizontal',
    ticks=[-0.5, -0.25, 0, 0.25, 0.5], extend='neither'
)
cbar.set_label("Externally forced SAT differences\n(°C per decade)",
               fontsize=20, labelpad=10, loc='center')
cbar.ax.tick_params(labelsize=20, direction='out', length=10, width=2)

# -------------------- Panel d: ICV Minus MMEM Map --------------------
ax_icv_map.coastlines(resolution='110m')
gl1 = ax_icv_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = True
gl1.left_labels = True
gl1.xformatter = cticker.LongitudeFormatter()
gl1.yformatter = cticker.LatitudeFormatter()
gl1.xlabel_style = {'size': 12}
gl1.ylabel_style = {'size': 12}
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])

levels_icv = np.arange(-0.25, 0.275, 0.025)
norm_icv = BoundaryNorm(boundaries=levels_icv, ncolors=len(levels_icv)-1)

p_icv = ICV_diff.sel(period="1979-2022").plot(
    ax=ax_icv_map,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=norm_icv,
    levels=levels_icv,
    add_colorbar=False
)

ax_icv_map.text(-0.02, 1.2, "b", transform=ax_icv_map.transAxes,
                fontsize=24, fontweight='bold', va='top')
ax_icv_map.set_title("MMLE - OBS Difference pattern\n(1979-2022)",
                     fontsize=20, pad=10, loc='center')
ax_icv_map.text(0.95, 1.05, f"{mean_ratio_ICV:.0f}%",
                transform=ax_icv_map.transAxes,
                fontsize=20, ha='center', va='center')

cbar_ax2 = fig.add_axes([0.61, 0.25, 0.3, 0.02])
cbar2 = plt.colorbar(
    p_icv, cax=cbar_ax2, orientation='horizontal',
    ticks=[-0.25, -0.125, 0, 0.125, 0.25], extend='neither'
)
cbar2.set_label("Internal variability SAT differences\n(°C per decade)",
                fontsize=20, labelpad=10, loc='center')
cbar2.ax.tick_params(labelsize=20, direction='out', length=10, width=2)

# Final layout and save
plt.tight_layout()
fig.savefig('1979-2022-MMLE-OBS-difference.png', dpi=300, bbox_inches='tight')
fig.savefig('1979-2022-MMLE-OBS-difference.pdf', dpi=300, bbox_inches='tight')
plt.show()
# In[]:
# plot the forced pattern of OBS and MMLE
# Create figure with two map panels (1 row × 2 cols)
fig, (ax_forced_map, ax_icv_map) = plt.subplots(
    1, 2,
    figsize=(25, 15),
    subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180))
)

# -------------------- Panel c: Forced Minus MMEM Map --------------------
# Coastlines and gridlines
ax_forced_map.coastlines(resolution='110m')
gl = ax_forced_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()
gl.xlabel_style = {'size': 22}
gl.ylabel_style = {'size': 22}
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])

# Colormap levels    
levels= np.arange(-1.0, 1.1, 0.1)
norm = BoundaryNorm(boundaries=levels, ncolors=len(levels)-1)

OBS_forced = HadCRUT5_trend['trend']*10.0
# Plot
p_forced = OBS_forced.plot(
    ax=ax_forced_map,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=norm,
    levels=levels,
    add_colorbar=False
)

# Annotations
ax_forced_map.text(-0.02, 1.2, "a", transform=ax_forced_map.transAxes,
                   fontsize=34, fontweight='bold', va='top')
ax_forced_map.set_title("OBS externally forced pattern\n(1980-2022)",
                        fontsize=28, pad=10, loc='center')
ax_forced_map.text(0.95, 1.05, f"{mean_ratio:.0f}%",
                   transform=ax_forced_map.transAxes,
                   fontsize=28, ha='center', va='center')

# Colorbar
cbar_ax = fig.add_axes([0.12, 0.25, 0.3, 0.02])
cbar = plt.colorbar(
    p_forced, cax=cbar_ax, orientation='horizontal',
    ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0],
    extend='neither'
)
cbar.set_label("°C per decade",
               fontsize=24, labelpad=10, loc='center')
cbar.ax.tick_params(labelsize=22, direction='out', length=10, width=2)

# -------------------- Panel d: ICV Minus MMEM Map --------------------
ax_icv_map.coastlines(resolution='110m')
gl1 = ax_icv_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = True
gl1.left_labels = True
gl1.xformatter = cticker.LongitudeFormatter()
gl1.yformatter = cticker.LatitudeFormatter()
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])

# levels_icv = np.arange(-0.25, 0.275, 0.025)
# norm_icv = BoundaryNorm(boundaries=levels_icv, ncolors=len(levels_icv)-1)
MMLE_forced = MMEM_annual_trend['trend']*10.0
p_icv = MMLE_forced.plot(
    ax=ax_icv_map,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=norm,
    levels=levels,
    add_colorbar=False
)

ax_icv_map.text(-0.02, 1.2, "b", transform=ax_icv_map.transAxes,
                fontsize=34, fontweight='bold', va='top')
ax_icv_map.set_title("MMLE externally forced pattern\n(1980-2022)",
                     fontsize=28, pad=10, loc='center')
ax_icv_map.text(0.95, 1.05, f"{mean_ratio_ICV:.0f}%",
                transform=ax_icv_map.transAxes,
                fontsize=28, ha='center', va='center')

cbar_ax2 = fig.add_axes([0.61, 0.25, 0.3, 0.02])
cbar2 = plt.colorbar(
    p_icv, cax=cbar_ax2, orientation='horizontal',
    ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], extend='neither'
)
cbar2.set_label("°C per decade",
                fontsize=24, labelpad=10, loc='center')
cbar2.ax.tick_params(labelsize=22, direction='out', length=10, width=2)

# Final layout and save
plt.tight_layout()
fig.savefig('1980-2022-MMLE-OBS-forced.png', dpi=300, bbox_inches='tight')
fig.savefig('1980-2022-MMLE-OBS-forced.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
intervals = np.arange(0.0, 1.05, 0.05)
cmap = mcolors.ListedColormap(palettable.cmocean.sequential.Amp_20.mpl_colors)
extend = 'max'

# plot the ICV pattern of OBS and MMEM: std
fig, (ax_forced_map, ax_icv_map) = plt.subplots(
    1, 2,
    figsize=(25, 15),
    subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180))
)

# -------------------- Panel c: Forced Minus MMEM Map --------------------
# Coastlines and gridlines
ax_forced_map.coastlines(resolution='110m')
gl = ax_forced_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()
gl.xlabel_style = {'size': 22}
gl.ylabel_style = {'size': 22}
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])


# Plot
p_forced = HadCRUT5_internal_trend['trend'].plot(
    ax=ax_forced_map,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    levels=intervals,
    add_colorbar=False,
    extend=extend
)

# Annotations
ax_forced_map.text(-0.02, 1.2, "a", transform=ax_forced_map.transAxes,
                   fontsize=34, fontweight='bold', va='top')
ax_forced_map.set_title("OBS internal variability\n(1980-2022)",
                        fontsize=28, pad=10, loc='center')
ax_forced_map.text(0.95, 1.05, f"{mean_ratio:.0f}%",
                   transform=ax_forced_map.transAxes,
                   fontsize=28, ha='center', va='center')

# Colorbar
cbar_ax = fig.add_axes([0.12, 0.25, 0.3, 0.02])
cbar = plt.colorbar(
    p_forced, cax=cbar_ax, orientation='horizontal',
    ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    extend='neither'
)
cbar.set_label("SAT trend Stddev.(°C per decade)",
               fontsize=24, labelpad=10, loc='center')
cbar.ax.tick_params(labelsize=22, direction='out', length=10, width=2)

# -------------------- Panel d: ICV Minus MMEM Map --------------------
ax_icv_map.coastlines(resolution='110m')
gl1 = ax_icv_map.gridlines(
    draw_labels=True, linestyle='--', color='gray', alpha=0.5
)
gl1.top_labels = False
gl1.right_labels = False
gl1.bottom_labels = True
gl1.left_labels = True
gl1.xformatter = cticker.LongitudeFormatter()
gl1.yformatter = cticker.LatitudeFormatter()
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])

# levels_icv = np.arange(-0.25, 0.275, 0.025)
# norm_icv = BoundaryNorm(boundaries=levels_icv, ncolors=len(levels_icv)-1)
p_icv = MMEM_internal_trend.plot(
    ax=ax_icv_map,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    levels=intervals,
    add_colorbar=False,
    extend=extend
)

ax_icv_map.text(-0.02, 1.2, "b", transform=ax_icv_map.transAxes,
                fontsize=34, fontweight='bold', va='top')
ax_icv_map.set_title("MMLE internal variability\n(1980-2022)",
                     fontsize=28, pad=10, loc='center')
ax_icv_map.text(0.95, 1.05, f"{mean_ratio_ICV:.0f}%",
                transform=ax_icv_map.transAxes,
                fontsize=28, ha='center', va='center')

cbar_ax2 = fig.add_axes([0.61, 0.25, 0.3, 0.02])
cbar2 = plt.colorbar(
    p_icv, cax=cbar_ax2, orientation='horizontal',
    ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], extend='neither'
)
cbar2.set_label("SAT trend Stddev.(°C per decade)",
                fontsize=24, labelpad=10, loc='center')
cbar2.ax.tick_params(labelsize=22, direction='out', length=10, width=2)

# Final layout and save
plt.tight_layout()
fig.savefig('1980-2022-MMLE-OBS-ICV.png', dpi=300, bbox_inches='tight')
fig.savefig('1980-2022-MMLE-OBS-ICV.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
