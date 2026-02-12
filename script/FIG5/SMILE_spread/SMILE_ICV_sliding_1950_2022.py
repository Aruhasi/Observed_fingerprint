# 
"""
Concatenate ICV residual patterns for all models: run dims
"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
import glob
import xarray as xr
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from src.Statistic_cal import mmle_mean_model_band_from_xarray

# %% 
MODELS = ['CanESM5', 'CESM2', 'IPSL_CM6A', 'EC_Earth3', 'ACCESS', 'MPI_ESM', 'MIROC6']
# MODELS = ['IPSL_CM6A']
# MODELS = ['ACCESS']

END_YEAR = 2022
START_MIN = 1950
TAU_MIN = 10
TAU_MAX = END_YEAR - START_MIN + 1  # 73
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)
key_regions = ['Arctic', 'Subpolar_gyre','SoutheastPacific','SOP', 'NPI', 'SO', 'SOP_original']
# %%
combined_by_model = {}

for model in MODELS:
    print(f"Loading model: {model}", flush=True)
    DIR_IN = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/{model}/regional_anomalies"
    files = sorted(glob.glob(f"{DIR_IN}/{model}_run*_regional_mean_trends_1950_2022_sliding.nc"))

    ds_list = [xr.open_dataset(f) for f in files]
    combined_ds = xr.concat(ds_list, dim="run")  # dims: run, period, region
    combined_by_model[model] = combined_ds 
# %%
# mean of seven models percentile
MMLE_icv_band = mmle_mean_model_band_from_xarray(
       combined_by_model,
       region_list=key_regions,
       qlo=0.05, qhi=0.95,# equal-model: use min available across models for each (region,period)
       # seed=None,
       varname="trend_region"
       )
# %%
out_nc = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/MMLE_avg_of_7LEs_5th_95th_quantile_1950_2022_sliding.nc"
MMLE_icv_band.to_netcdf(out_nc)
print("Saved:", out_nc)
# %%
variable_name = np.arange(1950,2014)
keys = [f'{i}-2022' for i in variable_name]
region_name = ['Arctic', 'Subpolar_gyre','SoutheastPacific','SOP']
# load data
dir_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/'
dir_SOP_trend = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/SOP_update/'
dir_input_percentile = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/percentile/'
dir_SOP_percentile = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/SOP_update/percentile/'
# %%
data_array = {}
for region in region_name:
    # if region == 'SOP':
    #     data_array[region] = {}
    #     data_array[region] = xr.open_dataset(f'{dir_SOP_trend}{region}_update_snr_gt_2_trend_variations.nc')
    # else:
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_input}{region}_trend_variations.nc')
# %%
# put the forced and unforced data into the same dictionary
da_Arctic  = xr.Dataset(
    {'raw':data_array['Arctic'].raw, 
    'forced': data_array['Arctic'].forced, 
    'unforced': data_array['Arctic'].internal})
# %%
da_NorthAtlantic    = xr.Dataset({'raw':data_array['Subpolar_gyre'].raw, 'forced': data_array['Subpolar_gyre'].forced, 'unforced': data_array['Subpolar_gyre'].internal})
da_SoutheastPacific = xr.Dataset({'raw':data_array['SoutheastPacific'].raw, 'forced': data_array['SoutheastPacific'].forced, 'unforced': data_array['SoutheastPacific'].internal})
da_SOP              = xr.Dataset({'raw':data_array['SOP'].raw, 'forced': data_array['SOP'].forced, 'unforced': data_array['SOP'].internal})
# %%
arctic_unforced_lower = MMLE_icv_band["q_low"].sel(region='Arctic')
NAWH_unforced_lower   = MMLE_icv_band["q_low"].sel(region='Subpolar_gyre')
SEP_unforced_lower    = MMLE_icv_band["q_low"].sel(region='SoutheastPacific')
SOP_unforced_lower    = MMLE_icv_band["q_low"].sel(region='SOP_original')

arctic_unforced_upper = MMLE_icv_band["q_high"].sel(region='Arctic')
NAWH_unforced_upper   = MMLE_icv_band["q_high"].sel(region='Subpolar_gyre')
SEP_unforced_upper    = MMLE_icv_band["q_high"].sel(region='SoutheastPacific')
SOP_unforced_upper    = MMLE_icv_band["q_high"].sel(region='SOP_original')
# %%
# input the MMLE forced trend data
dir_MMLE_forced = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/MMLE/'
# load the MMLE forced trend data
MMLE_forced_ARC_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_ARC_trend_1950-2022_sliding.nc').trend
MMLE_forced_NAWH_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_subpolar_gyre_trend_1950-2022_sliding.nc').trend
MMLE_forced_SEP_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_SoutheastPacific_trend_1950-2022_sliding.nc').trend
MMLE_forced_SOP_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_SOP_original_trend_1950-2022_sliding.nc').trend
# %%
"""
Plot to check ICV bands against obs trends
"""
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['legend.frameon'] = False

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
import seaborn as sns
# sns.set_theme(style="whitegrid")
# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '20', 'color': 'black', 'weight': 'normal',
                'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '20'}

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Assuming the data arrays da_Arctic, da_NorthAtlantic, da_SoutheastPacific, da_SOP, and the bounds are defined
# Create the plot
fig = plt.figure(figsize=(25, 15))
gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.7)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])

# Define colors, line widths, titles, linestyles
colors = ['#0F1023', '#B11927', '#407BD0', '#B7D0EA']
linestyles = ['-', '-', '-.', ':']
titles = ['Arctic (ARC)', 'North Atlantic Warming Hole (NAWH)', 'Southeast Pacific (SEP)', 'Southern Ocean Pacific sector (SOP)']
vars = ['raw', 'forced', 'unforced', 'MMLE_forced']

for i, var in enumerate(vars):
    if var == 'MMLE_forced':
        da_Arctic[var] = MMLE_forced_ARC_trend
        da_NorthAtlantic[var] = MMLE_forced_NAWH_trend
        da_SoutheastPacific[var] = MMLE_forced_SEP_trend
        da_SOP[var] = MMLE_forced_SOP_trend
        sns.lineplot(x=np.arange(1950, 2014), y=da_Arctic[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax1)
        sns.lineplot(x=np.arange(1950, 2014), y=da_NorthAtlantic[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax2)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SoutheastPacific[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax3)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SOP[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax4)
    else:
        print(f"{var} shape: ", da_Arctic[var].shape)
        sns.lineplot(x=np.arange(1950, 2014), y=da_Arctic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax1)
        sns.lineplot(x=np.arange(1950, 2014), y=da_NorthAtlantic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax2)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SoutheastPacific[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax3)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SOP[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax4)

# Add the unforced shading region
# add the unforced shading region with reversed upper and lower bounds
ax1.fill_between(np.arange(1950,2014), arctic_unforced_lower.values, arctic_unforced_upper.values, color=colors[3])
ax2.fill_between(np.arange(1950,2014), NAWH_unforced_lower.values, NAWH_unforced_upper.values, color=colors[3])
ax3.fill_between(np.arange(1950,2014), SEP_unforced_lower.values, SEP_unforced_upper.values, color=colors[3])
ax4.fill_between(np.arange(1950,2014), SOP_unforced_lower.values, SOP_unforced_upper.values, color=colors[3])

# Add the unforced shading region filled with hatched lines
years = np.arange(1950, 2014)

for ax, low, high in [
    (ax1, arctic_unforced_lower.values, arctic_unforced_upper.values),
    (ax2, NAWH_unforced_lower.values,    NAWH_unforced_upper.values),
    (ax3, SEP_unforced_lower.values,     SEP_unforced_upper.values),
    (ax4, SOP_unforced_lower.values,     SOP_unforced_upper.values),
]:
    ax.fill_between(years,
                    low, high,
                    facecolor='none',
                    edgecolor='grey',
                    hatch='//',
                    linewidth=0)
ax1.set_ylim([-1.0, 1.0])
ax2.set_ylim([-1.0, 1.0])
ax3.set_ylim([-1.5, 1.0])
ax4.set_ylim([-0.6, 0.6])

ax1.set_xlim([1948, 2015])
ax2.set_xlim([1948, 2015])
ax3.set_xlim([1948, 2015])
ax4.set_xlim([1948, 2015])

# Set x and y axis limits
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim([1948, 2015])
    
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xticks([1953, 1963, 1973, 1983, 1993, 2003, 2013])
    ax.set_xticklabels(['1953', '1963', '1973', '1983', '1993', '2003', '2013'])
    ax.set_ylabel('Trend (°C/decade)', fontsize=30)
    ax.set_xlabel('Start year of linear trend', fontsize=30)
    # ax.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')
    # ax.tick_params(axis='y', which='minor', length=8, width=2.5, direction='in')
    ax.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
    # ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    
# Add top x axis label
ax1_upper = ax1.twiny()
ax2_upper = ax2.twiny()
ax3_upper = ax3.twiny()
ax4_upper = ax4.twiny()

for ax_upper in [ax1_upper, ax2_upper, ax3_upper, ax4_upper]:
    ax_upper.invert_xaxis()
    ax_upper.set_xlim([75, 8])
    ax_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
    ax_upper.set_xticks([70, 60, 50, 40, 30, 20, 10])
    ax_upper.set_xticklabels(['70', '60', '50', '40', '30', '20', '10'])
    ax_upper.tick_params(axis='x', labelsize=26)
    ax_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
    ax_upper.xaxis.set_minor_locator(MultipleLocator(2))
    ax_upper.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    
# Customize spines
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

# Add zero line
for ax in [ax1, ax2, ax3, ax4]:
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)

# Add vertical lines
for ax in [ax1, ax2, ax3, ax4]:
    for year in [2013, 1993, 1963]:
        ax.axvline(x=year, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
        
# move the title to the left and move up the title
ax1.set_title(titles[0], loc='left',fontsize=32,pad=20)
ax2.set_title(titles[1], loc='left',fontsize=32,pad=20)
ax3.set_title(titles[2], loc='left',fontsize=32,pad=20)
ax4.set_title(titles[3], loc='left',fontsize=32,pad=20)
# Add text labels for the vertical lines
# ax1.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax2.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax3.text(1979.1, -1.46, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax4.text(1979.1, -0.56, '1979-2022', fontsize=26, rotation=90, color='#999A9E')

# Add subplot order text
ax1.text(1945, 1.58, 'a', fontsize=35, ha='center', va='center', fontweight='bold')
ax2.text(1945, 1.58, 'b', fontsize=35, ha='center', va='center', fontweight='bold')
ax3.text(1945, 1.74, 'c', fontsize=38, ha='center', va='center', fontweight='bold')
ax4.text(1945, 0.95, 'd', fontsize=35, ha='center', va='center', fontweight='bold')

# Create custom legend
custom_lines = [Line2D([0], [0], color=colors[0], lw=3.5),
                Line2D([0], [0], color=colors[1], lw=3.5),
                Line2D([0], [0], color=colors[2], lw=3.5, linestyle='-.'),
                Line2D([0], [0], color=colors[1], lw=3.5, linestyle=':')]
leg1 = ax1.legend(custom_lines, ['total', 'external forcing', 'internal variability', 'MMLE forced'], 
                  loc='lower left', fontsize=18)
ax1.add_artist(leg1)
# add the legend for the unforced shading region
# --- custom legend for the two shaded regions ---
shade_handles = [
    # solid blue fill for the OBS‐ICV envelope
    Patch(
        facecolor=colors[3],
        edgecolor='none',
        alpha=0.4,          # match the alpha you used in fill_between
        label='OBS-ICV'
    ),
    # hatched patch for the piControl‐ICV region
    Patch(
        facecolor='none',
        edgecolor='grey',
        hatch='//',
        label='LEs-ICV'
    ),
]

leg2 = ax2.legend(
    handles=shade_handles,
    loc='lower left',      # choose a free corner
    fontsize=20
)
ax2.add_artist(leg2)
dir_fig_output = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG5/"
os.makedirs(dir_fig_output, exist_ok=True)
plt.savefig(f'{dir_fig_output}FIG5_SMILE_icv_7LE_averaged_percentile.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{dir_fig_output}FIG5_SMILE_icv_7LE_averaged_percentile.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{dir_fig_output}FIG5.eps', dpi=300, bbox_inches='tight')
plt.show()
# %%