# In[1]:
import os
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import xarray as xr
# import seaborn as sns
# print(sns.__version__)
# My modules 
import src.SAT_function as data_process
# from area_wgt_func import *
# In[2]:
# import src.slurm_cluster as scluster
# client, scluster = scluster.init_dask_slurm_cluster(scale=1)

# In[3]:
variable_name = np.arange(1950,2014)
keys = [f'{i}-2022' for i in variable_name]
region_name = ['Arctic', 'subpolar_gyre','SoutheastPacific','SOP']
# load data
dir_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/data/'
dir_SOP_trend = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/check_SO_region/data/'
dir_NAT_trend = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/data/NAT_wrt_nh_Keil/'
data_array = {}

# for region in region_name:
#     data_array[region] = {}
#     data_array[region] = xr.open_dataset(f'{dir_input}{region}_trend_variations.nc')
# %%
for region in region_name:
    if region == 'subpolar_gyre':
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_NAT_trend}{region}_wrt_nh_trend_variations.nc')
    elif region == 'SOP':
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_SOP_trend}{region}_trend_variations.nc')
    else:
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_input}{region}_trend_variations.nc')
# %%
# put the forced and unforced data into the same dictionary
da_Arctic           = xr.Dataset(
    {'raw':data_array['Arctic'].raw, 
    'forced': data_array['Arctic'].forced, 
    'unforced': data_array['Arctic'].internal})
# %%
da_NorthAtlantic    = xr.Dataset({'raw':data_array['subpolar_gyre'].raw, 'forced': data_array['subpolar_gyre'].forced, 'unforced': data_array['subpolar_gyre'].internal})
da_SoutheastPacific = xr.Dataset({'raw':data_array['SoutheastPacific'].raw, 'forced': data_array['SoutheastPacific'].forced, 'unforced': data_array['SoutheastPacific'].internal})
da_SOP              = xr.Dataset({'raw':data_array['SOP'].raw, 'forced': data_array['SOP'].forced, 'unforced': data_array['SOP'].internal})
# da_SouthernOcean = xr.Dataset({'raw':raw_da['SouthernOcean'].trend, 'forced': forced_da['SouthernOcean'].trend, 'unforced': unforced_da['SouthernOcean'].trend})
# In[6]:
# check the minimum value of the trend
print(da_Arctic.min(), da_Arctic.max())
# %%
print(da_NorthAtlantic.min(), da_NorthAtlantic.max())
# print(da_SouthernOcean.min())
# In[7]:
print(da_SoutheastPacific.min(), da_SoutheastPacific.max())
# %%
print(da_SOP.min(), da_SOP.max())
# In[11]:
# plot the year length vs trend value in four key regions for 1959-2022
"""
    Figure settings: x-axis: present trend length in years, y-axis: trend value
    Lines: different colours for forced and unforced trends
     forced trends: solid lines (), unforced trends: dashed lines
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
# import the shaded region for unforced trend
arctic_unforced_lower = xr.open_dataset(f'{dir_input}internal_arctic_trend_lower_percentile.nc')
NAWH_unforced_lower = xr.open_dataset(f'{dir_NAT_trend}internal_subpolar_gyre_trend_lower_percentile.nc')
SEP_unforced_lower = xr.open_dataset(f'{dir_input}internal_SEP_trend_lower_percentile.nc')
SOP_unforced_lower = xr.open_dataset(f'{dir_SOP_trend}internal_SOP_trend_lower_percentile.nc')

arctic_unforced_upper = xr.open_dataset(f'{dir_input}internal_arctic_trend_upper_percentile.nc')
NAWH_unforced_upper = xr.open_dataset(f'{dir_NAT_trend}internal_subpolar_gyre_trend_upper_percentile.nc')
SEP_unforced_upper = xr.open_dataset(f'{dir_input}internal_SEP_trend_upper_percentile.nc')
SOP_unforced_upper = xr.open_dataset(f'{dir_SOP_trend}internal_SOP_trend_upper_percentile.nc')

# %%
import seaborn as sns
# sns.set_theme(style="whitegrid")
# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '20', 'color': 'black', 'weight': 'normal',
                'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '20'}

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
vars = ['raw', 'forced', 'unforced']

# Loop over the variables and plot the data
for i, var in enumerate(vars):
    print(f"{var} shape: ", da_Arctic[var].shape)
    sns.lineplot(x=np.arange(1950, 2014), y=da_Arctic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax1)
    sns.lineplot(x=np.arange(1950, 2014), y=da_NorthAtlantic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax2)
    sns.lineplot(x=np.arange(1950, 2014), y=da_SoutheastPacific[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax3)
    sns.lineplot(x=np.arange(1950, 2014), y=da_SOP[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax4)

# Add the unforced shading region
ax1.fill_between(np.arange(1950, 2014), 
                 arctic_unforced_lower.trend.values[::-1], 
                 arctic_unforced_upper.trend.values[::-1], 
                 color=colors[3])
ax2.fill_between(np.arange(1950, 2014), NAWH_unforced_lower.trend.values[::-1], NAWH_unforced_upper.trend.values[::-1], color=colors[3])
ax3.fill_between(np.arange(1950, 2014), SEP_unforced_lower.trend.values[::-1], SEP_unforced_upper.trend.values[::-1], color=colors[3])
ax4.fill_between(np.arange(1950, 2014), SOP_unforced_lower.trend.values[::-1], SOP_unforced_upper.trend.values[::-1], color=colors[3])

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
    for year in [2013, 1993, 1979, 1963]:
        ax.axvline(x=year, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
        
# move the title to the left and move up the title
ax1.set_title(titles[0], loc='left',fontsize=32,pad=20)
ax2.set_title(titles[1], loc='left',fontsize=32,pad=20)
ax3.set_title(titles[2], loc='left',fontsize=32,pad=20)
ax4.set_title(titles[3], loc='left',fontsize=32,pad=20)
# Add text labels for the vertical lines
ax1.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
ax2.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
ax3.text(1979.1, -1.46, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
ax4.text(1979.1, -0.56, '1979-2022', fontsize=26, rotation=90, color='#999A9E')

# Add subplot order text
ax1.text(1945, 1.58, 'a', fontsize=35, ha='center', va='center', fontweight='bold')
ax2.text(1945, 1.58, 'b', fontsize=35, ha='center', va='center', fontweight='bold')
ax3.text(1945, 1.74, 'c', fontsize=38, ha='center', va='center', fontweight='bold')
ax4.text(1945, 0.95, 'd', fontsize=35, ha='center', va='center', fontweight='bold')

# Create custom legend
custom_lines = [Line2D([0], [0], color=colors[0], lw=3.5),
                Line2D([0], [0], color=colors[1], lw=3.5),
                Line2D([0], [0], color=colors[2], lw=3.5, linestyle='-.')]
leg2 = ax1.legend(custom_lines, ['total', 'human forced', 'internal variability'], 
                  loc='lower left', fontsize=26)
ax1.add_artist(leg2)

plt.savefig('Figure4.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure4.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Figure4.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

# %%
