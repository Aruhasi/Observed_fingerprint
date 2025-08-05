# In[1]:
import os
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import xarray as xr

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
dir_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/end_year_2013/data/'
dir_NAT_trend = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/end_year_2013/data/'
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
        data_array[region] = xr.open_dataset(f'{dir_input}{region}_revised_trend_variations_end2013.nc')
    else:
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_input}{region}_trend_variations_end2013.nc')
# %%
# put the forced and unforced data into the same dictionary
da_Arctic           = xr.Dataset({'raw':data_array['Arctic'].raw, 'forced': data_array['Arctic'].forced, 'unforced': data_array['Arctic'].internal})
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
# # calculate the regional mean 
# def wgt_mean(var,lon,lat):
#     """
#     Calculate weighted mean
#     Parameters
#     ----------
#     var : 3-D array 
#     lat, lon : 1-D arrays
#     """
#     #Mask nans
#     var_ma = ma.masked_invalid(var)
#     #Weight matrix
#     #lat60 = lat.sel(lat=slice(60,-60))
#     wgtmat = np.cos(np.tile(abs(lat.values[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,...] #(time,lat,lon)
#     #Apply
#     #var_mean = np.ma.sum((var_ma*wgtmat*~var_ma.mask)/(np.ma.sum(wgtmat * ~var_ma.mask)))
#     var_mean = var_ma*wgtmat*~var_ma.mask
#     var_m = np.nanmean(np.nanmean(var_mean,axis=-1),axis=-1)
#     return var_m
# In[14]:
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
dir_percentile_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/data/'
dir_percentile_NAT_trend = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/data/NAT_wrt_nh_Keil/'
dir_percentile_SOP = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure4/check_SO_region/data/'
arctic_unforced_lower = xr.open_dataset(f'{dir_percentile_input}internal_arctic_trend_lower_percentile.nc')
NAWH_unforced_lower = xr.open_dataset(f'{dir_percentile_NAT_trend}internal_subpolar_gyre_trend_lower_percentile.nc')
SEP_unforced_lower = xr.open_dataset(f'{dir_percentile_input}internal_SEP_trend_lower_percentile.nc')
SOP_unforced_lower = xr.open_dataset(f'{dir_percentile_SOP}internal_SOP_trend_lower_percentile.nc')

arctic_unforced_upper = xr.open_dataset(f'{dir_percentile_input}internal_arctic_trend_upper_percentile.nc')
NAWH_unforced_upper = xr.open_dataset(f'{dir_percentile_NAT_trend}internal_subpolar_gyre_trend_upper_percentile.nc')
SEP_unforced_upper = xr.open_dataset(f'{dir_percentile_input}internal_SEP_trend_upper_percentile.nc')
SOP_unforced_upper = xr.open_dataset(f'{dir_percentile_SOP}internal_SOP_trend_upper_percentile.nc')
# %%
# rename the variable name
arctic_unforced_lower = arctic_unforced_lower.rename_vars({'__xarray_dataarray_variable__':'trend'})
NAWH_unforced_lower = NAWH_unforced_lower.rename_vars({'__xarray_dataarray_variable__':'trend'})
SEP_unforced_lower = SEP_unforced_lower.rename_vars({'__xarray_dataarray_variable__':'trend'})
SOP_unforced_lower = SOP_unforced_lower.rename_vars({'__xarray_dataarray_variable__':'trend'})

arctic_unforced_upper = arctic_unforced_upper.rename_vars({'__xarray_dataarray_variable__':'trend'})
NAWH_unforced_upper = NAWH_unforced_upper.rename_vars({'__xarray_dataarray_variable__':'trend'})
SEP_unforced_upper = SEP_unforced_upper.rename_vars({'__xarray_dataarray_variable__':'trend'})
SOP_unforced_upper = SOP_unforced_upper.rename_vars({'__xarray_dataarray_variable__':'trend'})
# %%
# end in 2013
arctic_unforced_lower_2013 = arctic_unforced_lower.trend.values[0:55]
arctic_unforced_upper_2013 = arctic_unforced_upper.trend.values[0:55]

NAWH_unforced_lower_2013 = NAWH_unforced_lower.trend.values[0:55]
NAWH_unforced_upper_2013 = NAWH_unforced_upper.trend.values[0:55]

SEP_unforced_lower_2013 = SEP_unforced_lower.trend.values[0:55]
SEP_unforced_upper_2013 = SEP_unforced_upper.trend.values[0:55]

SOP_unforced_lower_2013 = SOP_unforced_lower.trend.values[0:55]
SOP_unforced_upper_2013 = SOP_unforced_upper.trend.values[0:55]
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
# Create the plot
fig = plt.figure(figsize=(25, 15))
gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.7)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])

# define rgb colors for the outlines
# colors = [(32,120,180), #blue
#           (106,61,154), #purple
#           (173,23,88), #magenta
#           (255,127,0), #orange
#           (226,26,27),#red
#           (49,160,45) #green
#          ]
# colors_set = [(r / 255, g / 255, b / 255) for r, g, b in colors]
colors = ['#0F1023','#B11927', '#407BD0', '#B7D0EA']
line_widths = [5.5, 5.5, 5.5, 5.5, 1.5, 1.5, 1.5]
titles = ['Arctic (ARC)', 'North Atlantic Warming Hole (NAWH)', 'Southeast Pacific (SEP)', 'Southern Ocean Pacific sector (SOP)']
linestyles = ['-', '-', '-.', ':']

vars = ['raw', 'forced', 'unforced']
# We use a loop to simulate multiple lines for each category
for i, var in enumerate(vars):
    # Plot the forced trends for the EU region
    sns.lineplot(x=np.arange(1950,2005), y=da_Arctic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax1)
    sns.lineplot(x=np.arange(1950,2005), y=da_NorthAtlantic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax2)
    sns.lineplot(x=np.arange(1950,2005), y=da_SoutheastPacific[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax3)
    sns.lineplot(x=np.arange(1950,2005), y=da_SOP[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax4)

# add the unforced shading region with reversed upper and lower bounds
ax1.fill_between(np.arange(1950,2005), arctic_unforced_lower_2013[::-1], arctic_unforced_upper_2013[::-1], color=colors[3])
ax2.fill_between(np.arange(1950,2005), NAWH_unforced_lower_2013[::-1], NAWH_unforced_upper_2013[::-1], color=colors[3])
ax3.fill_between(np.arange(1950,2005), SEP_unforced_lower_2013[::-1], SEP_unforced_upper_2013[::-1], color=colors[3])
ax4.fill_between(np.arange(1950,2005), SOP_unforced_lower_2013[::-1], SOP_unforced_upper_2013[::-1], color=colors[3])

ax1.set_xlim([1948, 2006])
ax2.set_xlim([1948, 2006])
ax3.set_xlim([1948, 2006])
ax4.set_xlim([1948, 2006])

# set the y axis limit
ax1.set_ylim([-1.0, 1.2])
ax2.set_ylim([-1.0, 1.0])
ax3.set_ylim([-1.0, 1.0])
ax4.set_ylim([-0.6, 0.6])

# ax1.set_xticks([1954, 1964, 1974, 1984, 1994, 2004])
# ax2.set_xticks([1954, 1964, 1974, 1984, 1994, 2004])
# ax3.set_xticks([1954, 1964, 1974, 1984, 1994, 2004])
# ax4.set_xticks([1954, 1964, 1974, 1984, 1994, 2004])

# ax1.set_xticklabels(['1954', '1964', '1974', '1984', '1994', '2004'])
# ax2.set_xticklabels(['1954', '1964', '1974', '1984', '1994', '2004'])
# ax3.set_xticklabels(['1954', '1964', '1974', '1984', '1994', '2004'])
# ax4.set_xticklabels(['1954', '1964', '1974', '1984', '1994', '2004'])
# Set x and y axis limits
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim([1948, 2006])
    
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xticks([1954, 1964, 1974, 1984, 1994, 2004])
    ax.set_xticklabels(['1954', '1964', '1974', '1984', '1994', '2004'])
    ax.set_ylabel('Trend (°C/decade)', fontsize=30)
    ax.set_xlabel('Start year of linear trend', fontsize=30)
    # ax.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')
    # ax.tick_params(axis='y', which='minor', length=8, width=2.5, direction='in')
    ax.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
    # ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    
# add the top x axis label: length of the trend
ax1_upper = ax1.twiny()
ax2_upper = ax2.twiny()
ax3_upper = ax3.twiny()
ax4_upper = ax4.twiny()

# Define the bottom axis range
start_years = np.array([1954, 1964, 1974, 1984, 1994, 2004])  # start years
trend_lengths = 2004 - start_years + 10                      # exact match with figure design

# For each top twin axis
for ax, ax_upper in zip([ax1, ax2, ax3, ax4], [ax1_upper, ax2_upper, ax3_upper, ax4_upper]):
    ax_upper.set_xlim(ax.get_xlim())  # Align exactly with bottom axis
    ax_upper.set_xticks(start_years)
    ax_upper.set_xticklabels(trend_lengths.astype(str))  # Label as trend length
    ax_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
    ax_upper.tick_params(axis='x', labelsize=26, which='major', length=12, width=2.5, direction='in')
    ax_upper.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    ax_upper.xaxis.set_minor_locator(MultipleLocator(2))

# reverse the top x axis
# ax1_upper.invert_xaxis()
# ax2_upper.invert_xaxis()
# ax3_upper.invert_xaxis()
# ax4_upper.invert_xaxis()

# ax1_upper.set_xlim([67,8])
# ax2_upper.set_xlim([67,8])
# ax3_upper.set_xlim([67,8])
# ax4_upper.set_xlim([67,8])

# ax1_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
# ax2_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
# ax3_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
# ax4_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)

# ax1_upper.set_xticks([60, 50, 40, 30, 20, 10])
# ax2_upper.set_xticks([60, 50, 40, 30, 20, 10])
# ax3_upper.set_xticks([60, 50, 40, 30, 20, 10])
# ax4_upper.set_xticks([60, 50, 40, 30, 20, 10])

# # set the x axis top label 
# ax1_upper.set_xticklabels(['60', '50', '40', '30', '20', '10'])
# ax2_upper.set_xticklabels(['60', '50', '40', '30', '20', '10'])
# ax3_upper.set_xticklabels(['60', '50', '40', '30', '20', '10'])
# ax4_upper.set_xticklabels(['60', '50', '40', '30', '20', '10'])

# set the frame border width all around the subplot
ax1.spines['top'].set_linewidth(2.5)
ax1.spines['right'].set_linewidth(2.5)
ax1.spines['bottom'].set_linewidth(2.5)
ax1.spines['left'].set_linewidth(2.5)

ax2.spines['top'].set_linewidth(2.5)
ax2.spines['right'].set_linewidth(2.5)
ax2.spines['bottom'].set_linewidth(2.5)
ax2.spines['left'].set_linewidth(2.5)

ax3.spines['top'].set_linewidth(2.5)
ax3.spines['right'].set_linewidth(2.5)
ax3.spines['bottom'].set_linewidth(2.5)
ax3.spines['left'].set_linewidth(2.5)

ax4.spines['top'].set_linewidth(2.5)
ax4.spines['right'].set_linewidth(2.5)
ax4.spines['bottom'].set_linewidth(2.5)
ax4.spines['left'].set_linewidth(2.5)

# close the right and top spines
# ax1.spines['right'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax5.spines['right'].set_visible(False)

ax1_upper.tick_params(axis='x', labelsize=26)
ax2_upper.tick_params(axis='x', labelsize=26)
ax3_upper.tick_params(axis='x', labelsize=26)
ax4_upper.tick_params(axis='x', labelsize=26)

# show the ticks on the top x axis
ax1_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax2_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax3_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax4_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')

# leg1 = ax1.legend(['Forced', 'Unforced'], loc='upper right', fontsize=24, frameon=False)
# Add the zero line
ax1.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)
ax2.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)
ax3.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)
ax4.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)

# Customizing the plot
# Correctly set labels and titles using set_* methods
ax1.set_ylabel('Trend (°C/decade)', fontsize=30)
ax2.set_ylabel('Trend (°C/decade)', fontsize=30)
ax3.set_ylabel('Trend (°C/decade)', fontsize=30)
ax4.set_ylabel('Trend (°C/decade)', fontsize=30)

ax1.set_xlabel('Start year of linear trend', fontsize=30)
ax2.set_xlabel('Start year of linear trend', fontsize=30)
ax3.set_xlabel('Start year of linear trend', fontsize=30)
ax4.set_xlabel('Start year of linear trend', fontsize=30)

# move the title to the left and move up the title
ax1.set_title(titles[0], loc='left',fontsize=32,pad=20)
ax2.set_title(titles[1], loc='left',fontsize=32,pad=20)
ax3.set_title(titles[2], loc='left',fontsize=32,pad=20)
ax4.set_title(titles[3], loc='left',fontsize=32,pad=20)
# Set x y axis label size
# ax1.tick_params(axis='x', labelsize=32)
# ax2.tick_params(axis='x', labelsize=32)
# ax3.tick_params(axis='x', labelsize=32)
# ax4.tick_params(axis='x', labelsize=32)
# ax5.tick_params(axis='x', labelsize=32)

# ax1.tick_params(axis='y', labelsize=32)
# ax2.tick_params(axis='y', labelsize=32)
# ax3.tick_params(axis='y', labelsize=32)
# ax4.tick_params(axis='y', labelsize=32)
# ax5.tick_params(axis='y', labelsize=32)

# set the x, y axes ticks length
ax1.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax1.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')

ax2.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax2.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')

ax3.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax3.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')

ax4.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
ax4.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')

# ax3.yaxis.set_ticks([])
# add three vertical lines of the x-axis: 10-year[2013],30yr[1993],60yr[1963]
ax1.axvline(x=2004, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax1.axvline(x=1984, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax1.axvline(x=1954, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)

ax2.axvline(x=2004, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax2.axvline(x=1984, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax2.axvline(x=1954, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)

ax3.axvline(x=2004, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax3.axvline(x=1984, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax3.axvline(x=1954, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)

ax4.axvline(x=2004, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax4.axvline(x=1984, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
ax4.axvline(x=1954, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)

# Add the text labels for the vertical lines

# add the order of the subplot
ax1.text(1945, 1.85, 'a', fontsize=35, ha='center', va='center', fontweight='bold')
ax2.text(1945, 1.58, 'b', fontsize=35, ha='center', va='center', fontweight='bold')
ax3.text(1945, 1.58, 'c', fontsize=38, ha='center', va='center', fontweight='bold')
ax4.text(1945, 0.95, 'd', fontsize=35, ha='center', va='center', fontweight='bold')

# ax4.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# ax4.set_xticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
# ax4.set_ylim([0., 0.6])
# ax4.set_xlim([0, 100])
# Set the x-axis label size
ax1.tick_params(axis='x', labelsize=26)
ax2.tick_params(axis='x', labelsize=26)
ax3.tick_params(axis='x', labelsize=26)
ax4.tick_params(axis='x', labelsize=26)

# Set the y-axis label size
ax1.tick_params(axis='y', which ='both', right= False,labelsize=26)
ax2.tick_params(axis='y', which ='both', right= False,labelsize=26)
ax3.tick_params(axis='y', which ='both', right= False,labelsize=26)
ax4.tick_params(axis='y', which ='both', right= False,labelsize=26)

# Create custom legend entries for the regions
custom_lines = [Line2D([0], [0], color=colors[0], lw=3.5),
                Line2D([0], [0], color=colors[1], lw=3.5),
                Line2D([0], [0], color=colors[2], lw=3.5, linestyle='-.'),
                ]
# Define a custom patch for the shaded area
# blue_shading = mpatches.Patch(color=colors[2], alpha=0.3, label='ICV 5th-95th percentile')

# Create the legend
leg2 = ax1.legend(custom_lines, ['total', 'external forcing', 'internal variability'], loc='lower left', fontsize=26)
ax1.add_artist(leg2)

# Add the patch for the blue shading
# blue_shading_legend = ax2.legend(handles=[blue_shading], loc='lower left', fontsize=26)
# ax2.add_artist(blue_shading_legend)
# leg2 = ax2.legend(custom_lines, ['total', 'human forced', 'ICV 5th-95th percentile'], 
#                   loc='lower left', fontsize=26)
# ax2.add_artist(leg2)

# fig_legend = plt.figure(figsize=(10,10))
# ax_legend = fig_legend.add_subplot(111)
# ax_legend.axis('off')
# ax_legend.legend(*ax1.get_legend_handles_labels(), title='', loc='center', fontsize=30, ncol=3)

# fig_legend.savefig('legend4.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Extended_Fig11.png', dpi=300, bbox_inches='tight')
plt.savefig('Extended_Fig11.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Extended_Fig11.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

# %%
