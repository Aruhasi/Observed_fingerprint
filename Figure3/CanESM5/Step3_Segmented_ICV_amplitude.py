#!/usr/bin/env python
# coding: utf-8

# ### Import the necessary database

# In[1]:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#In[2]:
# define function
import src.SAT_function as data_process
import src.Data_Preprocess as preprosess

# In[3]:
import src.slurm_cluster as scluster
client, scluster = scluster.init_dask_slurm_cluster(walltime="01:30:00", memory="128GiB")

# In[4]:
def func_mk(x):
    """
    Mann-Kendall test for trend
    """
    results = data_process.apply_mannkendall(x)
    slope = results[0]
    p_val = results[1]
    return slope, p_val

# In[5]:
# Input the MMEM of SAT-OBS internal variability
dir_residuals = './Figure3/CanESM5/'
ds_CanESM5_1850_2022 = xr.open_mfdataset(dir_residuals + 'GSAT_CanESM5_Internal_Variability_anomalies_1850_2022.nc',chunks={'run':1})

# In[6]:
ds_CanESM5_1850_2022

# In[7]:
# ds_CanESM5_1850_2022 = ds_CanESM5_1850_2022.rename({'__xarray_dataarray_variable__': 'tas'})
# In[8]:
# Generate the running windows of the residuals of SAT-OBS
#       with a series of equal length with an interval of 5 years starting from 10 years to 100 years
#       and calculate the trend pattern of each segment
#       and calculate the ensemble standard deviation of the trend pattern of each interval of segments

# define the function to generate the running windows of the residuals of SAT-OBS
def generate_segments(data, segment_length):
    """
    data: 3D array with dimensions [year, lat, lon]
    segment_length: length of each segment in years
    """
    years = range(int(data['year'].min().item()), int(data['year'].max().item()) - segment_length + 2)
    print(years)
    # Initialize an empty list to store the segments
    segments = []
    
    # For each year in the range
    for year in years:
        # Extract the segment of data from that year to year + segment_length
        segment = data.sel(year=slice(str(year), str(year + segment_length - 1)))
        
        # Append this segment to the list of segments
        segments.append(segment)
    
    return segments


# In[9]:
# Generate the running windows of the residuals of SAT-OBS
time_interval = [60]

ICV_segments = {}
for i in time_interval:
    ICV_segments[i] = generate_segments(ds_CanESM5_1850_2022['tas'], segment_length=i)
# In[10]:
# Assuming ICV_segments is a dictionary with segment_length as keys and list of DataArray segments as values
max_num_segments = max(len(segments) for segments in ICV_segments.values())
segment_lengths = ICV_segments.keys()

# Create a new Dataset to hold the new arrays
new_ds = xr.Dataset()

for segment_length in segment_lengths:
    segments_list = ICV_segments[segment_length]
    # print(segments_list)
    
    # Pad the segments list to have the same number of segments
    padded_segments = segments_list.copy()
    while len(padded_segments) < max_num_segments:
        # Create a DataArray filled with NaNs to match the shape of the segments
        nan_segment = xr.full_like(padded_segments[0], np.nan)
        padded_segments.append(nan_segment)
    
    # Create a coordinate for the new segment dimension
    segment_coord = range(max_num_segments)
    
    # Concatenate the padded segments with the new segment coordinate
    concatenated = xr.concat(padded_segments, dim=segment_coord)
    
    # Assign a specific name to the new dimension
    concatenated = concatenated.rename({'concat_dim': 'segment'})
    
    # Add the new DataArray to the new dataset
    new_ds[f'ICV_segments_{segment_length}yr'] = concatenated


# In[11]:
new_ds
# In[12]:
ds_combined = xr.merge([ds_CanESM5_1850_2022, new_ds])
# In[13]:
ds_combined
# In[14]:
# check the minimum and maximum of the new variable
# ds_combined['ICV_segments_30yr'].min().values, ds_combined['ICV_segments_30yr'].max().values
# In[15]:
# define function to calculate the standard deviation of the trend pattern of each interval of segments
def std_trend_pattern(data):
    """
    data: 4D array with dimensions [year, lat, lon, segment]
    segment_length: length of each segment in years
    """
    # calculate the standard deviation of the trend pattern of each interval of segments
    std_trend_pattern = np.nanstd(data, axis=0)
    
    return std_trend_pattern
# In[16]:
# Calculate the trend pattern of each segment
#       and calculate the ensemble standard deviation of the trend pattern of each interval of segments
for segment_length in segment_lengths:
    # Calculate the trend pattern of each segment
    ds_combined[f'ICV_segments_{segment_length}yr_trend'], ds_combined[f'ICV_segments_{segment_length}yr_p_values'] = xr.apply_ufunc(
        func_mk,
        ds_combined[f'ICV_segments_{segment_length}yr'],
        input_core_dims=[['year']],
        output_core_dims=[[],[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )
    # multiply the trend pattern of each segment with 10.0 to get the trend pattern in degC/decade
    ds_combined[f'ICV_segments_{segment_length}yr_trend'] = ds_combined[f'ICV_segments_{segment_length}yr_trend']*10.0
# In[17]:
for segment_length in segment_lengths:
    # Calculate the standard deviation of the trend pattern of each interval of segments
    ds_combined[f'ICV_segments_{segment_length}yr_std_trend_pattern'] = xr.apply_ufunc(
        std_trend_pattern,
        ds_combined[f'ICV_segments_{segment_length}yr_trend'],
        input_core_dims=[['segment']],
        output_core_dims=[[ ]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
# In[18]:
ds_combined
# In[19]:
# calculate the ensemble mean of the trend pattern of each interval of segments;
#     and save the ensemble mean of the trend pattern of each interval of segments to the dataset
# for segment_length in segment_lengths:
#     key_trend = f'ICV_segments_{segment_length}yr_trend'
#     key_mean = f'ICV_segments_{segment_length}yr_trend_mean'

#     if key_trend in ds_combined:
#         # Calculate mean
#         data = np.nanmean(ds_combined[key_trend], axis=0)
        
#         # Check if the mean key exists, if not, initialize it
#         if key_mean not in ds_combined:
#             ds_combined[key_mean] = []

#         # Append data
#         ds_combined[key_mean]= (['lat', 'lon'], data)
# In[20]:
ds_output = './Figure3/CanESM5/'
# ds_combined['ICV_segments_10yr_std_trend_pattern'].to_netcdf(ds_output + 'ICV_segments_10yr_std_trend_pattern.nc')
# ds_combined['ICV_segments_30yr_std_trend_pattern'].to_netcdf(ds_output + 'ICV_segments_30yr_std_trend_pattern.nc')
ds_combined['ICV_segments_60yr_std_trend_pattern'].to_netcdf(ds_output + 'ICV_segments_60yr_std_trend_pattern.nc')

# In[21]:

ds_combined['ICV_segments_60yr_std_trend_pattern']

# In[22]:
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['savefig.transparent'] = True

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
# In[23]:

def plot_trend(trend_data, lats, lons, levels=None, extend=None,cmap=None, 
                                 title="", ax=None, show_xticks=False, show_yticks=False):
    """
    Plot the trend spatial pattern using Robinson projection with significance overlaid.

    Parameters:
    - trend_data: 2D numpy array with the trend values.
    - lats, lons: 1D arrays of latitudes and longitudes.
    - p_values: 2D array with p-values for each grid point.
    - GMST_p_values: 2D array with GMST p-values for each grid point.
    - title: Title for the plot.
    - ax: Existing axis to plot on. If None, a new axis will be created.
    - show_xticks, show_yticks: Boolean flags to show x and y axis ticks.
    
    Returns:
    - contour_obj: The contour object from the plot.
    """
# Create a new figure/axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()
        
    contour_obj = ax.contourf(lons, lats, trend_data, levels=levels, extend=extend, cmap=cmap, transform=ccrs.PlateCarree(central_longitude=0))
    # Plot significance masks with different hatches
    # ax.contourf(lons, lats, significance_mask, levels=[0.05, 1.0],hatches=['///'], colors='none', transform=ccrs.PlateCarree())

    ax.coastlines(resolution='110m')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.35)

    # Disable labels on the top and right of the plot
    gl.top_labels = False
    gl.right_labels = False

    # Enable labels on the bottom and left of the plot
    gl.bottom_labels = show_xticks
    gl.left_labels = show_yticks
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    
    if show_xticks:
        gl.bottom_labels = True
    if show_yticks:
        gl.left_labels = True
    
    ax.set_title(title, loc='center', fontsize=18, pad=5.0)

    return contour_obj


# In[24]:

# define an asymmetric colormap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import BoundaryNorm

intervals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# Normalizing the intervals to [0, 1]
min_interval = min(intervals)
max_interval = max(intervals)
normalized_intervals = [(val - min_interval) / (max_interval - min_interval) for val in intervals]

# colors = ['#2616D3', '#005EFF', '#0084FF', '#00A2FF', '#00BCDB', (1.0, 1.0, 1.0, 1.0),(1.0, 1.0, 1.0, 1.0),(1.0, 0.8, 0.5, 1.0),
#     (1.0, 0.803921568627451, 0.607843137254902, 1.0), (1.0, 0.6000000000000001, 0.20000000000000018, 1.0),(1.0, 0.4039215686274509, 0.0, 1.0),(0.8999999999999999, 0.19999999999999996, 0.0, 1.0),
#     (0.7470588235294118, 0.0, 0.0, 1.0), (0.6000000000000001, 0.0, 0.0, 1.0),(0.44705882352941173, 0.0, 0.0, 1.0),(0.30000000000000004, 0.0, 0.0, 1.0),(0.14705882352941177, 0.0, 0.0, 1.0),
#     (0.0, 0.0, 0.0, 1.0)]

# Creating a list of tuples with normalized positions and corresponding colors
# color_list = list(zip(normalized_intervals, colors))

# # Create the colormap
# custom_cmap = LinearSegmentedColormap.from_list('my_custom_cmap', color_list)

# # Create a normalization
# norm = Normalize(vmin=min_interval, vmax=max_interval)

# In[25]:
import seaborn as sns
import palettable
from palettable.colorbrewer.diverging import RdBu_11_r
import matplotlib.colors as mcolors

cmap = mcolors.ListedColormap(palettable.cmocean.sequential.Amp_20.mpl_colors)
# Plot 10yr 50 runs std pattern
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# levels = np.arange(-0.5, 0.55, 0.05)
levels = np.arange(0.0, 1.1, 0.1)
extend = 'both'

# Define the number of plots per page
num_plots_per_page = 4
num_subplots_x = 2  # Number of subplots in the x direction (columns)
num_subplots_y = 2  # Number of subplots in the y direction (rows)

# Define the dimensions of the figure for each page
figsize_x = 20
figsize_y = 12

lat = ds_combined['lat'].values
lon = ds_combined['lon'].values

# In[26]:
start_year = 1950
end_year = 2022
min_length = 10

extend = 'max'
"""
plot 50 runs of the trend pattern of the ICV segments of 10 years
"""
with PdfPages('./10yr_ICV_SAT_trend_figures.pdf') as pdf:
    for start_page in range(0, len(ds_combined['ICV_segments_10yr_std_trend_pattern'].run), num_plots_per_page):
        fig, axes = plt.subplots(num_subplots_y, num_subplots_x, figsize=(figsize_x, figsize_y),
                                 subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        for i in range(num_plots_per_page):
            idx = start_page + i
            if idx >= len(ds_combined['ICV_segments_10yr_std_trend_pattern'].run):
                break
            data = ds_combined['ICV_segments_10yr_std_trend_pattern'].isel(run=idx)
            
            ix = i % num_subplots_x
            iy = i // num_subplots_x
            
            ax = axes[iy, ix]
         
            # # Add cyclic point to data
            # data_with_cyclic, lon_cyclic = cutil.add_cyclic_point(data, coord=lon)
            # p_values_with_cyclic, _ = cutil.add_cyclic_point(pvalue_annual_da[interval], coord=lon)
            
            # Plotting the data with significance
            contour_obj = plot_trend(data, lat, lon, levels=levels, extend=extend, cmap=cmap,
                                     title=" ", ax=ax, show_xticks=True, show_yticks=True)
            ax.set_title(f"Trend for run {idx}", fontsize=18)
            ax.plot([-50,-10,-10,-50, -50], [42, 42, 60, 60, 42],
            color='tab:blue', linewidth=2.0,
            transform=ccrs.PlateCarree())

        # Add colorbar for each page
        cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
        cbar = plt.colorbar(contour_obj, cax=cbar_ax, orientation='horizontal', extend=extend)
        cbar.set_label('Annual SAT trend (°C per decade)', fontsize=16)
        
        # Save the page
        pdf.savefig(fig)
        plt.close(fig)

# In[27]:
client.close()
scluster.close()

# In[ ]:




