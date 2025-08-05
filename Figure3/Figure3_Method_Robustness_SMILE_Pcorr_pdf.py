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
# In[2]:
# input pattern correlation:
def read_correlations(file_path):
    correlations = {
        '10-year': [],
        '30-year': [],
        '60-year': []
    }
    
    current_period = '10-year'  # Initialize with the first period

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Try to convert the line into a float and add it to the current period list
                correlations[current_period].append(float(line.strip()))
            except ValueError:
                # If it fails, it's probably a header line; update the current period based on the header
                if '10-year' in line:
                    current_period = '10-year'
                elif '30-year' in line:
                    current_period = '30-year'
                elif '60-year' in line:
                    current_period = '60-year'
                # If it's not a header, the line is skipped

    return correlations
# In[3]:
# input the pattern correlation of each models: single realization and the ensemble mean
dir_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure2/'

CanESM5_run_ens_corr = read_correlations(dir_in + 'CanESM5/pattern_correlations.txt')
IPSL_run_ens_corr = read_correlations(dir_in + 'IPSL/pattern_correlations.txt')
EC_Earth_run_ens_corr = read_correlations(dir_in + 'EC_Earth/pattern_correlations.txt')
ACCESS_run_ens_corr = read_correlations(dir_in + 'ACCESS/pattern_correlations.txt')
MPI_ESM_run_ens_corr = read_correlations(dir_in + 'MPI_ESM/pattern_correlations.txt')
MIROC6_run_ens_corr = read_correlations(dir_in + 'MIROC6/pattern_correlations.txt')
# In[4]:
MMEM_corr = {'10-year': [0.82], '30-year': [0.86], '60-year': [0.87]}
# %%
unforced_CanESM5_run_obs_corr = read_correlations(dir_in + 'CanESM5/pattern_correlations_unforced_STD.txt')
unforced_IPSL_run_obs_corr = read_correlations(dir_in + 'IPSL/pattern_correlations_unforced_STD.txt')
unforced_EC_Earth_run_obs_corr = read_correlations(dir_in + 'EC_Earth/pattern_correlations_unforced_STD.txt')
unforced_ACCESS_run_obs_corr = read_correlations(dir_in + 'ACCESS/pattern_correlations_unforced_STD.txt')
unforced_MPI_ESM_run_obs_corr = read_correlations(dir_in + 'MPI_ESM/pattern_correlations_unforced_STD.txt')
unforced_MIROC6_run_obs_corr = read_correlations(dir_in + 'MIROC6/pattern_correlations_unforced_STD.txt')
# %%
models_data = {
    'CanESM5': CanESM5_run_ens_corr,
    'IPSL-CM6A-LR': IPSL_run_ens_corr,
    'EC-Earth3': EC_Earth_run_ens_corr,
    'ACCESS-ESM1.5': ACCESS_run_ens_corr,
    'MPI-ESM1.2-LR': MPI_ESM_run_ens_corr,
    'MIROC6': MIROC6_run_ens_corr
}

models_unforced_data = {
    'CanESM5': unforced_CanESM5_run_obs_corr,
    'IPSL-CM6A-LR': unforced_IPSL_run_obs_corr,
    'EC-Earth3': unforced_EC_Earth_run_obs_corr,
    'ACCESS-ESM1.5': unforced_ACCESS_run_obs_corr,
    'MPI-ESM1.2-LR': unforced_MPI_ESM_run_obs_corr,
    'MIROC6': unforced_MIROC6_run_obs_corr,
}

# %%
long_data = []

# Loop through the dictionary and append each data point as a row in the list
for model, correlations_dict in models_data.items():
    for time_period, correlations in correlations_dict.items():
        for correlation in correlations:
            long_data.append({
                'Model': model,
                'Time Period': time_period,
                'Correlation': correlation
            })

# Convert the list to a DataFrame
long_df = pd.DataFrame(long_data)
# %%
ICV_long_data = []

# Loop through the dictionary and append each data point as a row in the list
for model, correlations_dict in models_unforced_data.items():
    for time_period, correlations in correlations_dict.items():
        for correlation in correlations:
            ICV_long_data.append({
                'Model': model,
                'Time Period': time_period,
                'Correlation': correlation
            })
        
# Convert the list to a DataFrame
ICV_long_df = pd.DataFrame(ICV_long_data)
# %%
# mmem correlation
# Find the maximum size among all arrays for each time period
def pad_to_max_length(arrays, fill_value=np.nan):
    max_length = max([len(arr) for arr in arrays])
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), constant_values=fill_value) for arr in arrays]
    return padded_arrays

# %%
# Pad the arrays for each time period
ENS_corr = {
    '10-year': np.concatenate(pad_to_max_length([
        np.array(CanESM5_run_ens_corr['10-year']),
        np.array(IPSL_run_ens_corr['10-year']),
        np.array(EC_Earth_run_ens_corr['10-year']),
        np.array(ACCESS_run_ens_corr['10-year']),
        np.array(MPI_ESM_run_ens_corr['10-year']),
        np.array(MIROC6_run_ens_corr['10-year']),
    ])),
    '30-year': np.concatenate(pad_to_max_length([
        np.array(CanESM5_run_ens_corr['30-year']),
        np.array(IPSL_run_ens_corr['30-year']),
        np.array(EC_Earth_run_ens_corr['30-year']),
        np.array(ACCESS_run_ens_corr['30-year']),
        np.array(MPI_ESM_run_ens_corr['30-year']),
        np.array(MIROC6_run_ens_corr['30-year']),
    ])),
    '60-year': np.concatenate(pad_to_max_length([
        np.array(CanESM5_run_ens_corr['60-year']),
        np.array(IPSL_run_ens_corr['60-year']),
        np.array(EC_Earth_run_ens_corr['60-year']),
        np.array(ACCESS_run_ens_corr['60-year']),
        np.array(MPI_ESM_run_ens_corr['60-year']),
        np.array(MIROC6_run_ens_corr['60-year']),
    ])),
}

# Verify the resulting arrays
print("10-year correlations:", ENS_corr['10-year'])
print("30-year correlations:", ENS_corr['30-year'])
print("60-year correlations:", ENS_corr['60-year'])
# %%
unforced_ENS_corr = {
    '10-year': np.concatenate(pad_to_max_length([
        np.array(unforced_CanESM5_run_obs_corr['10-year']),
        np.array(unforced_IPSL_run_obs_corr['10-year']),
        np.array(unforced_EC_Earth_run_obs_corr['10-year']),
        np.array(unforced_ACCESS_run_obs_corr['10-year']),
        np.array(unforced_MPI_ESM_run_obs_corr['10-year']),
        np.array(unforced_MIROC6_run_obs_corr['10-year']),
    ])),
    '30-year': np.concatenate(pad_to_max_length([
        np.array(unforced_CanESM5_run_obs_corr['30-year']),
        np.array(unforced_IPSL_run_obs_corr['30-year']),
        np.array(unforced_EC_Earth_run_obs_corr['30-year']),
        np.array(unforced_ACCESS_run_obs_corr['30-year']),
        np.array(unforced_MPI_ESM_run_obs_corr['30-year']),
        np.array(unforced_MIROC6_run_obs_corr['30-year']),
    ])),
    '60-year': np.concatenate(pad_to_max_length([
        np.array(unforced_CanESM5_run_obs_corr['60-year']),
        np.array(unforced_IPSL_run_obs_corr['60-year']),
        np.array(unforced_EC_Earth_run_obs_corr['60-year']),
        np.array(unforced_ACCESS_run_obs_corr['60-year']),
        np.array(unforced_MPI_ESM_run_obs_corr['60-year']),
        np.array(unforced_MIROC6_run_obs_corr['60-year']),
    ])),
}
# %%
MMEM_unforced_corr = {'10-year': [0.67], '30-year': [0.60], '60-year': [0.58]}
# Subplots: c, d
# In[1]:
# input the 30-year forced trend patterns of OBS and MMEM
dir_forced_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure1/data_revision/'

HadCRUT5_trend = xr.open_dataset(dir_forced_input + 
                                 'HadCRUT5_annual_forced_30yr_trend.nc')
# In[2]:
dir_model_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure6_Forced/data/Smiles_ensemble/'
MMEM_annual_trend = xr.open_dataset(dir_model_in + 
                                    'MMEM_annual_forced_30yr_trend.nc')
# rename the variable name
MMEM_annual_trend = MMEM_annual_trend.rename({'__xarray_dataarray_variable__': 'trend'})
HadCRUT5_trend  = HadCRUT5_trend.rename({'__xarray_dataarray_variable__': 'trend'})
# %%
# Obs_trend_da = HadCRUT5_trend.trend*10.0
# Model_trend_da = MMEM_annual_trend.trend*10.0

# In[3]:
# calculate the pattern difference between OBS and MMEM
pattern_diff = MMEM_annual_trend.trend - HadCRUT5_trend.trend

# In[4]:
def cal_ratio(data,pattern_diff):
    data = pattern_diff/data
    return data
# %%
pattern_ratio = cal_ratio(HadCRUT5_trend.trend, pattern_diff)
print(pattern_ratio)
# %%
print(pattern_ratio.min().values)
# calculate the global mean values
mean_ratio = pattern_ratio.mean().values*100
# In[5]:
# Input the Observational internal trend (wrt MMEM GSAT)
dir_internal_input = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure3/data/ICV_std/'
HadCRUT5_internal_trend = xr.open_dataset(dir_internal_input +'ICV_segments_30yr_std_trend_pattern.nc')

dir_model_internal = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure7/data/'
MMEM_internal_trend = xr.open_dataset(dir_model_internal + 'MMEM_annual_30yr_noise_trend_std.nc')
# %%
HadCRUT5_internal_trend = HadCRUT5_internal_trend.rename({'ICV_segments_30yr_std_trend_pattern': 'trend'})
MMEM_internal_trend = MMEM_internal_trend.rename({'tas': 'trend'})
# %%
ICV_diff = MMEM_internal_trend.trend - HadCRUT5_internal_trend.trend
print(ICV_diff)
# %%
Ratio_ICV = cal_ratio(HadCRUT5_internal_trend.trend, ICV_diff)
print(Ratio_ICV)
# %%
# calculate the global mean values
mean_ratio_ICV = Ratio_ICV.mean().values*100
# %%
forced_diff = pattern_diff*10.0 
# %%
print(f"Forced_diff min: {forced_diff.min()}, max: {forced_diff.max()}")
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
fig = plt.figure(figsize=(25, 18))
gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.25, wspace=0.3)

# Time segments and colors for the PDFs
time_segments = ["10-year", "30-year", "60-year"]
colors = ['#F9AE78', '#E47159', '#3D5C6F']

# -------------------- Subplot a: Forced Correlation PDF --------------------
ax_pdf_forced = plt.subplot(gs[0, 0])
for i, segment in enumerate(time_segments):
    segment_data = long_df[long_df['Time Period'] == segment]['Correlation']
    sns.kdeplot(
        segment_data,
        label=f"r(Single realization, ENS)({segment})",
        fill=False,
        # alpha = 0.8,
        linewidth=4.5,
        color=colors[i],
        ax=ax_pdf_forced,
        bw_adjust=1.0,
        cut=0,
        clip=(0.4, 1)
    )
    mmem_value = MMEM_corr[segment][0]
    ax_pdf_forced.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, label=f"r(Obs, MMEM)({segment})")

# Add title and label
ax_pdf_forced.text(-0.12, 1.2, "a", transform=ax_pdf_forced.transAxes, fontsize=34, fontweight='bold', va='top')
ax_pdf_forced.text(0.5, 1.1, "Externally forced pattern correlation", fontsize=30, ha='center', va='center', transform=ax_pdf_forced.transAxes)
ax_pdf_forced.set_xlabel("Pattern Correlation", fontsize=26)
ax_pdf_forced.set_ylabel("Density", fontsize=26)
# ax_pdf_forced.grid(True)
ax_pdf_forced.spines['top'].set_visible(False)
ax_pdf_forced.spines['right'].set_visible(False)
ax_pdf_forced.tick_params(axis='x', labelsize=26)
ax_pdf_forced.tick_params(axis='y', labelsize=26)
ax_pdf_forced.set_xlim(0.4, 1.0)
ax_pdf_forced.set_ylim(0., 20)
# make the ticker longer
ax_pdf_forced.tick_params(axis='x', direction='out', length=6, width=2)
ax_pdf_forced.tick_params(axis='y', direction='out', length=6, width=2)

# -------------------- Subplot b: Internal Variability Correlation PDF --------------------
ax_pdf_icv = plt.subplot(gs[0, 1])
for i, segment in enumerate(time_segments):
    segment_data = ICV_long_df[ICV_long_df['Time Period'] == segment]['Correlation']
    sns.kdeplot(
        segment_data,
        label=f"r(Single realization, ENS)({segment})",
        fill=False,
        # alpha = 0.8,
        linewidth=4.5,
        color=colors[i],
        ax=ax_pdf_icv,
        bw_adjust=1.0,
        cut=0,
        clip=(0.4, 1)
    )
    mmem_value = MMEM_unforced_corr[segment][0]
    ax_pdf_icv.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, label=f"r(Obs, MMEM)({segment})")

# Add title and label
ax_pdf_icv.text(-0.12, 1.2, "b", transform=ax_pdf_icv.transAxes, fontsize=34, fontweight='bold', va='top')
ax_pdf_icv.text(0.5, 1.1, "Internal variability pattern correlation", fontsize=30, ha='center', va='center', transform=ax_pdf_icv.transAxes)
ax_pdf_icv.set_xlabel("Pattern Correlation", fontsize=26)
ax_pdf_icv.set_ylabel("Density", fontsize=26)
# ax_pdf_icv.grid(True)

ax_pdf_icv.spines['top'].set_visible(False)
ax_pdf_icv.spines['right'].set_visible(False)
ax_pdf_icv.tick_params(axis='x', labelsize=26)
ax_pdf_icv.tick_params(axis='y', labelsize=26)
ax_pdf_icv.set_xlim(0.4, 1.0)
ax_pdf_icv.set_ylim(0., 20)
# make the ticker longer
ax_pdf_icv.tick_params(axis='x', direction='out', length=8, width=2)
ax_pdf_icv.tick_params(axis='y', direction='out', length=8, width=2)
# -------------------- Add Legends --------------------
legend_elements_pattern = [
    Line2D([0], [0], color=colors[0], lw=0, label='10-year'),
    Line2D([0], [0], color=colors[1], lw=0, label='30-year'),
    Line2D([0], [0], color=colors[2], lw=0, label='60-year')
]

legend_elements_time = [
    Line2D([0], [0], color='black', lw=4.5, linestyle='-.', label='OBS'),
    Line2D([0], [0], color='black', lw=4.5, linestyle='-', label='Large Ensembles'),
]

fig.legend(
    handles=legend_elements_pattern,
    loc='upper left',
    fontsize=26,
    title="",
    title_fontsize=24,
    labelcolor='linecolor',
    ncol=3,
    bbox_to_anchor=(0.09, 0.88),
    frameon=False,
    columnspacing=0.05,  # Reduce space between columns
    handletextpad=0.2,  # Reduce space between marker and text
    borderaxespad=0.01  # Reduce padding around the legend
)

fig.legend(
    handles=legend_elements_time,
    loc='upper left',
    fontsize=24,
    title="",
    title_fontsize=24,
    ncol=2,
    bbox_to_anchor=(0.12, 0.85),
    frameon=False,
    columnspacing=0.5,  # Reduce space between columns
)
# shift the legend lower

# -------------------- Subplot c: Forced Minus MMEM Map --------------------
ax_forced_map = plt.subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180))
ax_forced_map.coastlines(resolution='110m')
gl = ax_forced_map.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.15, linewidth=0.25)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = cticker.LongitudeFormatter()
gl.yformatter = cticker.LatitudeFormatter()
gl.xlabel_style = {'size': 22}
gl.ylabel_style = {'size': 22}
gl.bottom_labels = True
gl.left_labels = True
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])

levels_forced = np.arange(-0.5, 0.55, 0.05)  # symmetric around 0
n_bins = len(levels_forced) - 1  # number of color intervals = 21

norm_forced = BoundaryNorm(boundaries=levels_forced, ncolors=n_bins)
cmap_forced = "RdBu_r" #mcolors.ListedColormap(palettable.cmocean.diverging.Curl_20.mpl_colors)

p_forced = forced_diff.plot(
    ax=ax_forced_map,
    transform=ccrs.PlateCarree(),
    cmap=cmap_forced,
    norm=norm_forced,
    levels=levels_forced,
    add_colorbar=False
)

# Add titleHuman-forced SAT difference\n
ax_forced_map.text(-0.12, 1.2, "c", transform=ax_forced_map.transAxes, fontsize=34, fontweight='bold', va='top')
ax_forced_map.set_title("MMLE - OBS\n(1993-2022)", fontsize=28, pad=10, loc='center')

# Add percentage annotation in the upper-right corner
ax_forced_map.text(0.95, 1.05, f"{mean_ratio:.0f}%", fontsize=28, ha='center', va='center',
                   transform=ax_forced_map.transAxes)

# Add colorbar
cbar_ax_forced = fig.add_axes([0.175, 0.1, 0.25, 0.02])
# cbar_forced = plt.colorbar(
#     p_forced,
#     cax=cbar_ax_forced,
#     orientation='horizontal',
#     ticks=[-0.5, -0.25, 0, 0.25, 0.5],
#     extend='neither'  
# )
cbar_forced = plt.colorbar(
    p_forced,
    cax=cbar_ax_forced,
    orientation='horizontal',
    extend='neither',  # <-- Enable both ends for symmetry
    ticks=[-0.5, -0.25, 0, 0.25, 0.5]
)
cbar_forced.set_label("Externally forced SAT differences\n(°C per decade)", fontsize=24, labelpad=10, loc='center')
cbar_forced.ax.tick_params(labelsize=22)
cbar_forced.ax.tick_params(direction='out', length=10, width=2)  # Adjust values as needed
cbar_forced.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

# -------------------- Subplot d: ICV Minus MMEM Map --------------------
ax_icv_map = plt.subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180))
ax_icv_map.coastlines(resolution='110m')
gl1 = ax_icv_map.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.15, linewidth=0.25)
gl1.top_labels = False
gl1.right_labels = False
gl1.xformatter = cticker.LongitudeFormatter()
gl1.yformatter = cticker.LatitudeFormatter()
gl1.xlabel_style = {'size': 18}
gl1.ylabel_style = {'size': 18}
gl1.bottom_labels = True
gl1.left_labels = True
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
# Dynamic levels based on data range
levels_icv = np.arange(-0.25, 0.275, 0.025)  # symmetric
n_bins_icv = len(levels_icv)-1

norm_icv = BoundaryNorm(boundaries=levels_icv, ncolors=n_bins_icv)
cmap_icv = "RdBu_r" #mcolors.ListedColormap(palettable.cmocean.diverging.Curl_20.mpl_colors)
# cmap_icv = plt.get_cmap("RdBu_r") # Use a diverging colormap

p_icv = ICV_diff.plot(
    ax=ax_icv_map,
    transform=ccrs.PlateCarree(),
    cmap=cmap_icv,
    norm=norm_icv,
    levels=levels_icv,
    add_colorbar=False
)

# Add titleInternal variability SAT difference\n
ax_icv_map.text(-0.12, 1.2, "d", transform=ax_icv_map.transAxes, fontsize=34, fontweight='bold', va='top')
ax_icv_map.set_title("MMLE - OBS\n(1993-2022)", fontsize=28, pad=10, loc='center')

# Add percentage annotation in the upper-right corner
ax_icv_map.text(0.95, 1.05, f"{mean_ratio_ICV:.0f}%", fontsize=28, ha='center', va='center',
                transform=ax_icv_map.transAxes)

# Add colorbar
cbar_ax_icv = fig.add_axes([0.61, 0.1, 0.25, 0.02])
# cbar_icv = plt.colorbar(
#     p_icv,
#     cax=cbar_ax_icv,
#     orientation='horizontal',
#     ticks=[-0.25, -0.125, 0, 0.125, 0.25],
#     extend='neither'  # No extension
# )
cbar_icv = plt.colorbar(
    p_icv,
    cax=cbar_ax_icv,
    orientation='horizontal',
    extend='neither',  # No extension
    ticks=[-0.25, -0.125, 0, 0.125, 0.25]
)

cbar_icv.set_label("Internal variability SAT differences\n(°C per decade)", fontsize=24, labelpad=10, loc='center')
cbar_icv.ax.tick_params(labelsize=22)
# Make ticks longer
cbar_icv.ax.tick_params(direction='out', length=10, width=2)  # Adjust values as needed
# tick labels for the colorbar in the subplot c and d
cbar_icv.set_ticks([-0.25, -0.125, 0, 0.125, 0.25])

# Save and display the figure
plt.tight_layout()
fig.savefig('Fig3.png', dpi=300, bbox_inches='tight')
fig.savefig('Fig3.pdf', dpi=300, bbox_inches='tight')
fig.savefig('Fig3.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

# In[]:
