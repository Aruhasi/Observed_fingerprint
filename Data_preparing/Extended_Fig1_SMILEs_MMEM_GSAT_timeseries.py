# In[1]
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
import glob
# define function
import src.SAT_function as data_process
import src.Data_Preprocess as preprocess
# %%
import src.slurm_cluster as scluster
client, scluster = scluster.init_dask_slurm_cluster()
# In[2]:
# DIRECTORIES
# Path to input files
# HadCRUT5_dir    = '/work/mh0033/m301036/Data_storage/Data/Regrid/'
CanESM5_dir     = '/work/mh0033/m301036/Data_storage/CMIP6-CanESM5/MergeOut/ssp245/'
IPSL_dir        = '/work/mh0033/m301036/Data_storage/CMIP6-IPSL-CM6A-LR/Rawdata/regrid/'
EC_Earth_dir    = '/work/mh0033/m301036/Data_storage/CMIP6-EC-Earth/MergeOut/ssp245/'
ACCESS_dir      = '/work/mh0033/m301036/Data_storage/CMIP6-ACCESS/Rawdata/regrid/'
MPI_ESM_dir     = '/work/mh0033/m301036/Data_storage/CMIP6-MPI-ESM-LR/MergeOut/SSP245/'
MIROC6_dir      = '/work/mh0033/m301036/Data_storage/CMIP6-MIROC/MergeOut/ssp245/'
# Path to output files
# In[3]:
# read in the data
# ds_HadCRUT5 = xr.open_dataset(HadCRUT5_dir + 'tas_HadCRUT5_regrid.nc').tas_mean

ds_CanESM5  = xr.open_mfdataset(CanESM5_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
ds_IPSL     = xr.open_mfdataset(IPSL_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
ds_EC_Earth = xr.open_mfdataset(EC_Earth_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
ds_ACCESS   = xr.open_mfdataset(ACCESS_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
ds_MPI_ESM  = xr.open_mfdataset(MPI_ESM_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
ds_MIROC6   = xr.open_mfdataset(MIROC6_dir + 'tas_Amon_1850-2022_*.nc', combine='nested', concat_dim='run').tas
# In[4]:
# Preprocess the data
# calculate the annual mean anomalies
ds_CanESM5_1850_2022 = ds_CanESM5.sel(time=slice('1850', '2022')).squeeze() 
ds_CanESM5_1850_2022 = ds_CanESM5_1850_2022 - 273.15 
ds_IPSL_1850_2022 = ds_IPSL.sel(time=slice('1850', '2022')).squeeze() 
ds_IPSL_1850_2022 = ds_IPSL_1850_2022 - 273.15 
ds_EC_Earth_1850_2022 = ds_EC_Earth.sel(time=slice('1850', '2022')).squeeze()
ds_EC_Earth_1850_2022 = ds_EC_Earth_1850_2022 - 273.15 
ds_ACCESS_1850_2022 = ds_ACCESS.sel(time=slice('1850', '2022')).squeeze() 
ds_ACCESS_1850_2022 = ds_ACCESS_1850_2022 - 273.15 
ds_MPI_ESM_1850_2022 = ds_MPI_ESM.sel(time=slice('1850', '2022')).squeeze() 
ds_MPI_ESM_1850_2022 = ds_MPI_ESM_1850_2022 - 273.15 
ds_MIROC6_1850_2022 = ds_MIROC6.sel(time=slice('1850', '2022')).squeeze() 
ds_MIROC6_1850_2022 = ds_MIROC6_1850_2022 - 273.15 
# %%
# calculate the anomalies
ds_CanESM5_ano   = data_process.calc_anom_1961_1990(ds_CanESM5_1850_2022)
ds_IPSL_ano      = data_process.calc_anom_1961_1990(ds_IPSL_1850_2022)
ds_EC_Earth_ano  = data_process.calc_anom_1961_1990(ds_EC_Earth_1850_2022)
ds_ACCESS_ano    = data_process.calc_anom_1961_1990(ds_ACCESS_1850_2022)
ds_MPI_ESM_ano   = data_process.calc_anom_1961_1990(ds_MPI_ESM_1850_2022)
ds_MIROC6_ano    = data_process.calc_anom_1961_1990(ds_MIROC6_1850_2022)
# In[5]:
# annual mean
ds_HadCRUT5_annual_mean  = ds_HadCRUT5_ano.groupby('time.year').mean(dim='time')
ds_CanESM5_annual_mean   = ds_CanESM5_ano.groupby('time.year').mean(dim='time')
ds_IPSL_annual_mean      = ds_IPSL_ano.groupby('time.year').mean(dim='time')
ds_EC_Earth_annual_mean  = ds_EC_Earth_ano.groupby('time.year').mean(dim='time')
ds_ACCESS_annual_mean    = ds_ACCESS_ano.groupby('time.year').mean(dim='time')
ds_MPI_ESM_annual_mean   = ds_MPI_ESM_ano.groupby('time.year').mean(dim='time')
ds_MIROC6_annual_mean    = ds_MIROC6_ano.groupby('time.year').mean(dim='time')
# %%
# set the dimensional coordinates
ds_HadCRUT5_annual_mean = ds_HadCRUT5_annual_mean.assign_coords({'year': np.arange(1850, 2023), 'lat': ds_HadCRUT5['lat'], 'lon': ds_HadCRUT5['lon']})
ds_CanESM5_annual_mean = ds_CanESM5_annual_mean.assign_coords({'run':np.arange(1,51),
                                                                     'year': np.arange(1850, 2023), 'lat': ds_CanESM5['lat'], 'lon': ds_CanESM5['lon']})
ds_IPSL_annual_mean = ds_IPSL_annual_mean.assign_coords({'run':np.arange(1,33),
                                                                'year': np.arange(1850, 2023), 'lat': ds_IPSL['lat'], 'lon': ds_IPSL['lon']})
ds_EC_Earth_annual_mean = ds_EC_Earth_annual_mean.assign_coords({'run':np.arange(1,22),
                                                                        'year': np.arange(1850, 2023), 'lat': ds_EC_Earth['lat'], 'lon': ds_EC_Earth['lon']})
ds_ACCESS_annual_mean = ds_ACCESS_annual_mean.assign_coords({'run':np.arange(1,41),
                                                                    'year': np.arange(1850, 2023), 'lat': ds_ACCESS['lat'], 'lon': ds_ACCESS['lon']})
ds_MPI_ESM_annual_mean = ds_MPI_ESM_annual_mean.assign_coords({'run':np.arange(1,51),
                                                                    'year': np.arange(1850, 2023), 'lat': ds_MPI_ESM['lat'], 'lon': ds_MPI_ESM['lon']})
ds_MIROC6_annual_mean = ds_MIROC6_annual_mean.assign_coords({'run':np.arange(1,51),
                                                                    'year': np.arange(1850, 2023), 'lat': ds_MIROC6['lat'], 'lon': ds_MIROC6['lon']})
# %%
dir_out = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Datafiles/'
ds_HadCRUT5_annual_mean.to_netcdf(dir_out + 'tas_HadCRUT5_annual_ano_1850_2022.nc')
ds_CanESM5_annual_mean.to_netcdf(dir_out + 'tas_CanESM5_annual_ano_1850_2022.nc')
ds_IPSL_annual_mean.to_netcdf(dir_out + 'tas_IPSL_annual_ano_1850_2022.nc')
ds_EC_Earth_annual_mean.to_netcdf(dir_out + 'tas_EC_Earth_annual_ano_1850_2022.nc')
ds_ACCESS_annual_mean.to_netcdf(dir_out + 'tas_ACCESS_annual_ano_1850_2022.nc')
ds_MPI_ESM_annual_mean.to_netcdf(dir_out + 'tas_MPI_ESM_annual_ano_1850_2022.nc')
ds_MIROC6_annual_mean.to_netcdf(dir_out + 'tas_MIROC6_annual_ano_1850_2022.nc')
# %%
client.close()
scluster.close()
# In[6]:
dir_in = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Datafiles/'
# ds_HadCRUT5_annual_mean = xr.open_mfdataset(dir_in + 'tas_HadCRUT5_annual_ano_1850_2022.nc')
ds_CanESM5_annual_mean = xr.open_mfdataset(dir_in + 'tas_CanESM5_annual_ano_1850_2022.nc',chunks={'run':1})
ds_IPSL_annual_mean = xr.open_mfdataset(dir_in + 'tas_IPSL_annual_ano_1850_2022.nc',chunks={'run':1})
ds_EC_Earth_annual_mean = xr.open_mfdataset(dir_in + 'tas_EC_Earth_annual_ano_1850_2022.nc',chunks={'run':1})
ds_ACCESS_annual_mean = xr.open_mfdataset(dir_in + 'tas_ACCESS_annual_ano_1850_2022.nc',chunks={'run':1})
ds_MPI_ESM_annual_mean = xr.open_mfdataset(dir_in + 'tas_MPI_ESM_annual_ano_1850_2022.nc',chunks={'run':1})
ds_MIROC6_annual_mean = xr.open_mfdataset(dir_in + 'tas_MIROC6_annual_ano_1850_2022.nc',chunks={'run':1})

# %%
# calculate the global mean
# tas_HadCRUT5_annual_global  = data_process.calc_weighted_mean(ds_HadCRUT5_annual_mean['tas_mean'])
tas_CanESM5_annual_global   = data_process.calc_weighted_mean(ds_CanESM5_annual_mean['tas'])
tas_IPSL_annual_global      = data_process.calc_weighted_mean(ds_IPSL_annual_mean['tas'])
tas_EC_Earth_annual_global = data_process.calc_weighted_mean(ds_EC_Earth_annual_mean['tas'])
tas_ACCESS_annual_global   = data_process.calc_weighted_mean(ds_ACCESS_annual_mean['tas'])
tas_MPI_ESM_annual_global  = data_process.calc_weighted_mean(ds_MPI_ESM_annual_mean['tas'])
tas_MIROC6_annual_global   = data_process.calc_weighted_mean(ds_MIROC6_annual_mean['tas'])

# In[7]:
# plot the time series from 1850 to 2022
# Calculate the ensemble mean of the time series
tas_CanESM5_annual_global_ENS = tas_CanESM5_annual_global.mean(dim='run')
tas_IPSL_annual_global_ENS = tas_IPSL_annual_global.mean(dim='run')
tas_EC_Earth_annual_global_ENS = tas_EC_Earth_annual_global.mean(dim='run')
tas_ACCESS_annual_global_ENS = tas_ACCESS_annual_global.mean(dim='run')
tas_MPI_ESM_annual_global_ENS = tas_MPI_ESM_annual_global.mean(dim='run')
tas_MIROC6_annual_global_ENS = tas_MIROC6_annual_global.mean(dim='run')
# %%
# Calculate the SIMLEs mean of the global mean temperature
SMILE_data = xr.concat([tas_CanESM5_annual_global_ENS, tas_IPSL_annual_global_ENS, tas_EC_Earth_annual_global_ENS, 
                        tas_ACCESS_annual_global_ENS, tas_MPI_ESM_annual_global_ENS, tas_MIROC6_annual_global_ENS], dim='run')
# %%
SMILE_data_mean = SMILE_data.mean(dim='run')
# %%
dir_out = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure1/'
# tas_HadCRUT5_annual_global.to_netcdf(dir_out+'GMSAT_HadCRUT5_annual_timeseries.nc')
# tas_CanESM5_annual_global.to_netcdf(dir_out+'GMSAT_CanESM5_annual_timeseries.nc')
# tas_IPSL_annual_global.to_netcdf(dir_out+'GMSAT_IPSL_CM6A_annual_timeseries.nc')
# tas_EC_Earth_annual_global.to_netcdf(dir_out+'GMSAT_EC_Earth_annual_timeseries.nc')
# tas_ACCESS_annual_global.to_netcdf(dir_out+'GMSAT_ACCESS_annual_timeseries.nc')
# tas_MPI_ESM_annual_global.to_netcdf(dir_out+'GMSAT_MPI_ESM_annual_timeseries.nc')
# tas_MIROC6_annual_global.to_netcdf(dir_out+'GMSAT_MIROC6_annual_timeseries.nc')
# %%
# output the ensemble mean timeseries
# tas_CanESM5_annual_global_ENS.to_netcdf(dir_out+'GMSAT_CanESM5_annual_timeseries_ENS.nc')
# tas_IPSL_annual_global_ENS.to_netcdf(dir_out+'GMSAT_IPSL_CM6A_annual_timeseries_ENS.nc')
# tas_EC_Earth_annual_global_ENS.to_netcdf(dir_out+'GMSAT_EC_Earth_annual_timeseries_ENS.nc')
# tas_ACCESS_annual_global_ENS.to_netcdf(dir_out+'GMSAT_ACCESS_annual_timeseries_ENS.nc')
# tas_MPI_ESM_annual_global_ENS.to_netcdf(dir_out+'GMSAT_MPI_ESM_annual_timeseries_ENS.nc')
# tas_MIROC6_annual_global_ENS.to_netcdf(dir_out+'GMSAT_MIROC6_annual_timeseries_ENS.nc')
# %%
SMILE_data_mean.to_netcdf(dir_out+'GMSAT_SMILEs_ENS_annual_timeseries_obtained_basedOn_ModelENS.nc')
# %%
tas_HadCRUT5_annual_global = xr.open_dataset(dir_out+'tas_HadCRUT5_global_annual_mean_timeseries.nc')
tas_NOAA_annual_global = xr.open_dataset(dir_out+'tas_NOAA_global_annual_mean_timeseries.nc')
tas_Berkeley_annual_global = xr.open_dataset(dir_out+'tas_Berkeley_global_annual_mean_timeseries.nc')
# In[7]:
# tas_HadCRUT5_annual_global = tas_HadCRUT5_annual_global.rename({'__xarray_dataarray_variable__':'tas'})
# tas_NOAA_annual_global = tas_NOAA_annual_global.rename({'__xarray_dataarray_variable__':'tas'})
# tas_Berkeley_annual_global = tas_Berkeley_annual_global.rename({'__xarray_dataarray_variable__':'tas'})

# In[8]:
#Plotting
# setting the parameters for the figure
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True
x = np.arange(1850, 2023, 1)
# %%
obs_name = ["HadCRUT5", "NOAAGlobalTemp", "BerkeleyEarth"]
model_names = ["CanESM5(50)", "IPSL-CM6A-LR(14)","EC-Earth3(21)", "ACCESS-ESM1-5(37)", "MPI-ESM-1-2(LR)(50)","MIROC6(50)"]
scenarios = ['SSP245', 'SSP585']

RGB_dict = {'CanESM5(50)':np.array([50, 34, 136])/255., 
            'IPSL-CM6A-LR(14)':np.array([68, 170, 152])/255., 
            'EC-Earth3(21)':np.array([221, 204, 118])/255., 
            'ACCESS-ESM1-5(37)':np.array([204, 101, 119])/255.,
            'MPI-ESM-1-2(LR)(50)':np.array([170, 67, 153])/255., 
            'MIROC6(50)':np.array([136, 33, 85])/255., 
            'MME':np.array([0, 0, 0])/255.}
# RGB_dict = {'CanESM5(50)':np.array([67, 178, 216])/255., 
#             'IPSL-CM6A-LR(14)':np.array([122, 139, 38])/255.,
#             'EC-Earth3(21)':np.array([124, 99, 184])/255., 
#             'ACCESS-ESM1-5(37)':np.array([30, 76, 36])/255.,
#             'MPI-ESM-1-2(LR)(50)':np.array([93, 161,162])/255., 
#             'MIROC6(50)':np.array([35, 54, 109])/255., 
#             'MME':np.array([0, 0, 0])/255.}
# %%
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.legend import Legend

# %%
num_time_series = 50
num_time_series_1 = 14
num_time_series_2 = 21
num_time_series_3 = 37
num_time_series_4 = 50
num_time_series_5 = 50

# c_MPI = 'c'#'black'
# c_MIROC6 = 'tab:brown'
# c_CanESM5 = 'y'
# c_MME = 'black'
c_obs = "orange"
c_obs_1 = "blue"
c_obs_2 = "green" #'darkred'

obs_color = {
    "NOAAGlobalTemp": c_obs,
    "HadCRUT5": c_obs_1,
    "BerkeleyEarth": c_obs_2
}

for obs in obs_name:
    print(obs)
    obs_color[obs]
    print(obs_color[obs])
    
lw_obs = 1.5
lw_model = 2.5
xmin, xmax = 1850, 2022
ymin, ymax = -1.5, 2.0


import matplotlib.lines as Line2D

fig, ax = plt.subplots(figsize=(15, 6))
# f = plt.gcf()

plt.plot([xmin,xmax],[0,0], color='grey', linestyle='-', linewidth=0.75)

for i in range(num_time_series):
    label = 'CanESM5(50)' if i == 0 else None
    plt.plot(x, tas_CanESM5_annual_global[i,:], color=RGB_dict['CanESM5(50)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)

for i in range(num_time_series_1):
    label = 'IPSL-CM6A-LR(14)' if i == 0 else None
    plt.plot(x, tas_IPSL_annual_global[i,:], color=RGB_dict['IPSL-CM6A-LR(14)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)
   
for i in range(num_time_series_2):
    label = 'EC-Earth3(21)' if i == 0 else None
    plt.plot(x, tas_EC_Earth_annual_global[i,:], color=RGB_dict['EC-Earth3(21)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)

for i in range(num_time_series_3):
    label = 'ACCESS-ESM1-5(37)' if i == 0 else None
    plt.plot(x, tas_ACCESS_annual_global[i,:], color=RGB_dict['ACCESS-ESM1-5(37)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)

for i in range(num_time_series_4):
    label = 'MPI-ESM-1-2(LR)(50)' if i == 0 else None
    plt.plot(x, tas_MPI_ESM_annual_global[i,:], color=RGB_dict['MPI-ESM-1-2(LR)(50)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)

for i in range(num_time_series_5):
    label = 'MIROC6(50)' if i == 0 else None
    plt.plot(x, tas_MIROC6_annual_global[i,:], color=RGB_dict['MIROC6(50)'], linestyle= '--', linewidth=0.75, alpha=0.75, label=label)
# plt.legend(loc='best')

plt.plot(x, tas_NOAA_annual_global['tas'],  marker='o', linestyle='-', markerfacecolor='none', markeredgewidth=1.5, label='NOAAGlobalTemp', color=c_obs, linewidth=lw_obs)
plt.plot(x, tas_HadCRUT5_annual_global['tas'], marker='o', linestyle='-', markerfacecolor='none', markeredgewidth=1.5, label='HadCRUT5', color=c_obs_1, linewidth=lw_obs)
plt.plot(x, tas_Berkeley_annual_global['tas'], marker='o', linestyle='-', markerfacecolor='none', markeredgewidth=1.5, label='BerkeleyEarth', color=c_obs_2, linewidth=lw_obs)

plt.plot(x, SMILE_data_mean, color=RGB_dict['MME'], linestyle= '-', label='Multi-mean', linewidth=lw_model)

plt.axvspan(1950, 2022, alpha=0.25, color='grey')
plt.text(1950, 2.10, '1950-2022', fontsize=18)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
# plt.title('a 1850-2022 GSAT ', loc='left', fontweight='bold', pad=10, fontsize=22)

plt.ylabel('SAT anomaly relative to 1961-1990(°C)')
legend = plt.legend(loc='upper left', fontsize=12, ncol=3)

for line in legend.get_lines():
    line.set_linewidth(1.5)  # Set this to your desired line width

# custom_lines_models = [Line2D([0], [0], color=RGB_dict[name], linestyle='-', label=name) for name in model_names]
# legend2 = ax.legend(handles=custom_lines_models, loc='upper left', bbox_to_anchor=(0, 1 - len(obs_name) * 0.12), title='SMILEs')
# ax.add_artists(legend2)
plt.tight_layout()
plt.savefig("test-Fig-GSAT-annual-timeseries-1850-2022.png", dpi=300, bbox_inches='tight')

plt.show()
# %%
client.close()
scluster.close()
# %%
