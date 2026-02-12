#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute OBS regional SAT raw trend anomalies

Method:
- Input the Compute raw trend anomalies of SAT from observations.
- Sliding windows regional averaged calculation: 2013–2022, 2012–2022, ..., 1950–2022
  (window lengths 10–73 years); Arctic, NAWH, SEP, SOP, NPI.
- For each window:
    * compute regional mean SAT trend
    * adopted sub function from SAT_function_Obs_Fingerprint.py: selreg; calc_weighted_mean;
    * convert to K/decade
Usage:
    mpirun -np N python -u Step1_regional_OBS_SAT_Raw_trend_anomalies.py {region_name}
"""
# In[1]:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# %%
# define function
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess

# %%
dir_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG2'

region_name = 'SOP'  # 'Arctic', 'NAWH', 'SEP', 'SOP', 'NPI'

reg_latlon_dict = {
    'Arctic': {'lat': [66, 90], 'lon': [-180, 180]},
    'NAWH': {'lat': [20, 65], 'lon': [-160, -20]},
    'SEP': {'lat': [-45, -15], 'lon': [110, 180]},
    'SOP': {'lat': [-45, -15], 'lon': [0, 70]},
    'NPI': {'lat': [30, 65], 'lon': [0, 60]},
}
OBS_ds = xr.open_dataset(f'{dir_in}/Raw_HadCRUT5_MK_trend_1950-2022_sliding.nc')

# %%
# SO box 
so_lon_mid = (0 + 360) / 2
so_lat_mid = (-65 + -50) / 2
ax.plot([0, 360, 360, 0, 0], [-70, -70, -50, -50, -70],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(so_lon_mid, so_lat_mid, 'SO', color='black', fontsize=18, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SO
# Extratropical South Pacific box
sop_lon_mid = (230 + 280) / 2
sop_lat_mid = (-40 + -70) / 2
ax.plot([230%360, 280%360, 280%360, 230%360, 230%360], [-40, -40, -70, -70, -40],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(sop_lon_mid, sop_lat_mid, 'SOP', color='black', fontsize=18, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for SOP

# North Pacific box
npac_lon_mid = (175 + 220) / 2
npac_lat_mid = (30 + 50) / 2
ax.plot([175%360, 220%360, 220%360, 175%360, 175%360], [30, 30, 50, 50, 30],
        color='grey', linewidth=2.0, transform=ccrs.PlateCarree())
ax.text(npac_lon_mid, npac_lat_mid, 'NPI', color='black', fontsize=18, transform=ccrs.PlateCarree(),
        ha='center', va='center')  # Label for NPI 