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
# dir_forced_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/cesm2_100/trend_ICV_HadCRUT5_annual/'
dir_icv_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std/concatenate/'

HadCRUT5_trend = xr.open_dataset(dir_icv_input + 
                                 'OBS_ICV_MK_trend_STD_1950_2022_sliding.nc')

# The OBS field has an extra trend_length dim; pick the diagonal so period[i] uses trend_length[i]
obs_raw = HadCRUT5_trend["icv_trend_std"]
if "trend_length" in obs_raw.dims:
    n_match = min(obs_raw.sizes["period"], obs_raw.sizes["trend_length"])
    idx = xr.DataArray(np.arange(n_match), dims=["period"])
    obs_icv = obs_raw.isel(period=idx, trend_length=idx)
    obs_icv = obs_icv.drop_vars([v for v in obs_icv.coords if v == "trend_length"], errors="ignore")
else:
    obs_icv = obs_raw
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
dir_LE_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}/SMILE_internal/'

LE_annual_trend = {}
for model in MODELS:
    LE_annual_trend[model] = xr.open_dataset(
        dir_LE_in.format(model=model) +
        f'{model}_SMILE_noise_trend_std_sliding_1950_2022.nc'
    )
# In[3]:
def cal_ratio(data, pattern_diff):
    return pattern_diff / data
dir_output = "/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Pattern_diff/"
os.makedirs(dir_output, exist_ok=True)
# %%
dir_model_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_internal/'
# '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure6_Forced/data/Smiles_ensemble/'
MMLE_annual_trend = xr.open_dataset(dir_model_in + 
                                    'MMLE_internal_trend_std_1950-2022_sliding.nc')
# In[6]:
# calculate the pattern difference between LE_ens and OBS
pattern_diff_LE = {}
for model in MODELS:
    # Align OBS and model on common periods and spatial grid
    obs_aln, le_aln = xr.align(obs_icv, LE_annual_trend[model].noise_trend_std, join="inner")

    pattern_diff = le_aln - obs_aln
    pattern_diff_LE[model] = pattern_diff

    print(f"{model} pattern diff min: {pattern_diff.min().values}, max: {pattern_diff.max().values}")

    dir_output_LE = f"/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Pattern_diff/{model}/"
    os.makedirs(dir_output_LE, exist_ok=True)

    pattern_diff.to_dataset(name='trend_diff').to_netcdf(
        dir_output_LE + f'{model}_OBS_internal_pattern_diff_1950_2022.nc'
    )

    pattern_ratio = cal_ratio(obs_aln, pattern_diff)
    pattern_ratio.to_dataset(name='trend_ratio').to_netcdf(
        dir_output_LE + f'{model}_OBS_internal_pattern_ratio_1950_2022.nc'
    )

    mean_ratio_LE = pattern_ratio.sel(period="1979-2022").mean().values * 100
    print(f"{model} mean ratio (%): {mean_ratio_LE}")
# %%
