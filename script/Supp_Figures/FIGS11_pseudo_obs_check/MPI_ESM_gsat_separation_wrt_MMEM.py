# %%
import numpy as np
import xarray as xr
import pandas as pd
import os
import logging
from mpi4py import MPI
# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# %%
# Load function modules
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess

# %%
# Load MMEM SAT data
input_model = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data/'
MPI_ESM_data = xr.open_mfdataset(input_model + 'tas_MPI_ESM_annual_ano_1850_2022.nc', chunks={'run': 1})
# %%
# Load GMST timeseries for different models
gmst_ENsemble = 'GMSAT_SMILEs_ENS_annual_timeseries_obtained_basedOn_ModelENS_CESM2_100realizations.nc'
input_ts_dir = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_timeseries/'
gmst_dict = xr.open_dataset(input_ts_dir+ gmst_ENsemble)
# %%
# Output directory
output_dir = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/pseudo_obs_check/'
os.makedirs(output_dir, exist_ok=True)

# Perform regression of SAT against this model's GMST
slope_array, intercept_array = data_process.linear_regression_single_gmst_multi_sat(
        gmst_dict['tas'], MPI_ESM_data['tas'].values
    )

    # Convert to xarray DataArray
slope_data = xr.DataArray(
        slope_array,
        coords={'run': MPI_ESM_data['run'], 'lat': MPI_ESM_data['lat'], 'lon': MPI_ESM_data['lon']},
        dims=['run', 'lat', 'lon'],
    )
intercept_data = xr.DataArray(
        intercept_array,
        coords={'run': MPI_ESM_data['run'], 'lat': MPI_ESM_data['lat'], 'lon': MPI_ESM_data['lon']},
        dims=['run', 'lat', 'lon'],
    )

# Reconstruct forced signal and unforced residual
var_scaled = slope_data * gmst_dict['tas'] + intercept_data
var_unforced = MPI_ESM_data['tas'] - var_scaled
    
# define variable names
var_scaled.name = 'tas'
var_unforced.name = 'tas'

    # Save results
output_path = f"{output_dir}/MPI_ESM_tas_forced_ano_wrt_MMEM_GSAT_1850_2022.nc"
output_ICV = f"{output_dir}/MPI_ESM_tas_unforced_ano_wrt_MMEM_GSAT_1850_2022.nc"
var_scaled.to_netcdf(output_path)
var_unforced.to_netcdf(output_ICV)
print(f"Finished writing outputs for MMEM")
# %%