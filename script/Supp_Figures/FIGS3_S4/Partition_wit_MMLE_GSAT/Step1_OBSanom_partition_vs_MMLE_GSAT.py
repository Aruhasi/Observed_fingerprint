# In[1]:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# define function
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess
# %% load data
input_observation = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/OBS_data/'
OBS_data = ['Berkeley', 'HadCRUT5', 'NOAA']
OBS = OBS_data[0]  # 'NOAA'

OBS_annual_ano = xr.open_dataset(input_observation + f'tas_{OBS}_annual_anomalies.nc')
# %%
input_model = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_timeseries/'
ACCESS_GMSAT_annual_ENS     = xr.open_dataset(input_model + 'GMSAT_ACCESS_annual_timeseries_ENS.nc')
CanESM_GMSAT_annual_ENS     = xr.open_dataset(input_model + 'GMSAT_CanESM5_annual_timeseries_ENS.nc')
CESM2_GMSAT_annual_ENS      = xr.open_dataset(input_model + 'GMSAT_CESM2LE_CMIP6_SMBB_annual_timeseries_ENS.nc')
EC_Earth_GMSAT_annual_ENS   = xr.open_dataset(input_model + 'GMSAT_EC_Earth_annual_timeseries_ENS.nc')
IPSL_GMSAT_annual_ENS   = xr.open_dataset(input_model + 'GMSAT_IPSL_CM6A_annual_timeseries_ENS.nc')
MIROC_GMSAT_annual_ENS  = xr.open_dataset(input_model + 'GMSAT_MIROC6_annual_timeseries_ENS.nc')
MPI_GMSAT_annual_ENS    = xr.open_dataset(input_model + 'GMSAT_MPI_ESM_annual_timeseries_ENS.nc')
# %%
# input_cesm2le = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/CESM2_LE/data/Timeseries/'
# tas_annual_gsat = xr.open_dataset(input_cesm2le + 'GMSAT_CESM2LE_CMIP6_SMBB_annual_timeseries_ENS_1850-2022.nc')['TREFHT']
# CESM2_GMSAT_annual_ENS = tas_annual_gsat.mean(dim='member').to_dataset(name='tas')
# %%
# dir_out = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_timeseries/"
# CESM2_GMSAT_annual_ENS.to_netcdf(dir_out + 'GMSAT_CESM2LE_cmip6+smbb_annual_timeseries_ENS.nc')
# %%
INPUT_MMLE_LE = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/CESM2_LE/'
MMLE_GMSAT_annual_ENS = xr.open_dataset(INPUT_MMLE_LE + 'GMSAT_SMILEs_ENS_annual_timeseries_obtained_basedOn_ModelENS_CESM2_100realizations.nc')
# In[3]:
# ------ configure model GSAT dictionary ------
GSAT_ENS = {
    "MIROC6":    MIROC_GMSAT_annual_ENS["tas"],
    "MPI_ESM":   MPI_GMSAT_annual_ENS["tas"],
    "ACCESS":    ACCESS_GMSAT_annual_ENS["tas"],
    "EC_Earth3": EC_Earth_GMSAT_annual_ENS["tas"],
    "IPSL_CM6A": IPSL_GMSAT_annual_ENS["tas"],
    "CESM2":     CESM2_GMSAT_annual_ENS["tas"],
    "CanESM5":   CanESM_GMSAT_annual_ENS["tas"],
    "MMLE":   MMLE_GMSAT_annual_ENS["tas"],  # if needed
}
import numpy as np

def forced_signal_reconstruction(gsat_1d, slope_2d, intercept_2d):
    """
    Reconstruct forced SAT signal from GSAT and regression coefficients.

    Parameters
    ----------
    gsat_1d : array-like, shape (time,)
        Global-mean SAT time series (e.g. GSAT_ENS['tas'].values).
    slope_2d : array-like, shape (lat, lon)
        Regression slope (lat, lon).
    intercept_2d : array-like, shape (lat, lon)
        Regression intercept (lat, lon).

    Returns
    -------
    forced_3d : np.ndarray, shape (time, lat, lon)
        Reconstructed forced SAT field.
    """
    g = np.asarray(gsat_1d)          # (time,)
    b = np.asarray(slope_2d)         # (lat, lon)
    a = np.asarray(intercept_2d)     # (lat, lon)

    # Broadcast:
    # a[None, :, :]   -> (1,   lat, lon)
    # g[:, None, None]-> (time,1,   1)
    # result          -> (time,lat, lon)
    forced = a[None, :, :] + b[None, :, :] * g[:, None, None]
    return forced

def forced_signal_reconstruction_xr(gsat_da, slope_da, intercept_da):
    """
    xarray-based version. Assumes
      - gsat_da dims: ('year',) or ('time',)
      - slope_da, intercept_da dims: ('lat', 'lon')
    Returns DataArray with dims ('year', 'lat', 'lon') or ('time','lat','lon').
    """
    # make sure time dim has a consistent name
    time_dim = "year" if "year" in gsat_da.dims else "time"
    g = gsat_da.rename({time_dim: "year"})

    # xarray broadcasting does the magic:
    forced = intercept_da + slope_da * g
    return forced.transpose("year", "lat", "lon")

# %%
# apply the "OBS-LPS" framework to partition the observed SAT anomalies
MODEL_LIST = ["MIROC6", "MPI_ESM", "ACCESS", "EC_Earth3", "IPSL_CM6A", "CESM2", "CanESM5", "MMLE"]

lat = OBS_annual_ano["lat"]
lon = OBS_annual_ano["lon"]
year = OBS_annual_ano["year"]

OBS_slope             = xr.Dataset()
OBS_intercept         = xr.Dataset()
OBS_forced_signal     = xr.Dataset()
OBS_internal_variab   = xr.Dataset()

for model in MODEL_LIST:
    gsat = GSAT_ENS[model]

    # Optional: enforce same year range
    common_years = np.intersect1d(gsat["year"], year)
    gsat_common = gsat.sel(year=common_years)
    obs_common  = OBS_annual_ano["tas"].sel(year=common_years)

    # 1) regression: returns 2D arrays (lat, lon)
    slope, intercept = data_process.linear_regression_gmst(
        gsat_common.values,      # time
        obs_common.values        # time, lat, lon
    )

    # wrap into DataArrays with coords
    slope_da = xr.DataArray(
        slope,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name=f"slope_{model}",
    )
    intercept_da = xr.DataArray(
        intercept,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name=f"intercept_{model}",
    )

    OBS_slope[model]     = slope_da
    OBS_intercept[model] = intercept_da

    # 2) forced signal reconstruction (time, lat, lon)
    forced = forced_signal_reconstruction(
        gsat_common.values,  # time
        slope,
        intercept,
    )
    forced_da = xr.DataArray(
        forced,
        coords={"year": common_years, "lat": lat, "lon": lon},
        dims=("year", "lat", "lon"),
        name=f"forced_{model}",
    )
    OBS_forced_signal[model] = forced_da

    # 3) internal variability = obs - forced
    internal_da = obs_common - forced_da
    internal_da.name = f"internal_{model}"
    OBS_internal_variab[model] = internal_da
# In[4]:
# save data
output_path = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS_OBS_records/'
os.makedirs(output_path, exist_ok=True)

for Model in MODEL_LIST:
    ds_to_save = xr.Dataset({
        'slope': OBS_slope[Model],
        'intercept': OBS_intercept[Model],
        'forced_signal': OBS_forced_signal[Model],
        'internal_variability': OBS_internal_variab[Model],
    })
    output_file = output_path + f'{OBS}_SAT_anomaly_partition_wrt_{Model}_ENS.nc'
    ds_to_save.to_netcdf(output_file)
# %%
