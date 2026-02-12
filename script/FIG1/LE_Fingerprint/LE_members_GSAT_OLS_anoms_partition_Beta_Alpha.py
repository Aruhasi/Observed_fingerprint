#!/usr/bin/env python3
"""
Step 1: Use each LE's ENSEMBLE-MEAN GSAT to separate each realization's
SAT anomalies into forced signal and internal variability (residual).
"""

import os
from pathlib import Path
import numpy as np
import xarray as xr
from mpi4py import MPI

# --------------------------------------------------------------------
# MPI setup
# --------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess  # optional

# ---------------------- Settings ----------------------
START_YEAR = 1850
END_YEAR   = 2022

FIELD_DIR = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data")
GSAT_ENS_DIR = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_timeseries")
FIG3_ROOT = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3")

if rank == 0:
    FIG3_ROOT.mkdir(parents=True, exist_ok=True)
comm.Barrier()

MODEL_LIST = ["MIROC6", "MPI_ESM", "ACCESS", "EC_Earth3", "IPSL_CM6A", "CESM2", "CanESM5"]

FIELD_FILE_PATTERN = {
    "MIROC6":   "tas_MIROC6_annual_ano_1850_2022.nc",
    "MPI_ESM":  "tas_MPI_ESM_annual_ano_1850_2022.nc",
    "ACCESS":   "tas_ACCESS_annual_ano_1850_2022.nc", 
    "EC_Earth3": "tas_EC_Earth3_annual_ano_1850_2022.nc",
    "IPSL_CM6A": "tas_IPSL_CM6A_annual_ano_1850_2022.nc",
    "CESM2":     "tas_CESM2_CMIP6_SMBB_annual_ano_1850-2022.nc",
    "CanESM5":   "tas_CanESM5_annual_ano_1850_2022.nc",
}
# %%
# # read in the SAT anomalies for each model and Check
# ds_CESM2LE = xr.open_dataset(f'{FIELD_DIR}/CESM2LE_CMIP6_SMBB_tas_ano_1961-1990_annual_mean_1850-2022.nc')
# tas_CESM2LE_ano = ds_CESM2LE['TREFHT']

# ds_CanESM5 = xr.open_dataset(f'{FIELD_DIR}/tas_CanESM5_annual_ano_1850_2022.nc')
# tas_CanESM5_ano = ds_CanESM5['tas']

# %%
GSAT_ENS_FILE = {
    "MIROC6":   "GMSAT_MIROC6_annual_timeseries_ENS.nc",
    "MPI_ESM":  "GMSAT_MPI_ESM_annual_timeseries_ENS.nc",
    "ACCESS":   "GMSAT_ACCESS_annual_timeseries_ENS.nc",
    "EC_Earth3": "GMSAT_EC_Earth_annual_timeseries_ENS.nc",
    "IPSL_CM6A": "GMSAT_IPSL_CM6A_annual_timeseries_ENS.nc",
    "CESM2":    "GMSAT_CESM2LE_CMIP6_SMBB_annual_timeseries_ENS.nc",
    "CanESM5":  "GMSAT_CanESM5_annual_timeseries_ENS.nc",
}

def detect_dims(da):
    cand_time = [d for d in da.dims if d in ("time", "year")]
    cand_lat  = [d for d in da.dims if d.lower() in ("lat", "latitude", "y")]
    cand_lon  = [d for d in da.dims if d.lower() in ("lon", "longitude", "x")]

    if len(cand_time) != 1 or len(cand_lat) != 1 or len(cand_lon) != 1:
        raise ValueError(f"Could not infer dims for DataArray with dims {da.dims}")

    time_dim = cand_time[0]
    lat_dim  = cand_lat[0]
    lon_dim  = cand_lon[0]

    member_dims = [d for d in da.dims if d not in (time_dim, lat_dim, lon_dim)]
    if len(member_dims) != 1:
        raise ValueError(f"Expected exactly one member dim, got {member_dims}")
    member_dim = member_dims[0]

    return time_dim, lat_dim, lon_dim, member_dim

def get_gsat_da(model_name):
    fname = GSAT_ENS_DIR / GSAT_ENS_FILE[model_name]
    ds = xr.open_dataset(fname)

    vname = list(ds.data_vars)[0]
    da = ds[vname]

    if "year" in da.dims:
        pass
    elif "time" in da.dims:
        da = da.rename({"time": "year"})
    else:
        raise ValueError(f"GSAT file {fname} has no 'time' or 'year' dimension")

    da = da.sel(year=slice(START_YEAR, END_YEAR))
    return da

def process_model(model_name):
    print(f"\n[rank {rank}] === Processing model: {model_name} ===", flush=True)

    field_file = FIELD_DIR / FIELD_FILE_PATTERN[model_name]
    if not field_file.exists():
        raise FileNotFoundError(f"SAT anomaly file not found: {field_file}")

    ds_sat = xr.open_dataset(field_file)

    # pick SAT variable
    sat_var_candidates = [
        name for name, var in ds_sat.data_vars.items()
        if set(var.dims) & {"time", "year"}
    ]
    if not sat_var_candidates:
        raise ValueError(f"No SAT-like variable found in {field_file}")
    sat_name = sat_var_candidates[0]
    tas_all = ds_sat[sat_name]

    time_dim, lat_dim, lon_dim, member_dim = detect_dims(tas_all)
    print(f"[rank {rank}] dims: time={time_dim}, lat={lat_dim}, lon={lon_dim}, member={member_dim}", flush=True)

    if time_dim != "year":
        tas_all = tas_all.rename({time_dim: "year"})
        time_dim = "year"

    tas_all = tas_all.sel(year=slice(START_YEAR, END_YEAR))

    gsat_ens = get_gsat_da(model_name)
    common_years = np.intersect1d(tas_all["year"].values, gsat_ens["year"].values)
    tas_all = tas_all.sel(year=common_years)
    gsat_ens = gsat_ens.sel(year=common_years)

    members = tas_all[member_dim].values
    slope_forced_signal_da_list   = []
    intercept_forced_signal_da_list = []

    for mem in members:
        print(f"[rank {rank}]    member: {mem}", flush=True)
        tas_m = tas_all.sel({member_dim: mem}).transpose("year", lat_dim, lon_dim)

        # regression
        slope_2d, intercept_2d = data_process.linear_regression_gmst(
            gsat_ens.values,   # (ntime,)
            tas_m.values       # (ntime, nlat, nlon)
        )

        slope_forced_signal_da = xr.DataArray(
            slope_2d,
            dims=(lat_dim, lon_dim),
            coords={
                lat_dim: tas_m[lat_dim],
                lon_dim: tas_m[lon_dim],
            },
            name="forced_slope",
        )
        intercept_forced_signal_da = xr.DataArray(
            intercept_2d,
            dims=(lat_dim, lon_dim),
            coords={
                lat_dim: tas_m[lat_dim],
                lon_dim: tas_m[lon_dim],
            },
            name="forced_intercept",
        )
        slope_forced_signal_da_list.append(slope_forced_signal_da.expand_dims({member_dim: [mem]}))
        intercept_forced_signal_da_list.append(intercept_forced_signal_da.expand_dims({member_dim: [mem]}))
    # Ensure member labels are a proper fixed-size dtype (string)
    members_arr = np.asarray(members).astype(str)

    # Concatenate along member dimension (DataArray-safe concat)
    slope_all = xr.concat(slope_forced_signal_da_list, dim=member_dim)
    intercept_all = xr.concat(intercept_forced_signal_da_list, dim=member_dim)

    # Attach member coordinate (not a data variable)
    slope_all = slope_all.assign_coords({member_dim: members_arr})
    intercept_all = intercept_all.assign_coords({member_dim: members_arr})  
    
    model_out_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG1/Fingerprint/cesm2_100"
    model_out_dir = Path(model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    slope_force = model_out_dir / f"GSAT_{model_name}_slope_Beta_coefficients_{START_YEAR}-{END_YEAR}.nc"
    print(f"[rank {rank}]  -> writing {slope_force}", flush=True)
    slope_all.to_dataset(name="coefficients").to_netcdf(slope_force)
    intercept_force = model_out_dir / f"GSAT_{model_name}_intercept_Alpha_constants_{START_YEAR}-{END_YEAR}.nc"
    print(f"[rank {rank}]  -> writing {intercept_force}", flush=True)
    intercept_all.to_dataset(name="intercept").to_netcdf(intercept_force)
# %%
# ---------- main MPI driver ----------
def main():
    # round-robin: each rank gets a subset of models
    for i, model in enumerate(MODEL_LIST):
        if i % size == rank:
            process_model(model)

    comm.Barrier()
    if rank == 0:
        print("All ranks finished LE partition.", flush=True)

if __name__ == "__main__":
    main()
# %%