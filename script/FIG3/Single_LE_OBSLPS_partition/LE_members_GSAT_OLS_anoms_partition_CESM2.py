#!/usr/bin/env python3
"""
Step 1: Use each LE's ENSEMBLE-MEAN GSAT to separate each realization's
SAT anomalies into forced signal and internal variability (residual).
"""
# %%
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

MODEL_LIST = ["CESM2"]

FIELD_FILE_PATTERN = {
    "MIROC6":   "tas_MIROC6_annual_ano_1850_2022.nc",
    "MPI_ESM":  "tas_MPI_ESM_annual_ano_1850_2022.nc",
    "ACCESS":   "tas_ACCESS_annual_ano_1850_2022.nc", 
    "EC_Earth3": "tas_EC_Earth_annual_ano_1850_2022.nc",
    "IPSL_CM6A": "tas_IPSL_annual_ano_1850_2022.nc",
    "CESM2":    "tas_CESM2_CMIP6_SMBB_annual_ano_1850_2022.nc",
    "CanESM5":  "tas_CanESM5_annual_ano_1850_2022.nc",
}

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
    forced_list   = []
    residual_list = []

    for mem in members:
        print(f"[rank {rank}]    member: {mem}", flush=True)
        tas_m = tas_all.sel({member_dim: mem}).transpose("year", lat_dim, lon_dim)

        # regression
        slope_2d, intercept_2d = data_process.linear_regression_gmst(
            gsat_ens.values,   # (ntime,)
            tas_m.values       # (ntime, nlat, nlon)
        )

        forced_arr = data_process.forced_signal_reconstruction(
            gsat_ens.values,   # (ntime,)
            slope_2d,          # (nlat, nlon)
            intercept_2d,      # (nlat, nlon)
        )

        forced_da = xr.DataArray(
            forced_arr,
            dims=("year", lat_dim, lon_dim),
            coords={
                "year": tas_m["year"],
                lat_dim: tas_m[lat_dim],
                lon_dim: tas_m[lon_dim],
            },
            name="forced_signal",
        )
        residual_da = (tas_m - forced_da).rename("internal_variability")

        forced_list.append(forced_da.expand_dims({member_dim: [mem]}))
        residual_list.append(residual_da.expand_dims({member_dim: [mem]}))
    # Ensure member labels are a proper fixed-size dtype (string)
    members_arr = np.asarray(members).astype(str)

    # Concatenate along member dimension (DataArray-safe concat)
    forced_all = xr.concat(forced_list, dim=member_dim)
    residual_all = xr.concat(residual_list, dim=member_dim)

    # Attach member coordinate (not a data variable)
    forced_all = forced_all.assign_coords({member_dim: members_arr})
    residual_all = residual_all.assign_coords({member_dim: members_arr})

    # Build output Dataset
    ds_out = xr.Dataset(
        {
            "forced_signal": forced_all,
            "internal_variability": residual_all,
        }
    )

    model_out_dir = FIG3_ROOT / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = model_out_dir / f"SAT_anomaly_partition_wrt_{model_name}_CMIP6_SMBB_ENS_GSAT_{START_YEAR}-{END_YEAR}.nc"
    print(f"[rank {rank}]  -> writing {out_file}", flush=True)
    ds_out.to_netcdf(out_file)

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