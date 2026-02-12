#!/usr/bin/env python3
"""
Sanity check for LE forced vs total SAT anomalies.

For a chosen model + member:
  - load the original SAT anomaly field (all members)
  - load the partitioned file (forced_signal + internal_variability)
  - compute global-mean (area-weighted) SAT for:
        * total anomalies
        * forced_signal
  - plot them together as a time series
  - print correlation between forced GM and model-ENS GSAT (optional)
"""
# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- SETTINGS (edit here) -----------------
MODEL = "MPI_ESM"          # e.g. "MPI_ESM", "CESM2", "CanESM5", ...
MEMBER_SEL = 1    # member label to check (see printout below) - use integer not string

START_YEAR = 1850
END_YEAR   = 2022

# Directories (adapt to your paths)
FIELD_DIR   = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_data")
GSAT_ENS_DIR = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/LE_timeseries")
FIG3_ROOT   = Path("/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3")

# Map model -> SAT anomaly file (all members)
FIELD_FILE_PATTERN = {
    "CanESM5":   "tas_CanESM5_annual_ano_1850_2022.nc",
    "ACCESS":    "tas_ACCESS_annual_ano_1850_2022.nc",
    "IPSL_CM6A": "tas_IPSL_annual_ano_1850_2022.nc",
    "MPI_ESM":   "tas_MPI_ESM_annual_ano_1850_2022.nc",
    "MIROC6":    "tas_MIROC6_annual_ano_1850_2022.nc",
    "EC_Earth3": "tas_EC_Earth_annual_ano_1850_2022.nc",
    "CESM2":     "CMIP6_tas_annual_mean_all_members.nc",
}

# Map model -> ensemble-mean GSAT file (optional, for extra check)
GSAT_ENS_FILE = {
    "ACCESS":    "GMSAT_ACCESS_annual_timeseries_ENS.nc",
    "CanESM5":   "GMSAT_CanESM5_annual_timeseries_ENS.nc",
    "EC_Earth3": "GMSAT_EC_Earth_annual_timeseries_ENS.nc",
    "IPSL_CM6A": "GMSAT_IPSL_CM6A_annual_timeseries_ENS.nc",
    "MIROC6":    "GMSAT_MIROC6_annual_timeseries_ENS.nc",
    "MPI_ESM":   "GMSAT_MPI_ESM_annual_timeseries_ENS.nc",
    "CESM2":     "GMSAT_CESM2LE_annual_timeseries_ENS.nc",
}
# %%
# ----------------- Helpers -----------------
def detect_dims(da):
    """
    Infer time, lat, lon, member dims from a 4D DataArray.
    """
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
        raise ValueError(f"Expected one member dim, got {member_dims}")
    member_dim = member_dims[0]
    return time_dim, lat_dim, lon_dim, member_dim
def get_gsat_ens(model_name):
    """Load ensemble-mean GSAT (1D, year)."""
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

# %%
# ----------------- Load data -----------------
# 1) total SAT anomalies (all members)
field_file = FIELD_DIR / FIELD_FILE_PATTERN[MODEL]
ds_sat = xr.open_dataset(field_file)

# SAT variable
sat_var_candidates = [
    name for name, var in ds_sat.data_vars.items()
    if set(var.dims) & {"time", "year"}
]
if not sat_var_candidates:
    raise ValueError(f"No SAT-like variable found in {field_file}")
sat_name = sat_var_candidates[0]
tas_all = ds_sat[sat_name]

time_dim, lat_dim, lon_dim, member_dim = detect_dims(tas_all)

if time_dim != "year":
    tas_all = tas_all.rename({time_dim: "year"})
    time_dim = "year"

tas_all = tas_all.sel(year=slice(START_YEAR, END_YEAR))

# Show available members so you can pick one
print(f"Available members for {MODEL}:")
print(tas_all[member_dim].values)

# Pick requested member
tas_m = tas_all.sel({member_dim: MEMBER_SEL}).transpose("year", lat_dim, lon_dim)

# 2) partition file (forced + residual for this model)
part_file = FIG3_ROOT / MODEL / f"SAT_anomaly_partition_wrt_{MODEL}_ENS_GSAT_{START_YEAR}-{END_YEAR}.nc"
ds_part = xr.open_dataset(part_file)

forced_m = ds_part["forced_signal"].sel({member_dim: MEMBER_SEL}).transpose("year", lat_dim, lon_dim)
# %%
# Align years (just in case)
common_years = np.intersect1d(tas_m["year"].values, forced_m["year"].values)
tas_m    = tas_m.sel(year=common_years)
forced_m = forced_m.sel(year=common_years)

lat = tas_m[lat_dim]
lon = tas_m[lon_dim]
# %%
# ----------------- Global means -----------------
# simple area weights ~ cos(lat)
coslat = np.cos(np.deg2rad(lat))
weights = xr.DataArray(coslat, dims=(lat_dim,), coords={lat_dim: lat})

gm_total  = tas_m.weighted(weights).mean(dim=(lat_dim, lon_dim))
gm_forced = forced_m.weighted(weights).mean(dim=(lat_dim, lon_dim))

# Optional: compare forced GM with ensemble-mean GSAT used in regression
gsat_ens = get_gsat_ens(MODEL).sel(year=gm_forced["year"])
corr_forced_gsat = np.corrcoef(gm_forced.values, gsat_ens.values)[0, 1]
print(f"Correlation(global-mean forced SAT, ensemble-mean GSAT) for {MODEL}, {MEMBER_SEL}: {corr_forced_gsat:.3f}")
# %%
# ----------------- Plot -----------------
plt.figure(figsize=(10, 5))
plt.plot(gm_total["year"], gm_total, label="Total SAT anomaly (GM)", lw=1.8)
plt.plot(gm_forced["year"], gm_forced, label="Forced SAT (GM, regression)", lw=1.8)
plt.xlabel("Year")
plt.ylabel("Global-mean SAT anomaly (°C)")
plt.title(f"{MODEL} – {MEMBER_SEL}: total vs forced global-mean SAT")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%