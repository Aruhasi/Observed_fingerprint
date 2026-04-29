#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute regional-mean trends from per-run MK trend PATTERNS (period, lat, lon).

Input per model (recommended merged file):
  <BASE>/<MODEL>/per_run_patterns/<MODEL>_ALLRUNS_resid_MKtrend_patterns_1950_2022_sliding.nc
with variable:
  trend(run_id, period, lat, lon)

This script runs ONE run_id (Slurm array task):
  - loads only that run's trend(period, lat, lon)
  - converts lon to [-180,180]
  - computes area-weighted regional mean trends for 6 regions:
      Arctic, Subpolar_gyre (NAWH-style), SoutheastPacific (polygon), SOP, NPI, SO
  - saves a small NetCDF:
      trend_region(period, region) [K/decade]
"""
# %%
import os
import sys
import numpy as np
import xarray as xr
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprocess
import src.Polygon_region as poly
# %%
# ----------------------
# PATHS / CONFIG
# ----------------------
BASE = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/"
LSM_FILE = "/work/mh0033/m301036/Data_storage/CMIP6-MPI-ESM-LR/GR15_lsm_regrid.nc"  # used for NAWH ocean mask

REGIONS = ["Arctic", "Subpolar_gyre", "SoutheastPacific", "SOP", "NPI", "SO", "SOP_original"]

# Boxes
BOX = {
    "Arctic": dict(lat1=66.5, lat2=90,  lon1=-180, lon2=180),
    "SO":     dict(lat1=-65,  lat2=-50, lon1=-180, lon2=180),
    "NPI":    dict(lat1=30,   lat2=50,  lon1=175,  lon2=220),   # confirm lon convention after conversion
    "SOP":    dict(lat1=-70,  lat2=-40, lon1=230,  lon2=280),   # in 0-360; will convert later
    "SOP_original": dict(lat1=-70,  lat2=-55, lon1=180,  lon2=260),   # original definition
}

# SEP polygon definition (lon,lat) (in -180..180 coords)
SEP_POLY_X = np.array([-110, -160, -80, -80, -110], dtype=float)
SEP_POLY_Y = np.array([-25,   0,    0, -25,  -25], dtype=float)
SEP_BBOX   = dict(lat1=-25, lat2=0, lon1=-160, lon2=-80)

# ----------------------
# ARGS
# ----------------------
if len(sys.argv) < 3:
    raise SystemExit("Usage: python regional_trends_one_run.py <RUN_ID> <MODEL>")

run_id = int(sys.argv[1])   # 1..N
model  = sys.argv[2]
# model = "IPSL_CM6A"
# input merged file
IN_FILE = f"{BASE}/{model}/forced_{model}_MK_trend_1950-2022_sliding.nc"

# output per-run
OUT_DIR = f"{BASE}/{model}/regional_anomalies"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/{model}_run{run_id:03d}_regional_mean_trends_1950_2022_sliding.nc"

# ----------------------
# HELPERS
# ----------------------
def area_weighted_mean_latlon(da):
    """Area-weighted mean over lat/lon for da(period, lat, lon)."""
    w = np.cos(np.deg2rad(da["lat"]))
    w = w / w.mean()
    return da.weighted(w).mean(("lat", "lon"))
"""
# depercated due to the xarray indices selection always brittle and wrong
1. It cannot handle dateline-crossing regions.
2. It sometimes misinterprets lon bounds if lon is not sorted.
The biggest bug: it chooses indices using “closest point” and then slices lon[ind_start_lon:ind_end_lon].
"""
# def select_box(da, lat, lon, lat1, lat2, lon1, lon2):
#     """Use your selreg to match existing pipeline."""
#     sub, _, _ = data_process.selreg(da, lat, lon, lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
#     return sub
def select_box_xr(da, lat1, lat2, lon1, lon2):
    # ensure increasing slices regardless of coord order
    if da.lat[0] < da.lat[-1]:
        lat_slice = slice(lat1, lat2)
    else:
        lat_slice = slice(lat2, lat1)

    if da.lon[0] < da.lon[-1]:
        lon_slice = slice(lon1, lon2)
    else:
        lon_slice = slice(lon2, lon1)

    return da.sel(lat=lat_slice, lon=lon_slice)

def select_box_dateline_xr(da, lat1, lat2, lon1, lon2):
    """
    da.lon must be in [-180,180] and sorted.
    If lon1 > lon2 => crosses dateline => union of two slices.
    """
    if lon1 <= lon2:
        return select_box_xr(da, lat1, lat2, lon1, lon2)

    part1 = select_box_xr(da, lat1, lat2, lon1, 180)
    part2 = select_box_xr(da, lat1, lat2, -180, lon2)
    return xr.concat([part1, part2], dim="lon").sortby("lon")

def build_sep_mask(lat_vals, lon_vals):
    """Create SEP polygon mask (lat,lon) inside bbox already selected."""
    mask = np.zeros((lat_vals.size, lon_vals.size), dtype=bool)
    poly.get_mask(mask, lon_vals, lat_vals, SEP_POLY_X, SEP_POLY_Y)
    return xr.DataArray(mask, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))

def calc_subpolar_gyre_index(trend_da, lsm_da):
    """
    Your NAWH-style index on trend patterns:
      subpolar gyre mean (ocean only) minus NH mean (?) you used (0..90).
    Here we reproduce your function structure, but applied to trend patterns.

    trend_da: (period, lat, lon) with lon in [-180,180]
    lsm_da:   land-sea mask (lat,lon) ideally same grid and lon in [-180,180]
             ocean assumed where mask==0 (as your code).
    """
    # ocean mask
    trend_ocean = trend_da.where(lsm_da == 0)

    ds_WH = trend_ocean.sel(lat=slice(42, 60), lon=slice(-50, -10))
    wh_mean = area_weighted_mean_latlon(ds_WH)

    # NH reference (0..90, -180..180) — match your code logic
    ds_sel = trend_ocean.sel(lat=slice(0, 90), lon=slice(-180, 180))
    nh_mean = area_weighted_mean_latlon(ds_sel)

    return wh_mean - nh_mean  # (period)

# ----------------------
# LOAD ONE RUN TREND PATTERN
# ----------------------
if not os.path.exists(IN_FILE):
    raise FileNotFoundError(IN_FILE)

ds_full = xr.open_dataset(IN_FILE)
print(ds_full)
# %%
# IMPORTANT: accommodate either "run" or "run_id" naming
if "run" in ds_full.dims:
    coord = ds_full["run"] if "run" in ds_full.coords else None
    if coord is not None and np.min(coord.values) == 1:
        trend = ds_full["trend"].sel(run=run_id)
    else:
        trend = ds_full["trend"].isel(run=run_id - 1)
elif "run_id" in ds_full.dims:
    coord = ds_full["run_id"] if "run_id" in ds_full.coords else None
    if coord is not None and np.min(coord.values) == 1:
        trend = ds_full["trend"].sel(run_id=run_id)
    else:
        trend = ds_full["trend"].isel(run_id=run_id - 1)
else:
    raise KeyError("Input dataset missing 'run' or 'run_id' dimension")

# lon to [-180,180] to be consistent with your region definitions
trend_adj = preprocess.convert_longitude(trend).sortby("lon")

lat = trend_adj["lat"]
lon = trend_adj["lon"]

# ----------------------
# REGION CALCS
# ----------------------
out_series = {}
# ---- Arctic, SO ----
for r in ["Arctic", "SO"]:
    bb = BOX[r]
    sub = select_box_xr(trend_adj, bb["lat1"], bb["lat2"], bb["lon1"], bb["lon2"])
    out_series[r] = area_weighted_mean_latlon(sub)

# ---- NPI (keep your explicit split or use lonaware with converted bounds) ----
# Convert 175..220 (0..360) to -180..180:
# 175 stays 175; 220 -> -140, so it crosses the dateline.
# NPI: 175..-140 crosses dateline
sub_npi = select_box_dateline_xr(trend_adj, 30, 50, 175, -140)
out_series["NPI"] = area_weighted_mean_latlon(sub_npi)
# SOP 230..280 => -130..-80
sub_sop = select_box_xr(trend_adj, -70, -40, -130, -80)
out_series["SOP"] = area_weighted_mean_latlon(sub_sop)
# SOP_original 180..260 => -180..-100
sub_sop0 = select_box_xr(trend_adj, -70, -55, -180, -100)
out_series["SOP_original"] = area_weighted_mean_latlon(sub_sop0)
# ---- DEBUG print region info ----
def check_region(name, da):
    print(name, "size=", da.size,
          "lon[min,max]=", float(da.lon.min()), float(da.lon.max()),
          "lat[min,max]=", float(da.lat.min()), float(da.lat.max()))

check_region("NPI", sub_npi)
check_region("SOP", sub_sop)
check_region("SOP_original", sub_sop0)
# %%
# SEP polygon
sep_box = trend_adj.sel(lat=slice(SEP_BBOX["lat1"], SEP_BBOX["lat2"]),
                        lon=slice(SEP_BBOX["lon1"], SEP_BBOX["lon2"]))
mask_da = build_sep_mask(sep_box["lat"].values, sep_box["lon"].values)
sep_poly = sep_box.where(mask_da)
out_series["SoutheastPacific"] = area_weighted_mean_latlon(sep_poly)  # (period)

# Subpolar gyre (needs land-sea mask on same lon convention)
lsm = xr.open_dataset(LSM_FILE)
# your file used var1[0,:,:]; convert to DataArray(lat,lon)
if "var1" in lsm:
    lsm_da = lsm["var1"].isel(time=0) if "time" in lsm["var1"].dims else lsm["var1"].isel(dim_0=0) if "dim_0" in lsm["var1"].dims else lsm["var1"][0]
else:
    # adjust if your lsm variable name differs
    lsm_da = list(lsm.data_vars.values())[0]
lsm_da = preprocess.convert_longitude(lsm_da)

# regrid mask if needed (simple nearest); best is to precompute on same grid
lsm_da = lsm_da.interp(lat=lat, lon=lon, method="nearest")
out_series["Subpolar_gyre"] = calc_subpolar_gyre_index(trend_adj, lsm_da)

lsm.close()
ds_full.close()
# ----------------------
# SAVE
# ----------------------
# stack into (period, region) using xarray alignment to avoid shape mismatches
period_coord = trend_adj["period"]
region_coord = xr.DataArray(REGIONS, dims=("region",), name="region")

da_list = []
for r in REGIONS:
    da = out_series[r]
    # ensure a period dimension exists and align to common coord
    if "period" not in da.dims:
        da = da.expand_dims(period=period_coord)
    da = da.reindex(period=period_coord)
    da_list.append(da)

stacked = xr.concat(da_list, dim=region_coord)

ds_out = xr.Dataset(
    data_vars=dict(
        trend_region=stacked.astype(np.float32),
    ),
    coords=dict(
        period=period_coord,
        region=region_coord,
        run_id=run_id,
        model=model,
    )
)
ds_out["trend_region"].attrs["units"] = "degC/decade"
ds_out = ds_out.assign_coords(run_id=run_id)
ds_out.attrs["model"] = model
ds_out.attrs["description"] = "Regional mean trends computed from per-run residual MK trend patterns."

tmp = OUT_FILE + ".tmp"
ds_out.to_netcdf(tmp)
os.replace(tmp, OUT_FILE)

print("Wrote:", OUT_FILE, flush=True)
# %%