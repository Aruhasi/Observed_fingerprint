#!/usr/bin/env python3
# %%
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # avoids Lustre/HDF5 locking issues

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import src.SAT_function_Obs_Fingerprint as data_process
# %%
# -----------------
model = "MPI_ESM"
run = 1
start_year, end_year = 1950, 2022
min_length = 10
STEP = 5  # use 1 for all windows
# -----------------

dir_in = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}"
in_file = f"{dir_in}/SAT_anomaly_partition_wrt_{model}_ENS_GSAT_1850-2022.nc"

def func_mk(x_1d):
    slope, p = data_process.mk_test(x_1d)
    return slope, p

# ---------- load one run into memory (debug-mode, simplest + safest) ----------
with xr.open_dataset(in_file) as ds:
    forced = ds["forced_signal"].sel(run=run)

    # robust time->year handling
    if "year" not in forced.dims and "time" in forced.dims:
        # if time is datetime64, map to integer year coordinate
        if np.issubdtype(forced["time"].dtype, np.datetime64):
            forced = forced.assign_coords(year=forced["time"].dt.year).swap_dims({"time": "year"}).drop_vars("time")
        else:
            forced = forced.rename({"time": "year"})

    forced = forced.sel(year=slice(start_year, end_year)).load()  # load for quick debug

lat = forced["lat"].values
lon = forced["lon"].values

# expected last begin_year from min_length
last_begin = end_year - min_length + 1

begin_years = list(range(start_year, last_begin + 1, STEP))
if begin_years[-1] != last_begin:
    begin_years.append(last_begin)

print(f"Compute windows: {len(begin_years)} (begin from {begin_years[0]} to {begin_years[-1]})")

# ---------- single gridpoint sanity check ----------
iy = int(np.argmin(np.abs(lat - 0)))     # near equator
ix = int(np.argmin(np.abs(lon - 180)))   # near dateline
ts_gp = forced.isel(lat=iy, lon=ix).values
slope_gp, p_gp = func_mk(ts_gp)
print(f"Gridpoint check @lat={lat[iy]:.2f}, lon={lon[ix]:.2f}: MK slope={slope_gp:.4g}/yr, p={p_gp:.3g}")
# %%
# ---------- compute + plot running trend maps ----------
out_png_dir = f"./debug_plots_{model}_run{run}"
os.makedirs(out_png_dir, exist_ok=True)

for by in begin_years:
    da = forced.sel(year=slice(by, end_year))

    trend, pval = xr.apply_ufunc(
        func_mk,
        da,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="forbidden",   # debug: eager + deterministic
        output_dtypes=[float, float],
    )

    trend_decade = (trend * 10.0).astype("float32")  # per decade

    # ---- plot ----
    fig = plt.figure(figsize=(10, 4.8))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    # quick symmetric limits for visibility (robust to NaNs)
    vmax = np.nanpercentile(np.abs(trend_decade.values), 98)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.1

    im = ax.pcolormesh(
        lon, lat, trend_decade.values,
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=-vmax, vmax=vmax
    )
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
    cb.set_label("Trend (units/decade)")

    ax.set_title(f"{model} run{run} | MK trend {by}-{end_year} (forced_signal)")

    fn = f"{out_png_dir}/trend_{model}_run{run}_{by}-{end_year}.png"
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Saved PNGs to: {out_png_dir}")
# ----------
# %%
