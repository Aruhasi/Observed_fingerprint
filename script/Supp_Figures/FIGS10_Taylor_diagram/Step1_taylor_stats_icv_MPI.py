#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Taylor-diagram stats (R, std, CRMSE) for:
  1) Each LE realization trend map vs the LE ensemble-mean trend map
  2) OBS trend map vs the LE ensemble-mean trend map

for ALL sliding periods (period dimension), using area-weighted (coslat) statistics.

Expected inputs:
  - members file: trend(run, period, lat, lon)
  - ensmean file: trend(period, lat, lon)
  - obs file (optional): trend(period, lat, lon)

Outputs (NetCDF):
  - corr_run_vs_ens(run, period)
  - std_run(run, period)
  - std_ref(period)
  - crmse_run_vs_ens(run, period)
  - bias_run_vs_ens(run, period)         [optional but written]
  - rmse_total_run_vs_ens(run, period)   [optional but written]
  - corr_obs_vs_ens(period)              [if obs provided]
  - std_obs(period)                      [if obs provided]
  - crmse_obs_vs_ens(period)             [if obs provided]
  - bias_obs_vs_ens(period)              [if obs provided]
  - rmse_total_obs_vs_ens(period)        [if obs provided]

Run example:
  mpirun -np 8 python -u StepX_taylor_stats_MPI.py \
    --model CESM2 --component forced \
    --members /.../forced_CESM2_MK_trend_1950-2022_sliding.nc \
    --ensmean /.../CESM2_ENSmean_forced_MK_trend_1950-2022_sliding.nc \
    --obs /.../forced_HadCRUT5_MMLE_MK_trend_1950-2022_sliding.nc \
    --out /.../TaylorStats/CESM2_forced_taylor_stats_1950-2022_sliding.nc
"""

import os
import argparse
import warnings
import numpy as np
import xarray as xr
from mpi4py import MPI
# %%
# -----------------------
# MPI
# -----------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()

def _coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    """Return cos(lat) weights as DataArray(lat) (not normalized here)."""
    return xr.DataArray(np.cos(np.deg2rad(lat.values)), coords={"lat": lat}, dims=("lat",))

def _broadcast_w2d(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Broadcast cos(lat) to (lat, lon)."""
    wlat = _coslat_weights(lat)
    w2d = wlat.broadcast_like(xr.DataArray(np.zeros((lat.size, lon.size)), coords={"lat": lat, "lon": lon}, dims=("lat","lon")))
    return w2d

def _pair_mask(x: xr.DataArray, y: xr.DataArray) -> xr.Dataset:
    ds = xr.Dataset({"x": x, "y": y})
    return ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]))

def _wnormalize(w2d: xr.DataArray) -> xr.DataArray:
    """Normalize weights to sum to 1 over (lat, lon)."""
    s = w2d.sum(dim=("lat","lon"))
    return w2d / s

def _wmean(da2d: xr.DataArray, w2d_norm: xr.DataArray) -> xr.DataArray:
    return (da2d * w2d_norm).sum(dim=("lat","lon"))

def _wstd_centered(da2d: xr.DataArray, w2d_norm: xr.DataArray) -> xr.DataArray:
    mu = _wmean(da2d, w2d_norm)
    return np.sqrt(_wmean((da2d - mu)**2, w2d_norm))
# %%
def taylor_stats_2d(test: xr.DataArray, ref: xr.DataArray) -> dict:
    """
    Compute centered Taylor stats (corr, std_test, std_ref, crmse) plus bias and total rmse.
    Uses cos(lat) area weights. All stats computed on common valid mask.
    """
    ds = _pair_mask(test, ref)
    x = ds["x"]
    y = ds["y"]

    # build and normalize weights on the same valid mask
    w2d = _broadcast_w2d(x["lat"], x["lon"])
    w2d = w2d.where(np.isfinite(x) & np.isfinite(y))
    w2d_norm = _wnormalize(w2d)

    # means (for bias)
    mx = _wmean(x, w2d_norm)
    my = _wmean(y, w2d_norm)
    bias = mx - my

    # centered anomalies
    xc = x - mx
    yc = y - my

    # std
    sx = np.sqrt(_wmean(xc**2, w2d_norm))
    sy = np.sqrt(_wmean(yc**2, w2d_norm))

    # corr (centered Pearson)
    cov = _wmean(xc * yc, w2d_norm)
    corr = cov / (sx * sy)

    # centered RMSE (CRMSE)
    crmse = np.sqrt(_wmean((xc - yc)**2, w2d_norm))

    # total RMSE (includes bias)
    rmse_total = np.sqrt(_wmean((x - y)**2, w2d_norm))

    return {
        "corr": float(corr),
        "std_test": float(sx),
        "std_ref": float(sy),
        "crmse": float(crmse),
        "bias": float(bias),
        "rmse_total": float(rmse_total),
    }
# %%
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--component", required=True, help="forced or internal (used for metadata only)")
    p.add_argument("--members", required=True, help="NetCDF: trend(run, period, lat, lon)")
    p.add_argument("--ensmean", required=True, help="NetCDF: trend(period, lat, lon)")
    p.add_argument("--obs", default=None, help="Optional NetCDF: trend(period, lat, lon) or with extra trend_length dim")
    p.add_argument("--var", default="trend", help="Default variable name if per-file vars not set")
    p.add_argument("--var_members", default="trend", help="Variable name in members file (default: trend)")
    p.add_argument("--var_ensmean", default="noise_trend_std", help="Variable name in ensmean file (default: noise_trend_std)")
    p.add_argument("--var_obs", default="icv_trend_std", help="Variable name in obs file (default: icv_trend_std)")
    p.add_argument("--obs_trend_length_index", type=int, default=-1, help="If obs has trend_length dim, pick this index (default: match period index when -1)")
    p.add_argument("--out", required=True, help="Output NetCDF path")
    args = p.parse_args()

    # Resolve variable names per file
    var_mem = args.var_members or args.var
    var_ref = args.var_ensmean or args.var
    var_obs = args.var_obs or args.var  # 
    if rank == 0:
        print(f"[rank 0] model={args.model} component={args.component}")
        print(f"[rank 0] members={args.members}")
        print(f"[rank 0] ensmean={args.ensmean}")
        print(f"[rank 0] obs={args.obs}")
        print(f"[rank 0] var members={var_mem}  ensmean={var_ref}  obs={var_obs}")
        print(f"[rank 0] out={args.out}")

    ds_mem = xr.open_dataset(args.members)
    ds_ref = xr.open_dataset(args.ensmean)

    # Helper to get a variable with a clear error if missing
    def _get_var(ds, name, label):
        if name in ds:
            return ds[name]
        raise KeyError(f"Variable '{name}' not found in {label}. Available: {list(ds.data_vars)}")

    mem = _get_var(ds_mem, var_mem, "members")   # (run, period, lat, lon)
    ref = _get_var(ds_ref, var_ref, "ensmean")   # (period, lat, lon)

    # Align coords
    mem, ref = xr.align(mem, ref, join="inner")

    runs = mem["run"].values
    periods = mem["period"].values
    n_run = runs.size
    n_per = periods.size

    # Allocate local arrays (full size, fill only assigned runs)
    corr_local = np.full((n_run, n_per), np.nan, dtype=np.float64)
    std_local  = np.full((n_run, n_per), np.nan, dtype=np.float64)
    crmse_local = np.full((n_run, n_per), np.nan, dtype=np.float64)
    bias_local  = np.full((n_run, n_per), np.nan, dtype=np.float64)
    rmse_local  = np.full((n_run, n_per), np.nan, dtype=np.float64)

    # Each rank handles runs: ir = rank, rank+npro, ...
    for ir in range(rank, n_run, npro):
        rname = runs[ir]
        if (ir % max(1, (n_run // 10))) == 0:
            print(f"[rank {rank}] run index {ir}/{n_run} (run={rname})", flush=True)

        mem_run = mem.sel(run=rname)  # (period, lat, lon)

        for ip in range(n_per):
            per = periods[ip]
            test_map = mem_run.sel(period=per)
            ref_map  = ref.sel(period=per)

            st = taylor_stats_2d(test_map, ref_map)

            corr_local[ir, ip] = st["corr"]
            std_local[ir, ip]  = st["std_test"]
            crmse_local[ir, ip]= st["crmse"]
            bias_local[ir, ip] = st["bias"]
            rmse_local[ir, ip] = st["rmse_total"]

    # Gather on rank 0
    corr_list = comm.gather(corr_local, root=0)
    std_list  = comm.gather(std_local, root=0)
    crmse_list= comm.gather(crmse_local, root=0)
    bias_list = comm.gather(bias_local, root=0)
    rmse_list = comm.gather(rmse_local, root=0)

    if rank == 0:
        corr = np.nanmax(np.stack(corr_list, axis=0), axis=0)
        stdt = np.nanmax(np.stack(std_list, axis=0), axis=0)
        crmse= np.nanmax(np.stack(crmse_list, axis=0), axis=0)
        bias = np.nanmax(np.stack(bias_list, axis=0), axis=0)
        rmse = np.nanmax(np.stack(rmse_list, axis=0), axis=0)

        # Reference std per period (centered std of ref vs itself)
        std_ref = np.full((n_per,), np.nan, dtype=np.float64)
        for ip in range(n_per):
            per = periods[ip]
            ref_map = ref.sel(period=per)
            # std of centered ref
            w2d = _broadcast_w2d(ref_map["lat"], ref_map["lon"])
            w2d = w2d.where(np.isfinite(ref_map))
            w2d_norm = _wnormalize(w2d)
            std_ref[ip] = float(_wstd_centered(ref_map, w2d_norm))

        out = xr.Dataset(
            data_vars=dict(
                corr_run_vs_ens=(("run","period"), corr),
                std_run=(("run","period"), stdt),
                std_ref=(("period",), std_ref),
                crmse_run_vs_ens=(("run","period"), crmse),
                bias_run_vs_ens=(("run","period"), bias),
                rmse_total_run_vs_ens=(("run","period"), rmse),
            ),
            coords=dict(run=runs, period=periods),
            attrs=dict(
                model=args.model,
                component=args.component,
                note="Taylor stats are centered (means removed) and area-weighted by cos(lat). CRMSE is centered RMSE."
            )
        )

        # OBS vs ENSmean (optional)
        if args.obs is not None and os.path.exists(args.obs):
            ds_obs = xr.open_dataset(args.obs)
            obs_raw = _get_var(ds_obs, var_obs, "obs")

            # If obs has an extra trend_length dimension, match it to period by default
            # if "trend_length" in obs_raw.dims:
            if "trend_length" in obs.coords:
                obs = obs.drop_vars("trend_length")
            else:
                obs = obs_raw

            obs, ref2 = xr.align(obs, ref, join="inner")

            corr_obs = np.full((n_per,), np.nan, dtype=np.float64)
            std_obs  = np.full((n_per,), np.nan, dtype=np.float64)
            crmse_obs= np.full((n_per,), np.nan, dtype=np.float64)
            bias_obs = np.full((n_per,), np.nan, dtype=np.float64)
            rmse_obs = np.full((n_per,), np.nan, dtype=np.float64)

            for ip in range(n_per):
                per = periods[ip]
                st = taylor_stats_2d(obs.sel(period=per), ref2.sel(period=per))
                corr_obs[ip] = st["corr"]
                std_obs[ip]  = st["std_test"]
                crmse_obs[ip]= st["crmse"]
                bias_obs[ip] = st["bias"]
                rmse_obs[ip] = st["rmse_total"]

            out["corr_obs_vs_ens"] = (("period",), corr_obs)
            out["std_obs"]         = (("period",), std_obs)
            out["crmse_obs_vs_ens"]= (("period",), crmse_obs)
            out["bias_obs_vs_ens"] = (("period",), bias_obs)
            out["rmse_total_obs_vs_ens"] = (("period",), rmse_obs)
            ds_obs.close()

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_netcdf(args.out)
        print(f"[rank 0] wrote: {args.out}", flush=True)

    ds_mem.close()
    ds_ref.close()

    comm.Barrier()
    if rank != 0:
        print(f"[rank {rank}] done", flush=True)
# %%
if __name__ == "__main__":
    main()
