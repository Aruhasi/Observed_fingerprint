#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Mann–Kendall trends for all end-fixed windows (length 10..73 years)
for one CESM2 internal-variability realization and write NetCDF output.

Usage
  python run_single_CESM2_ICV_segments.py <RUN_ID>

RUN_ID is 1-based (matches SLURM_ARRAY_TASK_ID). Output is saved to
  docs/data/FIG3/CESM2/output/CESM2_ICV_segments_1850_2022_run<NNN>_trend.nc
"""
import os
import sys
import argparse
import numpy as np
import xarray as xr
import src.SAT_function_Obs_Fingerprint as data_process
# %%
import src.slurm_cluster as scluster
client, scluster = scluster.init_dask_slurm_cluster(scale=4, walltime="02:00:00")
# %%
# Paths
DIR_RESID = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/CESM2"
FILE_RESID = os.path.join(
    DIR_RESID, "SAT_anomaly_partition_wrt_CESM2_CMIP6_SMBB_ENS_GSAT_1850-2022.nc"
)
DIR_OUT = os.path.join(DIR_RESID, "output")
SEGMENT_LENGTHS = range(10, 74, 1)  # 10..73 years
CHUNKS = {"run": 1, "lat": 45, "lon": 90}

def func_mk(x):
    slope, _p = data_process.apply_mannkendall(x)
    return slope * 10.0  # K/decade

def generate_segments(data, segment_length):
    num_segments = data.sizes["year"] - segment_length + 1
    segments = [data.isel(year=slice(i, i + segment_length)) for i in range(num_segments)]
    return xr.concat(segments, dim="segment")
# %%
def compute_trend(data, segment_lengths):
    out = xr.Dataset()
    max_segments = 0
    for seg_len in segment_lengths:
        segments = generate_segments(data, segment_length=seg_len)
        trend_da = xr.apply_ufunc(
            func_mk,
            segments.chunk(dict(year=-1)),
            input_core_dims=[["year"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        if isinstance(trend_da, xr.Dataset):
            trend_da = trend_da.to_array().squeeze()
        num_segments = trend_da.sizes["segment"]
        max_segments = max(max_segments, num_segments)
        trend_da = trend_da.assign_coords(segment=range(num_segments))
        if num_segments < max_segments:
            padding = xr.DataArray(
                np.full((max_segments - num_segments, *trend_da.shape[1:]), np.nan),
                dims=["segment", *trend_da.dims[1:]],
                coords={**trend_da.coords, "segment": range(num_segments, max_segments)},
            )
            trend_da = xr.concat([trend_da, padding], dim="segment")
        out[f"trend_{seg_len}"] = trend_da
    return out
# %%
def parse_run_id(cli_args=None, default=1):
    """Resolve run_id from function arg, env, or CLI (tolerates Jupyter args)."""
    env_val = os.getenv("RUN_ID")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("run_id", nargs="?", type=int)
    parser.add_argument("--run-id", dest="run_id_kw", type=int)
    known, _ = parser.parse_known_args(cli_args)
    return (
        known.run_id_kw
        or known.run_id
        or (int(env_val) if env_val is not None else None)
        or default
    )
# %%
def main(run_id=None):
    run_id = parse_run_id() if run_id is None else run_id
    if run_id is None:
        raise SystemExit("Run ID is required. Provide CLI arg, --run-id, or RUN_ID env.")
    run_index = run_id - 1  # convert 1-based to 0-based

    os.makedirs(DIR_OUT, exist_ok=True)
    ds = xr.open_dataset(FILE_RESID, chunks=CHUNKS)["internal_variability"].isel(run=run_index)
    drop_coords = [c for c in ["run", "realization"] if c in ds.coords]
    if drop_coords:
        ds = ds.reset_coords(names=drop_coords, drop=True)
    # Compute trends
    print(f"Computing trends for run ID {run_id} (index {run_index})...")
    trends = compute_trend(ds, SEGMENT_LENGTHS)
    out_path = os.path.join(DIR_OUT, f"CESM2_ICV_segments_1850_2022_run{run_id:03d}_trend.nc")
    for coord in ("member", "realization"):
        if coord in trends.coords:
            trends = trends.reset_coords(names=coord, drop=True)
    trends.to_netcdf(out_path)
    print(f"Wrote {out_path}")
# %%
if __name__ == "__main__":
    main()
# %%