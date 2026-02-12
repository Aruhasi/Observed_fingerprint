#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast computation of ICV segmented stddev of MK trend patterns for OBS internal variability.

- Input:
    OBS_SAT_anomaly_partition_wrt_MMLE_ENS.nc
    variable: internal_variability(year, lat, lon)

- For each segment length L = 10..73 years:
    * generate all sliding windows of length L along 'year'
    * compute MK slope (K/decade) for each segment
    * compute stddev over 'segment' -> icv_trend_std(L, lat, lon)

MPI parallelization:
    - segment_lengths are split across ranks, e.g. rank 0: L=10,20,...; rank 1: L=11,21,... etc.
    - each rank writes a small temporary file
    - rank 0 merges them at the end
"""
# %%
import sys
import numpy as np
import xarray as xr
import os

import src.SAT_function_Obs_Fingerprint as data_process
from mpi4py import MPI

# ---------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()
print(f"[rank {rank}] MPI world size = {npro}", flush=True)

# CLI args: segment_length or "merge" mode
if len(sys.argv) > 1:
    if sys.argv[1].lower() == "merge":
        segment_length = None  # Merge mode
    else:
        try:
            segment_length = int(sys.argv[1])
        except ValueError:
            segment_length = None
else:
    segment_length = None

start_year = 1850
end_year   = 2022
# %%
dir_in  = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6"
dir_out = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std"

# ---------------------------------------------------------------------
# MK helper: slope in K/decade
# ---------------------------------------------------------------------
def func_mk(x):
    """
    Mann-Kendall test for trend along 'year'.
    Returns slope in K/decade.
    """
    slope, p_val = data_process.apply_mannkendall(x)
    return slope * 10.0

# ---------------------------------------------------------------------
# Generate sliding segments
# ---------------------------------------------------------------------
def generate_segments(data, segment_length):
    """
    Generate sliding time segments for a given segment length.

    data: DataArray with dimension 'year'
    segment_length: number of years in each segment

    Returns: DataArray with dims ('segment', 'year', 'lat', 'lon')
    """
    num_segments = data.sizes["year"] - segment_length + 1
    segments = [
        data.isel(year=slice(i, i + segment_length)) for i in range(num_segments)
    ]
    return xr.concat(segments, dim="segment")
# %%
# ---------------------------------------------------------------------
# Local computation on each rank
# ---------------------------------------------------------------------
def process_icv_rank(icv_da, all_segment_lengths):
    """
    Each rank computes icv_trend_std(L, lat, lon) for a subset of segment_lengths.
    Returns a Dataset on this rank with:
        icv_trend_std(trend_length, lat, lon)
    or an empty Dataset if this rank has no assigned lengths.
    """
    # Assign L values to this rank in round-robin fashion
    local_lengths = [L for i, L in enumerate(all_segment_lengths) if i % npro == rank]

    if not local_lengths:
        print(f"[rank {rank}] No segment lengths assigned.", flush=True)
        return xr.Dataset()

    std_list = []

    for seg_len in local_lengths:
        print(f"[rank {rank}] Computing MK std for segment length L={seg_len}", flush=True)

        # 1) build segments: (segment, year, lat, lon)
        segments = generate_segments(icv_da, segment_length=seg_len)

        # 2) MK slopes along year -> (segment, lat, lon)
        slope_da = xr.apply_ufunc(
            func_mk,
            segments,
            input_core_dims=[["year"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",           # fine to keep; or set to None if you prefer no dask
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[float],
        )

        if isinstance(slope_da, xr.Dataset):
            slope_da = slope_da.to_array().squeeze()
        slope_da = slope_da.rename("trend")

        # add trend_length coord and save segment trends for this length
        slope_da = slope_da.expand_dims(trend_length=[seg_len])
        slope_file = os.path.join(
            dir_out, f"OBS_ICV_MK_trend_segments_L{seg_len}_rank{rank}.nc"
        )
        slope_da.to_netcdf(slope_file)

        # 3) std over segments -> (lat, lon)
        std_da = slope_da.std(dim="segment", skipna=True)
        std_da = std_da.expand_dims(trend_length=[seg_len])
        std_list.append(std_da)

        # free memory: we don't keep segments or slopes
        del segments, slope_da

    # concat all local L's
    icv_trend_std_local = xr.concat(std_list, dim="trend_length").sortby("trend_length")
    icv_trend_std_local.name = "icv_trend_std"

    icv_trend_std_local.attrs["units"] = "K/decade"
    icv_trend_std_local.attrs["description"] = (
        "Stddev over sliding segments of MK trend of observed internal "
        f"variability for assigned segment lengths, years {start_year}–{end_year}."
    )

    ds_local = xr.Dataset({"icv_trend_std": icv_trend_std_local})
    return ds_local
# %%
# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def main():
    # All ranks open the same file (reading in parallel is fine on Levante)
    in_file = f"{dir_in}/OBS_SAT_anomaly_partition_wrt_MMLE_ENS.nc"
    print(f"[rank {rank}] Reading {in_file}", flush=True)

    ds_in = xr.open_dataset(
        in_file,
        chunks={"year": 50, "lat": 45, "lon": 45},
    )
    icv = ds_in["internal_variability"].sel(year=slice(start_year, end_year))

    # segment lengths 10..73
    all_segment_lengths = list(range(10, 74))

    # Each rank computes its subset
    ds_local = process_icv_rank(icv, all_segment_lengths)

    # Write temporary file for this rank (even if empty, we can skip)
    tmp_file = os.path.join(
        dir_out,
        f"OBS_ICV_MK_trend_STD_{start_year}-{end_year}_len10-73_rank{rank}.nc",
    )

    if ds_local is not None and len(ds_local.data_vars) > 0:
        print(f"[rank {rank}] Writing local output to {tmp_file}", flush=True)
        ds_local.to_netcdf(tmp_file)

    ds_in.close()
    comm.Barrier()

    # Rank 0 merges all rank files
    if rank == 0:
        print("[rank 0] Merging rank files into final output...", flush=True)

        # merge std files
        tmp_files = [
            os.path.join(
                dir_out,
                f"OBS_ICV_MK_trend_STD_{start_year}-{end_year}_len10-73_rank{r}.nc",
            )
            for r in range(npro)
            if os.path.exists(
                os.path.join(
                    dir_out,
                    f"OBS_ICV_MK_trend_STD_{start_year}-{end_year}_len10-73_rank{r}.nc",
                )
            )
        ]

        if not tmp_files:
            raise RuntimeError("No temporary rank files found to merge.")

        ds_all = xr.open_mfdataset(tmp_files, combine="nested", concat_dim="trend_length")
        ds_all = ds_all.sortby("trend_length")

        out_file = os.path.join(
            dir_out,
            f"OBS_ICV_MK_trend_STD_{start_year}-{end_year}_len10-73.nc",
        )
        print(f"[rank 0] Writing final output: {out_file}", flush=True)
        ds_all.to_netcdf(out_file)
        ds_all.close()

        # merge segment-trend files
        seg_files = sorted(
            f for f in os.listdir(dir_out) if f.startswith("OBS_ICV_MK_trend_segments_L")
        )
        seg_paths = [os.path.join(dir_out, f) for f in seg_files]
        if seg_paths:
            ds_seg = xr.open_mfdataset(seg_paths, combine="nested", concat_dim="trend_length")
            ds_seg = ds_seg.sortby("trend_length")
            seg_out = os.path.join(
                dir_out,
                f"OBS_ICV_MK_trend_segments_all_{start_year}-{end_year}_len10-73.nc",
            )
            print(f"[rank 0] Writing segment trends: {seg_out}", flush=True)
            ds_seg.to_netcdf(seg_out)
            ds_seg.close()

        # Clean up temporary files
        for tf in tmp_files:
            os.remove(tf)
            print(f"[rank 0] Removed {tf}", flush=True)
        for fp in seg_paths:
            os.remove(fp)
            print(f"[rank 0] Removed {fp}", flush=True)
# %%
# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
if rank == 0:
    os.makedirs(dir_out, exist_ok=True)

comm.Barrier()
main()
