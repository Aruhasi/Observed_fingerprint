#!/usr/bin/env python3
# %%
# -*- coding: utf-8 -*-

import sys
import numpy as np
import xarray as xr
import os
import src.SAT_function_Obs_Fingerprint as data_process
# %%
start_year = 1850
end_year   = 2022

dir_in  = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/cesm2_100/"
dir_out = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std"
os.makedirs(dir_out, exist_ok=True)

def func_mk(x):
    slope, p_val = data_process.apply_mannkendall(x)
    return slope * 10.0

def generate_segments(data, segment_length):
    num_segments = data.sizes["year"] - segment_length + 1
    segments = [
        data.isel(year=slice(i, i + segment_length)) for i in range(num_segments)
    ]
    return xr.concat(segments, dim="segment")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python OBS_ICV_singleL.py <segment_length>")

    L = int(sys.argv[1])
    print(f"Computing OBS ICV MK trend + std for segment length L={L}")

    in_file = f"{dir_in}/OBS_SAT_anomaly_partition_wrt_MMLE_ENS.nc"
    ds_in = xr.open_dataset(in_file)
    icv = ds_in["internal_variability"].sel(year=slice(start_year, end_year))

    # segments: (segment, year, lat, lon)
    segments = generate_segments(icv, segment_length=L)

    # slopes: (segment, lat, lon)
    slope_da = xr.apply_ufunc(
        func_mk,
        segments,
        input_core_dims=[["year"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",  # or None
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[float],
    )
    if isinstance(slope_da, xr.Dataset):
        slope_da = slope_da.to_array().squeeze()
    slope_da = slope_da.rename("trend")

    # std over segment
    std_da = slope_da.std(dim="segment", skipna=True)
    std_da = std_da.rename("icv_trend_std")
    std_da = std_da.expand_dims(trend_length=[L])

    # write outputs
    seg_file = os.path.join(dir_out, f"OBS_ICV_MK_trend_segments_L{L}.nc")
    std_file = os.path.join(dir_out, f"OBS_ICV_MK_trend_STD_L{L}.nc")

    xr.Dataset({"trend": slope_da}).to_netcdf(seg_file)
    xr.Dataset({"icv_trend_std": std_da}).to_netcdf(std_file)

    ds_in.close()
    print(f"Done L={L}")
# %%
if __name__ == "__main__":
    os.makedirs(dir_out, exist_ok=True)
    main()
# %%