# In[1]:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
# In[2]:
# define function
import src.SAT_function as data_process
import src.Data_Preprocess as preprosess

# %%
import src.slurm_cluster as scluster
client, scluster = scluster.init_dask_slurm_cluster(walltime="02:30:00")
# In[3]:
import dask.array as da
# %%
def func_mk(x):
    """
    Mann-Kendall test for trend
    """
    results = data_process.apply_mannkendall(x)
    slope = results[0]
    # slope_x = x.isel(time = 0).copy(slope)
    return slope * 10
# %%
def generate_segments(data, segment_length):
    """
    Generate time segments for each segment length with `apply_ufunc` support for parallel processing.
    """
    # Calculate start years for each segment
    num_segments = data.sizes["year"] - segment_length + 1
    segments = [data.isel(year=slice(i, i + segment_length)) for i in range(num_segments)]
    return xr.concat(segments, dim="segment")
# %%
def compute_trend(data, segment_lengths, trend_function):
    """Compute the trend for each segment length using Dask parallelization."""
    ICV_segments_ds = xr.Dataset()
    max_segments = 0
    # Ensure `segment_lengths` is iterable
    if isinstance(segment_lengths, int):
        segment_lengths = [segment_lengths]
    
    for seg_len in segment_lengths:
        segments = generate_segments(data, segment_length=seg_len)
        
        # Calculate the trend for each segment
        trend_da = xr.apply_ufunc(
            trend_function,
            segments.chunk(dict(year=-1)),
            input_core_dims=[["year"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float]
        )
        # Convert trend_da to DataArray if needed
        if isinstance(trend_da, xr.Dataset):
            trend_da = trend_da.to_array().squeeze()

        # Determine the maximum number of segments
        num_segments = trend_da.sizes['segment']
        max_segments = max(max_segments, num_segments)

        # Assign the 'segment' coordinate if missing and pad
        trend_da = trend_da.assign_coords(segment=range(num_segments))
        if num_segments < max_segments:
            padding = xr.DataArray(
                np.full((max_segments - num_segments, *trend_da.shape[1:]), np.nan),
                dims=["segment", *trend_da.dims[1:]],
                coords={**trend_da.coords, "segment": range(num_segments, max_segments)}
            )
            padded_trend_da = xr.concat([trend_da, padding], dim="segment")
        else:
            padded_trend_da = trend_da

        # Append to dataset
        ICV_segments_ds[f"trend_{seg_len}"] = padded_trend_da

    return ICV_segments_ds
# %%
def main(runs):
    # Directory setup
    dir_residuals = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Figure2/CanESM5/'
    dir_output = '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/LE_evaluation/Fig3_CanESM5/output/'
    
    # Load dataset
    ds_CanESM5_1850_2022 = xr.open_mfdataset(
        dir_residuals + 'GSAT_CanESM5_Internal_Variability_anomalies_1850_2022.nc',
        chunks={'run': 1, 'lat': 45, 'lon': 90}
    ).isel(run=runs)
    # run = runs
    # Define segment lengths and compute trends
    segment_lengths = range(10, 74, 1)
    combined_results = compute_trend(ds_CanESM5_1850_2022, segment_lengths, func_mk)
    # Initialize a list to hold results
    # delayed_trend_results = []

    # for run in runs:  # Iterate over runs
    #     # Select the current run
    #     ds_run = ds_CanESM5_1850_2022.sel(run=run)
    #     # Compute the trend for the current run
    #     delayed_trend = compute_trend(ds_run, segment_lengths, func_mk)
        
    #     # Add the run dimension if not present
    #     delayed_trend = delayed_trend.expand_dims(run=[run])
        
    #     # Append to the list
    #     delayed_trend_results.append(delayed_trend)
    #     print(f'Run {run} done')

    # # Concatenate all runs
    # combined_results = xr.concat(delayed_trend_results, dim='run')

    # Save result to NetCDF
    output_path = os.path.join(dir_output, f'CanESM5_ICV_segments_1850_2022_run{runs}_trend.nc')
    combined_results.to_netcdf(output_path)
    print(f'Results saved to {output_path}')
# %%
if __name__ == "__main__":
    import sys
    run_index = int(sys.argv[1])  # Accept the run index from the command-line argument
    main(runs=run_index)

# %%