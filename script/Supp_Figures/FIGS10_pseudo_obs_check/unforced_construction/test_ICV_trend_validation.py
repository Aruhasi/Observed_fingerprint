# %%
import numpy as np
import xarray as xr
import os
import sys
import src.SAT_function as data_process
# %%
# === Settings ===
segment_lengths = [30]
model = "MIROC6"  # <-- can change to another model
test_runs = [1, 2]  # Small subset of runs for validation
dir_residuals = "/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Revision_check/pseudo_obs_check/data"
dir_output = os.path.join(dir_residuals, "trend/test")
os.makedirs(dir_output, exist_ok=True)

# === Functions ===
def func_mk(x):
    results = data_process.apply_mannkendall(x)
    return results[0] * 10

def separate_interval(data, segment_length):
    return [data.isel(year=slice(i, i + segment_length)) for i in range(data.sizes['year'] - segment_length + 1)]

def compute_trend_for_segments(data, segment_lengths):
    trend_ds = xr.Dataset()
    for seg_len in segment_lengths:
        segments = separate_interval(data, seg_len)
        trend_list = []
        for seg in segments:
            slope = xr.apply_ufunc(
                func_mk,
                seg,
                input_core_dims=[["year"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float]
            )
            # Ensure it's a DataArray
            if isinstance(slope, xr.Dataset):
                slope = list(slope.data_vars.values())[0]
            trend_list.append(slope)

        trend_da = xr.concat(trend_list, dim="segment")
        trend_da = trend_da.assign_coords(segment=np.arange(len(trend_list)))
        trend_ds[f"trend_{seg_len}"] = trend_da
    return trend_ds
# %%
file_path = os.path.join(dir_residuals, f"MPI_ESM_tas_unforced_ano_wrt_{model}_1850_2022.nc")
ds = xr.open_dataset(file_path, chunks={"lat": 45, "lon": 90, "year": -1})

# === Process test runs ===
results = []
for run in test_runs:
    print(f"Testing run {run}")
    ds_run = ds.sel(run=run)
    trend_result = compute_trend_for_segments(ds_run, segment_lengths)
    trend_result = trend_result.expand_dims(run=[run])
    results.append(trend_result)

# === Concatenate and save ===
if results:
    final_result = xr.concat(results, dim="run")
    output_file = os.path.join(dir_output, f"ICV_30yr_trend_test_{model}.nc")
    final_result.to_netcdf(output_file)
    print(f"[Test] Saved test output to: {output_file}")
else:
    print("No test runs processed.")
# %%
# plotting the results
def plot_trend_segments(trend_da, lat, lon, run, segment_indices=[0, 1, 2], cmap="coolwarm", levels=np.linspace(-1, 1, 21)):
    """
    Plot multiple segments of a given run.
    
    Parameters:
    - trend_da: DataArray with dims ('segment', 'lat', 'lon')
    - lat, lon: coordinate arrays
    - run: run number
    - segment_indices: list of segment indices to plot
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    n = len(segment_indices)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(5*n, 4),
                             subplot_kw={"projection": ccrs.Robinson()})

    if n == 1:
        axes = [axes]  # make iterable

    for ax, seg_idx in zip(axes, segment_indices):
        ax.set_title(f"Run {run}, Segment {seg_idx}", fontsize=12)
        segment_data = trend_da.sel(segment=seg_idx)
        segment_data.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, levels=levels, add_colorbar=False
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', linestyle='--')

    # Add shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=levels[0], vmax=levels[-1])),
        ax=axes, orientation='horizontal', fraction=0.05, pad=0.1
    )
    cbar.set_label("Trend (°C/decade)")
    plt.tight_layout()
    plt.show()
# %%
# Example: visualize trend_30 for run=1, segments 0-2
plot_trend_segments(
    trend_da=final_result['trend_30'].sel(run=1),
    lat=final_result['lat'],
    lon=final_result['lon'],
    run=1,
    segment_indices=[0, 1, 2]  # or any list of indices
)
