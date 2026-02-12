# %%
import numpy as np
import xarray as xr
import os
from scipy.stats import pearsonr
# %%
dir_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MPI_ESM/SMILE_internal/'
run_indir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/pseudo_obs_check/trend/"
model_names = ["MIROC6", "ACCESS", "EC_Earth", "IPSL_CM6A", "CanESM5", "CESM2", "MMEM"]
# model_names = []

# === Load ensemble mean trend ===
ensemble_ds = xr.open_dataset(os.path.join(dir_input, 'MPI_ESM_SMILE_noise_trend_std_sliding_1950_2022.nc'))
# ensemble_ds = ensemble_ds.rename({'__xarray_dataarray_variable__': 'tas'})
# %%
ensemble_trend = ensemble_ds.sel(period="1993-2022").noise_trend_std  # shape: (lat, lon)
# %%
# === Initialize correlation result storage ===
correlation_results = {}

# === Loop over models and compute correlations ===
for model in model_names:
    file_path = os.path.join(run_indir, f"MPI_ESM_ICV_noise_std_trend_pattern_{model}.nc")
    ds_model = xr.open_dataset(file_path)  # shape: (run, lat, lon)

    correlations = []
    for run in ds_model.run.values:
        run_data = ds_model.sel(run=run)['std_trend_30']
        # Flatten both and mask NaNs
        flat_model = run_data.values.flatten()
        flat_ens = ensemble_trend.values.flatten()
        valid = ~np.isnan(flat_model) & ~np.isnan(flat_ens)
        corr, _ = pearsonr(flat_model[valid], flat_ens[valid])
        correlations.append((run, corr))

    correlation_results[model] = correlations
# %%
# === Save results to a text file ===
for model, corrs in correlation_results.items():
    file_out = os.path.join(run_indir, f"{model}_ICV_std_correlation_with_ensemble.txt")
    with open(file_out, 'w') as f:
        f.write("run\tcorrelation\n")
        for run, corr in corrs:
            f.write(f"{run}\t{corr:.4f}\n")
# %%
correlation_results.keys()  # show which models were processed
# %%
