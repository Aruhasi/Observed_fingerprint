# In[1]:
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
# In[2]:
# define function
# NOTE: apply_mannkendall lives in SAT_function_Obs_Fingerprint.py
import src.SAT_function_Obs_Fingerprint as data_process
import src.Data_Preprocess as preprosess
# In[3]:
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npro = comm.Get_size()
# In[4]:
start_year = 1850
end_year   = 2022
print(f"[rank {rank}] MPI world size = {npro}", flush=True)

model = sys.argv[1]

# Number of realizations per LE
N_RUNS_DICT = {
    # "MPI_ESM"  : 50,
    # "MIROC6"   : 50,
    "CESM2"    : 100,
    # "CanESM5"  : 50,
    # "ACCESS"   : 40,
    # "EC_Earth3": 21,
    # "IPSL_CM6A": 32,
}

N_RUNS = N_RUNS_DICT[model]
print(f"[rank {rank}] Number of realizations for {model}: {N_RUNS}", flush=True)
# %%
# Input the standard deviation of SAT-OBS residuals
dir_members = (f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}" )
mem_file = os.path.join(dir_members, f"{model}_ICV_noise_std_trend_pattern_1850_2022.nc")

len_segments = np.arange(10, 74, 1)
start_year = 1950
end_year   = 2022
min_length = 10
period_starts = list(range(end_year - min_length + 1, start_year - 1, -1))  # 2013..1950
period_labels = [f"{by}-{end_year}" for by in period_starts]

ds_in = xr.open_dataset(mem_file, chunks={"lat": 10, "lon": 10})

trend_list = []
for L, period_label in zip(len_segments, period_labels):
    var_name = f"std_trend_{L}"
    if var_name not in ds_in:
        raise KeyError(f"Variable {var_name} not found in {mem_file}")

    da = ds_in[var_name]  # (run, lat, lon)
    da = da.rename("trend")
    da = da.expand_dims(period=[period_label])  # add period dim
    trend_list.append(da)

# Concatenate over period (one period per segment length)
ICV_STD_LE_all = xr.concat(trend_list, dim="period")
ICV_STD_LE_all = ICV_STD_LE_all.assign_coords(period=("period", period_labels))
# %%
# Save concatenated dataset
out_dir = dir_members
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_file = os.path.join(out_dir, f"{model}_ICV_noise_std_trend_pattern_1950_2022_sliding.nc")
ICV_STD_LE_all.to_netcdf(out_file)
print(f"[rank {rank}] Saved concatenated ICV std trend dataset to {out_file}", flush=True)
# In[ ]:
# End of script
comm.Barrier()
print(f"[rank {rank}] All done.", flush=True)
