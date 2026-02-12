# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# In[2]:
# input pattern RMSE from NetCDF files:
def read_RMSE(file_path):
    """
    Read pattern RMSE data from a NetCDF file with variables:
      - pattern_RMSE_run_vs_mmem(run, period)
      - pattern_RMSE_obs_vs_mmem(period)
    Returns
    -------
    run_RMSE : dict
        { '2013-2022': [list of run RMSE across all runs in this model],
          '1993-2022': [...],
          '1963-2022': [...],
          '1979-2022': [...] }

    obs_RMSE : dict
        Same keys, but each entry is a list of obs-vs-MMEM values
        (usually length 1 per model per period).
    """
    import xarray as xr
    import numpy as np

    target_periods = ['2013-2022', '1993-2022', '1963-2022', '1979-2022']

    run_RMSE = {k: [] for k in target_periods}
    obs_RMSE = {k: [] for k in target_periods}

    ds = xr.open_dataset(file_path)

    periods = ds['period'].values.astype(str)
    run_vs_mmem = ds['pattern_rmse_run_vs_mmem'].values   # (run, period)
    obs_vs_mmem = ds['pattern_rmse_obs_vs_mmem'].values   # (period,)

    for idx, period_str in enumerate(periods):
        period_str = period_str.strip()
        # period_str should be like "1950-2022", "2013-2022", etc.
        try:
            start_str, end_str = period_str.split('-')
        except ValueError:
            # unexpected format, skip
            continue

        label = f"{start_str}-{end_str}"
        if label not in run_RMSE:
            continue  # we only care about the four chosen windows

        # collect all runs for this period
        vals = run_vs_mmem[:, idx]
        run_RMSE[label].extend([float(v) for v in vals if np.isfinite(v)])

        # obs vs MMEM (single value for this period)
        obs_val = float(obs_vs_mmem[idx])
        if np.isfinite(obs_val):
            obs_RMSE[label].append(obs_val)

    ds.close()
    return run_RMSE, obs_RMSE
# In[3]: paths & models
dir_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Pattern_correlation_pearson/'
MODELS = ['CanESM5', 'CESM2', 'IPSL_CM6A', 'EC_Earth3', 'ACCESS', 'MPI_ESM', 'MIROC6']

# pretty labels for plotting
MODEL_LABELS = {
    'CanESM5':        'CanESM5',
    'CESM2':          'CESM2',
    'IPSL_CM6A':      'IPSL-CM6A-LR',
    'EC_Earth3':      'EC-Earth3',
    'ACCESS':         'ACCESS-ESM1.5',
    'MPI_ESM':        'MPI-ESM1.2-LR',
    'MIROC6':         'MIROC6',
}

# ------------------------------------------------------------------
# 1) Read FORCED pattern RMSE (run vs MMEM, obs vs MMEM)
#    from: <model>/<model>_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc
# ------------------------------------------------------------------
forced_run_rmse   = {}  # model -> dict(period -> list of run RMSE)
forced_obs_rmse   = {}  # model -> dict(period -> list of obs RMSE)
for m in MODELS:
    nc_path = os.path.join(
        dir_in,
        m,
        f"{m}_forced_pattern_rmse_run_and_obs_vs_MMEM_1950-2022_sliding.nc",
    )
    run_rmse, obs_rmse = read_RMSE(nc_path)
    forced_run_rmse[m] = run_rmse
    forced_obs_rmse[m] = obs_rmse

# ------------------------------------------------------------------
# 2) Read UNFORCED (ICV std) pattern RMSE
#    from: <model>/<model>_ICVstd_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc
# ------------------------------------------------------------------
unforced_run_rmse = {}
unforced_obs_rmse = {}
for m in MODELS:
    nc_path = os.path.join(
        dir_in,
        m,
        f"{m}_ICVstd_pattern_rmse_run_and_obs_vs_MMEM_1950-2022_sliding.nc",
    )
    run_rmse, obs_rmse = read_RMSE(nc_path)
    unforced_run_rmse[m] = run_rmse
    unforced_obs_rmse[m] = obs_rmse

# ------------------------------------------------------------------
# 3) Build "models_data" and "models_unforced_data" in pretty-label space
#    These are what you later feed into create_long_df
# ------------------------------------------------------------------
models_data = {
    MODEL_LABELS[m]: forced_run_rmse[m]
    for m in MODELS
}

models_unforced_data = {
    MODEL_LABELS[m]: unforced_run_rmse[m]
    for m in MODELS
}
# In[4]: create long-format DataFrames
def create_long_df(models_dict):
    long_data = []
    for model_label, corr_dict in models_dict.items():
        for time_period, RMSE in corr_dict.items():
            for correlation in RMSE:
                long_data.append({
                    'Model': model_label,
                    'Time Period': time_period,
                    'Correlation': correlation,
                })
    return pd.DataFrame(long_data)

long_df      = create_long_df(models_data)          # forced
ICV_long_df  = create_long_df(models_unforced_data) # unforced
# In[5]: helper for padding concatenation (unchanged)
def pad_to_max_length(arrays, fill_value=np.nan):
    max_length = max(len(arr) for arr in arrays)
    padded = [
        np.pad(a, (0, max_length - len(a)), constant_values=fill_value)
        for a in arrays
    ]
    return padded
# In[6]:
# ------------------------------------------------------------------
# 4) Build multi-model ENS distributions of run RMSE
#    ENS_rmse       : for FORCED run vs MMEM
#    unforced_ENS_rmse : for UNFORCED run vs MMEM
# ------------------------------------------------------------------
TARGET_PERIODS = ['2013-2022', '1993-2022', '1963-2022', '1979-2022']

def build_ens_rmse(run_rmse_dict_by_model):
    """
    run_rmse_dict_by_model: dict[model] -> dict[period] -> list of RMSE
    Returns ENS_rmse: dict[period] -> 1D np.array of all runs across all models
    """
    ens = {}
    for period in TARGET_PERIODS:
        arrays = [
            np.array(run_rmse_dict_by_model[m][period])
            for m in run_rmse_dict_by_model.keys()
        ]
        padded = pad_to_max_length(arrays)
        ens[period] = np.concatenate(padded)
    return ens

ENS_rmse          = build_ens_rmse(forced_run_rmse)
unforced_ENS_rmse = build_ens_rmse(unforced_run_rmse)

# ------------------------------------------------------------------
# 5) "MMEM_rmse" and "MMEM_unforced_rmse" from obs_RMSE
#    These now come from obs_RMSE returned by read_RMSE,
#    rather than being hard-coded. Here I compute the *multi-model mean*
#    per period; you can switch to np.median if you prefer.
# ------------------------------------------------------------------
def build_mmem_obs_rmse(obs_rmse_dict_by_model, agg='mean'):
    """
    obs_rmse_dict_by_model: dict[model] -> dict[period] -> list of obs-vs-MMEM values
    Returns: dict[period] -> [single aggregated value]  (kept as [val] for compatibility)
    """
    mmem = {}
    for period in TARGET_PERIODS:
        vals = []
        for m in obs_rmse_dict_by_model.keys():
            vals.extend(obs_rmse_dict_by_model[m][period])
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mmem[period] = [np.nan]
        elif agg == 'mean':
            mmem[period] = [float(vals.mean())]
        elif agg == 'median':
            mmem[period] = [float(np.median(vals))]
        else:
            raise ValueError("agg must be 'mean' or 'median'")
    return mmem
# %%
# FORCED MMEM–OBS correlation (formerly your hard-coded MMEM_rmse)
MMEM_rmse = build_mmem_obs_rmse(forced_obs_rmse, agg='mean')

# UNFORCED (ICV std) MMEM–OBS correlation (formerly MMEM_unforced_rmse)
MMEM_unforced_rmse = build_mmem_obs_rmse(unforced_obs_rmse, agg='mean')
# In[7]:
# This is the mean out of 7LEs' pattern correlation RMSE values
MMEM_rmse, MMEM_unforced_rmse
# %%
"""
12.Jan.2026: update calculation of OBS-vs-MMEM RMSE values
"""
# add the OBS-vs-MMEM RMSE values here for plotting
dir_OBS_RMSE = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Pattern_correlation_pearson/MMLE/'
# Read OBS-vs-MMEM RMSE from NetCDF files
OBS_FORCED_RMSE_file = os.path.join(
    dir_OBS_RMSE,
    'MMLE_forced_pattern_rmse_vs_forced_OBS_1950-2022_sliding.nc'
)
OBS_UNFORCED_RMSE_file = os.path.join(
    dir_OBS_RMSE,
    'OBS_ICVstd_pattern_rmse_obs_vs_MMEM_1950-2022_sliding.nc'
)
# INPUT the MMEM_rmse and MMEM_unforced_rmse values manually
MMEM_rmse = {'2013-2022': [0.81], '1993-2022': [0.85], '1963-2022': [0.87], '1979-2022': [0.87]}
MMEM_unforced_rmse = {'2013-2022': [0.68], '1993-2022': [0.61], '1963-2022': [0.59], '1979-2022': [0.61]}
# %%
# ============================================================================
# PLOTTING FUNCTION - Create two versions (with and without 1979-2022)
# ============================================================================
def create_figure(include_1979=False):
    """
    Create Figure 3 with or without 1979-2022 period.
    
    Parameters:
    include_1979 (bool): If True, includes 1979-2022 period (4 panels). 
                         If False, uses only 3 periods (2013-2022, 1993-2022, 1963-2022)
    """
    
    if include_1979:
        time_segments = ['2013-2022', '1993-2022', '1979-2022', '1963-2022']
        time_segments_labels = [f"{seg}-year" for seg in ['10', '30', '44', '60']]
        colors = ['#F9AE78', '#E47159', "#94BFD0", '#161E54']
        # FF986A, F16D34, , #3D5C6F(original)
        n_cols = 4
    else:
        time_segments = ['2013-2022', '1993-2022', '1963-2022']
        time_segments_labels = [f"{seg}-year" for seg in ['10', '30', '60']]
        colors = ['#F9AE78', '#E47159', '#3D5C6F']
        n_cols = 3
    
    
    # Filter DataFrames to include only selected periods
    long_df_filtered = long_df[long_df['Time Period'].isin(time_segments)]
    ICV_long_df_filtered = ICV_long_df[ICV_long_df['Time Period'].isin(time_segments)]
    
    # Set up the figure with a 2x2 layout
    fig = plt.figure(figsize=(25, 18))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.25, wspace=0.3)

    # -------------------- Subplot a: Forced Correlation PDF --------------------
    ax_pdf_forced = plt.subplot(gs[0, 0])
    for i, segment in enumerate(time_segments):
        segment_data = long_df_filtered[long_df_filtered['Time Period'] == segment]['Correlation']
        sns.kdeplot(
            segment_data,
            label=f"r(Single realization, ENS)({segment})",
            fill=False,
            linewidth=4.5,
            color=colors[i],
            ax=ax_pdf_forced,
            bw_adjust=1.0,
            cut=0,
            clip=(0., 0.6)
        )
        mmem_value = MMEM_rmse[segment][0]
        if (i == 2) and include_1979:
            mmem_value = MMEM_rmse[segment][0]   # slight offset for visibility
            ax_pdf_forced.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=5.5, 
                                 label=f"RMSE(Obs, MMEM)({segment}) (offset)")
        else:
            ax_pdf_forced.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, 
                             label=f"RMSE(Obs, MMEM)({segment})")

    # Add title and label
    ax_pdf_forced.text(-0.12, 1.2, "A", transform=ax_pdf_forced.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_pdf_forced.text(0.5, 1.1, "Externally forced RMSE", fontsize=30, 
                      ha='center', va='center', transform=ax_pdf_forced.transAxes)
    ax_pdf_forced.set_xlabel("RMSE", fontsize=26)
    ax_pdf_forced.set_ylabel("Density", fontsize=26)
    ax_pdf_forced.spines['top'].set_visible(False)
    ax_pdf_forced.spines['right'].set_visible(False)
    ax_pdf_forced.spines['bottom'].set_linewidth(2)
    ax_pdf_forced.spines['left'].set_linewidth(2)
    ax_pdf_forced.tick_params(axis='x', labelsize=26)
    ax_pdf_forced.tick_params(axis='y', labelsize=26)
    ax_pdf_forced.set_xlim(0, 0.6)
    ax_pdf_forced.set_ylim(0., 12.0)
    ax_pdf_forced.tick_params(axis='x', direction='out', length=6, width=2)
    ax_pdf_forced.tick_params(axis='y', direction='out', length=6, width=2)

    # -------------------- Subplot b: Internal Variability Correlation PDF --------------------
    ax_pdf_icv = plt.subplot(gs[0, 1])
    for i, segment in enumerate(time_segments):
        segment_data = ICV_long_df_filtered[ICV_long_df_filtered['Time Period'] == segment]['Correlation']
        sns.kdeplot(
            segment_data,
            label=f"r(Single realization, ENS)({segment})",
            fill=False,
            linewidth=4.5,
            color=colors[i],
            ax=ax_pdf_icv,
            bw_adjust=1.0,
            cut=0,
            clip=(0., 0.6)
        )
        mmem_value = MMEM_unforced_rmse[segment][0]
        ax_pdf_icv.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, 
                           label=f"RMSE(Obs, MMEM)({segment})")

    # Add title and label
    ax_pdf_icv.text(-0.12, 1.2, "B", transform=ax_pdf_icv.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_pdf_icv.text(0.5, 1.1, "Internal variability RMSE", fontsize=30, 
                   ha='center', va='center', transform=ax_pdf_icv.transAxes)
    ax_pdf_icv.set_xlabel("RMSE", fontsize=26)
    ax_pdf_icv.set_ylabel("Density", fontsize=26)
    ax_pdf_icv.spines['top'].set_visible(False)
    ax_pdf_icv.spines['right'].set_visible(False)
    ax_pdf_icv.spines['bottom'].set_linewidth(2)
    ax_pdf_icv.spines['left'].set_linewidth(2)
    ax_pdf_icv.tick_params(axis='x', labelsize=26)
    ax_pdf_icv.tick_params(axis='y', labelsize=26)
    ax_pdf_icv.set_xlim(0., 0.6)
    ax_pdf_icv.set_ylim(0., 60)
    ax_pdf_icv.tick_params(axis='x', direction='out', length=8, width=2)
    ax_pdf_icv.tick_params(axis='y', direction='out', length=8, width=2)
    # make the axes thicker


    # -------------------- Add Legends --------------------

    legend_elements_pattern = [
        Line2D([0], [0], color=colors[i], lw=0, label=time_segments_labels[i])
        for i in range(len(time_segments))
    ]

    legend_elements_time = [
        Line2D([0], [0], color='black', lw=4.5, linestyle='-.', label='OBS'),
        Line2D([0], [0], color='black', lw=4.5, linestyle='-', label='Large Ensembles'),
    ]

    fig.legend(
        handles=legend_elements_pattern,
        loc='upper left',
        fontsize=26,
        title="",
        title_fontsize=24,
        labelcolor='linecolor',
        ncol=4,
        bbox_to_anchor=(0.1, 0.45),
        frameon=False,
        columnspacing=0.05,
        handletextpad=0.2,
        borderaxespad=0.01
    )

    fig.legend(
        handles=legend_elements_time,
        loc='upper left',
        fontsize=24,
        title="",
        title_fontsize=24,
        ncol=2,
        bbox_to_anchor=(0.6, 0.45),
        frameon=False,
        columnspacing=0.5,
    )

    return fig

# %%
# Import required libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.colors as mcolors
import palettable

# %%
# Create both figures
fig_without_1979 = create_figure(include_1979=False)
fig_with_1979 = create_figure(include_1979=True)

# Save both figures
figure_output = '/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG3/'
os.makedirs(figure_output, exist_ok=True)

for ext in ("png", "pdf"):
    fig_without_1979.savefig(figure_output + f"FIG_RMSE.{ext}", dpi=300, bbox_inches='tight')
    fig_with_1979.savefig(figure_output + f"FIG_RMSE_with_1979.{ext}", dpi=300, bbox_inches='tight')

plt.show()
# %%
