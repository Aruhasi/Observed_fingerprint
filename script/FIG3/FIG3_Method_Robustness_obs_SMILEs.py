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
# input pattern correlation from NetCDF files:
def read_correlations(file_path):
    """
    Read pattern correlation data from a NetCDF file with variables:
      - pattern_corr_run_vs_mmem(run, period)
      - pattern_corr_obs_vs_mmem(period)

    Returns
    -------
    run_correlations : dict
        { '2013-2022': [list of run correlations across all runs in this model],
          '1993-2022': [...],
          '1963-2022': [...],
          '1979-2022': [...] }

    obs_correlations : dict
        Same keys, but each entry is a list of obs-vs-MMEM values
        (usually length 1 per model per period).
    """
    import xarray as xr
    import numpy as np

    target_periods = ['2013-2022', '1993-2022', '1963-2022', '1979-2022']

    run_correlations = {k: [] for k in target_periods}
    obs_correlations = {k: [] for k in target_periods}

    ds = xr.open_dataset(file_path)

    periods = ds['period'].values.astype(str)
    run_vs_mmem = ds['pattern_corr_run_vs_mmem'].values   # (run, period)
    obs_vs_mmem = ds['pattern_corr_obs_vs_mmem'].values   # (period,)

    for idx, period_str in enumerate(periods):
        period_str = period_str.strip()
        # period_str should be like "1950-2022", "2013-2022", etc.
        try:
            start_str, end_str = period_str.split('-')
        except ValueError:
            # unexpected format, skip
            continue

        label = f"{start_str}-{end_str}"
        if label not in run_correlations:
            continue  # we only care about the four chosen windows

        # collect all runs for this period
        vals = run_vs_mmem[:, idx]
        run_correlations[label].extend([float(v) for v in vals if np.isfinite(v)])

        # obs vs MMEM (single value for this period)
        obs_val = float(obs_vs_mmem[idx])
        if np.isfinite(obs_val):
            obs_correlations[label].append(obs_val)

    ds.close()
    return run_correlations, obs_correlations
# In[3]: paths & models
dir_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Pattern_correlation/'
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
# 1) Read FORCED pattern correlations (run vs MMEM, obs vs MMEM)
#    from: <model>/<model>_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc
# ------------------------------------------------------------------
forced_run_corr   = {}  # model -> dict(period -> list of run correlations)
forced_obs_corr   = {}  # model -> dict(period -> list of obs correlations)
for m in MODELS:
    nc_path = os.path.join(
        dir_in,
        m,
        f"{m}_forced_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc",
    )
    run_corr, obs_corr = read_correlations(nc_path)
    forced_run_corr[m] = run_corr
    forced_obs_corr[m] = obs_corr

# ------------------------------------------------------------------
# 2) Read UNFORCED (ICV std) pattern correlations
#    from: <model>/<model>_ICVstd_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc
# ------------------------------------------------------------------
unforced_run_corr = {}
unforced_obs_corr = {}
for m in MODELS:
    nc_path = os.path.join(
        dir_in,
        m,
        f"{m}_ICVstd_pattern_correlation_run_and_obs_vs_MMEM_1950-2022_sliding.nc",
    )
    run_corr, obs_corr = read_correlations(nc_path)
    unforced_run_corr[m] = run_corr
    unforced_obs_corr[m] = obs_corr

# ------------------------------------------------------------------
# 3) Build "models_data" and "models_unforced_data" in pretty-label space
#    These are what you later feed into create_long_df
# ------------------------------------------------------------------
models_data = {
    MODEL_LABELS[m]: forced_run_corr[m]
    for m in MODELS
}

models_unforced_data = {
    MODEL_LABELS[m]: unforced_run_corr[m]
    for m in MODELS
}
# In[4]: create long-format DataFrames
def create_long_df(models_dict):
    long_data = []
    for model_label, corr_dict in models_dict.items():
        for time_period, correlations in corr_dict.items():
            for correlation in correlations:
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
# 4) Build multi-model ENS distributions of run correlations
#    ENS_corr       : for FORCED run vs MMEM
#    unforced_ENS_corr : for UNFORCED run vs MMEM
# ------------------------------------------------------------------
TARGET_PERIODS = ['2013-2022', '1993-2022', '1963-2022', '1979-2022']

def build_ens_corr(run_corr_dict_by_model):
    """
    run_corr_dict_by_model: dict[model] -> dict[period] -> list of correlations
    Returns ENS_corr: dict[period] -> 1D np.array of all runs across all models
    """
    ens = {}
    for period in TARGET_PERIODS:
        arrays = [
            np.array(run_corr_dict_by_model[m][period])
            for m in run_corr_dict_by_model.keys()
        ]
        padded = pad_to_max_length(arrays)
        ens[period] = np.concatenate(padded)
    return ens

ENS_corr          = build_ens_corr(forced_run_corr)
unforced_ENS_corr = build_ens_corr(unforced_run_corr)

# ------------------------------------------------------------------
# 5) "MMEM_corr" and "MMEM_unforced_corr" from obs_correlations
#    These now come from obs_correlations returned by read_correlations,
#    rather than being hard-coded. Here I compute the *multi-model mean*
#    per period; you can switch to np.median if you prefer.
# ------------------------------------------------------------------
def build_mmem_obs_corr(obs_corr_dict_by_model, agg='mean'):
    """
    obs_corr_dict_by_model: dict[model] -> dict[period] -> list of obs-vs-MMEM values
    Returns: dict[period] -> [single aggregated value]  (kept as [val] for compatibility)
    """
    mmem = {}
    for period in TARGET_PERIODS:
        vals = []
        for m in obs_corr_dict_by_model.keys():
            vals.extend(obs_corr_dict_by_model[m][period])
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
# FORCED MMEM–OBS correlation (formerly your hard-coded MMEM_corr)
MMEM_corr = build_mmem_obs_corr(forced_obs_corr, agg='mean')

# UNFORCED (ICV std) MMEM–OBS correlation (formerly MMEM_unforced_corr)
MMEM_unforced_corr = build_mmem_obs_corr(unforced_obs_corr, agg='mean')
# In[7]:
MMEM_corr, MMEM_unforced_corr
# %%
# INPUT the MMEM_corr and MMEM_unforced_corr values manually
MMEM_corr = {'2013-2022': [0.75], '1993-2022': [0.79], '1963-2022': [0.81], '1979-2022': [0.81]}
MMEM_unforced_corr = {'2013-2022': [0.72], '1993-2022': [0.57], '1963-2022': [0.51], '1979-2022': [0.54]}
# %%
# input the 30-year forced trend patterns of OBS and MMEM
dir_forced_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS5_S6/trend_forced_HadCRUT5_annual/'

HadCRUT5_trend = xr.open_dataset(dir_forced_input + 
                                 'forced_HadCRUT5_MMLE_MK_trend_1950-2022_sliding.nc')
# In[2]:
dir_model_in = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_forced/'
# '/work/mh0033/m301036/Land_surf_temp/Disentangling_OBS_SAT_trend/Supp_Figure6_Forced/data/Smiles_ensemble/'
MMLE_annual_trend = xr.open_dataset(dir_model_in + 
                                    'MMLE_ENSmean_forced_MK_trend_1950-2022_sliding.nc')
# In[3]:
# calculate the pattern difference between OBS and MMLE
pattern_diff = MMLE_annual_trend.trend - HadCRUT5_trend.trend
print(pattern_diff)
# In[4]:
def cal_ratio(data,pattern_diff):
    data = pattern_diff/data
    return data
# %%
pattern_ratio = cal_ratio(HadCRUT5_trend.trend, pattern_diff)
print(pattern_ratio)
# %%
# dir_output = "/work/mh0033/m301036/OBS_LPS_revision/script/FIG3/Pattern_diff/"
# pattern_diff.to_dataset(name='trend_diff').to_netcdf(dir_output + 'MMLE_OBS_forced_pattern_diff_1950_2022.nc')
# pattern_ratio.to_dataset(name='trend_ratio').to_netcdf(dir_output + 'MMLE_OBS_forced_pattern_ratio_1950_2022.nc')
# %%
print(pattern_ratio.min().values)
# calculate the global mean values
mean_ratio = pattern_ratio.sel(period="1993-2022").mean().values*100
# In[5]:
# Input the Observational internal trend (wrt MMEM GSAT)
dir_internal_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/OBS_ICV_std/concatenate/'
HadCRUT5_internal_trend = xr.open_dataset(dir_internal_input +'OBS_ICV_MK_trend_STD_1950_2022_sliding.nc')['icv_trend_std']
# In[6]:
dir_model_internal = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/MMLE/SMILE_internal/'
MMEM_internal_trend = xr.open_dataset(dir_model_internal + 'MMLE_internal_trend_std_1950-2022_sliding.nc')['trend']
# %%
ICV_diff = MMEM_internal_trend - HadCRUT5_internal_trend
print(ICV_diff)
# %%
Ratio_ICV = cal_ratio(HadCRUT5_internal_trend, ICV_diff)
print(Ratio_ICV)
# %%
# save the data into netcdf file
# dir_output = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/Plot_output/"
# ICV_diff.to_dataset(name='ICV_trend_std_diff').to_netcdf(dir_output + 'MMLE_OBS_ICV_pattern_diff_1950_2022.nc')
# Ratio_ICV.to_dataset(name='ICV_trend_std_ratio').to_netcdf(dir_output + 'MMLE_OBS_ICV_pattern_ratio_1950_2022.nc')
# %%
# calculate the global mean values
mean_ratio_ICV = Ratio_ICV.sel(period="1993-2022", trend_length=30).mean().values*100
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
            bw_adjust=0.8,
            cut=0,
            clip=(0.2, 1)
        )
        mmem_value = MMEM_corr[segment][0]
        if (i == 2) and include_1979:
            mmem_value = MMEM_corr[segment][0] + 0.005  # slight offset for visibility
            ax_pdf_forced.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=5.5, 
                                 label=f"r(Obs, MMEM)({segment}) (offset)")
        else:
            ax_pdf_forced.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, 
                             label=f"r(Obs, MMEM)({segment})")

    # Add title and label
    ax_pdf_forced.text(-0.12, 1.2, "A", transform=ax_pdf_forced.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_pdf_forced.text(0.5, 1.1, "Externally forced pattern correlation", fontsize=30, 
                      ha='center', va='center', transform=ax_pdf_forced.transAxes)
    ax_pdf_forced.set_xlabel("Pattern Correlation", fontsize=26)
    ax_pdf_forced.set_ylabel("Density", fontsize=26)
    ax_pdf_forced.spines['top'].set_visible(False)
    ax_pdf_forced.spines['right'].set_visible(False)
    ax_pdf_forced.tick_params(axis='x', labelsize=26)
    ax_pdf_forced.tick_params(axis='y', labelsize=26)
    ax_pdf_forced.set_xlim(0.2, 1.0)
    ax_pdf_forced.set_ylim(0., 12.5)
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
            bw_adjust=0.8,
            cut=0,
            clip=(0.4, 1)
        )
        mmem_value = MMEM_unforced_corr[segment][0]
        ax_pdf_icv.axvline(mmem_value, color=colors[i], linestyle='-.', linewidth=4.5, 
                           label=f"r(Obs, MMEM)({segment})")

    # Add title and label
    ax_pdf_icv.text(-0.12, 1.2, "B", transform=ax_pdf_icv.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_pdf_icv.text(0.5, 1.1, "Internal variability pattern correlation", fontsize=30, 
                   ha='center', va='center', transform=ax_pdf_icv.transAxes)
    ax_pdf_icv.set_xlabel("Pattern Correlation", fontsize=26)
    ax_pdf_icv.set_ylabel("Density", fontsize=26)
    ax_pdf_icv.spines['top'].set_visible(False)
    ax_pdf_icv.spines['right'].set_visible(False)
    ax_pdf_icv.tick_params(axis='x', labelsize=26)
    ax_pdf_icv.tick_params(axis='y', labelsize=26)
    ax_pdf_icv.set_xlim(0.4, 1.0)
    ax_pdf_icv.set_ylim(0., 20)
    ax_pdf_icv.tick_params(axis='x', direction='out', length=8, width=2)
    ax_pdf_icv.tick_params(axis='y', direction='out', length=8, width=2)

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
        fontsize=28,
        title="",
        title_fontsize=24,
        labelcolor='linecolor',
        ncol=2,
        bbox_to_anchor=(0.12, 0.88),
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
        bbox_to_anchor=(0.12, 0.8),
        frameon=False,
        columnspacing=0.5,
    )

    # -------------------- Subplot c: Forced Minus MMEM Map --------------------
    ax_forced_map = plt.subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180))
    ax_forced_map.coastlines(resolution='110m')
    gl = ax_forced_map.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.15, linewidth=0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {'size': 22}
    gl.ylabel_style = {'size': 22}
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])

    levels_forced = np.arange(-0.5, 0.55, 0.05)
    n_bins = len(levels_forced) - 1

    norm_forced = BoundaryNorm(boundaries=levels_forced, ncolors=n_bins)
    cmap_forced = "RdBu_r"

    p_forced = pattern_diff.sel(period="1993-2022").plot(
        ax=ax_forced_map,
        transform=ccrs.PlateCarree(),
        cmap=cmap_forced,
        norm=norm_forced,
        levels=levels_forced,
        add_colorbar=False
    )

    ax_forced_map.text(-0.12, 1.2, "C", transform=ax_forced_map.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_forced_map.set_title("MMLE - OBS\n(1993-2022)", fontsize=28, pad=10, loc='center')
    ax_forced_map.text(0.95, 1.05, f"{mean_ratio:.0f}%", fontsize=28, ha='center', va='center',
                       transform=ax_forced_map.transAxes)

    cbar_ax_forced = fig.add_axes([0.175, 0.1, 0.25, 0.02])
    cbar_forced = plt.colorbar(
        p_forced,
        cax=cbar_ax_forced,
        orientation='horizontal',
        extend='neither',
        ticks=[-0.5, -0.25, 0, 0.25, 0.5]
    )
    cbar_forced.set_label("Externally forced SAT differences\n(°C per decade)", fontsize=24, labelpad=10, loc='center')
    cbar_forced.ax.tick_params(labelsize=22)
    cbar_forced.ax.tick_params(direction='out', length=10, width=2)

    # -------------------- Subplot d: ICV Minus MMEM Map --------------------
    ax_icv_map = plt.subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180))
    ax_icv_map.coastlines(resolution='110m')
    gl1 = ax_icv_map.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.15, linewidth=0.25)
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xformatter = cticker.LongitudeFormatter()
    gl1.yformatter = cticker.LatitudeFormatter()
    gl1.xlabel_style = {'size': 18}
    gl1.ylabel_style = {'size': 18}
    gl1.bottom_labels = True
    gl1.left_labels = True
    gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])

    levels_icv = np.arange(-0.25, 0.275, 0.025)
    n_bins_icv = len(levels_icv) - 1

    norm_icv = BoundaryNorm(boundaries=levels_icv, ncolors=n_bins_icv)
    cmap_icv = "RdBu_r"

    p_icv = ICV_diff.sel(period="1993-2022", trend_length=30).plot(
        ax=ax_icv_map,
        transform=ccrs.PlateCarree(),
        cmap=cmap_icv,
        norm=norm_icv,
        levels=levels_icv,
        add_colorbar=False
    )

    ax_icv_map.text(-0.12, 1.2, "D", transform=ax_icv_map.transAxes, fontsize=34, fontweight='bold', va='top')
    ax_icv_map.set_title("MMLE - OBS\n(1993-2022)", fontsize=28, pad=10, loc='center')
    ax_icv_map.text(0.95, 1.05, f"{mean_ratio_ICV:.0f}%", fontsize=28, ha='center', va='center',
                    transform=ax_icv_map.transAxes)

    cbar_ax_icv = fig.add_axes([0.61, 0.1, 0.25, 0.02])
    cbar_icv = plt.colorbar(
        p_icv,
        cax=cbar_ax_icv,
        orientation='horizontal',
        extend='neither',
        ticks=[-0.25, -0.125, 0, 0.125, 0.25]
    )
    cbar_icv.set_label("Internal variability SAT differences\n(°C per decade)", fontsize=24, labelpad=10, loc='center')
    cbar_icv.ax.tick_params(labelsize=22)
    cbar_icv.ax.tick_params(direction='out', length=10, width=2)

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
    fig_without_1979.savefig(figure_output + f"Fig3_1993_2022_C&D.{ext}", dpi=300, bbox_inches='tight')
    fig_with_1979.savefig(figure_output + f"Fig3_with_1979_2022.{ext}", dpi=300, bbox_inches='tight')

plt.show()
# %%
