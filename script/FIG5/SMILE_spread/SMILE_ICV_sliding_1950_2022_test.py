# 
"""
Concatenate ICV residual patterns for all models: run dims
"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
import glob
import xarray as xr
import numpy as np
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# from src.Statistic_cal import 
import numpy as np
import pandas as pd

# def mmle_equal_model_percentile_band(
#     df: pd.DataFrame,
#     tau_col: str = "tau",
#     model_col: str = "model",
#     value_col: str = "trend",
#     q=(0.05, 0.95),
#     K: int | None = None,
#     n_boot: int = 0,
#     seed: int = 0,
# ) -> pd.DataFrame:
#     """
#     Compute MMLE 5–95% (or any q) percentile band using an equal-model mixture.

#     Parameters
#     ----------
#     df : DataFrame with columns [tau_col, model_col, value_col]
#          Each row is one internal-variability trend sample (e.g., residual trend).
#     K : number of samples drawn PER MODEL per tau.
#         If None, uses the minimum sample count across models for each tau
#         (so no model dominates and no oversampling is required unless some model is short).
#     n_boot : if >0, bootstrap the equal-model mixture to stabilize quantiles.
#              Returns median quantiles across bootstraps + optional bootstrap spread.
#     """
#     rng = np.random.default_rng(seed)

#     out_rows = []
#     for tau, g_tau in df.groupby(tau_col):
#         # collect per-model arrays
#         by_model = {
#             m: g[value_col].to_numpy(dtype=float)
#             for m, g in g_tau.groupby(model_col)
#         }

#         # drop models with no data
#         by_model = {m: a[np.isfinite(a)] for m, a in by_model.items() if np.isfinite(a).any()}
#         if len(by_model) == 0:
#             continue

#         # choose K per tau if not provided: smallest available model sample size
#         K_tau = K if K is not None else min(len(a) for a in by_model.values())
#         if K_tau < 2:
#             continue

#         def draw_and_quantile():
#             pooled = []
#             for m, a in by_model.items():
#                 replace = len(a) < K_tau
#                 pooled.append(rng.choice(a, size=K_tau, replace=replace))
#             pooled = np.concatenate(pooled)
#             return np.quantile(pooled, q)

#         if n_boot and n_boot > 0:
#             qs = np.vstack([draw_and_quantile() for _ in range(n_boot)])
#             q_lo, q_hi = np.median(qs[:, 0]), np.median(qs[:, 1])

#             # Optional: bootstrap uncertainty of the quantile estimates themselves
#             q_lo_05, q_lo_95 = np.quantile(qs[:, 0], [0.05, 0.95])
#             q_hi_05, q_hi_95 = np.quantile(qs[:, 1], [0.05, 0.95])

#             out_rows.append({
#                 tau_col: tau,
#                 "q_low": q_lo,
#                 "q_high": q_hi,
#                 "q_low_boot_05": q_lo_05,
#                 "q_low_boot_95": q_lo_95,
#                 "q_high_boot_05": q_hi_05,
#                 "q_high_boot_95": q_hi_95,
#                 "K_per_model": K_tau,
#                 "n_models": len(by_model),
#             })
#         else:
#             q_lo, q_hi = draw_and_quantile()
#             out_rows.append({
#                 tau_col: tau,
#                 "q_low": q_lo,
#                 "q_high": q_hi,
#                 "K_per_model": K_tau,
#                 "n_models": len(by_model),
#             })

#     out = pd.DataFrame(out_rows).sort_values(tau_col).reset_index(drop=True)
#     return out

# =========================
# Example usage
# =========================
# Your input dataframe should look like:
# df columns: ["model", "tau", "trend"]
#   model: e.g. "CanESM5", "CESM2", ...
#   tau  : trend length (10..73)
#   trend: one sample of internal-variability regional trend (e.g., K/decade)

# band = mmle_equal_model_percentile_band(df, q=(0.05, 0.95), K=None, n_boot=500, seed=42)
# band now contains per-tau MMLE 5–95% range (q_low, q_high), plus K_per_model and model count.

# If you want a single band for one fixed period (no tau dimension),
# set tau_col to a constant column or just group by a dummy tau:
# df["tau"] = 73  # or "fixed"
# band = mmle_equal_model_percentile_band(df, q=(0.05, 0.95), n_boot=500, seed=42)
# MODELS = ['CanESM5', 'CESM2', 'IPSL_CM6A', 'EC_Earth3', 'ACCESS', 'MPI_ESM', 'MIROC6']
MODELS = ['CESM2']
# MODELS = ['ACCESS']

END_YEAR = 2022
START_MIN = 1950
TAU_MIN = 10
TAU_MAX = END_YEAR - START_MIN + 1  # 73
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)
key_regions = ['Arctic', 'Subpolar_gyre','SoutheastPacific','SOP', 'NPI', 'SO']
# %%
for model in MODELS:
    print(f"Processing model: {model}", flush=True)
    DIR_IN = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/{model}/regional_anomalies"
    files = sorted(glob.glob(f"{DIR_IN}/{model}_run*_regional_mean_trends_1950_2022_sliding.nc"))
    
    ds_list = []
    for file in files:
        ds = xr.open_dataset(file)
        ds_list.append(ds)
    
    combined_ds = xr.concat(ds_list, dim='run')
    # out_file = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/{model}/regional_anomalies/{model}_all_runs_regional_mean_trends_1950_2022_sliding.nc"
    # combined_ds.to_netcdf(out_file)
    # print(f"Saved combined dataset for {model} to {out_file}", flush=True)
# %%
SMILE_icv_pctl = {}  # model -> region -> DataArray(period, quantile)

for model in MODELS:
    print(f"Processing model: {model}", flush=True)
    DIR_IN = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV/{model}/regional_anomalies"
    files = sorted(glob.glob(f"{DIR_IN}/{model}_run*_regional_mean_trends_1950_2022_sliding.nc"))
    ds_list = [xr.open_dataset(f) for f in files]
    combined_ds = xr.concat(ds_list, dim="run")  # dims: run, period, region

    SMILE_icv_pctl[model] = {}
    for region in key_regions:
        da = combined_ds["trend_region"].sel(region=region)
        q = da.quantile([0.05, 0.95], dim="run")  # coords: quantile, period
        SMILE_icv_pctl[model][region] = q
        # Access like q.sel(quantile=0.05) for lower, q.sel(quantile=0.95) for upper
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
variable_name = np.arange(1950,2014)
keys = [f'{i}-2022' for i in variable_name]
region_name = ['Arctic', 'Subpolar_gyre','SoutheastPacific','SOP']
# load data
dir_input = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/'
dir_SOP_trend = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/SOP_update/'
dir_input_percentile = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/percentile/'
dir_SOP_percentile = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/SOP_update/percentile/'
# %%
data_array = {}
for region in region_name:
    if region == 'SOP':
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_SOP_trend}{region}_update_snr_gt_2_trend_variations.nc')
    else:
        data_array[region] = {}
        data_array[region] = xr.open_dataset(f'{dir_input}{region}_trend_variations.nc')
# %%
# put the forced and unforced data into the same dictionary
da_Arctic  = xr.Dataset(
    {'raw':data_array['Arctic'].raw, 
    'forced': data_array['Arctic'].forced, 
    'unforced': data_array['Arctic'].internal})
# %%
da_NorthAtlantic    = xr.Dataset({'raw':data_array['Subpolar_gyre'].raw, 'forced': data_array['Subpolar_gyre'].forced, 'unforced': data_array['Subpolar_gyre'].internal})
da_SoutheastPacific = xr.Dataset({'raw':data_array['SoutheastPacific'].raw, 'forced': data_array['SoutheastPacific'].forced, 'unforced': data_array['SoutheastPacific'].internal})
da_SOP              = xr.Dataset({'raw':data_array['SOP'].raw, 'forced': data_array['SOP'].forced, 'unforced': data_array['SOP'].internal})
# %%
arctic_unforced_lower = SMILE_icv_pctl['CESM2']['Arctic'].sel(quantile=0.05)
NAWH_unforced_lower   = SMILE_icv_pctl['CESM2']['Subpolar_gyre'].sel(quantile=0.05)
SEP_unforced_lower    = SMILE_icv_pctl['CESM2']['SoutheastPacific'].sel(quantile=0.05)
SOP_unforced_lower    = SMILE_icv_pctl['CESM2']['SOP'].sel(quantile=0.05)

arctic_unforced_upper = SMILE_icv_pctl['CESM2']['Arctic'].sel(quantile=0.95)
NAWH_unforced_upper   = SMILE_icv_pctl['CESM2']['Subpolar_gyre'].sel(quantile=0.95)
SEP_unforced_upper    = SMILE_icv_pctl['CESM2']['SoutheastPacific'].sel(quantile=0.95)
SOP_unforced_upper    = SMILE_icv_pctl['CESM2']['SOP'].sel(quantile=0.95)
# %%
# input the MMLE forced trend data
dir_MMLE_forced = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/MMLE/'
# load the MMLE forced trend data
MMLE_forced_ARC_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_ARC_trend_1950-2022_sliding.nc').trend
MMLE_forced_NAWH_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_subpolar_gyre_trend_1950-2022_sliding.nc').trend
MMLE_forced_SEP_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_SoutheastPacific_trend_1950-2022_sliding.nc').trend
MMLE_forced_SOP_trend = xr.open_dataset(f'{dir_MMLE_forced}MMLE_ENSforced_SOP_trend_1950-2022_sliding.nc').trend
# %%
"""
Plot to check ICV bands against obs trends
"""
plt.rcParams['figure.figsize'] = (8, 10)
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['savefig.transparent'] = True 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['legend.frameon'] = False

import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
# %%
import seaborn as sns
# sns.set_theme(style="whitegrid")
# Set the font dictionaries (for plot title and axis titles)
title_font = {'fontname': 'Arial', 'size': '20', 'color': 'black', 'weight': 'normal',
                'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
axis_font = {'fontname': 'Arial', 'size': '20'}

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Assuming the data arrays da_Arctic, da_NorthAtlantic, da_SoutheastPacific, da_SOP, and the bounds are defined
# Create the plot
fig = plt.figure(figsize=(25, 15))
gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.7)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])

# Define colors, line widths, titles, linestyles
colors = ['#0F1023', '#B11927', '#407BD0', '#B7D0EA']
linestyles = ['-', '-', '-.', ':']
titles = ['Arctic (ARC)', 'North Atlantic Warming Hole (NAWH)', 'Southeast Pacific (SEP)', 'Southern Ocean Pacific sector (SOP)']
vars = ['raw', 'forced', 'unforced', 'MMLE_forced']

for i, var in enumerate(vars):
    if var == 'MMLE_forced':
        da_Arctic[var] = MMLE_forced_ARC_trend
        da_NorthAtlantic[var] = MMLE_forced_NAWH_trend
        da_SoutheastPacific[var] = MMLE_forced_SEP_trend
        da_SOP[var] = MMLE_forced_SOP_trend
        sns.lineplot(x=np.arange(1950, 2014), y=da_Arctic[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax1)
        sns.lineplot(x=np.arange(1950, 2014), y=da_NorthAtlantic[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax2)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SoutheastPacific[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax3)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SOP[var].values, color="#B11927", linestyle=":", linewidth=5.5, ax=ax4)
    else:
        print(f"{var} shape: ", da_Arctic[var].shape)
        sns.lineplot(x=np.arange(1950, 2014), y=da_Arctic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax1)
        sns.lineplot(x=np.arange(1950, 2014), y=da_NorthAtlantic[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax2)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SoutheastPacific[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax3)
        sns.lineplot(x=np.arange(1950, 2014), y=da_SOP[var].values, color=colors[i], linestyle=linestyles[i], linewidth=3.5, ax=ax4)

# Add the unforced shading region
# add the unforced shading region with reversed upper and lower bounds
ax1.fill_between(np.arange(1950,2014), arctic_unforced_lower.values[::-1], arctic_unforced_upper.values[::-1], color=colors[3])
ax2.fill_between(np.arange(1950,2014), NAWH_unforced_lower.values[::-1], NAWH_unforced_upper.values[::-1], color=colors[3])
ax3.fill_between(np.arange(1950,2014), SEP_unforced_lower.values[::-1], SEP_unforced_upper.values[::-1], color=colors[3])
ax4.fill_between(np.arange(1950,2014), SOP_unforced_lower.values[::-1], SOP_unforced_upper.values[::-1], color=colors[3])

# Add the unforced shading region filled with hatched lines
years = np.arange(1950, 2014)

# for ax, low, high in [
#     (ax1, arctic_PI_unforced_lower.values[::-1], arctic_PI_unforced_upper.values[::-1]),
#     (ax2, NAWH_PI_unforced_lower.values[::-1],    NAWH_PI_unforced_upper.values[::-1]),
#     (ax3, SEP_PI_unforced_lower.values[::-1],     SEP_PI_unforced_upper.values[::-1]),
#     (ax4, SOP_PI_unforced_lower.values[::-1],     SOP_PI_unforced_upper.values[::-1]),
# ]:
#     ax.fill_between(years,
#                     low, high,
#                     facecolor='none',
#                     edgecolor='grey',
#                     hatch='//',
#                     linewidth=0)
ax1.set_ylim([-1.0, 1.0])
ax2.set_ylim([-1.0, 1.0])
ax3.set_ylim([-1.5, 1.0])
ax4.set_ylim([-0.6, 0.6])

ax1.set_xlim([1948, 2015])
ax2.set_xlim([1948, 2015])
ax3.set_xlim([1948, 2015])
ax4.set_xlim([1948, 2015])

# Set x and y axis limits
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim([1948, 2015])
    
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xticks([1953, 1963, 1973, 1983, 1993, 2003, 2013])
    ax.set_xticklabels(['1953', '1963', '1973', '1983', '1993', '2003', '2013'])
    ax.set_ylabel('Trend (°C/decade)', fontsize=30)
    ax.set_xlabel('Start year of linear trend', fontsize=30)
    # ax.tick_params(axis='y', which='major', length=12, width=2.5, direction='in')
    # ax.tick_params(axis='y', which='minor', length=8, width=2.5, direction='in')
    ax.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
    # ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    
# Add top x axis label
ax1_upper = ax1.twiny()
ax2_upper = ax2.twiny()
ax3_upper = ax3.twiny()
ax4_upper = ax4.twiny()

for ax_upper in [ax1_upper, ax2_upper, ax3_upper, ax4_upper]:
    ax_upper.invert_xaxis()
    ax_upper.set_xlim([75, 8])
    ax_upper.set_xlabel('Length of trends', fontsize=28, labelpad=10)
    ax_upper.set_xticks([70, 60, 50, 40, 30, 20, 10])
    ax_upper.set_xticklabels(['70', '60', '50', '40', '30', '20', '10'])
    ax_upper.tick_params(axis='x', labelsize=26)
    ax_upper.tick_params(axis='x', which='major', length=12, width=2.5, direction='in')
    ax_upper.xaxis.set_minor_locator(MultipleLocator(2))
    ax_upper.tick_params(axis='x', which='minor', length=8, width=2.5, direction='in')
    
# Customize spines
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

# Add zero line
for ax in [ax1, ax2, ax3, ax4]:
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=2.5, alpha=0.75)

# Add vertical lines
for ax in [ax1, ax2, ax3, ax4]:
    for year in [2013, 1993, 1963]:
        ax.axvline(x=year, color='#999A9E', linestyle='-', linewidth=2.5, alpha=0.75)
        
# move the title to the left and move up the title
ax1.set_title(titles[0], loc='left',fontsize=32,pad=20)
ax2.set_title(titles[1], loc='left',fontsize=32,pad=20)
ax3.set_title(titles[2], loc='left',fontsize=32,pad=20)
ax4.set_title(titles[3], loc='left',fontsize=32,pad=20)
# Add text labels for the vertical lines
# ax1.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax2.text(1979.1, -0.96, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax3.text(1979.1, -1.46, '1979-2022', fontsize=26, rotation=90, color='#999A9E')
# ax4.text(1979.1, -0.56, '1979-2022', fontsize=26, rotation=90, color='#999A9E')

# Add subplot order text
ax1.text(1945, 1.58, 'a', fontsize=35, ha='center', va='center', fontweight='bold')
ax2.text(1945, 1.58, 'b', fontsize=35, ha='center', va='center', fontweight='bold')
ax3.text(1945, 1.74, 'c', fontsize=38, ha='center', va='center', fontweight='bold')
ax4.text(1945, 0.95, 'd', fontsize=35, ha='center', va='center', fontweight='bold')

# Create custom legend
custom_lines = [Line2D([0], [0], color=colors[0], lw=3.5),
                Line2D([0], [0], color=colors[1], lw=3.5),
                Line2D([0], [0], color=colors[2], lw=3.5, linestyle='-.'),
                Line2D([0], [0], color=colors[1], lw=3.5, linestyle=':')]
leg1 = ax1.legend(custom_lines, ['total', 'external forcing', 'internal variability', 'MMLE forced'], 
                  loc='lower left', fontsize=18)
ax1.add_artist(leg1)
# add the legend for the unforced shading region
# --- custom legend for the two shaded regions ---
shade_handles = [
    # solid blue fill for the OBS‐ICV envelope
    Patch(
        facecolor=colors[3],
        edgecolor='none',
        alpha=0.4,          # match the alpha you used in fill_between
        label='OBS-ICV'
    ),
    # hatched patch for the piControl‐ICV region
    Patch(
        facecolor='none',
        edgecolor='grey',
        hatch='//',
        label='piControl-ICV'
    ),
]

leg2 = ax2.legend(
    handles=shade_handles,
    loc='lower left',      # choose a free corner
    fontsize=20
)
ax2.add_artist(leg2)
dir_fig_output = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG5/"
os.makedirs(dir_fig_output, exist_ok=True)
plt.savefig(f'{dir_fig_output}FIG5_CESM2_SMILE_range.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{dir_fig_output}FIG5_CESM2_SMILE_range.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{dir_fig_output}FIG5.eps', dpi=300, bbox_inches='tight')
plt.show()
# %%
def optionC2_equal_model_band(trends_by_model, taus=TAUS, qlo=0.05, qhi=0.95, K=None, nboot=200, seed=0):
    """
    trends_by_model: dict model -> dict tau -> array(member_trends)
    Returns arrays: lo(tau), hi(tau), med(tau) using equal-model mixture quantiles (bootstrapped).
    """
    rng = np.random.default_rng(seed)

    # choose K = min available member counts across models and taus (safe)
    if K is None:
        K = np.inf
        for m in trends_by_model:
            for tau in taus:
                arr = trends_by_model[m][tau]
                K = min(K, np.isfinite(arr).sum())
        K = int(max(5, min(K, 50)))  # cap for stability; adjust as you like

    lo_all = []
    hi_all = []
    med_all = []

    for _ in range(nboot):
        lo = []
        hi = []
        med = []
        for tau in taus:
            pooled = []
            for m in trends_by_model:
                arr = trends_by_model[m][tau]
                arr = arr[np.isfinite(arr)]
                # equal-model sampling
                samp = rng.choice(arr, size=K, replace=(arr.size < K))
                pooled.append(samp)
            pooled = np.concatenate(pooled)
            lo.append(np.quantile(pooled, qlo))
            hi.append(np.quantile(pooled, qhi))
            med.append(np.quantile(pooled, 0.5))
        lo_all.append(lo)
        hi_all.append(hi)
        med_all.append(med)

    # use median across bootstraps as the final band (stable)
    lo = np.nanmedian(np.array(lo_all), axis=0)
    hi = np.nanmedian(np.array(hi_all), axis=0)
    med = np.nanmedian(np.array(med_all), axis=0)
    return lo, hi, med, K

def add_startyear_axis(ax, end_year=END_YEAR):
    # secondary axis showing start year = 2023 - tau
    def tau_to_start(tau):  # tau -> start year
        return (end_year + 1) - tau
    def start_to_tau(start):  # start year -> tau
        return (end_year + 1) - start

    sec = ax.secondary_xaxis('bottom', functions=(tau_to_start, start_to_tau))
    sec.set_xlabel('Start year (end year fixed at 2022)')
    # Put primary label on top or keep on bottom; your choice:
    ax.set_xlabel(r'Trend length $\tau$ (years)')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Make start-year ticks nice (1950..2022)
    start_ticks = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010])
    sec.set_xticks(start_ticks)
    return sec

# You would:
# 1) build region_ts_members[model] = DataArray(member,year) for each model over 1950–2022
# 2) compute trends_by_model[model] = residual_member_trends_endfixed(region_ts_members[model])
# 3) mmle_lo, mmle_hi, mmle_med, K = optionC2_equal_model_band(trends_by_model)
# 4) compute obs_trend(tau) similarly but as a single trend (no member dimension)
# 5) plot
# %%
"""
Save the concatenated data out for each region of each LE model
"""

# ds.to_netcdf(out_file)
# print("Merged:", out_file)