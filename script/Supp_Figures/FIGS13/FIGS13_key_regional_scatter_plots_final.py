# %%
"""
Plotting key regional scatter plots: 
1. Arctic
2. Subpolar gyre
3. Southeast Pacific
4. Southern Ocean Pacific sector (SOP)
Two panels for each region:
- Panel a: Scatter plot of predicted regional trends due to external forcing vs. Actual regional trends "OBS-LPS" partitioned trend
    X-axis: predicted regional trends due to external forcing (MMLE forced trend)
    Y-axis: Actual regional trends "OBS-LPS" partitioned trend
- Panel b: Scatter plot of predicted regional trends due to internal variability (residual trends) vs. predicted regional internal variability "residual of the OBS-LPS" partitioned trend
    X-axis: predicted regional trends due to internal variability (SMILE unforced trend) 
    Y-axis: Actual regional trends due to internal variability (residual of the OBS-LPS forced trend)
"""
# %%
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# %%
def _linfit(x, y):
    """Simple least squares fit y = a*x + b, returns (a, b)."""
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    A = np.vstack([x[m], np.ones(m.sum())]).T
    a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
    return float(a), float(b)

def _rmse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - yhat[m])**2)))
# -----------------------------
file_id_to_label = {
    "CanESM5":   "CanESM5(50)",
    "CESM2":     "CESM2(100)",
    "IPSL_CM6A": "IPSL-CM6A-LR(32)",
    "IPSL":      "IPSL-CM6A-LR(32)",
    "EC_Earth3": "EC-Earth3(21)",
    "EC":        "EC-Earth3(21)",
    "ACCESS":    "ACCESS-ESM1.5(40)",
    "MIROC6":    "MIROC6(50)",
    "MPI_ESM":   "MPI-ESM1.2-LR(50)",
    "MPI":       "MPI-ESM1.2-LR(50)",
}

# RGB_dict = {
#     "CanESM5(50)": "#A60E16",
#     "CESM2(100)": "#EE3B2A",
#     "IPSL-CM6A-LR(32)": "#FC9171",
#     "EC-Earth3(21)": "#FDDFCF",
#     "ACCESS-ESM1.5(40)": "#5FB7B5",
#     "MPI-ESM1.2-LR(50)": "#7AB0DF",
#     "MIROC6(50)": "#0F55C5",
#     "MMLE": "black",
# }
RGB_dict = {'CanESM5(50)':'#A60E16', 
            'CESM2(100)':'#EE3B2A',
            'IPSL-CM6A-LR(32)':'#FC9171', 
            'EC-Earth3(21)':"#F5BFA2", 
            'ACCESS-ESM1.5(40)':"#5FB7B5",
            'MPI-ESM1.2-LR(50)':"#246BA1", 
            'MIROC6(50)':"#073278", 
            'MMLE':'black'}

category_style = {
    "member":         dict(marker="o", mfc="none", mec="k", mew=0.9, ms=30, alpha=0.85, linestyle=""),
    "HadCRUT5":       dict(marker="s", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
    "NOAAGlobalTemp": dict(marker="8", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
    "BEST":           dict(marker="^", mfc="k",    mec="k", mew=1.0, ms=32.5, alpha=0.95, linestyle=""),
}
MEMBER_SCATTER_STYLE = category_style["member"]
MODELS = [
    "CanESM5",
    "CESM2",
    "IPSL_CM6A",
    "EC_Earth3",
    "ACCESS",
    "MPI_ESM",
    "MIROC6",
]

END_YEAR = 2022
START_MIN = 1950
TAU_MIN = 10
TAU_MAX = END_YEAR - START_MIN + 1  # 73
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)
KEY_REGIONS = ["ARC", "subpolar_gyre", "SoutheastPacific", "SOP"]

PERIODS_TO_PLOT = {
    "10-year": "2013-2022",
    "30-year": "1993-2022",
    "60-year": "1963-2022",
}

REGION_ROWS = [
    ("Arctic", "Arctic"),
    ("Subpolar gyre", "subpolar_gyre"),
    ("Southeastern Pacific", "SoutheastPacific"),
    ("SOP", "SOP_original"),
]

FIG_OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/FIG5_regional_scatter/"
os.makedirs(FIG_OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)

# -----------------------------
# Data loaders
# -----------------------------
def _region_variants(region):
    # try a handful of common filename variants
    r = region
    return list(dict.fromkeys([
        r,
        r.lower(),
        r.upper(),
        r.replace("_", ""),
        r.replace("_", "-"),
        r.replace("-", "_"),
        r.title(),
        r.capitalize(),
    ]))

def load_actual_forced():
    """
    Returns:
      combined[model][region] = xr.Dataset  (NOT a list)
    Supports two layouts:
      A) file per (model, region): {model}_ENSforced_{region}_trend_1950-2022_sliding.nc
      B) file per model (contains region dim): {model}_ENSforced_*_trend_1950-2022_sliding.nc
    """
    combined = {}
    base = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_predict_forced/"

    missing = []
    for model in MODELS:
        combined[model] = {}
        # first: discover any model-level files (layout B)
        # {model}_ENSforced_SoutheastPacific_trend_1950-2022_sliding.nc
        model_level_files = sorted(glob.glob(f"{base}/{model}_ENSforced_*_trend_1950-2022_sliding.nc"))
        ds_model_level = None
        if len(model_level_files) > 0:
            # If there are multiple, pick the first; you can tighten this if needed
            ds_model_level = xr.open_dataset(model_level_files[0])

        for region in KEY_REGIONS:
            files = []
            # layout A: exact region in filename (try variants)
            for rv in _region_variants(region):
                files = sorted(glob.glob(f"{base}/{model}_ENSforced_{rv}_trend_1950-2022_sliding.nc"))
                if len(files) > 0:
                    break

            if len(files) > 0:
                combined[model][region] = xr.open_dataset(files[0])
            elif ds_model_level is not None:
                # layout B: use model-level file (select region later)
                combined[model][region] = ds_model_level
            else:
                combined[model][region] = None
                missing.append((model, region))

    if len(missing) > 0:
        print("WARNING: Missing predicted-forced files for:")
        for m, r in missing:
            print(f"  - {m} / {r}")
        print("Those (model, region) will be skipped in forced panels.")
    return combined

def load_estimated_forced():
    combined = {}
    for model in MODELS:
        dir_in = f"/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/{model}/regional_anomalies/"
        files = sorted(glob.glob(f"{dir_in}/{model}_run*_regional_mean_trends_1950_2022_sliding.nc"))
        ds_list = [xr.open_dataset(f) for f in files]
        combined[model] = xr.concat(ds_list, dim="run")
    return combined

def load_estimated_icv():
    combined = {}
    base_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/Revision_check/SMILE_actual_residual_ICV"
    for model in MODELS:
        dir_in = f"{base_dir}/{model}/regional_anomalies/"
        files = sorted(glob.glob(f"{dir_in}/{model}_run*_regional_mean_trends_ICV_1950_2022_sliding.nc"))
        ds_list = [xr.open_dataset(f) for f in files]
        combined[model] = xr.concat(ds_list, dim="run")
    return combined

def load_actual_icv():
    combined = {}
    base_dir = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/SMILE_residual_ICV"
    for model in MODELS:
        dir_in = f"{base_dir}/{model}/regional_anomalies"
        files = sorted(glob.glob(f"{dir_in}/{model}_run*_regional_mean_trends_1950_2022_sliding.nc"))
        ds_list = [xr.open_dataset(f) for f in files]
        combined[model] = xr.concat(ds_list, dim="run")
    return combined

# load once globally with clearer names
# - SMILE forced prediction (scalar per model/region)
combined_SMILE_forced_pred      = load_actual_forced()
# - SMILE internal variability prediction (per-run residuals)
combined_SMILE_ICV_pred         = load_actual_icv()
# - OBS-LPS forced (actual) per-run trends
combined_OBS_LPS_forced_actual  = load_estimated_forced()
# - OBS-LPS internal variability (actual) per-run residuals
combined_OBS_LPS_ICV_actual     = load_estimated_icv()
# %%
"""
Read in the OBS data forced and ICV trends
"""
OBS_data = ['Berkeley', 'NOAA', 'HadCRUT5']
region_name = ['Arctic', 'Subpolar_gyre', 'SoutheastPacific','SOP'] 

data_OBS_array = {}
dir_input_HadCRUT = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/'
dir_SOP_trend_HadCRUT = '/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG5/data/SOP_update/'
for OBS in OBS_data:
    data_OBS_array[OBS] = {}
    for region in region_name:
        if OBS == 'HadCRUT5' and region != 'SOP':
            data_OBS_array[OBS][region] = xr.open_dataset(f'{dir_input_HadCRUT}{region}_trend_variations.nc')
        elif OBS == 'HadCRUT5' and region == 'SOP':
            data_OBS_array[OBS][region] = xr.open_dataset(f'{dir_SOP_trend_HadCRUT}{region}_update_snr_gt_2_trend_variations.nc')
        else:
            DIR_INPUT = f'/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIGS_OBS_records/regional_data/{OBS}/'
            data_OBS_array[OBS][region] = xr.open_dataset(f'{DIR_INPUT}{region}_{OBS}_trend_variations.nc')
# %%
# -----------------------------
# Helpers (robust selection)
# -----------------------------
def _first_data_var(ds):
    return list(ds.data_vars)[0]

def _pick_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    return _first_data_var(ds)

def _norm_str(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _sel_like(da, dim, key):
    """Select along a coord dim with robust matching."""
    if dim not in da.dims and dim not in da.coords:
        return da
    vals = da[dim].values
    keyn = _norm_str(key)
    # exact match first
    if key in vals:
        return da.sel({dim: key})
    # normalized match
    for v in vals:
        if _norm_str(v) == keyn:
            return da.sel({dim: v})
    # try contains match
    for v in vals:
        if keyn in _norm_str(v):
            return da.sel({dim: v})
    raise KeyError(f"Could not match {key} on dim/coord '{dim}'. Available: {list(vals)[:10]}...")

def _to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

def _nan_pearsonr(x, y):
    x = _to_1d(x); y = _to_1d(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def _get_obs_scalar(region, period_str, obs_name, kind):
    """Return scalar obs value for given region/period/kind ('forced'|'icv')."""
    ds = data_OBS_array.get(obs_name, {}).get(region)
    if ds is None:
        return np.nan

    if kind == "forced":
        candidates = ["forced"]
    else:
        candidates = ["internal"]

    v = _pick_var(ds, candidates)
    da = ds[v]

    if ("region" in da.dims) or ("region" in da.coords):
        da = _sel_like(da, "region", region)
    # select period coordinate; obs files use "index" for period-like labels
    if ("period" in da.dims) or ("period" in da.coords):
        da = _sel_like(da, "period", period_str)
    elif ("index" in da.dims) or ("index" in da.coords):
        da = _sel_like(da, "index", period_str)

    try:
        return float(np.asarray(da).squeeze())
    except Exception:
        return np.nan

def _get_pred_forced_scalar(model, region, period_str):
    ds = combined_SMILE_forced_pred[model][region]
    if ds is None:
        return np.nan

    v = _pick_var(ds, candidates=["trend_region", "forced", "trend", "pred", "ENSforced"])
    da = ds[v]

    # If region exists inside the dataset, select it
    if ("region" in da.dims) or ("region" in da.coords):
        da = _sel_like(da, "region", region)

    # Select period
    if ("period" in da.dims) or ("period" in da.coords):
        da = _sel_like(da, "period", period_str)

    return float(np.asarray(da).squeeze())

def _get_actual_forced_runs(model, region, period_str):
    ds = combined_OBS_LPS_forced_actual[model]
    v = _pick_var(ds, candidates=[
        "forced", "trend_forced", "forced_trend", "trend_region_forced",
        "trend_region", "trend"
    ])
    da = ds[v]
    da = _sel_like(da, "region", region)
    da = _sel_like(da, "period", period_str)
    # expect dims include run
    return _to_1d(np.asarray(da))

def _get_pred_icv_runs(model, region, period_str):
    ds = combined_SMILE_ICV_pred[model]
    v = _pick_var(ds, candidates=[
        "trend_region", "trend"
    ])
    da = ds[v]
    da = _sel_like(da, "region", region)
    da = _sel_like(da, "period", period_str)
    return _to_1d(np.asarray(da))

def _get_actual_icv_runs(model, region, period_str):
    ds = combined_OBS_LPS_ICV_actual[model]
    v = _pick_var(ds, candidates=[
        "trend_region", "trend"
    ])
    da = ds[v]
    da = _sel_like(da, "region", region)
    da = _sel_like(da, "period", period_str)
    return _to_1d(np.asarray(da))
# %%
# -----------------------------
# Plot settings
# -----------------------------
# ============================================================
# Clean plotting: swap axes + add OBS markers + OBS range shade
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# --- which obs products to show
OBS_PRODUCTS = ["HadCRUT5"]  # match your keys in data_OBS_array
OBS_MARKERS = {"HadCRUT5": "s", "NOAA": "8", "Berkeley": "^"}  # consistent w/ your category_style
OBS_marker_STYLE = {
    "HadCRUT5": dict(color="blue", mfc="blue",  mec="blue", lw=1.4, alpha=0.9),
    # "NOAA":     dict(color="#1b9e77", mfc="#1b9e77", mec="#1b9e77", lw=1.4, alpha=0.9),
    # "Berkeley": dict(color="#d95f02", mfc="#d95f02", mec="#d95f02", lw=1.4, alpha=0.9),
}

DEBUG_OBS = True

def _pooled_stats_and_line(ax, x, y, draw=True):
    """Compute pooled r and RMSE to pooled linear fit; optionally draw pooled line."""
    r = _nan_pearsonr(x, y)
    a, b = _linfit(x, y)
    yhat = a * x + b
    rmse = _rmse(y, yhat)

    if draw and np.isfinite(a) and np.isfinite(b):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 1:
            xx = np.linspace(np.min(x[m]), np.max(x[m]), 80)
            ax.plot(xx, a * xx + b, color="0.35", lw=2.2, alpha=0.9, zorder=3)
    return r, rmse

def _add_one_to_one(ax, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return
    lo = min(np.min(x[m]), np.min(y[m]))
    hi = max(np.max(x[m]), np.max(y[m]))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.5, color="0.35", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

def _add_obs_xaxis_band_and_markers(ax, region_obs_name, period_str, kind):
    """
    Shade x-range among obs products + add markers exactly on the x-axis (bottom border).
    kind must match _get_obs_scalar: "forced" or "internal"
    """
    xs = []
    for obs in OBS_PRODUCTS:
        xs.append(_get_obs_scalar(region_obs_name, period_str, obs, kind))
    xs = np.array(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        if DEBUG_OBS:
            print(f"[OBS missing] region={region_obs_name}, period={period_str}, kind={kind}")
        return

    xlo, xhi = float(np.min(xs)), float(np.max(xs))

    # shade obs range (vertical band)
    ax.axvspan(xlo, xhi, color="0.85", alpha=0.6, zorder=0)

    # Put markers exactly at the bottom of current y-limits (x-axis line)
    y_axis = ax.get_ylim()[0]

    for obs in OBS_PRODUCTS:
        x = _get_obs_scalar(region_obs_name, period_str, obs, kind)
        if not np.isfinite(x):
            if DEBUG_OBS:
                print(f"[OBS NaN] {obs} region={region_obs_name}, period={period_str}, kind={kind}")
            continue

        mk = OBS_MARKERS.get(obs, "o")
        style = dict(OBS_marker_STYLE.get(obs, {}))
        style.setdefault("ms", 9.5)
        style.setdefault("mew", 1.1)
        # IMPORTANT: clip_on=False so it can sit on the axis border
        ax.plot([x], [y_axis], marker=mk, linestyle="", zorder=6, clip_on=False, **style)

# %%
# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
periods_to_plot = {
    "30-year": "1993-2022",
    "44-year": "1979-2022",
    "60-year": "1963-2022",
}

# IMPORTANT: OBS region naming differs for Subpolar gyre in your obs reads
# model region_key -> obs region_name
REGION_ROWS = [
    ("Arctic",               "ARC",          "Arctic"),
    ("NAWH",        "subpolar_gyre",   "Subpolar_gyre"),
    ("SEP", "SoutheastPacific","SoutheastPacific"),
    ("SOP",                  "SOP",             "SOP"),
]

FIG_OUT_DIR = "/work/mh0033/m301036/OBS_LPS_revision/docs/Figs/Revision_check/FIG5_regional_scatter/"
os.makedirs(FIG_OUT_DIR, exist_ok=True)
rng = np.random.default_rng(0)

for tag, period_str in periods_to_plot.items():
    nrows = len(REGION_ROWS)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2,
        figsize=(12, 4.0 * nrows),
        sharex=False, sharey=False,
        constrained_layout=True
    )

    # -------------------------
    # FIRST PASS: Collect all data to determine global axis limits
    # -------------------------
    all_data_forced = {'x': [], 'y': []}
    all_data_icv = {'x': [], 'y': []}
    
    for irow, (row_label, region_key, obs_region) in enumerate(REGION_ROWS):
        # Collect forced data
        for model in MODELS:
            y_pred = _get_pred_forced_scalar(model, region_key, period_str)
            if not np.isfinite(y_pred):
                continue
            x_act = _get_actual_forced_runs(model, region_key, period_str)
            if x_act.size == 0:
                continue
            all_data_forced['x'].append(x_act)
            all_data_forced['y'].append(np.full_like(x_act, y_pred, dtype=float))
        
        # Collect ICV data
        for model in MODELS:
            y_pred = _get_pred_icv_runs(model, region_key, period_str)
            x_act = _get_actual_icv_runs(model, region_key, period_str)
            n = min(y_pred.size, x_act.size)
            if n < 2:
                continue
            all_data_icv['x'].append(x_act[:n])
            all_data_icv['y'].append(y_pred[:n])
    
    # Calculate global limits
    xF_global = np.concatenate(all_data_forced['x']) if len(all_data_forced['x']) else np.array([])
    yF_global = np.concatenate(all_data_forced['y']) if len(all_data_forced['y']) else np.array([])
    xI_global = np.concatenate(all_data_icv['x']) if len(all_data_icv['x']) else np.array([])
    yI_global = np.concatenate(all_data_icv['y']) if len(all_data_icv['y']) else np.array([])
    
    # Calculate axis limits for forced
    if xF_global.size > 0 and yF_global.size > 0:
        mF = np.isfinite(xF_global) & np.isfinite(yF_global)
        loF = min(np.min(xF_global[mF]), np.min(yF_global[mF]))
        hiF = max(np.max(xF_global[mF]), np.max(yF_global[mF]))
        padF = 0.05 * (hiF - loF) if hiF > loF else 0.1
        loF -= padF; hiF += padF
    else:
        loF, hiF = -1, 1
    
    # Calculate axis limits for ICV
    if xI_global.size > 0 and yI_global.size > 0:
        mI = np.isfinite(xI_global) & np.isfinite(yI_global)
        loI = min(np.min(xI_global[mI]), np.min(yI_global[mI]))
        hiI = max(np.max(xI_global[mI]), np.max(yI_global[mI]))
        padI = 0.05 * (hiI - loI) if hiI > loI else 0.1
        loI -= padI; hiI += padI
    else:
        loI, hiI = -1, 1
    
    # -------------------------
    # SECOND PASS: Plot with consistent axis limits
    # -------------------------
    for irow, (row_label, region_key, obs_region) in enumerate(REGION_ROWS):
        axF = axes[irow, 0]  # forced
        axI = axes[irow, 1]  # internal variability

        # pooled for stats (swap axes!)
        xF_all, yF_all = [], []
        xI_all, yI_all = [], []

        # -------------------------
        # Forced: x = OBS-LPS forced (per run), y = LE ENS mean forced (scalar)
        # -------------------------
        for model in MODELS:
            y_pred = _get_pred_forced_scalar(model, region_key, period_str)  # LE ensemble mean forced
            if not np.isfinite(y_pred):
                continue

            x_act = _get_actual_forced_runs(model, region_key, period_str)   # OBS-LPS forced per run (estimated)
            if x_act.size == 0:
                continue

            # jitter y slightly (because y is scalar per model)
            jitter = rng.normal(0.0, 0.002, size=x_act.size)
            y_plot = y_pred + jitter

            color = RGB_dict.get(file_id_to_label.get(model, model), "k")
            axF.scatter(x_act, y_plot, s=25, color=color, edgecolor=color, facecolor="none", zorder=1)

            xF_all.append(x_act)
            yF_all.append(np.full_like(x_act, y_pred, dtype=float))

        xF_all = np.concatenate(xF_all) if len(xF_all) else np.array([])
        yF_all = np.concatenate(yF_all) if len(yF_all) else np.array([])

        # -------------------------
        # ICV: x = OBS-LPS ICV actual (per run), y = LE residual predicted (per run)
        # -------------------------
        for model in MODELS:
            y_pred = _get_pred_icv_runs(model, region_key, period_str)   # LE residual (per run)
            x_act  = _get_actual_icv_runs(model, region_key, period_str) # OBS-LPS ICV (per run)

            n = min(y_pred.size, x_act.size)
            if n < 2:
                continue
            y_pred = y_pred[:n]
            x_act  = x_act[:n]

            color = RGB_dict.get(file_id_to_label.get(model, model), "k")
            axI.scatter(x_act, y_pred, s=25, color=color, edgecolor=color, facecolor="none", zorder=1)

            # per-model fit line (ICV): fit y_pred ~ x_act
            a_m, b_m = _linfit(x_act, y_pred)
            if np.isfinite(a_m) and np.isfinite(b_m):
                m = np.isfinite(x_act) & np.isfinite(y_pred)
                if m.sum() > 1:
                    xx = np.linspace(np.min(x_act[m]), np.max(x_act[m]), 60)
                    axI.plot(xx, a_m * xx + b_m, color=color, lw=1.4, alpha=0.9, zorder=3)

            xI_all.append(x_act)
            yI_all.append(y_pred)

        xI_all = np.concatenate(xI_all) if len(xI_all) else np.array([])
        yI_all = np.concatenate(yI_all) if len(yI_all) else np.array([])
        
        # -------------------------
        # Apply consistent axis limits and 1:1 lines
        # -------------------------
        # Forced panel
        axF.plot([loF, hiF], [loF, hiF], ls="--", lw=1.5, color="0.35", zorder=2)
        axF.set_xlim(loF, hiF)
        axF.set_ylim(loF, hiF)
        
        # ICV panel  
        axI.plot([loI, hiI], [loI, hiI], ls="--", lw=1.5, color="0.35", zorder=2)
        axI.set_xlim(loI, hiI)
        axI.set_ylim(loI, hiI)

        # pooled stats + pooled line
        rF, rmseF = _pooled_stats_and_line(axF, xF_all, yF_all, draw=True)
        rI, rmseI = _pooled_stats_and_line(axI, xI_all, yI_all, draw=True)
        # -------------------------
        # Add observational markers + shading
        # -------------------------
        _add_obs_xaxis_band_and_markers(axF, obs_region, period_str, kind="forced")
        _add_obs_xaxis_band_and_markers(axI, obs_region, period_str, kind="internal")

        # -------------------------
        # Titles, labels, annotations
        # -------------------------
        Panel_label = chr(ord("A") + irow * 2)
        axF.text(-0.08, 1.1, Panel_label, transform=axF.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="right")
        Panel_label = chr(ord("A") + irow * 2 + 1)
        axI.text(-0.08, 1.1, Panel_label, transform=axI.transAxes,
                 fontsize=18, fontweight="bold", va="top", ha="right")
        
        axF.set_title(f"{row_label} — External forced", loc="center", fontsize=16)
        axI.set_title(f"{row_label} — Internal variability", loc="center", fontsize=16)

        axF.text(0.02, 0.95,
                 (f"r = {rF:.2f}\nRMSE(fit) = {rmseF:.3g}" if np.isfinite(rF) else "r = n/a"),
                 transform=axF.transAxes, va="top", ha="left", fontsize=16)
        axI.text(0.02, 0.95,
                 (f"r = {rI:.2f}\nRMSE(fit) = {rmseI:.3g}" if np.isfinite(rI) else "r = n/a"),
                 transform=axI.transAxes, va="top", ha="left", fontsize=16)

        # swap-axis labels (your requested text)
        if irow == nrows - 1:
            axF.set_xlabel("Estimated Trend from OBS-LPS (°C/decade)", fontsize=12)
            axI.set_xlabel("Estimated Trend from OBS-LPS (°C/decade)", fontsize=12)

        axF.set_ylabel("LE ensemble mean (°C/decade)", fontsize=12)
        axI.set_ylabel("LE residual (°C/decade)", fontsize=12)
        # clean spines/ticks
        for ax in (axF, axI):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

    # --- shared legend for models (colors)
    handles, labels = [], []
    for model in MODELS:
        lab = file_id_to_label.get(model, model)
        col = RGB_dict.get(lab, "k")
        h = plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=col, markeredgecolor="none",
                       markersize=7, alpha=0.9)
        handles.append(h); labels.append(lab)

    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=12)
    # add observational legend
    obs_handles, obs_labels = [], []
    for obs in OBS_PRODUCTS:
        mk = OBS_MARKERS.get(obs, "o")
        style = dict(OBS_marker_STYLE.get(obs, {}))
        style.setdefault("ms", 7.5)
        style.setdefault("mew", 1.1)
        h = plt.Line2D([0], [0], marker=mk, linestyle="",
                       **style)
        obs_handles.append(h)
        obs_labels.append(obs)
    fig.legend(obs_handles, obs_labels, ncol=len(OBS_PRODUCTS), loc="lower center", frameon=False,
               bbox_to_anchor=(0.775, -0.06), fontsize=12)
    out_pdf = os.path.join(FIG_OUT_DIR, f"Regional_scatter_{tag}_{period_str}_aligned.pdf".replace(" ", ""))
    out_png = os.path.join(FIG_OUT_DIR, f"Regional_scatter_{tag}_{period_str}_aligned.png".replace(" ", ""))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

print("Done. Saved figures to:", FIG_OUT_DIR)
# %%