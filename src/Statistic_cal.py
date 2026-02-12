#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 15, 2023
__author__ = "Dr. Josie Aruhasi"
This script contains the functions used in the OBS vs MMLE analysis.
The functions are:

- pattern_rmse: Calculate area-weighted RMSE between 2D fields.
"""
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys

from matplotlib.projections import PolarAxes
from matplotlib.projections import PolarAxes
import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy.stats as stats
import scipy.signal as signal
from typing import Optional, Tuple
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pattern_corr_da(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Simple Pearson correlation between two 2D fields x(lat, lon) and y(lat, lon).

    - Drops grid points where either field is NaN.
    - Flattens the remaining points and applies np.corrcoef.
    - Returns NaN if either field is constant (avoids RuntimeWarning from corrcoef).
    """
    ds = xr.Dataset({"x": x, "y": y})

    # Drop points where either field is NaN
    ds = ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]), drop=True)

    if ds["x"].size < 2:
        return np.nan

    x_flat = ds["x"].values.ravel()
    y_flat = ds["y"].values.ravel()

    if x_flat.size < 2:
        return np.nan

    # Avoid invalid divide in corrcoef when one series is constant
    if np.nanstd(x_flat) == 0.0 or np.nanstd(y_flat) == 0.0:
        return np.nan

    corr_matrix = np.corrcoef(x_flat, y_flat)
    corr_val = corr_matrix[0, 1]
    return float(corr_val) if np.isfinite(corr_val) else np.nan
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _prep_xy(x: xr.DataArray, y: xr.DataArray, dims=("lat","lon")):
    x_aligned, y_aligned = xr.align(x, y, join="inner")
    ds = xr.Dataset({"x": x_aligned, "y": y_aligned})
    ds = ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]), drop=True)
    if ds["x"].size == 0:
        return None
    return ds["x"], ds["y"]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _wmean(
    data: xr.DataArray,
    weights: xr.DataArray,
    dims: tuple = ("lat", "lon"),
) -> xr.DataArray:
    """
    Calculate weighted mean over specified dimensions.
    """
    weighted_sum = (data * weights).sum(dim=dims)
    sum_of_weights = weights.sum(dim=dims)
    return weighted_sum / sum_of_weights

def _weights_coslat(lat: xr.DataArray, data: xr.DataArray) -> xr.DataArray:
    w = xr.DataArray(np.cos(np.deg2rad(lat)), coords=lat.coords, dims=lat.dims)
    return w.broadcast_like(data)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pattern_rmse(
    x: xr.DataArray,
    y: xr.DataArray,
    lat: xr.DataArray,
    dims=("lat", "lon"),
    use_weights: bool = True,
    centered: bool = False,
) -> float:
    """
    Area-weighted (coslat) RMSE between 2D fields.

    centered=False -> full RMSE (includes mean/bias differences)
    centered=True  -> centered RMSE (removes weighted spatial mean from each pattern first)
                    = sqrt( Var(x') + Var(y') - 2 Cov(x',y') )
    """
    xy = _prep_xy(x, y, dims=dims)
    if xy is None:
        return float("nan")
    x_, y_ = xy

    if use_weights:
        w = _weights_coslat(lat, x_)
    else:
        w = xr.ones_like(x_)

    if centered:
        xm = _wmean(x_, w, dims=dims)
        ym = _wmean(y_, w, dims=dims)
        dx = (y_ - ym) - (x_ - xm)
    else:
        dx = y_ - x_

    wsum = w.sum(dim=dims)
    mse = (w * dx ** 2).sum(dim=dims) / wsum
    return float(np.sqrt(mse).values)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %%
"""
defined for construction of the SMILE icv regional trend range analysis.
# """
# def mmle_equal_model_band_from_xarray(
#     combined_by_model: dict,
#     region_list,
#     qlo=0.05,
#     qhi=0.95,
#     K=None,
#     nboot=200,
#     seed=0,
#     varname="trend_region",
#     run_dim="run",
#     period_dim="period",
#     region_dim="region",
# ):
#     """
#     Build MMLE equal-model mixture 5-95% band for each (region, period).

#     combined_by_model[model][varname] must have dims (run, period, region).
#     Returns xr.Dataset with variables: q_low, q_high, q_med (period, region).
#     """
#     rng = np.random.default_rng(seed)

#     # Use common periods across all models (safe for concat/compare)
#     period_sets = [set(ds[period_dim].values.tolist()) for ds in combined_by_model.values()]
#     common_periods = sorted(list(set.intersection(*period_sets)))
#     periods = xr.DataArray(common_periods, dims=[period_dim], name=period_dim)

#     # Prepare output arrays
#     nP = len(common_periods)
#     nR = len(region_list)
#     q_low  = np.full((nP, nR), np.nan, dtype=float)
#     q_high = np.full((nP, nR), np.nan, dtype=float)
#     q_med  = np.full((nP, nR), np.nan, dtype=float)

#     # Loop region/period and compute equal-model mixture quantiles
#     for j, region in enumerate(region_list):
#         for i, per in enumerate(common_periods):

#             # collect arrays across models for this (region, period)
#             arrs = []
#             for m, ds in combined_by_model.items():
#                 a = ds[varname].sel({region_dim: region, period_dim: per}).values  # shape: (run,)
#                 a = a[np.isfinite(a)]
#                 if a.size == 0:
#                     a = None
#                 arrs.append(a)

#             # drop models with missing data for this point
#             arrs = [a for a in arrs if a is not None]
#             if len(arrs) < 2:
#                 continue

#             # choose K per (region, period) if not provided
#             K_here = K if K is not None else min(a.size for a in arrs)
#             if K_here < 2:
#                 continue

#             def draw_one():
#                 pooled = []
#                 for a in arrs:
#                     pooled.append(rng.choice(a, size=K_here, replace=(a.size < K_here)))
#                 pooled = np.concatenate(pooled)
#                 return (
#                     np.quantile(pooled, qlo),
#                     np.quantile(pooled, qhi),
#                     np.quantile(pooled, 0.5),
#                 )

#             if nboot and nboot > 0:
#                 qs = np.array([draw_one() for _ in range(nboot)])  # (nboot, 3)
#                 q_low[i, j]  = np.nanmedian(qs[:, 0])
#                 q_high[i, j] = np.nanmedian(qs[:, 1])
#                 q_med[i, j]  = np.nanmedian(qs[:, 2])
#             else:
#                 lo, hi, md = draw_one()
#                 q_low[i, j], q_high[i, j], q_med[i, j] = lo, hi, md

#     out = xr.Dataset(
#         data_vars=dict(
#             q_low=((period_dim, region_dim), q_low),
#             q_high=((period_dim, region_dim), q_high),
#             q_med=((period_dim, region_dim), q_med),
#         ),
#         coords={
#             period_dim: periods.values,
#             region_dim: region_list,
#         },
#         attrs=dict(
#             method="MMLE equal-model mixture quantiles",
#             qlo=qlo, qhi=qhi, K=K if K is not None else "min per (region,period)",
#             nboot=nboot, seed=seed,
#         ),
#     )
#     return out
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
# %%
def mmle_band_deterministic_median_of_models(combined_by_model, region_list,
                                            qlo=0.05, qhi=0.95, varname="trend_region"):
    models = list(combined_by_model.keys())
    common_periods = sorted(list(set.intersection(
        *[set(ds["period"].values.tolist()) for ds in combined_by_model.values()]
    )))

    q_low = np.full((len(common_periods), len(region_list)), np.nan)
    q_high = np.full_like(q_low, np.nan)

    for j, region in enumerate(region_list):
        for i, per in enumerate(common_periods):
            lows, highs = [], []
            for m in models:
                a = combined_by_model[m][varname].sel(region=region, period=per).values
                a = a[np.isfinite(a)]
                if a.size < 2:
                    continue
                lows.append(np.quantile(a, qlo))
                highs.append(np.quantile(a, qhi))
            if len(lows) >= 2:
                q_low[i, j] = np.median(lows)
                q_high[i, j] = np.median(highs)

    return xr.Dataset(
        {"q_low": (("period", "region"), q_low),
         "q_high": (("period", "region"), q_high)},
        coords={"period": common_periods, "region": region_list},
    )
# %%
# the envelope method for MMLE band construction
def mmle_band_outer_envelope_from_xarray(
    combined_by_model: dict,
    region_list,
    qlo=0.05,
    qhi=0.95,
    varname="trend_region",
    run_dim="run",
    period_dim="period",
    region_dim="region",
):
    """
    Deterministic MMLE 'outer envelope' band:
      - For each model m: compute L_m = qlo quantile across runs, U_m = qhi quantile across runs
      - MMLE lower = min_m(L_m), MMLE upper = max_m(U_m)

    combined_by_model[model][varname] must have dims (run, period, region).
    Returns xr.Dataset with variables: q_low_env, q_high_env, plus per-model bounds.
    """
    models = list(combined_by_model.keys())

    # Use common periods across all models
    period_sets = [set(ds[period_dim].values.tolist()) for ds in combined_by_model.values()]
    common_periods = sorted(list(set.intersection(*period_sets)))

    # Pre-allocate arrays: (period, region)
    nP = len(common_periods)
    nR = len(region_list)

    low_env  = np.full((nP, nR), np.nan, dtype=float)
    high_env = np.full((nP, nR), np.nan, dtype=float)

    # Optional: store per-model bounds for diagnostics (model, period, region)
    low_m  = np.full((len(models), nP, nR), np.nan, dtype=float)
    high_m = np.full((len(models), nP, nR), np.nan, dtype=float)

    for mi, m in enumerate(models):
        ds = combined_by_model[m]
        da = ds[varname].sel({period_dim: common_periods, region_dim: region_list})
        # Ensure (period, region) ordering before quantile to avoid shape flips
        da = da.transpose(period_dim, region_dim, ...)

        # Model-specific quantiles across run dimension: dims (quantile, period, region)
        q = da.quantile([qlo, qhi], dim=run_dim, skipna=True)
        low_m[mi, :, :]  = q.sel(quantile=qlo).values
        high_m[mi, :, :] = q.sel(quantile=qhi).values

    # Outer envelope across models
    low_env  = np.nanmin(low_m, axis=0)   # min over model
    high_env = np.nanmax(high_m, axis=0)  # max over model

    out = xr.Dataset(
        data_vars=dict(
            q_low_env=((period_dim, region_dim), low_env),
            q_high_env=((period_dim, region_dim), high_env),
            # keep these if you want to debug/plot per-model envelopes
            q_low_model=(( "model", period_dim, region_dim), low_m),
            q_high_model=(( "model", period_dim, region_dim), high_m),
        ),
        coords={
            "model": models,
            period_dim: common_periods,
            region_dim: region_list,
        },
        attrs=dict(
            method="Deterministic MMLE outer envelope of model-specific quantile bounds",
            qlo=qlo, qhi=qhi, varname=varname
        ),
    )
    return out
# %%
# update on Feb 6, 2026: calculate each model's range and the get the mean envelope of those
def mmle_mean_model_band_from_xarray(combined_by_model, region_list, periods=None,
                                    qlo=0.05, qhi=0.95, varname="trend_region"):
    models = list(combined_by_model.keys())
    # common periods       
    if periods is None:
        common = sorted(list(set.intersection(
            *[set(ds["period"].values.tolist()) for ds in combined_by_model.values()]
        )))
    else:
        common = list(periods)  
    nP, nR = len(common), len(region_list)
    q_low  = np.full((nP, nR), np.nan, float) 
    q_high = np.full((nP, nR), np.nan, float)
    q_med  = np.full((nP, nR), np.nan, float)
    
    for j, region in enumerate(region_list):
        for i, per in enumerate(common):
            lows, highs, meds = [], [], []
            for m in models:
                a = combined_by_model[m][varname].sel(region=region, period=per).values
                a = a[np.isfinite(a)]
                if a.size < 2:
                    continue
                lows.append(np.quantile(a, qlo))
                highs.append(np.quantile(a, qhi))
                meds.append(np.quantile(a, 0.5))
            if len(lows) >= 2:
                q_low[i, j] = np.mean(lows)
                q_high[i, j] = np.mean(highs)
                q_med[i, j] = np.mean(meds)
    return xr.Dataset(
        {"q_low": (("period", "region"), q_low),
         "q_high": (("period", "region"), q_high),
         "q_med": (("period", "region"), q_med)},
        coords={"period": common, "region": region_list},
        attrs={"method": "mean_model_quantiles", "qlo": qlo, "qhi": qhi}
    )
# %%
# update on Jan 21, 2026: equal weighting concatenation of samples from each model; and get the 5th--95th percentiles
def mmle_equal_model_band_from_xarray(combined_by_model, region_list, periods=None,
                                      qlo=0.05, qhi=0.95, varname="trend_region",
                                      run_dim="run", K=None, seed=0):
    models = list(combined_by_model.keys())

    def _weighted_quantile(values, weights, q):
        values = np.asarray(values)
        weights = np.asarray(weights)
        if values.size == 0 or weights.sum() == 0:
            return np.nan
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, values))

    # common periods       
    if periods is None:
        common = sorted(list(set.intersection(
            *[set(ds["period"].values.tolist()) for ds in combined_by_model.values()]
        )))
    else:
        common = list(periods)

    nP, nR = len(common), len(region_list)
    q_low  = np.full((nP, nR), np.nan, float) 
    q_high = np.full((nP, nR), np.nan, float)
    q_med  = np.full((nP, nR), np.nan, float)

    for j, region in enumerate(region_list):
        for i, per in enumerate(common):
            pooled_vals = []
            pooled_wts = []
            for m in models:
                a = combined_by_model[m][varname].sel(region=region, period=per).values
                a = a[np.isfinite(a)]
                if a.size == 0:
                    continue
                w = np.full_like(a, 1.0 / a.size, dtype=float)
                pooled_vals.append(a)
                pooled_wts.append(w)
            if len(pooled_vals) >= 2:
                vals = np.concatenate(pooled_vals)
                wts = np.concatenate(pooled_wts)
                q_low[i, j]  = _weighted_quantile(vals, wts, qlo)
                q_high[i, j] = _weighted_quantile(vals, wts, qhi)
                q_med[i, j]  = _weighted_quantile(vals, wts, 0.5)

    return xr.Dataset(
        {"q_low": (("period", "region"), q_low),
         "q_high": (("period", "region"), q_high),
         "q_med": (("period", "region"), q_med)},
        coords={"period": common, "region": region_list},
        attrs={"method": "equal_model_weighted_quantiles", "weights": "1/run_count_per_model", "qlo": qlo, "qhi": qhi}
    )
# %% 
def area_weighted_rms(field, lat):
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    w2 = w.broadcast_like(field)
    wsum = w2.sum(dim=("lat", "lon"))
    return np.sqrt((w2 * field**2).sum(dim=("lat", "lon")) / wsum)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++