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