# %%
# Source - https://stackoverflow.com/q
# Posted by apennq, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-13, License - CC BY-SA 4.0

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from math import sqrt

def normalized_rmse(y_true, y_pred):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between true and predicted values.

    NRMSE is defined as the RMSE divided by the range of the true values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The NRMSE value.
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    range_y = max(y_true) - min(y_true)
    
    if range_y == 0:
        raise ValueError("The range of true values is zero, cannot compute NRMSE.")
    
    nrmse = rmse / range_y
    return nrmse
# %%
import numpy as np
import xarray as xr

def _weights_like(da: xr.DataArray, lat_name="lat") -> xr.DataArray:
    w = np.cos(np.deg2rad(da[lat_name]))
    # broadcast to (lat, lon)
    return w / w.mean()

def _mask_pair(x: xr.DataArray, y: xr.DataArray) -> xr.Dataset:
    ds = xr.Dataset({"x": x, "y": y})
    return ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]))

def wmean(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    return (da * w).sum(dim=("lat","lon")) / w.sum(dim=("lat","lon"))

def wstd(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    mu = wmean(da, w)
    var = wmean((da - mu)**2, w)
    return np.sqrt(var)

def pattern_corr(x: xr.DataArray, y: xr.DataArray, centered=True) -> float:
    ds = _mask_pair(x, y)
    x2, y2 = ds["x"], ds["y"]

    w = _weights_like(x2)

    if centered:
        x2 = x2 - wmean(x2, w)
        y2 = y2 - wmean(y2, w)

    cov = wmean(x2 * y2, w)
    sx  = wstd(x2, w)
    sy  = wstd(y2, w)

    r = cov / (sx * sy)
    return float(r)

def rmsd(x: xr.DataArray, y: xr.DataArray) -> float:
    ds = _mask_pair(x, y)
    d = ds["x"] - ds["y"]
    w = _weights_like(d)
    return float(np.sqrt(wmean(d**2, w)))

def nrmse(x: xr.DataArray, y: xr.DataArray, ref="y") -> float:
    """NRMSE = RMSD / std(reference)"""
    ds = _mask_pair(x, y)
    w = _weights_like(ds["x"])
    r = rmsd(ds["x"], ds["y"])
    s = wstd(ds[ref], w)
    return float(r / s)
# %%