import numpy as np
import xarray as xr


def _prep_xy(
    x: xr.DataArray,
    y: xr.DataArray,
    dims=("lat", "lon"),
):
    """Align, mask NaNs consistently, and return (x, y)."""
    x, y = xr.align(x, y, join="inner")
    ds = xr.Dataset({"x": x, "y": y})
    ds = ds.where(np.isfinite(ds["x"]) & np.isfinite(ds["y"]), drop=True)
    if ds["x"].size == 0:
        return None, None
    return ds["x"], ds["y"]


def _weights_coslat(
    lat: xr.DataArray,
    like: xr.DataArray,
):
    """
    cos(lat) weights broadcast to field.
    Returns weights with same dims as `like`.
    """
    w = np.cos(np.deg2rad(lat))
    # Broadcast to x(lat,lon) (or any field with lat dim)
    w2 = w.broadcast_like(like)
    return w2


def _wmean(a: xr.DataArray, w: xr.DataArray, dims=("lat", "lon")) -> xr.DataArray:
    wsum = w.sum(dim=dims)
    return (a * w).sum(dim=dims) / wsum


def _wvar(a: xr.DataArray, w: xr.DataArray, dims=("lat", "lon")) -> xr.DataArray:
    am = _wmean(a, w, dims=dims)
    wsum = w.sum(dim=dims)
    return (w * (a - am) ** 2).sum(dim=dims) / wsum


def _wcov(a: xr.DataArray, b: xr.DataArray, w: xr.DataArray, dims=("lat", "lon")) -> xr.DataArray:
    am = _wmean(a, w, dims=dims)
    bm = _wmean(b, w, dims=dims)
    wsum = w.sum(dim=dims)
    return (w * (a - am) * (b - bm)).sum(dim=dims) / wsum


def pattern_correlation(
    x: xr.DataArray,
    y: xr.DataArray,
    lat: xr.DataArray,
    dims=("lat", "lon"),
    use_weights: bool = True,
    centered: bool = True,
) -> float:
    """
    Area-weighted (coslat) spatial pattern correlation between 2D fields.

    Parameters
    ----------
    centered : bool
        If True, remove weighted spatial mean before computing correlation
        (this is the usual "centered pattern correlation").
        If False, computes correlation without removing the mean (less common).
    """
    x_, y_ = _prep_xy(x, y, dims=dims)
    if x_ is None:
        return float("nan")

    if use_weights:
        w = _weights_coslat(lat, x_)
    else:
        w = xr.ones_like(x_)

    if centered:
        cov = _wcov(x_, y_, w, dims=dims)
        vx = _wvar(x_, w, dims=dims)
        vy = _wvar(y_, w, dims=dims)
    else:
        # uncentered correlation: cov = E[xy] - E[x]E[y] (still subtracts mean implicitly);
        # to truly do "uncentered", use E[xy]/sqrt(E[x^2]E[y^2]).
        wsum = w.sum(dim=dims)
        Exy = (w * x_ * y_).sum(dim=dims) / wsum
        Ex2 = (w * x_ * x_).sum(dim=dims) / wsum
        Ey2 = (w * y_ * y_).sum(dim=dims) / wsum
        denom = np.sqrt(Ex2 * Ey2)
        return float((Exy / denom).values) if np.isfinite(denom) else float("nan")

    denom = np.sqrt(vx * vy)
    if not np.isfinite(denom):
        return float("nan")

    return float((cov / denom).values)


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
    x_, y_ = _prep_xy(x, y, dims=dims)
    if x_ is None:
        return float("nan")

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


# -------------------------
# Example usage (2D fields)
# -------------------------
# r = pattern_correlation(obs_map, sim_map, lat=obs_map["lat"], centered=True, use_weights=True)
# rmse = pattern_rmse(obs_map, sim_map, lat=obs_map["lat"], centered=False, use_weights=True)
# crmse = pattern_rmse(obs_map, sim_map, lat=obs_map["lat"], centered=True, use_weights=True)
# bias = float((_wmean(sim_map, _weights_coslat(obs_map["lat"], obs_map), dims=("lat","lon"))
#              - _wmean(obs_map, _weights_coslat(obs_map["lat"], obs_map), dims=("lat","lon"))).values)
