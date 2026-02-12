#!/usr/bin/env python3
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.util as cutil

import src.plot_func as plot_func
# import palettable & colors if you need that colormap
import matplotlib.colors as mcolors
import palettable.cmocean.diverging as cmocean_div
# %%
dir_trend = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/CESM2/SMILE_forced"

model = "CESM2"  # choose one model to plot; or loop over models
ds_trend = xr.open_dataset(
    f"{dir_trend}/{model}_ENSmean_forced_MK_trend_1950-2022_sliding.nc"
)

trend_all = ds_trend["trend"]    # dims: period, lat, lon
pval_all  = ds_trend["p_value"] # dims: period, lat, lon

lat = trend_all["lat"].values
lon = trend_all["lon"].values
periods = trend_all["period"].values

levels = np.arange(-0.5, 0.55, 0.05)
extend = "both"
cmap = mcolors.ListedColormap(cmocean_div.Balance_20.mpl_colors)

num_plots_per_page = 4
num_subplots_x = 2
num_subplots_y = 2
figsize_x = 20
figsize_y = 12

with PdfPages(f"./{model}_sliding_MK_trends.pdf") as pdf:
    for start_page in range(0, len(periods), num_plots_per_page):
        fig, axes = plt.subplots(
            num_subplots_y,
            num_subplots_x,
            figsize=(figsize_x, figsize_y),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=180)},
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        contour_obj = None

        for i in range(num_plots_per_page):
            idx = start_page + i
            if idx >= len(periods):
                break

            interval = periods[idx]
            data = trend_all.sel(period=interval)
            pvals = pval_all.sel(period=interval)

            iy = i // num_subplots_x
            ix = i % num_subplots_x
            ax = axes[iy, ix]

            data_cyc, lon_cyc = cutil.add_cyclic_point(data.values, coord=lon)
            pval_cyc, _       = cutil.add_cyclic_point(pvals.values, coord=lon)

            contour_obj = plot_func.plot_data_with_significance(
                data_cyc,
                lat,
                lon_cyc,
                pval_cyc,
                levels=levels,
                extend=extend,
                cmap=cmap,
                title=" ",
                ax=ax,
                show_xticks=False,
                show_yticks=False,
            )
            ax.set_title(f"Trend for {interval}", fontsize=18)

        # Colorbar shared per page
        cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
        cbar = plt.colorbar(contour_obj, cax=cbar_ax, orientation="horizontal", extend=extend)
        cbar.set_label("Annual SAT trend (°C per decade)", fontsize=16)

        pdf.savefig(fig)
        plt.close(fig)
# %%