#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taylor diagram plot (normalized) for forced/internal components across 7 LEs
using the "FloatingSubplot" Taylor layout (matches common published style).

Inputs (per model file):
  - corr_run_vs_ens(run, period)
  - std_run(run, period)
  - crmse_run_vs_ens(run, period)
  - std_ref(period)
  - optional obs: corr_obs_vs_ens(period), std_obs(period), crmse_obs_vs_ens(period)

Options:
  - normalized: plot std/std_ref and crmse/std_ref (refstd plotted at 1.0)
  - corr_min: zoom the wedge (e.g., 0.5..1.0)
  - marker notation:
      hollow circle : members vs ENS
      filled circle : model mean member
      filled square : OBS vs ENS

Outputs:
  - screen display + PNG saved

Author: (polished for your workflow)
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf

# %% 
# ---------------------------------------------------------------------
# Model label + colors (keep insertion order for legend)
# ---------------------------------------------------------------------
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

RGB_dict = {
    "CanESM5(50)": "#A60E16",
    "CESM2(100)": "#EE3B2A",
    "IPSL-CM6A-LR(32)": "#FC9171",
    "EC-Earth3(21)": "#F5BFA2",
    "ACCESS-ESM1.5(40)": "#5FB7B5",
    "MPI-ESM1.2-LR(50)": "#7AB0DF",
    "MIROC6(50)": "#0F55C5",
    "MMLE": "black",
}

category_style = {
    "member":    dict(marker="o", mfc="none", mec="k", mew=0.9, ms=10, alpha=0.85, linestyle=""),
    "modelmean": dict(marker="o", mfc="k",    mec="k", mew=0.7, ms=15, alpha=0.95, linestyle=""),
    "obs":       dict(marker="s", mfc="k",    mec="k", mew=1.0, ms=17.5, alpha=0.95, linestyle=""),
}


def model_from_file(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def model_color_and_label(model_id: str):
    label = file_id_to_label.get(model_id, model_id)
    color = RGB_dict.get(label, "0.5")
    return color, label

# %%
# ---------------------------------------------------------------------
# Load points for a given file + period
# ---------------------------------------------------------------------
def load_points(nc_path: str, period: str):
    ds = xr.open_dataset(nc_path)
    d = ds.sel(period=period)

    corr  = d["corr_run_vs_ens"].values
    std   = d["std_run"].values
    crmse = d["crmse_run_vs_ens"].values
    refstd = float(d["std_ref"].values)

    obs = None
    if "corr_obs_vs_ens" in d.data_vars:
        obs = dict(
            corr=float(d["corr_obs_vs_ens"].values),
            std=float(d["std_obs"].values),
            crmse=float(d["crmse_obs_vs_ens"].values),
        )
    ds.close()
    return corr, std, crmse, refstd, obs
# %%
# ---------------------------------------------------------------------
# FloatingSubplot Taylor diagram class (published-style axes)
# ---------------------------------------------------------------------
class TaylorDiagramFloating:
    def __init__(self, fig, rect, refstd,
                 corr_min=0.0, std_max=None,
                 tick_fs=14,
                 label="Reference"):
        """
        Parameters
        ----------
        refstd : float
            Reference standard deviation (1.0 if normalized).
        corr_min : float
            Minimum correlation shown (e.g. 0.5).
        std_max : float
            Max radius (std axis upper limit).
        """
        self.refstd = float(refstd)

        tr = PolarAxes.PolarTransform()

        # Correlation tick locations
        if corr_min >= 0.5:
            rlocs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
        elif corr_min >= 0.0:
            rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
        else:
            rlocs = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])

        tlocs = np.arccos(rlocs)  # theta locations
        gl1 = gf.FixedLocator(tlocs)
        tf1 = gf.DictFormatter(dict(zip(tlocs, [f"{r:g}" for r in rlocs])))

        # Axis extents
        smin = 0.0
        if std_max is None:
            std_max = 1.6 * self.refstd
        smax = float(std_max)

        # Wedge limits: theta from 0 to arccos(corr_min)
        theta_max = float(np.arccos(corr_min))
        gh = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, theta_max, smin, smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)

        # --- Axis styling to match sample figure ---
        # Correlation axis on outer arc
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Pattern correlation")
        ax.axis["top"].label.set_fontsize(tick_fs + 2)
        ax.axis["top"].major_ticklabels.set_fontsize(tick_fs)

        # Left (std) axis
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].toggle(ticklabels=True, label=True)
        ax.axis["left"].major_ticklabels.set_axis_direction("bottom")
        ax.axis["left"].label.set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Normalized standard deviation (°C/decade)")
        ax.axis["left"].label.set_fontsize(tick_fs + 2)
        ax.axis["left"].major_ticklabels.set_fontsize(tick_fs)

        # Right (std) axis (optional, looks like many published diagrams)
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True, label=False)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].major_ticklabels.set_fontsize(tick_fs)

        # Bottom axis hidden
        ax.axis["bottom"].set_visible(False)

        # Grid
        ax.grid(True, linewidth=1.0, alpha=0.7)

        # Aux polar axes where we plot points
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Bold wedge outline (approx: bold outer arc via spine-like line)
        # We'll draw an outer arc manually:
        th = np.linspace(0, theta_max, 400)
        rr = np.zeros_like(th) + smax
        self.ax.plot(th, rr, color="k", lw=2.2, zorder=5)

        # Reference point + dashed reference std arc
        self.samplePoints = []
        lref, = self.ax.plot([0], [self.refstd], "k*", ms=12, label=label, zorder=30)
        self.samplePoints.append(lref)

        t = np.linspace(0, theta_max, 300)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, "k--", lw=1.5, alpha=0.9, zorder=10)

        # Remember limits for contours
        self.theta_max = theta_max
        self.smin = smin
        self.smax = smax

    def add_sample(self, std, corr, *args, **kwargs):
        """Add a point at (std, corr)."""
        th = np.arccos(np.clip(corr, -1, 1))
        l, = self.ax.plot(th, std, *args, **kwargs)
        self.samplePoints.append(l)
        return l

    def add_contours(self, levels=9, **kwargs):
        """Centered RMSE contours."""
        rs = np.linspace(self.smin, self.smax, 450)
        ts = np.linspace(0, self.theta_max, 450)
        T, R = np.meshgrid(ts, rs)
        Corr = np.cos(T)
        E = np.sqrt(self.refstd**2 + R**2 - 2*self.refstd*R*Corr)

        if isinstance(levels, int):
            emax = float(np.nanmax(E))
            # levels = np.linspace(0, emax, levels + 1)[1:]  # drop 0
            # Use equal spacing with interval of 0.2
            levels = np.arange(0.2, emax + 0.2, 0.2)

        contours = self.ax.contour(T, R, E, levels=levels, **kwargs)
        return contours
# %%
# ---------------------------------------------------------------------
# Panel plotter
# ---------------------------------------------------------------------
def plot_panel(fig, rect, files, period, title,
               corr_min=0.5, normalized=True, tick_fs=14,
               contour_levels=9):

    # Filter files that have the period
    good_files = []
    model_ids = []
    for f in files:
        try:
            ds = xr.open_dataset(f)
            ok = period in ds["period"].values
            ds.close()
            if ok:
                good_files.append(f)
                model_ids.append(model_from_file(f))
        except Exception as e:
            print(f"[WARN] cannot read {f}: {e}")

    if len(good_files) == 0:
        ax = fig.add_subplot(rect)
        ax.axis("off")
        ax.set_title(title + "\n(no matching files / period missing)")
        return None

    # Compute std_max for the panel
    std_all = []
    for f in good_files:
        corr, std, crmse, refstd, obs = load_points(f, period)
        if normalized:
            std_all.append(np.nanmax(std / refstd))
            if obs is not None:
                std_all.append(obs["std"] / refstd)
        else:
            std_all.append(np.nanmax(std))
            if obs is not None:
                std_all.append(obs["std"])

    # std_max = 1.25 * float(np.nanmax(std_all))
    std_max = 2.0  # fixed for better comparison across panels

    # Create diagram
    refstd_plot = 1.0 if normalized else float(np.nanmean([load_points(f, period)[3] for f in good_files]))
    dia = TaylorDiagramFloating(
        fig=fig, rect=rect,
        refstd=refstd_plot,
        corr_min=corr_min,
        std_max=std_max,
        tick_fs=tick_fs,
        label="Reference"
    )

    # Contours (bold)
    cs = dia.add_contours(levels=contour_levels, colors="0.35", linewidths=2.2)
    plt.clabel(cs, inline=1, fontsize=tick_fs-1, fmt="%.2f")

    # Plot points
    for f in good_files:
        mid = model_from_file(f)
        mcolor, mlabel = model_color_and_label(mid)

        corr, std, crmse, refstd, obs = load_points(f, period)

        if normalized:
            std_n = std / refstd
            if obs is not None:
                obs_std = obs["std"] / refstd
        else:
            std_n = std
            if obs is not None:
                obs_std = obs["std"]

        # members
        st = category_style["member"].copy()
        st["mec"] = mcolor
        st["zorder"] = 5
        dia.ax.plot(np.arccos(np.clip(corr, -1, 1)), std_n, **st)

        # model mean member
        corr_m = float(np.nanmean(corr))
        std_m  = float(np.nanmean(std_n))
        st = category_style["modelmean"].copy()
        st["mfc"] = mcolor
        st["mec"] = "k"
        st["zorder"] = 7
        dia.add_sample(std_m, corr_m, **st)

        # OBS vs ENS
        if obs is not None:
            st = category_style["obs"].copy()
            st["mfc"] = mcolor
            st["mec"] = "k"
            st["zorder"] = 9
            dia.add_sample(obs_std, obs["corr"], **st)

    # Title
    dia._ax.set_title(title, fontsize=tick_fs+2, fontweight="bold", pad=18)

    return {model_color_and_label(m)[1] for m in model_ids}
# %%
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    period_A = "2013-2022"  # 10-YR
    period_B = "1993-2022"  # 30-YR
    period_C = "1963-2022"  # 60-YR
    forced_glob   = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/TaylorStats/*_forced_taylor_stats_1950-2022_sliding.nc"
    internal_glob = "/work/mh0033/m301036/OBS_LPS_revision/docs/data/FIG3/TaylorStats/*_internal_taylor_stats_1950-2022_sliding.nc"

    forced_files   = sorted(glob.glob(forced_glob))
    internal_files = sorted(glob.glob(internal_glob))
    # check the "corr_obs_vs_ens" stats in the internal files; print details
    for f in internal_files:
        ds = xr.open_dataset(f)
        if "corr_obs_vs_ens" in ds.data_vars:
            print(ds['corr_obs_vs_ens'].sel(period=period_A).values)
            print(ds['corr_obs_vs_ens'].sel(period=period_B).values)
            print(ds['corr_obs_vs_ens'].sel(period=period_C).values)
            print(f"[INFO] {os.path.basename(f)} has obs stats for internal variability")
        ds.close()
    # Large figure because FloatingSubplot consumes space
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.35)

    present_labels = set()

    present_labels |= plot_panel(fig, gs[0, 0], forced_files, period_A,
                                 title=f"Externally forced patterns ({period_A})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()

    present_labels |= plot_panel(fig, gs[0, 1], internal_files, period_A,
                                 title=f"Internal variability patterns ({period_A})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()

    present_labels |= plot_panel(fig, gs[1, 0], forced_files, period_B,
                                 title=f"Externally forced patterns ({period_B})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()

    present_labels |= plot_panel(fig, gs[1, 1], internal_files, period_B,
                                 title=f"Internal variability patterns ({period_B})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()
    # Add text label "A, B, C, D" to each panel
    present_labels |= plot_panel(fig, gs[2, 0], forced_files, period_C,
                                 title=f"Externally forced patterns ({period_C})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()

    present_labels |= plot_panel(fig, gs[2, 1], internal_files, period_C,
                                 title=f"Internal variability patterns ({period_C})",
                                 corr_min=0.0, normalized=True, tick_fs=14, contour_levels=9) or set()
    # Add text label "A, B, C, D" to each panel
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    panel_positions = [(0.11, 0.89), (0.53, 0.89), (0.11, 0.62), (0.53, 0.62), (0.11, 0.35), (0.53, 0.35)]
    for label, (x, y) in zip(panel_labels, panel_positions):
        fig.text(x, y, label, fontsize=24, fontweight='bold')   
    
    # add one unified legend for models at the bottom center
    type_handles = [
        plt.Line2D([0],[0], marker='o', linestyle='', markersize=12,
                   markerfacecolor='none', markeredgecolor='k',
                   label='Members vs ENS (hollow)'),
        plt.Line2D([0],[0], marker='o', linestyle='', markersize=12,
                   markerfacecolor='k', markeredgecolor='k',
                   label='Model member mean(filled)'),
        plt.Line2D([0],[0], marker='s', linestyle='', markersize=12,
                   markerfacecolor='k', markeredgecolor='k',
                   label='OBS vs ENS (square)'),
    ]

    model_handles = []
    for lbl in RGB_dict.keys():
        if lbl == "MMLE":
            continue
        if present_labels and lbl not in present_labels:
            continue

        model_handles.append(
            plt.Line2D([0], [0],
                    marker="o", linestyle="",
                    markerfacecolor=RGB_dict[lbl],
                    markeredgecolor="k",
                    markersize=12,
                    label=lbl)
        )

    leg_models = fig.legend(
    handles=model_handles, title="CMIP6 Models",
    loc="center left", bbox_to_anchor=(0.25, 0.05),
    frameon=False, fontsize=20, title_fontsize=22,
    ncol=3, columnspacing=1.2
    )
    fig.add_artist(leg_models)

    # Color each legend label to match the marker color
    for t in leg_models.get_texts():
        name = t.get_text()
        if name in RGB_dict:
            t.set_color(RGB_dict[name])


    leg_types = fig.legend(handles=type_handles, title="Marker notation",
               loc="center left", bbox_to_anchor=(0.2, 0.005),
               frameon=False, fontsize=20, title_fontsize=22,
               ncol=3, columnspacing=1.5)

    # fig.subplots_adjust(right=0.8)
    # plt.tight_layout(rect=[0, 0.0, 0.8, 1])

    out = f"TaylorDiagram_forced_internal_{period_A}_{period_B}_{period_C}_floating.png"
    # save pdf and png
    
    fig.savefig(out, dpi=300, bbox_inches="tight")
    out_pdf = out.replace(".png", ".pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[saved] {out}")
    print(f"[saved] {out_pdf}")
# %%
if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------
# %%
