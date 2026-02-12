# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %%
class TaylorDiagram(object):
    """
    Taylor diagram.

    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                  fig=None, rect=111, label='_', srange=(0, 1.5), extend=False,
                  xlabel='Obs'):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters:

        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        
        # Adjust axes
        ax.axis["right"].major_ticklabels.set_fontsize(18)
        ax.axis["left"].major_ticklabels.set_fontsize(18)
        ax.axis["top"].major_ticklabels.set_fontsize(18)
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        
        ax.axis["right"].label.set_fontsize(22)
        ax.axis["right"].label.set_weight('bold')
        ax.axis["left"].label.set_fontsize(22)
        ax.axis["left"].label.set_weight('bold')
        ax.axis["top"].label.set_fontsize(22)
        ax.axis["top"].label.set_weight('bold')
        ax.axis["left"].label.set_color('red') 
        ax.axis["right"].label.set_color('k') 
        ax.axis["top"].label.set_color('b') 
        
        ax.axis["left"].label.set_text("Observation")
        ax.axis["right"].label.set_text("Standard deviation")
        ax.axis["top"].label.set_text("Correlation")
        
        ax.axis["left"].line.set_linewidth(2)
        ax.axis["top"].line.set_linewidth(2)
        ax.axis["right"].line.set_linewidth(2)
        
        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'r*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'r--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]
        
    def draw_outer_border(self, ax):
        """
        Draw an outer border around the polar plot.
        """
        # Create a circle patch
        outer_border = patches.Circle(
            (0, 0), self.smax,
            transform=ax.transData._b,  # Use the polar transformation
            color='black', linewidth=2, fill=False, linestyle='--'
        )
        
        # Add the patch to the plot
        ax.add_patch(outer_border)
        
    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)
        
        return l

    def add_grid(self, *args, **kwargs):
        grid = self._ax.grid(*args, **kwargs)
        """Add a grid."""
        rlocs = [0,0.04,0.08,0.12,0.16,0.20,0.24,0.28,]  
        for r in rlocs:
            print(r)
            circle = plt.Circle((0, 0), r, transform=self._ax.transData._b, color='k',
                                fill=False, linestyle='-',linewidth=1.5,zorder=2)
            self._ax.add_patch(circle)
        
        
        return grid

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                              np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels,colors='green', linestyles='--', **kwargs)

        return contours
# %%