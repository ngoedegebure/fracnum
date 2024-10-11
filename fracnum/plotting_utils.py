import warnings
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.collections import LineCollection

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, cmap_trunc = [0,1], **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    map = lc.get_cmap()
    new_cmap = truncate_colormap(map, cmap_trunc[0], cmap_trunc[1], n=len(x))
    lc.set_cmap(new_cmap)

    # breakpoint()

    return ax.add_collection(lc)

def truncate_colormap(cmap, minval=0.2, maxval=0.8, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class VdP_Plotter():
    def __init__(self, x, xder, t, params, alpha, dt, T, n_eval, comp_time, forcing_params = None, cmap_name="magma_r", cmap_trunc = [0.15, 0.75]):
        self.x = x
        self.xder = xder
        self.t = t
        self.params = params 
        self.alpha = alpha
        self.dt = dt
        self.T = T
        self.n_eval = n_eval 
        self.comp_time = comp_time
        self.forcing_params = forcing_params

        self.cmap_name = cmap_name
        self.cmap_trunc = cmap_trunc

        self.hd_size = (16/2, 9/2)

        self.title, self.param_desciption = self.generate_captions()

    def show_plots(self):
        plt.show()

    def generate_captions(self):
        forcing_string = ""
        forcing_settings_string = ""
        if self.forcing_params is not None:
            if self.forcing_params['A'] != 0:
                forcing_string = 'forced '
                forcing_settings_string = r", A="+str(self.forcing_params['A'])+r", \omega="+str(np.round(self.forcing_params['omega'],4))

        if self.alpha == 1:
            fractional_settings_string = ""
            fractional_string = ""
        else:
            fractional_settings_string = r"\alpha="+str(self.alpha)+r", "
            fractional_string = "fractionally "

        title = fractional_string+"dampened "+forcing_string+"VdP Oscillator"
        subtitle = "$" + fractional_settings_string + r"\mu="+str(self.params['mu'])+forcing_settings_string+r". T = "+str(np.round(self.T,2))+ r", h=" + str(np.round(self.dt,2)) + r", q = " + str(int(self.n_eval))+r"$"

        return title, subtitle

    def phase(self, save_filepath=None, hd_aspect = False, empty=False):
        if hd_aspect:
            fig_phase, ax_phase = plt.subplots(figsize = self.hd_size)
        else:
            fig_phase, ax_phase = plt.subplots()

        lines = colored_line(self.x, self.xder, self.t, ax_phase,cmap_trunc = self.cmap_trunc, cmap=self.cmap_name, label=f"Bernstein Splines ({self.comp_time:.4f} s)", linewidth=2)

        if not empty:
            margin_pct = 0.1    

            ax_phase.set(xlabel = r"$x$", ylabel =r"$\dot{x}$")

            plt.suptitle("Phase portrait " + self.title)
            plt.title(self.param_desciption)
            fig_phase.colorbar(lines, label=r'$t$')
        else:
            margin_pct = 0.01
            ax_phase.set_axis_off()

        x_dist = max(self.x) - min(self.x)
        margin_x = margin_pct * x_dist
        ax_phase.set_xlim(min(self.x)-margin_x, max(self.x)+margin_x)

        y_dist = max(self.xder) - min(self.xder)
        margin_y = margin_pct * y_dist
        ax_phase.set_ylim(min(self.xder)-margin_y, max(self.xder)+margin_y)

        if save_filepath is not None:
            plt.savefig(save_filepath, dpi=300)

    def signal(self, save_filepath = None, hd_aspect = False):
        if hd_aspect:
            fig, axs = plt.subplots(2, figsize = self.hd_size)
        else:
            fig, axs = plt.subplots(2)

        axs[0].plot(self.t, self.x)
        axs[0].set(ylabel=r"$x$")

        axs[1].plot(self.t, self.xder)
        axs[1].set(ylabel=r"$\dot{x}$")
                
        plt.xlabel('t')
        plt.suptitle(f"Signal of {self.title}\n" + self.param_desciption)

        if save_filepath is not None:
            plt.savefig(save_filepath)

    def threedee(self, save_filepath=None, hd_aspect = False):
        fig = plt.figure()
        ax_3d = plt.subplot(projection='3d')
        # colored_lines = colored_line(self.x, self.xder, self.t, ax_3d,cmap_trunc = [0.15, 0.75], cmap='magma_r', label=f"Bernstein Splines ({self.comp_time:.4f} s)", linewidth=2)

        cmap = mpl.colormaps[self.cmap_name]
        N_x = len(self.x)
        cmap_tr = truncate_colormap(cmap, self.cmap_trunc[0], self.cmap_trunc[1], n=N_x)
        # breakpoint()

        for i in range(N_x-1):
            ax_3d.plot(self.x[i:(i+2)], self.xder[i:(i+2)], self.t[i:(i+2)], color = cmap_tr(i/N_x))
        ax_3d.set_xlabel(r'$x$')
        ax_3d.set_ylabel(r'$\dot{x}$')
        ax_3d.set_zlabel(r'$t$')

    def fourier_spectrum(self, save_filepath=None, hd_aspect = False):
        if hd_aspect:
            fig, axs = plt.subplots(2, figsize = self.hd_size)
        else:
            fig, axs = plt.subplots(2)

        f_x, f_xder = fft(self.x), fft(self.xder)

        N = len(self.x)
        T_samplespacing = self.T/(N)*(1/(2*np.pi))
        w = blackman(N)

        ywf = fft(self.x*w)
        xf = fftfreq(N, T_samplespacing)[:N//2]

        axs[0].plot(xf[1:N//10], 2.0/N * np.abs(f_x[1:N//10]))
        axs[0].set(ylabel=r"$x$ Fourier spectrum")

        axs[1].plot(xf[1:N//10], 2.0/N * np.abs(f_xder[1:N//10]))
        axs[1].set(ylabel=r"$\dot{x}$ Fourier spectrum")
                
        plt.xlabel(r"Freq domain $2\pi t$")
        plt.suptitle(f"Fourier spectrum of {self.title}\n" + self.param_desciption)

        if save_filepath is not None:
            plt.savefig(save_filepath)