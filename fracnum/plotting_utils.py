import warnings
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl

from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.collections import LineCollection

from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["lightseagreen", "orange", "darkmagenta", "dodgerblue"]) 

font_families = ['Segoe UI', 'Arial', 'Helvetica']
rc('font',**{'family':'sans-serif','sans-serif':font_families})
rc('font',**{'family':'cursive','cursive':font_families})
mpl.rcParams['mathtext.fontset'] = 'stixsans' #Possible fonts: ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

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

def get_lin_line_colors(x, cmap_name = 'magma_r', cmap_trunc = [0.2, 0.8]):
    N_x = len(x)
    cmap = mpl.colormaps[cmap_name]
    cmap = truncate_colormap(cmap, minval = cmap_trunc[0], maxval =  cmap_trunc[1])
    colors = cmap(np.linspace(0,1, N_x))

    return colors, cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(x), vmax=np.max(x)), cmap=cmap)

class VdP_Plotter():
    def __init__(self, x, xder, t, params, alpha, dt, T, n_eval, comp_time, forcing_params = None, lims_override = None, cmap_name="magma_r", cmap_trunc = [0.15, 0.75], beta = 1):
        self.x = x
        self.xder = xder
        self.t = t
        self.params = params
        self.alpha = alpha
        self.dt = dt
        self.T = T
        self.n_eval = n_eval
        self.comp_time = comp_time
        if forcing_params is not None:
            self.forcing_params = forcing_params[0] # Take just the first element in this case. Can be made more general
        else:
            self.forcing_params = None
        self.beta = beta
        self.cmap_name = cmap_name
        self.cmap_trunc = cmap_trunc

        self.hd_size = (16/2, 9/2)

        self.forcing = False
        self.title, self.param_desciption = self.generate_captions()

        if lims_override is not None:
            if 'x' in lims_override.keys() and 'xder' in lims_override.keys():
                self.x_lims = lims_override['x']
                self.xder_lims = lims_override['xder']
            if 'fourier_amp' in lims_override.keys():
                self.fourier_amp_lim = lims_override['fourier_amp']
            else:
                self.fourier_amp_lim = None
        else:
            self.x_lims = [min(self.x), max(self.x)]
            self.xder_lims = [min(self.xder), max(self.xder)]
            self.fourier_amp_lim = None

    def show_plots(self):
        plt.show()

    def close_plots(self):
        plt.close('all')

    def generate_captions(self):
        fractional_string = "fractionally "
        forcing_string = "unforced "
        damping_string = "damped "
        oss_type = "VdP "

        forcing_settings_string = ""
        if self.forcing_params is not None:
            if self.forcing_params['A'] != 0 and self.forcing_params['omega'] != 0 :
                self.forcing = True
                forcing_string = 'forced '
                forcing_settings_string = r", A="+str(self.forcing_params['A'])+r", \omega="+str(np.round(self.forcing_params['omega'],4))

        if self.alpha == 1:
            fractional_settings_string = ""
            fractional_string = ""
        else:
            fractional_settings_string = rf"\alpha={ self.alpha}, \beta = {self.beta},"
            
        if self.params['mu'] == 0:
            # If mu = 0, the damping alpha does not affect anything
            damping_string = ""
            oss_type = "harmonic "
            fractional_string = ""
            fractional_settings_string = ""

        title = f"{fractional_string}{damping_string}{forcing_string}{oss_type}Oscillator"
        # title = fractional_string+"dampened "+forcing_string+"VdP Oscillator"
        subtitle = "$" + fractional_settings_string + r"\mu="+str(self.params['mu'])+forcing_settings_string+r", T = "+str(np.round(self.T,2))+ r", h=" + str(np.round(self.dt,2)) + r", q = " + str(int(self.n_eval))+r"$"

        return title, subtitle

    def phase(self, save_filepath=None, hd_aspect = False, empty=False, add_text = None):
        if hd_aspect:
            fig_phase, ax_phase = plt.subplots(figsize = self.hd_size, layout="constrained")
        else:
            fig_phase, ax_phase = plt.subplots()

        lines = colored_line(self.x, self.xder, self.t, ax_phase,cmap_trunc = self.cmap_trunc, cmap=self.cmap_name, label=f"Bernstein Splines ({self.comp_time:.4f} s)", linewidth=2)

        if not empty:
            margin_pct_x, margin_pct_y = 0.1, 0.1

            ax_phase.set(xlabel = r"$x$", ylabel =r"$\dot{x}$")

            plt.suptitle("Phase portrait " + self.title)
            plt.title(self.param_desciption)
            fig_phase.colorbar(lines, label=r'$t$')
        else:
            margin_pct_x, margin_pct_y = 0.3, 0.07
            ax_phase.set_axis_off()

        x_dist = self.x_lims[1] - self.x_lims[0]
        margin_x = margin_pct_x * x_dist
        ax_phase.set_xlim(self.x_lims[0]-margin_x, self.x_lims[1]+margin_x)

        y_dist = self.xder_lims[1] - self.xder_lims[0]
        margin_y = margin_pct_y * y_dist
        ax_phase.set_ylim(self.xder_lims[0]-margin_y, self.xder_lims[1]+margin_y)

        if add_text is not None:
            label_text = ax_phase.text(0.88, 0.075, add_text, transform=ax_phase.transAxes)

        if save_filepath is not None:
            plt.savefig(save_filepath, dpi=400)

        return ax_phase

    def signal(self, save_filepath = None, hd_aspect = False):
        if hd_aspect:
            fig, axs = plt.subplots(2, figsize = self.hd_size)
        else:
            fig, axs = plt.subplots(2)

        axs[0].plot(self.t, self.x)
        axs[0].set(ylabel=r"$x$")

        pct_margin = 0.10 

        margin_x = (self.x_lims[1] - self.x_lims[0])*pct_margin
        axs[0].set_ylim([self.x_lims[0]-margin_x, self.x_lims[1]+margin_x])

        axs[1].plot(self.t, self.xder)
        axs[1].set(ylabel=r"$\dot{x}$")

        margin_y = (self.xder_lims[1] - self.xder_lims[0])*pct_margin
        axs[1].set_ylim([self.xder_lims[0]-margin_y, self.xder_lims[1]+margin_y])

        plt.xlabel('t')
        plt.suptitle(f"Signal of {self.title}\n" + self.param_desciption)

        if save_filepath is not None:
            plt.savefig(save_filepath, dpi=300)

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

        if self.fourier_amp_lim is not None:
            axs[0].set_ylim(self.fourier_amp_lim)
            axs[1].set_ylim(self.fourier_amp_lim)

        plt.xlabel(r"Freq domain $2\pi t$")
        plt.suptitle(f"Fourier spectrum of {self.title}\n" + self.param_desciption)

        if save_filepath is not None:
            plt.savefig(save_filepath, dpi = 300)

    def phase_fourier(self, save_filepath=None, x_signal = True, t_cutoff = 100):
        if not x_signal:
            grid = [
                ['phase', 'fourier_x'],
                ['phase', 'fourier_xder']
            ]
            title_string = "\nPhase portrait and fourier spectrum of " + self.title + "\n" + self.param_desciption
        else:
            grid = [
                ['phase', 'x_signal'],
                ['phase', 'fourier_x']
            ]
            title_string = "\nPhase portrait, signal and fourier spectrum of " + self.title + "\n" + self.param_desciption
        
        fig, axs = plt.subplot_mosaic(grid, figsize = self.hd_size, layout="constrained")
        plt.suptitle(title_string)

        # TODO: MAKE MODULAR!
        # PHASE #
        lines = colored_line(self.x, self.xder, self.t, axs["phase"],cmap_trunc = self.cmap_trunc, cmap=self.cmap_name, label=f"Bernstein Splines ({self.comp_time:.4f} s)", linewidth=2)

        margin_pct = 0.1

        axs["phase"].set(xlabel = r"$x$", ylabel =r"$\dot{x}$")
        # axs["phase"].colorbar(lines, label=r'$t$')

        x_dist = self.x_lims[1] - self.x_lims[0]
        margin_x = margin_pct * x_dist
        axs["phase"].set_xlim(self.x_lims[0]-margin_x, self.x_lims[1]+margin_x)

        y_dist = self.xder_lims[1] - self.xder_lims[0]
        margin_y = margin_pct * y_dist
        axs["phase"].set_ylim(self.xder_lims[0]-margin_y, self.xder_lims[1]+margin_y)

        # Fourier #

        f_x = fft(self.x)
        if not x_signal:
            f_xder = fft(self.xder)

        N = len(self.x)
        T_samplespacing = self.T/(N)*(1/(2*np.pi))
        # w = blackman(N)

        # ywf = fft(self.x*w)
        xf = fftfreq(N, T_samplespacing)[:N//2]

        select_div = 15

        if self.forcing:
            omega = self.forcing_params['omega']
            axs['fourier_x'].axvline(x = omega, color = 'red', linestyle = '--', linewidth = 1.5, label = 'Forcing')
            axs['fourier_x'].legend()
            if not x_signal:
                axs['fourier_xder'].axvline(x = omega, color = 'red', linestyle = '--', linewidth = 1.5)

        axs['fourier_x'].plot(xf[1:N//select_div], 2.0/N * np.abs(f_x[1:N//select_div]))
        axs['fourier_x'].set(ylabel=r"$x$ amplitude")

        if x_signal:
            axs['x_signal'].plot(self.t, self.x)
            axs['x_signal'].set(ylabel=r"$x$")
            axs['x_signal'].set_xlabel(r"$t$")
            axs['fourier_x'].set_xlabel(r"Frequency ( $2\pi / T$ )")
        else:
            axs['fourier_xder'].plot(xf[1:N//select_div], 2.0/N * np.abs(f_xder[1:N//select_div]))
            axs['fourier_xder'].set(ylabel=r"$\dot{x}$ amplitude")
            axs['fourier_xder'].set_xlabel(r"Frequency ( $2\pi / T$ )")

        if self.fourier_amp_lim is not None:
            axs['fourier_x'].set_ylim(self.fourier_amp_lim)
            if x_signal:
                axs['x_signal'].set_ylim(self.x_lims)
            else:
                axs['fourier_xder'].set_ylim(self.fourier_amp_lim)
            
        if x_signal:
            if t_cutoff is not None:
                margin, top_freq = axs['fourier_x'].get_xlim()
                max_freq = top_freq - np.abs(margin)

                t_max = min(np.max(self.t), t_cutoff)
                x_margin = np.abs(margin) / max_freq * t_max
                
                new_lims = (-x_margin, t_max+x_margin)
                axs['x_signal'].set_xlim(new_lims)

        margin_pct_padding = 0.02
        engine = fig.get_layout_engine()
        engine.set(rect=(margin_pct_padding,margin_pct_padding,1-margin_pct_padding*2,1-margin_pct_padding*2))

        if save_filepath is not None:
            plt.savefig(save_filepath, dpi = 300)