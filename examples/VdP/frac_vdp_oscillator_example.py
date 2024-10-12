import numpy as np
from fracnum.splines import BernsteinSplines
from fracnum.plotting_utils import VdP_Plotter

"""
Example file of using Bernstein splines for time-integrating the (forced) fractional Van der Pol oscillator in the sense:

x'' - mu * (1 - x^2) * D^alpha x + x = A * sin(omega * t)

Where D represents the Caputo derivative of order alpha. Enjoy the chaos!

- Niels Goedegebure, 11 October 2024

"""

###################
# Spline settings #
###################

T = np.pi*40            # Integration max time
dt = 0.05               # Spline size (also called h) though varying size can also be used by creating a custom t_knot_vals
N_knot_int = int(T/dt)  # Number of knots
n_eval = 1              # Polynomial order

####
# A bit of a technicality: the order of the polynomial goes up by 3 since it gets multiplied 3 times in the VdP system equation.
# In general equal to m * n_eval, where m denotes the highest order of multiplication in x-components of f(x)
n_calc = 3 * n_eval


t_knot_vals = np.linspace(0, T, N_knot_int+1)  # Initialize equidistant knot values [t_0, t_1, t_2 ... t_k]. Can be generalized to anything.

bs = BernsteinSplines(t_knot_vals, n_calc, n_eval=n_eval)   # Initialize splines!

####################
# System function  #
####################

mult = bs.splines_multiply_matrix   # Bernstein multiplication method
upscale = bs.splines_upscale        # Bernstein upscale method to match the polynomial order (same as multiplying n times with identity splines of the same order) # noqa


def f(t_vals, x_vals, params, bernstein=False):
    mu = params['mu']
    if bernstein:
        A_x_in, A_y_in = x_vals

        # x' = y
        A_x_out = upscale(A_y_in, 3)

        # y' = mu * (1-x^2)y - x
        A_y_out = mu * mult(1 - mult(A_x_in, A_x_in), A_y_in) - upscale(A_x_in, 3)

        return np.array([A_x_out, A_y_out])
    else:   # The non-Bernstein spline one is added for quicker comparison options with e.g. Diethelm's method (see fracnum.numerical)
        x, y = x_vals[:, 0], x_vals[:, 1]

        # x' = y
        x_out = y
        # y' = mu * (1-x^2)y - x
        y_out = -x - mu * (x**2 - 1)*y

        return np.array([x_out, y_out]).T

####################
# Parameter Inputs #
####################


# Initial value (x, y).
# NOTE: for now, keep y to zero so that the derivative calculation still applies without need for a constant
x_0 = np.array([2, 0])


alpha = 0.9     # Fractional order of damping


params = {
    'mu': 1     # mu parameter of VdP oscillator
}

forcing_params = {
    'A': 3,        # Forcing amplitude,
    'omega': 5,    # Forcing frequency,
    'dim': 1       # Denotes the dimension the forcing gets applied to. In this case the "y" coordinate, hence 1
}

############
# Compute! #
############

system_alpha_vals = [alpha, 2-alpha]    # alpha for x and y component separately. For reference, see https://doi.org/10.1016/j.chaos.2006.05.010

# MAIN EXECUTION!:
results = bs.solve_ivp_local(
    f,
    x_0,
    system_alpha_vals,
    T,
    f_params=params,
    sin_forcing_params=forcing_params,
    verbose=False,
    tol=1e-8
)

# Save some results...
x = results['x']                # Function at evaluation points as vector
a_vals = results['a']           # Splines coefficients in matrix form
comp_time = results['time']     # Total computational time
t = results['t']                # Knot time values

x_der = bs.ddt(a_vals[0, :, :])     # Calculate the derivative of x

############
# Plotting #
############

plot_x = x[:-1, 0]
plot_x_der = x_der[:-1]
plot_t = t[:-1]

plot_object = VdP_Plotter(plot_x, plot_x_der, plot_t, params, alpha, dt, T, n_eval, comp_time, forcing_params=forcing_params)

plot_object.phase()
plot_object.signal()
plot_object.threedee()
plot_object.fourier_spectrum()
plot_object.show_plots()

############
# Enjoy!:) #
############
