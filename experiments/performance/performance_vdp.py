import numpy as np
import fracnum as fr
import copy

from fracnum.splines import BernsteinSplines
from fracnum.plotting_utils import VdP_Plotter
import matplotlib.pyplot as plt
from fracnum.numerical import build_hilf_knot_vals

###
# Performance parameters:
###

FORCE_NUM = True # Add forcing numerically each step? False => precompute
EQ_OPT = True    # Equidistant grid optimization (for Caputo (beta = 1))
# For cupy yes / no see environment.env

######################
# VdP function setup #
######################

alpha_damping = 0.5     # Fractional order of damping
beta = 1               # Hilfer Beta

params = {
    'mu': 1,     # mu parameter of VdP oscillator
}

# Forcing is a list of dictionaries applying A*sin(omega*t) + c to solution component dim (1 gives y in this case)
forcing_params = [{
    'dim' : 1,
    'A' : 3, #3
    'omega' : 3.3,#1.94,#4.0,#6.2,
    'c' : 0
}]

if FORCE_NUM:
    params['A'] = copy.copy(forcing_params[0]['A'])
    params['omega'] = copy.copy(forcing_params[0]['omega'])
    forcing_params[0]['A'] = 0

# Initialize the function for the Van der Pol oscillator
# This has to be done before the next steps in order to get the right N_upscale. See below.
VdP_bs = fr.ode_functions.VdP(params, bernstein = True, transpose = False) 

###################
# Spline settings #
###################

### Knot input values ###
T = 1500                  # Integration max time #1500 for profiling
n_eval = 1              # Polynomial order
##
eps = 10**(-15)         # Time shift epsilon, start of interval
dt = 0.002               # Spline size (also called h) though varying size can also be used by creating a custom t_knot_vals
c = 3/2                 # Knot size increase constant
##

gamm = alpha_damping + beta - alpha_damping*beta # Hilfer gamma parametrization

t_knot_vals = build_hilf_knot_vals(eps, T, c, gamm, dt) # np.linspace(0, T, N_knot_int+1)  # Initialize equidistant knot values [t_0, t_1, t_2 ... t_k]. Can be generalized to anything.

####

# A bit of a technicality: the order of the polynomial goes up by 3 since it gets multiplied 3 times in the VdP system equation.
# In general equal to m * n_eval, where m denotes the highest order of multiplication in x-components of f(x)
n_calc = VdP_bs.N_upscale * n_eval

#################
#################
#################

# alpha for x and y component separately. For reference, see https://doi.org/10.1016/j.chaos.2006.05.010
system_alpha_vals = [alpha_damping, 2-alpha_damping]    

bs = BernsteinSplines(t_knot_vals, n_calc, n_eval=n_eval, alpha_init=system_alpha_vals[0], eq_opt = EQ_OPT)  # Initialize splines setup!

# B_matrix = bs.B_I[system_alpha_vals[0]]
# B_sel = B_matrix[:, :, -1, :]
# B_size = getsizeof(B_sel)/(10**9)
# print(f"B size: {B_size:.3f} GB")

# B_full = B_matrix.mean(axis = (3))

# plt.plot(B_full[:, 0, -1])
# plt.plot(B_full[:, 0, -1])

# plt.matshow(B_matrix.mean(axis = (1,3)))
# plt.show()
# breakpoint()

#################
#################
#################

mult = bs.splines_multiply   # Bernstein multiplication method
upscale = bs.splines_upscale # Bernstein upscale method to match the polynomial order (same as multiplying n times with identity splines of the same order)

VdP_bs.set_bs_mult_upscale_functions(mult, upscale)  

############
# Compute! #
############

# Initial value (x, y).
x_0 = np.array([0, 0])

# Initialize the solving structure
spline_solver = bs.initialize_solver(VdP_bs.f, x_0, system_alpha_vals, beta_vals = beta, forcing_params = forcing_params) 
# Run the solver. method = 'local' uses knot-by-knot integration, method = 'global' at-large interval integration.
results = spline_solver.run(method='local', verbose = False)

print(f"average {results['n_it_per_knot']} it. / knot")

PLOTTING = False
if PLOTTING:
    ## BELOW THIS: ONLY FOR PLOTS

    # Save some results...
    x = results['x'][0]                   # Function at evaluation points as vector
    a_vals = results['a']                 # Splines coefficients in matrix form
    comp_time = results['total_time']     # Total computational time
    t = results['t']                      # Knot time values

    x_der = np.diff(x)/np.diff(t) # Calculate the derivative of x

    ############
    # Plotting #
    ############

    skip_n_vals = 20 # Skip first n vals in case of Hilfer derivative for plotting
    if np.all(beta == 1) :
        skip_n_vals = 0

    plot_x = x[skip_n_vals:-1] # Last value is discarded for the derivative calculation
    plot_x_der = x_der[skip_n_vals:]
    plot_t = t[skip_n_vals:-1]

    # If provided, can be used to keep fixed limits for the frame. Currently not inputted below.
    lims_override = {
        'x': [-2.0, 2.0],
        'xder': [-5.0, 5.0],
        'fourier_amp': [0, 2.8]
    }

    plot_object = VdP_Plotter(
        plot_x, 
        plot_x_der, 
        plot_t, 
        params, 
        alpha_damping, 
        dt, 
        T, 
        n_eval, 
        comp_time, 
        forcing_params=forcing_params, 
        lims_override = None,
        beta = beta
    )

    plot_object.phase()
    plot_object.phase(empty = True)
    plot_object.threedee()
    plot_object.fourier_spectrum()
    plot_object.phase_fourier()

    plot_object.show_plots()

    ############
    # Enjoy!:) #
    ############