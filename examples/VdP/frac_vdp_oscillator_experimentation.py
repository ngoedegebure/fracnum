import numpy as np
import time
import fracnum as fr
from fracnum.splines import BernsteinSplines
from fracnum.experimentation_handler import ExperimentationHandler

# Initialize the function for the Van der Pol oscillator
# This has to be done before the next steps in order to get the right N_upscale, see below
VdP_bs = fr.ode_functions.VdP(bernstein = True, transpose = False) 

###################
# Spline settings #
###################

T = 50                 # Integration max time
dt = 0.05               # Spline size (also called h) though varying size can also be used by creating a custom t_knot_vals
N_knot_int = int(T/dt)  # Number of knots
n_eval = 1              # Polynomial order. NOTE: when taking the spline derivative as done below, it is beneficial to keep this at 1 for plotting / spline cont. diff reasons

####
# A bit of a technicality: the order of the polynomial goes up by 3 since it gets multiplied 3 times in the VdP system equation.
# In general equal to m * n_eval, where m denotes the highest order of multiplication in x-components of f(x)
n_calc = VdP_bs.N_upscale * n_eval

t_knot_vals = np.linspace(0, T, N_knot_int+1)  # Initialize equidistant knot values [t_0, t_1, t_2 ... t_k]. Can be generalized to anything.

bs = BernsteinSplines(t_knot_vals, n_calc, n_eval=n_eval)  # Initialize splines setup!

mult = bs.splines_multiply   # Bernstein multiplication method
upscale = bs.splines_upscale # Bernstein upscale method to match the polynomial order (same as multiplying n times with identity splines of the same order)

VdP_bs.set_bs_mult_upscale_functions(mult, upscale)  

# Initial value (x, y).
# NOTE: for now, keep y to zero so that the derivative calculation still applies without need for a constant
x_0 = np.array([2, 0])

function_params = {
    'mu': 2     # mu parameter of VdP oscillator
}

# Forcing is a list of dictionaries applying A*sin(omega*t) + c to solution component dim (1 gives y in this case)
default_forcing_params = [{
    'dim' : 1,
    'A' : 3,
    'c' : 0
}]

# Here the logic of experimentation varying parameters is given
vary_params = {
    'alpha' : {
        'bounds': [1, 0.9], # Bounds and step size determine linear values through linspace ...
        'step_size': 0.1,   # ... can also be overrided by providing 'values' as key
        'type':'alpha'      # 'type' can be one of: 'alpha' / 'param' / 'x_0' / 'forcing'
    },
    'omega' : {
        'bounds': [1, 5],
        'step_size': 1,
        'type':'forcing',
        'forcing_index': 0  # Needs to be provided in forcing, refers to n'th element of default forcing params
    }                       # TODO: Perhaps change this structure, also for forcing in general
}

# Settings for storing results timeseries. For now, xarray dataset as .nc file is supported
storage_settings = {
    # NOTE! No file storage if no path provided, although the dataset will still be returned
    'path': f'results/{int(time.time())}_{"_".join(vary_params.keys())}.nc', 
    'save_ts': [0] ,    # Result dimensions of which to save timeseries
    'save_ts_der': [0], # Result  dimensions of which to save finite-differenced ts
    'autosave': True    # Saves to file on every run if True
}

# A lambda function mapping the scalar alpha to an alpha for each dim
# When not provided, alpha is taken equal for all dims
alpha_fun = lambda alpha: [alpha, 2-alpha] 

#############################
# Initialize the experiment #
#############################
exp = ExperimentationHandler(bs, VdP_bs, vary_params, alpha_fun = alpha_fun)
# Set default values as specified above
exp.set_defaults(x_0 = x_0, params = function_params, forcing_params = default_forcing_params)
# Run!
ds = exp.run_experiment(storage_settings = storage_settings)

print(ds)