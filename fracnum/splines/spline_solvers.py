from .static_spline_methods import SplineMethods
from fracnum.numerical import sin_I_a
import numpy as np
from scipy.special import gamma
import time
from tqdm import tqdm

class SplineSolver():
    # Solver method for boundary and initial value problems (BVP and IVP's)
    def __init__(self, bs, f, x_0, alpha_vals, forcing_parameters = {}):
        self.bs = bs        # Bernstein spline functions instance
        self.f = f          # System function f
        self.x_0 = x_0      # Initial value # TODO: maybe do this in running?
        self.d = len(x_0)   # Number of dimensions

        self.sin_forcing_storage = {} # Init. storage dictionary for forcing values

        # Get alpha for all dimensions
        self.alpha = SplineSolver.parse_alpha(alpha_vals, self.d) 
        # Build forcing values
        self.forcing_vals = self.build_forcing_values(forcing_parameters)
        # Initialize x storage values
        self.x_storage = None

    @staticmethod
    def parse_alpha(alpha_vals, d):
        if np.array(alpha_vals).size == 1:
            # If alpha is one-dimensional, turn it into d-dim. vector of alpha
            alpha_parsed = np.ones(d)*alpha_vals
        elif np.array(alpha_vals).size != d:
            # If not alpha and if not d, not unambiguous what to do
            print("ERROR! Either give one alpha or d (dimensions) alpha's!")
            assert 0
        elif np.array(alpha_vals).size == d:
            # If d values provided, all is fine!
            alpha_parsed = alpha_vals
        return alpha_parsed

    def build_forcing_values(self, forcing_parameters):
        # Initialize forcing values as 0 
        # Needs to be float in order to not create errors on addition later
        forcing_vals = [np.array([0], dtype='float64')] * self.d

        for forcing_element in forcing_parameters:
            # Initialize sin and c_vals as 0
            sin_vals, c_vals = 0,0
            
            dim = forcing_element['dim']    # The dimension to add forcing to
            alpha_f = self.alpha[dim]       # Alpha value

            # sin forcing: I^alpha{ A*sin(omega * t) }
            if 'A' in forcing_element.keys() and 'omega' in forcing_element.keys():
                A_f, omega_f = forcing_element['A'], forcing_element['omega']
                if A_f !=0 and omega_f != 0:
                    # If sin forcing enabled
                    sin_forcing_storage_key = (alpha_f, A_f, omega_f)
                    if sin_forcing_storage_key not in self.sin_forcing_storage.keys():
                        # Get sine values with sin_I_a method. Not possible to do vector-wise
                        # because of mpmath, hence the for-loop
                        sin_vals_vector = np.array([sin_I_a(t, alpha_f, omega_f) for t in self.bs.t_eval_vals_list])
                        sin_vals = A_f* SplineMethods.a_to_matrix(sin_vals_vector, self.bs.n_eval)
                    else:
                        # If stored, get from storage
                        sin_vals = self.sin_forcing_storage[sin_forcing_storage_key]
            # c (constant) forcing: I^alpha { c }
            if 'c' in forcing_element.keys():
                c_f = forcing_element['c']
                if c_f != 0:
                    # If constant forcing enabled
                    # The alpha-fractional integral of 1 is given by: t^alpha / gamma(alpha+1)
                    constant_int_vals = self.bs.t_eval_vals_list**(alpha_f)/gamma(alpha_f+1)
                    c_vals = SplineMethods.a_to_matrix(constant_int_vals, self.bs.n_eval) * c_f

            # Sum the two together
            forcing_vals[dim] = sin_vals + c_vals

        return forcing_vals
    
    def get_initial_x(self,save_x):
        # In this method, either an initial x is taken from storage or built from initial values
        # Mostly useful for the global solution method, where a good initial guess is important
        if self.x_storage is not None and save_x:
            # Get initial x from storage
            x = self.x_storage
        else:
            # Build as a constant function of the initial conditions
            N_tot = len(self.bs.t_eval_vals_list)
            # x_flat = np.ones([N_tot, self.d]) * self.x_0
            x_flat = self.bs.t_eval_vals_list*3/self.bs.t_eval_vals_list[-1]
            x = np.array([SplineMethods.a_to_matrix(self.x_0[i]+x_flat[:], self.bs.n_eval) for i in range(self.d)])

        return x

    def build_results_dict(self, x, n_tot_it, norm,it_norm, total_time, f_0, T = None, delta = None):
        # TODO: maybe change this to an object or xr / pandas?
        # For now: creates a dictionary of main results

        N_knots = x[0, :, :].shape[0] # SSOT: can be computed from x
        output_dict = {
            't': self.bs.t_eval_vals_list,
            # Below looks a bit confusing, but the logic is: a is the ordered coefficients and x the "flat" evaluation
            'x': np.array([SplineMethods.a_to_vector(x[i, :, :]) for i in range(self.d)]).T, #TODO: transpose? CHECK!
            'a': x, 
            'norm_type':norm,
            'norm_value':it_norm,
            'n_it':n_tot_it,
            'total_time':total_time,
            'time_per_it':total_time/n_tot_it,
            'n_it_per_knot':n_tot_it/N_knots,
            'time_per_knot':total_time/N_knots,
            'f_0':f_0,
            'T':T,
            'delta':delta
        }
        return output_dict
    
    @staticmethod
    def it_norm(x, x_prev, norm):
        # Computes the norm difference in the increments
        # Done in static method for standardization guarantee
        if norm == 'sup':
            it_norm = np.max(np.abs(x-x_prev))
        elif norm == 'L2':
            it_norm = np.linalg.norm(x-x_prev)
        return it_norm
    
    def get_delta(self, T, f_a_vals):
        # delta_T = _0I_T^alpha f(x) for selected T
        # Used for BVP approaches (or when a very accurate value is required at one T)
        # Note: x(T) = x_0 + delta_T
        delta = np.zeros(self.d)
        for k in range(self.d):
            delta[k] = self.bs.I_a_scalar(T, f_a_vals[k, :, :], self.alpha[k])
        return delta
    
    @staticmethod
    def print_summary(res):
        # Summary of selected results total
        print(f"~ {res['n_it']} iterations, {res['total_time']:.4f} s total, {res['time_per_it']:.6f} s/it")
        # Summary of selected results per knot
        print(f"~ {res['n_it_per_knot']:.2f} it/knot avg., {res['time_per_knot']:.6f} s/knot avg.")

    def iterate_local(self, x, conv_tol, conv_max_it, div_treshold, norm, verbose):
        ### "Local" iteration iterates knot-for knot ###

        N_knots = self.bs.t_eval_vals_ord.shape[0] # Number of knots
        n_eval = self.bs.t_eval_vals_ord.shape[1] # Spline polynomial evaluation of integral
        n_calc = self.bs.t_calc_vals_ord.shape[1] # Spline polynomial order calculations f

        f_a_vals_tot = np.zeros([self.d, N_knots, n_calc]) # Initialize the function value storage
        n_tot_it = 0 # Initialize total iteration counter

        for i_knot in tqdm(range(N_knots), desc='Iterating IVP for knots', disable = (not verbose)):
            # Initialize the integral values for this knot
            int_vals_base = np.zeros([self.d, n_eval])
            if i_knot > 0:
                f_no_last = f_a_vals_tot[:, 0:(i_knot+1), :] # Get f value for knots including current
                f_no_last[:, -1, :] = np.zeros(f_no_last[:, -1, :].shape) # Override current with zeros
                # Calculate the influence of previous knots on this knot
                # This is why the current knot is taken as 0 but still included:
                # ... since fractional integrals are not in general constant after their support
                for k in range(self.d):
                    # Int_vals_base keeps track of the influence of the integral of all previous knots
                    # Hence, will not change in the next steps
                    int_vals_base[k, :] = self.bs.I_a(f_no_last[k, :, :], alpha = self.alpha[k], knot_sel = ('to', i_knot))

                # Take the previous knot value as an initial guess for the to-be-calculated one
                x[:, i_knot, :] = x[:, i_knot-1, :]

            for conv_it in range(conv_max_it):
                ### Main iteration: converge the value for the to-be-calculated knot t_M ###

                x_prev = x.copy() # Saves previous estimate for convergence statistics

                # Compute function values f(t_M, x_M)
                f_a_vals_tot[:, i_knot:(i_knot+1), :] = self.f(self.bs.t_eval_vals_ord[i_knot, :], x[:, i_knot:(i_knot+1), :])

                ### Compute integration for each element ###
                for k in range(self.d):
                    # Get I^alpha { f(t_M, x_M) }
                    int_vals = self.bs.I_a(f_a_vals_tot[k, i_knot, :], alpha = self.alpha[k], knot_sel = ('at', i_knot))
                    # x_{M,j+1} = x_0 + _0I_t^alpha f(t_{M, j}, x_{M, j})
                    x[k, i_knot, :] = self.x_0[k] + int_vals_base[k, :] + int_vals
                    # Add forcing values if selected
                    if np.array(self.forcing_vals[k]).size > 1:
                        x[k, i_knot, :] += self.forcing_vals[k][i_knot, :]
                
                n_tot_it+=1 # Track iteration

                # Compute norm change
                it_norm = SplineSolver.it_norm(x, x_prev, norm)

                if verbose:
                    print(f'{norm}-norm change', it_norm)

                # Iteration logic based on norm change
                if it_norm < conv_tol:
                    if verbose:
                        print(f"Tolerance break, {norm}-norm below {conv_tol}!")
                    break
                elif it_norm > div_treshold:
                    print("Diverging with norm", it_norm, ", aborted!")
                    break
                
                # Notify if max iterations is reached
                if conv_it == (conv_max_it - 1):
                    print(f"WARNING: max iterations ({conv_max_it}) reached on one knot, increment norm {it_norm}")

        return x, f_a_vals_tot, n_tot_it, it_norm
    
    def iterate_global(self, x, conv_tol, conv_max_it, div_treshold, norm, verbose, bvp = None, T = None):
        # If bvp, keep track of the delta ( _0I^alpha_T f(x) )
        if bvp:
            delta = np.zeros(self.d)
        
        N_knots = self.bs.t_eval_vals_ord.shape[0] # Number of knots
        n_calc = self.bs.t_calc_vals_ord.shape[1] # Spline polynomial order calculations f
        f_a_vals = np.zeros([self.d, N_knots, n_calc])

        for n_tot_it in range(conv_max_it):
            x_prev = x.copy() # Store previous estimate for increment norm calculation later
            f_a_vals[:, :, :] = self.f(self.bs.t_eval_vals_ord, x) # Calculate f(x, t)

            for k in range(self.d):
                # Get _0I^alpha_t f(x,t) for all t_eval
                int_vals = self.bs.I_a(f_a_vals[k, :, :], alpha = self.alpha[k])
                # x = x_0 + _0I_t^alpha f(x,t)
                x[k, :, :] = self.x_0[k] + int_vals

                if np.array(self.forcing_vals[k]).size > 1:
                    if bvp:
                        # TODO: implement forcing with delta computation
                        print("WARNING! Forcing not yet supported for global BVP solver! No forcing added.")
                    else:
                        # Add forcing
                        x[k, :, :] += self.forcing_vals[k]

                if bvp:
                    # Get delta computation
                    delta[k] = self.bs.I_a_scalar(T, f_a_vals[k, :, :], self.alpha[k])
                    # Substract integrated delta for BVP requirement
                    x[k, :, :] -= (self.bs.t_eval_vals_ord/T)**self.alpha[k] * delta[k]

            # Compute the iteration norm increment
            it_norm = SplineSolver.it_norm(x, x_prev, norm)

            # Iteration logic based on norm change
            if verbose:
                print('Norm change', it_norm)
            if it_norm < conv_tol:
                if verbose:
                    print("Tolerance break!")
                break
            elif it_norm > div_treshold:
                print("Diverging with norm", it_norm, ", aborted!")
                break
            
            # Notify if max iterations is reached
            if n_tot_it == (conv_max_it - 1):
                print(f"WARNING: max iterations ({conv_max_it}) reached on one knot, increment norm {it_norm}")

        return x, f_a_vals, n_tot_it, it_norm

    def run(self, 
            method='local', 
            bvp = False, 
            T = None, 
            conv_tol = 1e-12, 
            conv_max_it = 500, 
            div_treshold = 5e2, 
            norm = 'sup', 
            save_x = True, 
            verbose = False):
        
        # Makes sure the integral basis is ready
        self.bs.build_and_save_integral_basis(self.alpha, verbose= verbose)
        # Gets initial value for x
        x = self.get_initial_x(save_x)

        tic = time.time() # Start timer
        ### Main solving: gets the solution and some statistics ###
        if method == 'local':
            ### LOCAL SOLUTION METHOD: knot-for knot ###
            if bvp:
                # TODO: implement with root-finding
                print("WARNING: NO BVP implementation in local solution method, reverting to regular IVP calculation!")

            x_sol, f_a_vals, n_tot_it, it_norm = self.iterate_local(x, conv_tol, conv_max_it, div_treshold, norm, verbose)
        if method == 'global':
            ### GLOBAL SOLUTION METHOD: whole interval at once ###
            x_sol, f_a_vals, n_tot_it, it_norm = self.iterate_global(x, conv_tol, conv_max_it, div_treshold, norm, verbose, bvp = bvp, T = T)
            
        toc = time.time() # Stop timer
        total_time = toc - tic # Computes solution finding time

        if T is not None:
            delta = self.get_delta(T, f_a_vals)
        else:
            delta = None

        # Gets the value for f_0. Can be useful for checking and period selection (e.g. y'(0)=0)
        f_0 = f_a_vals[:, 0, 0] 
        # Build all results here
        result_dict = self.build_results_dict(x_sol, n_tot_it,norm, it_norm, total_time, f_0, T = T, delta = delta)
        # Get a nice human-readable summary of run time statistics
        if verbose:
            SplineSolver.print_summary(result_dict)        

        # Save the solution if required
        if save_x:
            self.x_storage = x_sol

        return result_dict