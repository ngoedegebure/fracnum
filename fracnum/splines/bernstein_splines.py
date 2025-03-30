from .static_spline_methods import SplineMethods
from .static_bernstein_methods import BernsteinMethods
from .spline_solvers import SplineSolver
from .backend import np

class BernsteinSplines:
    def __init__(self, t_knot_vals, n, n_eval = None, alpha_init = None, silent_mode = False, magnify = None, eq_opt = True, eq_opt_tol = 1e-10):        
        self.t_knot_vals = t_knot_vals  # Knot values t_0, t_1, t_2 ... t_k
        self.h = np.diff(t_knot_vals)   # Knot sizes h_0 ... h_{k-1}
        self.n = n                      # Polynomial calculation order

        # Optimizes for equidistant grid: saves A LOT of memory. Mostly useful for Caputo (\beta = 1)
        if eq_opt:
            # Extra check: check whether grid is actually equidistant 
            if np.max(np.abs(np.diff(self.h))) < eq_opt_tol:
                self.eq_opt = True
                print('NOTE: Equidistant grid optimization enabled')
            else:
                self.eq_opt = False
                print(f'WARNING: Equidistant grid optimization disabled since grid not equidistant (tol: {eq_opt_tol:.1e})')
        else:
            self.eq_opt = False            

        # Get calculation t values
        self.t_calc_vals_ord, self.t_calc_vals_list = SplineMethods.build_total_t_vals(t_knot_vals, n, magnify= magnify)

        if n_eval is not None:
            # If a different evaluation order is specied, use this order and build t vals accordingly
            self.n_eval = n_eval
            self.t_eval_vals_ord, self.t_eval_vals_list = SplineMethods.build_total_t_vals(t_knot_vals, n_eval,  magnify= magnify)
        else:
            # If not, take both as n and use calculation t values
            self.n_eval = n
            self.t_eval_vals_ord, self.t_eval_vals_list = self.t_calc_vals_ord, self.t_calc_vals_list

        # Empty integral basis storage
        self.B_I, self.B_I_scalar = {}, {}

        if self.eq_opt:
            self.B_I_opt = {}
            self.B_b_B_I_opt = {}

        # If initialization request is given in arguments, use this
        if alpha_init is not None and self.eq_opt == False:
            self.B_I[alpha_init] = SplineMethods.build_integral_basis(alpha_init, self.t_calc_vals_ord, self.t_eval_vals_ord, progress_verbose = True, time_verbose = True)
        
        # Build binomial basis (converts Bernstein polynomials to monomials)
        self.B_b = BernsteinMethods.build_bernstein_binom_basis(self.n)

        # Initialize storage dictionaries of spline multiplication and scaling
        self.C_storage, self.upscale_storage, self.downscale_storage = {}, {}, {}

        # Initialize solution storage
        self.x_storage = None

        # Initialize forcing values storage
        self.forcing_storage = {}

        self.silent_mode = silent_mode

    def I_a_scalar(self, t, A, alpha, time_verbose = False):
        # A wrapper to more elegantly compute the integral at one specific time for a scalar t
        index_key = (alpha, t)
        if index_key not in self.B_I_scalar.keys():
            t_matrix_val = np.reshape(t, [1,1])
            self.B_I_scalar[index_key] = SplineMethods.build_integral_basis(alpha, self.t_calc_vals_ord, t_matrix_val, time_verbose=time_verbose, progress_verbose = False)

        int = np.einsum('kl,kl->',A@self.B_b, self.B_I_scalar[index_key][:, :, 0, 0])
        return int

    def I_a(self, A, alpha, knot_sel = None, to_vector = False, progress_verbose = True, time_verbose = False):
        q = self.n_eval

        alpha = float(alpha)
        if alpha == 0:
            print("WARNING: support for alpha = 0 needs new implementation!")
            return False
            if knot_sel is None:
                # All knots
                int = A
            elif knot_sel[0] == 'to':
                knot_index = knot_sel[1]
                int = A[:(knot_index+1)]
            elif knot_sel[0] == 'at':
                # Up to selected knot
                knot_index = knot_sel[1]
                if len(A.shape) == 1:
                    int = np.reshape(A, (1, A.shape[0]))
                else:
                    int = A[knot_index:(knot_index+1)]
            int = self.splines_downscale(int, self.n_eval)
        else:
            # Get or create the integration basis
            if (knot_sel is None or self.eq_opt == False) and alpha not in self.B_I.keys():
                self.B_I[alpha] = SplineMethods.build_integral_basis(alpha, self.t_calc_vals_ord, self.t_eval_vals_ord, progress_verbose = progress_verbose, time_verbose = time_verbose)
            elif (knot_sel is not None and self.eq_opt == True) and alpha not in self.B_I_opt.keys():
                N = self.t_eval_vals_ord.shape[0]
                self.B_I_opt[alpha] = SplineMethods.build_integral_basis(alpha, self.t_calc_vals_ord, self.t_eval_vals_ord[(N-1):N, :], progress_verbose = progress_verbose, time_verbose = time_verbose)
                self.B_b_B_I_opt[alpha] = self.B_b@self.B_I_opt[alpha][-1, :, :]
            
            if knot_sel is None:
                # All knots
                int = np.einsum('kl,kln->n', A@self.B_b, self.B_I[alpha])
            elif knot_sel[0] == 'to':
                # Up to selected knot
                knot_index = knot_sel[1]
                if self.eq_opt:
                    ## Eq opt here!
                    N_knots = self.B_I_opt[alpha].shape[0]
                    start_knot = N_knots-knot_index
                    # breakpoint()
                    
                    # int = np.einsum('kl,kln->n', A[:(knot_index+1), :]@self.B_b, self.B_I_opt[alpha][(start_knot-1):(N_knots), :, :])
                    int= np.tensordot(A[:(knot_index+1), :]@self.B_b, self.B_I_opt[alpha][(start_knot-1):(N_knots), :, :], axes=2)
                else:
                    # TODO: get index to non-optimized setup!! Needs smarter selection
                    int = np.einsum('kl,kln->n', A[:(knot_index+1), :]@self.B_b, self.B_I[alpha][:(knot_index+1), :, knot_index, :])
            elif knot_sel[0] == 'at':
                # Only selected knot
                knot_index = knot_sel[1]
                if self.eq_opt:
                    ## Eq opt here!
                    # breakpoint()
                    # int = np.einsum('l,ln->n', A@self.B_b, self.B_I_opt[alpha][-1, :, :])
                    int =  A@self.B_b_B_I_opt[alpha] 
                else:
                    # TODO: get index to non-optimized setup!! Needs smarter selection
                    int = np.einsum('l,ln->n', A@self.B_b, self.B_I[alpha][knot_index, :, knot_index, :])
                    
        if to_vector:
            return # SplineMethods.a_to_vector(int)
        else:
            return SplineMethods.a_to_matrix(int, q)# int

    def ddt(self, A, to_vector=True, upscale = True):
        # Computes the derivative of Bernstein splines...
        # ... easy to do since the derivative of Bernstein polynomial is the difference...
        # ... of coefficients of the two lower order polynomials
        n = A.shape[1]-1
        A_der = (((A[:, 1:] - A[:, :-1]) * n).T * 1/self.h).T 
        
        if upscale:
            # Upscale by one can be helpful since the derivative is a spline downscaled by one order
            A_der_output = self.splines_upscale(A_der, 0, override_plusone=True)
        else:
            A_der_output = A_der
        
        if to_vector:
            return SplineMethods.a_to_vector(A_der_output)
        else:
            return A_der_output

    def splines_multiply(self, A, B):
        # Multiplies two splines
        if A.shape[0] != B.shape[0]:
            print('Multiplication not allowed! Number of rows does not correspond!')
            assert 0
        else:
            n_rows = A.shape[0]

        n = A.shape[1] - 1
        m = B.shape[1] - 1

        # Get or create multiplication matrix C
        index_tuple = (n, m)
        if index_tuple in self.C_storage.keys():
            C = self.C_storage[index_tuple]
        else:
            C = BernsteinMethods.build_C_matrix(n, m)
            self.C_storage[index_tuple] = C

        D = np.einsum('kij,mi,mj->mk', C, A, B)

        return D

    def splines_upscale(self, A, n, override_plusone = False):
        # Upscale a spline * n times or + 1 times
        if override_plusone:
            mult_shape = 2
            n = "plusone" # A bit ugly now but it works for storage
        else:
            mult_shape = A.shape[1]*(n-1)-1

        index_tuple = (A.shape, n)
        # Get or create scaling matrix
        if index_tuple in self.upscale_storage.keys():
            scale_matrix = self.upscale_storage[index_tuple]
        else:
            scale_matrix = np.ones([A.shape[0], mult_shape])
            self.upscale_storage[index_tuple] = scale_matrix
        # Use multiplication method
        result = self.splines_multiply(A, scale_matrix)
        return result
    
    def splines_downscale(self, A, n_target):
        m = A.shape[-1]-1 # original order
        key = (n_target, m)
        if key in self.downscale_storage.keys():
            P = self.downscale_storage[key]
        else:
            P = BernsteinMethods.bernstein_projection_matrix(m, n_target)
            self.downscale_storage[key] = P
            
        res = np.einsum('nm,km->kn', P, A) # m: original. n: target. k: no. of knots
        
        return res
    
    def build_and_save_integral_basis(self, alpha_vals, verbose = False, time_verbose = False):
        if self.eq_opt == False:
            if self.silent_mode:
                verbose, time_verbose = False, False
            for alpha_val in alpha_vals:
                alpha_val = float(alpha_val)
                if alpha_val not in self.B_I.keys():
                    self.B_I[alpha_val] = SplineMethods.build_integral_basis(alpha_val, self.t_calc_vals_ord, self.t_eval_vals_ord, progress_verbose=verbose, time_verbose=time_verbose)
    
    def initialize_solver(self, f, x_0, alpha_vals, beta_vals = 1,forcing_params = {}):
        return SplineSolver(self, f, x_0, alpha_vals, beta_vals = beta_vals, forcing_parameters = forcing_params)
        
    