import numpy as np
from .static_spline_methods import SplineMethods
from .static_bernstein_methods import BernsteinMethods
from .spline_solvers import SplineSolver

class BernsteinSplines:
    def __init__(self, t_knot_vals, n, n_eval = None, alpha_init = None):        
        self.t_knot_vals = t_knot_vals  # Knot values t_0, t_1, t_2 ... t_k
        self.h = np.diff(t_knot_vals)   # Knot sizes h_0 ... h_{k-1}
        self.n = n                      # Polynomial calculation order

        # Get calculation t values
        self.t_calc_vals_ord, self.t_calc_vals_list = SplineMethods.build_total_t_vals(t_knot_vals, n)

        if n_eval is not None:
            # If a different evaluation order is specied, use this order and build t vals accordingly
            self.n_eval = n_eval
            self.t_eval_vals_ord, self.t_eval_vals_list = SplineMethods.build_total_t_vals(t_knot_vals, n_eval)
        else:
            # If not, take both as n and use calculation t values
            self.n_eval = n
            self.t_eval_vals_ord, self.t_eval_vals_list = self.t_calc_vals_ord, self.t_calc_vals_list

        # Empty integral basis storage
        self.B_I, self.B_I_scalar = {}, {}

        # If initialization request is given in arguments, use this
        if alpha_init is not None:
            self.B_I[self.alpha] = SplineMethods.build_integral_basis(self.alpha, self.t_calc_vals_ord, self.t_eval_vals_ord, verbose = True)
        
        # Build binomial basis (converts Bernstein polynomials to monomials)
        self.B_b = BernsteinMethods.build_bernstein_binom_basis(self.n)

        # Initialize storage dictionaries of spline multiplication and scaling
        self.C_storage, self.upscale_storage = {}, {}

        # Initialize solution storage
        self.x_storage = None

        # Initialize forcing values storage
        self.forcing_storage = {}

    def I_a_scalar(self, t, A, alpha):
        # A wrapper to more elegantly compute the integral at one specific time for a scalar t
        index_key = (alpha, t)
        if index_key not in self.B_I_scalar.keys():
            t_matrix_val = np.reshape(t, [1,1])
            self.B_I_scalar[index_key] = SplineMethods.build_integral_basis(alpha, self.t_calc_vals_ord, t_matrix_val, time_verbose=False, progress_verbose = False)

        int = np.einsum('kl,kl->',A@self.B_b, self.B_I_scalar[index_key][:, :, 0, 0])
        return int

    def I_a(self, A, alpha, knot_sel = None, to_vector = False):
        # Get or create the integration basis
        if alpha not in self.B_I.keys():
            self.B_I[alpha] = SplineMethods.build_integral_basis(alpha, self.t_calc_vals_ord, self.t_eval_vals_ord)

        if knot_sel is None:
            # All knots
            int = np.einsum('kl,klmn->mn', A@self.B_b, self.B_I[alpha])
        elif knot_sel[0] == 'to':
            # Up to selected knot
            knot_index = knot_sel[1]
            int = np.einsum('kl,kln->n', A[:(knot_index+1), :]@self.B_b, self.B_I[alpha][:(knot_index+1), :, knot_index, :])
        elif knot_sel[0] == 'at':
            # Only selected knot
            knot_index = knot_sel[1]
            int = np.einsum('l,ln->n', A@self.B_b, self.B_I[alpha][knot_index, :, knot_index, :])

        if to_vector:
            return SplineMethods.a_to_vector(int)
        else:
            return int

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
    
    def build_and_save_integral_basis(self, alpha_vals, verbose = False):
        for alpha_val in alpha_vals:
            if alpha_val not in self.B_I.keys():
                self.B_I[alpha_val] = SplineMethods.build_integral_basis(alpha_val, self.t_calc_vals_ord, self.t_eval_vals_ord, time_verbose=verbose)
    
    def initialize_solver(self, f, x_0, alpha_vals, forcing_params = {}):
        return SplineSolver(self, f, x_0, alpha_vals, forcing_parameters = forcing_params)
        