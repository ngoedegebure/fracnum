import numpy as np
from abc import ABC, abstractmethod

class ODEFunctions(ABC):
    def __init__(self, params, bernstein, transpose):
        self.params = params
        self.bernstein = bernstein
        self.transpose = transpose

        self.duplicate_dim = None

    def set_bs_mult_upscale_functions(self, bs_mult, bs_upscale):
        self.bs_mult, self.bs_upscale = bs_mult, bs_upscale

    @abstractmethod
    def f(self, t_vals, x_vals):
        pass

class VdP(ODEFunctions):
    def __init__(self, params = {}, bernstein=False, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 3
        self.three_system = False

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        mu = self.params['mu']
        if self.bernstein:
            A_x_in, A_y_in = x_vals[0, :], x_vals[1, :]

            if self.three_system == False:
                # x' = y
                A_x_out = self.bs_upscale(A_y_in, self.N_upscale)

                # y' = mu * (1-x^2)y - x
                A_y_out = mu * self.bs_mult(1 - self.bs_mult(A_x_in, A_x_in), A_y_in) - self.bs_upscale(A_x_in, self.N_upscale)

                return np.array([A_x_out, A_y_out])
            else:
                A_z_in = x_vals[2, :]

                # x' = z
                A_x_out = self.bs_upscale(A_z_in, self.N_upscale)

                # y' = mu * (1-x^2)z - x
                A_y_out = mu * self.bs_mult(1 - self.bs_mult(A_x_in, A_x_in), A_z_in) - self.bs_upscale(A_x_in, self.N_upscale)

                # z' = y
                A_z_out = self.bs_upscale(A_y_in, self.N_upscale)

                return np.array([A_x_out, A_y_out, A_z_out])
        else: # The non-Bernstein spline one is added for quicker comparison options with e.g. Diethelm's method (see fracnum.numerical)
            x, y = x_vals[0], x_vals[1]
            # x' = y
            x_out = y
            # y' = mu * (1-x^2)y - x
            y_out = -x - mu * (x**2 -1)*y

            # Forcing included here to use in e.g. Diethelm or another numerical piece
            if 'A' in self.params and 'omega' in self.params:
                A = float(self.params['A'])
                omega = float(self.params['omega'])
                if A != 0 and omega != 0:
                    # TODO: CHANGE TO GENERAL ELEMENT ADDITION
                    y_out+=A*np.sin(omega*t_vals)

            if 'c' in self.params.keys():
                c = self.params['c']
                if c != 0:
                    # TODO: CHANGE TO GENERAL ELEMENT ADDITION
                    y_out+=c

            return np.array([x_out, y_out]).T
        
class NegExp(ODEFunctions):
    def __init__(self, params = {}, bernstein=False, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        if self.bernstein:
            A_x_in = x_vals[0, :]

            A_x_out = -A_x_in

            return np.array([A_x_out])
        else: # The non-Bernstein spline one is added for quicker comparison options with e.g. Diethelm's method (see fracnum.numerical)
            x, y = x_vals[0]

            x_out = x

            return np.array([x_out]).T
        
class t_k(ODEFunctions):
    def __init__(self, params = {}, bernstein=False, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        k = np.array(self.params['k'])
        if 'c' in self.params.keys():
            c = self.params['c']
        else:
            c = 0
        res = np.array([np.sum([t_vals**k_i for k_i in k], axis=0) + c])

        if self.transpose:
            return res.T
        else:
            return res
    
class test_fun(ODEFunctions):
    def __init__(self, params = {}, bernstein=False, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        return np.array([t_vals**2 - (1-t_vals)**3])
    
class lin_damp_os(ODEFunctions):
    def __init__(self, params = {}, bernstein=True, transpose = False):
        super().__init__(params, bernstein, transpose)
        self.N_upscale = 1

    def f(self, t_vals, x_vals):
        if self.transpose:
            x_vals = x_vals.T

        eta, omega = self.params['eta'], self.params['omega']
        if self.bernstein:
            A_x_in, A_y_in = x_vals[0, :], x_vals[1, :]

            # x' = y
            A_x_out = A_y_in

            # y' = mu * (1-x^2)y - x
            A_y_out = -eta*A_y_in -omega**2*A_x_in

            return np.array([A_x_out, A_y_out])
        else:
            assert 0