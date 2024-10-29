from .static_spline_methods import SplineMethods
from fracnum.numerical import sin_I_a
import numpy as np
from scipy.special import gamma
import time
from tqdm import tqdm

# Here the main solving gets done
# TODO: Add comments to this file!

class SplineSolver():
    def __init__(self, bs, f, x_0, alpha_vals, forcing_parameters = {}):
        self.bs = bs
        self.f = f
        self.x_0 = x_0
        self.d = len(x_0)

        self.sin_forcing_storage = {}

        self.alpha = SplineSolver.parse_alpha(alpha_vals, self.d)
        self.forcing_vals = self.build_forcing_values(forcing_parameters)

        self.x_storage = None

    @staticmethod
    def parse_alpha(alpha_vals, d):
        if np.array(alpha_vals).size == 1:
            alpha_parsed = np.ones(d)*alpha_vals
        elif np.array(alpha_vals).size != d:
            print("ERROR! Either give one alpha or d (dimensions) alpha's!")
            return False
        elif np.array(alpha_vals).size == d:
            alpha_parsed = alpha_vals
        return alpha_parsed

    def build_forcing_values(self, forcing_parameters):
        forcing_vals = [np.array([0], dtype='float64')] * self.d
        for forcing_element in forcing_parameters:
            sin_vals, c_vals = 0,0

            dim = forcing_element['dim']
            alpha_f = self.alpha[dim]
            # Sin storage: A*sin(omega * t)
            if 'A' in forcing_element.keys() and 'omega' in forcing_element.keys():
                A_f, omega_f = forcing_element['A'], forcing_element['omega']
                if A_f !=0 and omega_f != 0:
                    sin_forcing_storage_key = (alpha_f, A_f, omega_f)
                    if sin_forcing_storage_key not in self.sin_forcing_storage.keys():
                        sin_vals = A_f* SplineMethods.a_to_matrix(np.array([sin_I_a(t, alpha_f, omega_f) for t in self.bs.t_eval_vals_list]), self.bs.n_eval)
                    else:
                        sin_vals = self.sin_forcing_storage[sin_forcing_storage_key]
            
            if 'c' in forcing_element.keys():
                c_f = forcing_element['c']
                if c_f != 0:
                    c_int_vals = self.bs.t_eval_vals_list**(alpha_f)/gamma(alpha_f+1)
                    c_vals = SplineMethods.a_to_matrix(c_int_vals, self.bs.n_eval) * c_f

            forcing_vals[dim] = sin_vals + c_vals

        return forcing_vals
    
    def get_initial_x(self,save_x):
        N_tot = len(self.bs.t_eval_vals_list)
        if self.x_storage is not None and save_x:
            x = self.x_storage
        else:
            x_flat = np.ones([N_tot, self.d]) * self.x_0
            x = np.array([SplineMethods.a_to_matrix(x_flat[:, i], self.bs.n_eval) for i in range(self.d)])

        return x

    def build_results_dict(self, x, n_tot_it, norm,it_norm, total_time, f_0, T = None, delta = None):
        N_knots = x[0, :, :].shape[0]

        output_dict = {
            't': self.bs.t_eval_vals_list,
            'x':np.array([SplineMethods.a_to_vector(x[i, :, :]) for i in range(self.d)]).T,
            'a':x,
            'norm_type':norm,
            'norm_value':it_norm,
            'n_it':n_tot_it,
            'total_time':total_time,
            'time_per_it':total_time/n_tot_it,
            'n_it_per_knot':n_tot_it/N_knots,
            'time_per_knot':total_time/N_knots,
            'f_0':f_0,
            'delta':delta
        }
        return output_dict
    
    @staticmethod
    def it_norm(x, x_prev, norm):
        if norm == 'sup':
            it_norm = np.max(np.abs(x-x_prev))
        elif norm == 'L2':
            it_norm = np.linalg.norm(x-x_prev)
        return it_norm
    
    def get_delta(self, T, f_a_vals):
        delta = np.zeros(self.d)
        for k in range(self.d):
            delta[k] = self.bs.I_a_scalar(T, f_a_vals[k, :, :], self.alpha[k])
        return delta
    
    @staticmethod
    def print_summary(res):
        print(f"~ {res['n_it']} iterations, {res['total_time']:.4f} s total, {res['time_per_it']:.6f} s/it")
        print(f"~ {res['n_it_per_knot']:.2f} it/knot avg., {res['time_per_knot']:.6f} s/knot avg.")

    def iterate_local(self, x, conv_tol, conv_max_it, div_treshold, norm, verbose):
        f_a_vals_tot = np.zeros([self.d, self.bs.t_calc_vals_ord.shape[0], self.bs.t_calc_vals_ord.shape[1]])
        n_tot_it = 0
        for i_knot in tqdm(range(self.bs.t_eval_vals_ord.shape[0]), desc='Iterating IVP for knots'):
            int_vals_base = np.zeros([self.d, self.bs.t_eval_vals_ord.shape[1]])
            if i_knot > 0:
                f_no_last = f_a_vals_tot[:, 0:(i_knot+1), :]
                f_no_last[:, -1, :] = np.zeros(f_no_last[:, -1, :].shape)
                for k in range(self.d):
                    int_vals_base[k, :] = self.bs.I_a(f_no_last[k, :, :], alpha = self.alpha[k], knot_sel = ('to', i_knot))
                x[:, i_knot, :] = x[:, i_knot-1, :]
            for conv_it in range(conv_max_it):
                n_tot_it+=1
                x_prev = x.copy()
                f_a_vals_tot[:, i_knot:(i_knot+1), :] = self.f(self.bs.t_eval_vals_ord[i_knot, :], x[:, i_knot:(i_knot+1), :])

                for k in range(self.d):
                    # Saves some time
                    int_vals = self.bs.I_a(f_a_vals_tot[k, i_knot, :], alpha = self.alpha[k], knot_sel = ('at', i_knot))
                    x[k, i_knot, :] = self.x_0[k] + int_vals_base[k, :] + int_vals

                    if np.array(self.forcing_vals[k]).size > 1:
                        x[k, i_knot, :] += self.forcing_vals[k][i_knot, :]

                it_norm = SplineSolver.it_norm(x, x_prev, norm)

                if verbose:
                    print('Norm change', it_norm)
                if it_norm < conv_tol:
                    if verbose:
                        print("Tolerance break!")
                    break
                elif it_norm > div_treshold:
                    print("Diverging with norm", it_norm, ", aborted!")
                    break

        return x, f_a_vals_tot, n_tot_it, it_norm
    
    def iterate_global(self, x, conv_tol, conv_max_it, div_treshold, norm, verbose, bvp = None, T = None):
        if bvp:
            delta = np.zeros(self.d)
        n_tot_it = 0

        for i in range(conv_max_it):
            x_prev = x.copy()

            f_a_vals = self.f(self.bs.t_eval_vals_ord, x)

            for k in range(self.d):
                int_vals = self.bs.I_a(f_a_vals[k, :, :], alpha = self.alpha[k])
                x[k, :, :] = self.x_0[k] + int_vals

                if np.array(self.forcing_vals[k]).size > 0:
                    if bvp:
                        print("WARNING! Forcing not yet supported for global BVP solver! No forcing added.")
                    else:
                        x[k, :, :] += self.forcing_vals[k][:, :]

                if bvp:
                    delta[k] = self.bs.I_a_scalar(T, f_a_vals[k, :, :], self.alpha[k])
                    x[k, :, :] -= (self.t_eval_vals_ord/T)**self.alpha[k] * self.delta[k]

            n_tot_it+=1

            it_norm = SplineSolver.it_norm(x, x_prev, norm)

            if verbose:
                print('Norm change', it_norm)
            if it_norm < conv_tol:
                if verbose:
                    print("Tolerance break!")
                break
            elif it_norm > div_treshold:
                print("Diverging with norm", it_norm, ", aborted!")
                break

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
        
        self.bs.build_and_save_integral_basis(self.alpha, verbose= True)
        x = self.get_initial_x(save_x)

        tic = time.time()
        if method == 'local':
            x_sol, f_a_vals, n_tot_it, it_norm = self.iterate_local(x, conv_tol, conv_max_it, div_treshold, norm, verbose)
        if method == 'global':
            x_sol, f_a_vals, n_tot_it, it_norm = self.iterate_global(x, conv_tol, conv_max_it, div_treshold, norm, verbose, bvp = bvp, T = T)
            
        toc = time.time()
        total_time = toc - tic

        if T is not None:
            delta = self.get_delta(T, f_a_vals)
        else:
            delta = None

        f_0 = f_a_vals[:, 0, 0]
        result_dict = self.build_results_dict(x_sol, n_tot_it,norm, it_norm, total_time, f_0, T = T, delta = delta)

        SplineSolver.print_summary(result_dict)        

        if save_x:
            self.x_storage = x_sol

        return result_dict