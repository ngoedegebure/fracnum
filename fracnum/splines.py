import math
import time
from scipy.special import gamma, betainc, beta, binom, comb
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from mpmath import hyp1f2

# AWAS! AWAS! AWAS!
import numpy as np
# import cupy as np

import scipy.sparse as sp
# import opt_einsum as oe


class BernsteinSplines: 
    def __init__(self, t_knot_vals, n, n_eval = None):
        self.t_knot_vals = t_knot_vals
        self.h = np.diff(t_knot_vals)
        self.n = n

        # self.alpha = alpha

        self.t_tot_vals_ord, self.t_eval_vals_list = self.build_total_t_vals(t_knot_vals, n)

        if n_eval is not None:
            self.n_eval = n_eval
            self.t_calc_vals_list = self.t_eval_vals_list.copy()
            # Overrides self.t_eval_vals_list !
            self.t_eval_vals_ord, self.t_eval_vals_list = self.build_total_t_vals(t_knot_vals, n_eval)
        else:
            self.n_eval = n
            self.t_eval_vals_ord = self.t_tot_vals_ord

        self.B_I = dict()
        # I took the following out so that it does not do anything double

        # self.B_I[self.alpha] = self.build_integral_basis(self.alpha) 
        self.B_b = self.build_binom_basis()

        # self.B_d_eval = self.build_der_basis(self.n_eval)

        # print(self.B_d_eval)
        # breakpoint()

        # self.B_bB_I = np.tensordot(self.B_b, self.B_I, axes=([1], [1]))
        # self.B_bB_I = np.swapaxes(self.B_bB_I, 0, 1)

        self.C_storage = dict()
        self.upscale_storage = dict()

        self.x_storage = None

        self.sin_forcing_storage = dict()

    def I_a_b_beta(t_vals, alpha, k, bounds):
        a, b = bounds
        t_pre = t_vals[t_vals <= a]
        t_in = t_vals[(t_vals > a)]
        t_trans = (t_in-a)/(b-a)

        # taken out (b-a)**(alpha*1)
        I_b = 1/gamma(alpha)*t_trans**(alpha+k)*betainc(k+1, alpha, np.fmin(1, t_trans)/t_trans)*beta(k+1, alpha)

        # I_b = 1/gamma(alpha)* t_in**(alpha+k)*(betainc(k+1, alpha, np.fmin(b,t_in)/t_in) - betainc(k+1, alpha, a/t_in))# * beta(k+1, alpha)
        return np.concatenate([np.zeros(t_pre.shape), I_b])

    def build_total_t_vals(self, t_knot_vals, n):
        total_t_vals_ord = np.zeros([len(t_knot_vals)-1, n+1])

        for i in range(len(t_knot_vals)-1):
            total_t_vals_ord[i, :] = np.linspace(t_knot_vals[i], t_knot_vals[i+1], n+1)

        t_eval_points = np.concatenate([np.array([t_knot_vals[0]]),np.reshape(total_t_vals_ord[:, 1:], (len(t_knot_vals)-1)*n)])
        # breakpoint()
        return total_t_vals_ord, t_eval_points

    @staticmethod
    def a_to_matrix(a_vector, n):
        N_rows = int((len(a_vector)-1)/max(n, 1)) #int((len(a_vector))/(max(n, 1)))
        N_columns = n+1
        # breakpoint()
        total_a_matrix = np.zeros([N_rows, N_columns])

        total_a_matrix[:, :-1] = a_vector[:-1].reshape([N_rows, N_columns-1]) # all columns except last
        total_a_matrix[:-1, -1] = total_a_matrix[1:, 0]    # last column except for last element
        total_a_matrix[-1, -1] = a_vector[-1]               # last column last row
        return total_a_matrix
    
    @staticmethod
    def a_to_vector(total_a_matrix):
        a_vector = np.zeros(total_a_matrix[:, :-1].size + 1)
        
        last_val = total_a_matrix[-1, -1]
        a_vector[:-1] = total_a_matrix[:, :-1].reshape(a_vector[:-1].shape)
        a_vector[-1] = last_val
        return a_vector
    
    def build_integral_basis(self, alpha):
        
        # B_I[knot_calc, order_calc, order_eval]
        B_I = np.zeros([self.t_tot_vals_ord.shape[0], self.t_tot_vals_ord.shape[1], self.t_eval_vals_ord.shape[0], self.t_eval_vals_ord.shape[1]])

        for i_knot in tqdm(range(self.t_tot_vals_ord.shape[0]), desc=f"Building integral basis alpha = {alpha} for knots"):
            t_knot_vals = self.t_tot_vals_ord[i_knot, :]

            t_a, t_b = t_knot_vals[0], t_knot_vals[-1]
            dt = t_b-t_a

            s_i = (self.t_eval_vals_list-t_a)/dt
            
            for order_k in range(self.t_tot_vals_ord.shape[1]):
                B_I_vals_list = dt**alpha * BernsteinSplines.I_a_b_beta(s_i, alpha, order_k, [0, 1])
                # The reshaping instead of doing a for loop makes everything much much faster and saves k computations
                B_I[i_knot, order_k, :, :] = BernsteinSplines.a_to_matrix(B_I_vals_list, self.n_eval)
        return B_I#, sp.coo_matrix(B_I.reshape(-1))
    
    def build_binom_basis(self):
        binom_basis = np.zeros([self.n+1, self.n+1])

        for j in range(self.n+1):
            for l in range(j, self.n+1):
                binom_basis[j, l] = binom(self.n, l) * binom(l, j) * (-1)**(l-j)
        return binom_basis#, sp.coo_matrix(binom_basis.reshape(-1))
        
    def I_a(self, A, alpha = None, knot_sel = None, to_vector = False):
        if alpha is not None:
            if alpha not in self.B_I.keys():
                self.B_I[alpha] = self.build_integral_basis(alpha)
        else:
            print('Provide an alpha!!!')
            return False

        # add new alpha functionality
        
        if knot_sel is None:
            # A = BernsteinSplines.a_to_matrix(a, self.n)
            # breakpoint()
            
            # int = np.einsum('kl,klm->km', A@self.B_b, self.B_I[alpha])
    
            # int = oe.contract('kl,klmn->mn', A@self.B_b_sparse, self.B_I_sparse[alpha])
            int = np.einsum('kl,klmn->mn', A@self.B_b, self.B_I[alpha])
        elif knot_sel[0] == 'to':
            # breakpoint()
            knot_index = knot_sel[1]
            int = np.einsum('kl,kln->n', A[:(knot_index+1), :]@self.B_b, self.B_I[alpha][:(knot_index+1), :, knot_index, :])
        elif knot_sel[0] == 'at':
            knot_index = knot_sel[1]
            # breakpoint()
            # breakpoint()
            int = np.einsum('l,ln->n', A@self.B_b, self.B_I[alpha][knot_index, :, knot_index, :])
        #     # A = BernsteinSplines.a_to_matrix(a, self.n)[:calc_sel]
        #     B_I = self.B_I[alpha][:calc_sel, :, eval_sel]
        #     coeff = A@self.B_b
        
        #     int = np.tensordot(coeff, B_I[alpha])
        if to_vector:
            return self.a_to_vector(int)
        else:
            return int
        # return BernsteinSplines.a_to_matrix(int, self.n_eval)
    
    def sin_I_a(self, t, alpha, omega):
        if alpha == 0:
            result = np.sin(omega*t)
        elif alpha == 1:
            result = -np.cos(omega*t)/omega + 1/omega
        else:
            # Define the first hypergeometric term
            term1 = hyp1f2(alpha / 2, 1 / 2, alpha / 2 + 1, -1 / 4 * omega**2 * t**2)

            # Define the second hypergeometric term
            term2 = hyp1f2(alpha / 2 + 1 / 2, 3 / 2, alpha / 2 + 3 / 2, -1 / 4 * omega**2 * t**2)

            # Define the full expression
            result = (t**alpha * ((np.sin(omega * t) * term1) / alpha - 
                            (omega * t * np.cos(omega * t) * term2) / (alpha + 1))) / gamma(alpha)
        
        return result

    def ddt(self, A, to_vector=True):
        n = A.shape[1] # check!
        A_der = (((A[:, 1:] - A[:, :-1]) * (n-1)).T * 1/self.h).T  # WHY 2 ???

        n_d = A_der.shape[1]
        A_der_output = self.splines_upscale(A_der, 0, override_plusone=True)
        # breakpoint()
        if to_vector:
            return self.a_to_vector(A_der_output)
        else:
            return A_der_output
        # der = np.einsum('kl,klmn->mn', A@self.B_b, self.B_I[alpha])

    def binomial(n, k):
        """Compute the binomial coefficient 'n choose k'."""
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    @staticmethod
    def construct_C_matrix(n, m):
        """Construct the matrix C of size (n+m+1) x (n+1) x (m+1)."""
        C = np.zeros((n+m+1, n+1, m+1))
        
        # Fill in the C matrix
        for i in range(n+1):
            for j in range(m+1):
                k = i + j
                if k <= n + m:
                    C[k, i, j] = (binom(n, i) * binom(m, j)) / binom(n+m, k)
        
        return C

    def splines_multiply_matrix(self, A, B, scale= False):
        if A.shape[0] != B.shape[0]:
            print('Multiplication not allowed! Number of rows does not correspond!')
            return False
        else:
            n_rows = A.shape[0]
        
        n = A.shape[1] - 1
        m = B.shape[1] - 1
        
        index_tuple = (n, m)
        if index_tuple in self.C_storage.keys():
            C = self.C_storage[index_tuple]
        else:
            C = self.construct_C_matrix(n, m)
            # C = sp.coo_matrix(C.reshape(-1))
            self.C_storage[index_tuple] = C
        D = np.zeros([n_rows, n+m+1])

        # breakpoint()
        D = np.einsum('kij,mi,mj->mk', C, A, B)

        # breakpoint()
        return D
    
    def splines_upscale(self, A, n, override_plusone = False):
        index_tuple = (A.shape, n)
        if (A.shape, n) in self.upscale_storage.keys():
            scale_matrix = self.upscale_storage[index_tuple]
        else:
            if override_plusone:
                mult_shape = 2
            else:
                mult_shape = A.shape[1]*(n-1)-1
                
            scale_matrix = np.ones([A.shape[0], mult_shape])
            self.upscale_storage[index_tuple] = scale_matrix

        # CAN BE OPTIMIZED MORE!
        result = self.splines_multiply_matrix(A, scale_matrix, scale=True)
        return result
        
    def solve_ivp_global(self, f, x_0, alpha, T, periodic = False, tol = 1e-12, div_treshold = 5e2,  N_it_max = 500, norm = 'sup', video = None, remember_x = True, f_params = None, verbose = True):
        if video is not None:
            video_plot = True
            video_path = 'video_path/video_'+ str(int(time.time()))+'/'
        else:
            video_plot = False

        d = len(x_0)
        N_tot = len(self.t_eval_vals_list)

        if remember_x and self.x_storage is not None:
            x = self.x_storage
        else:
            x_flat = np.ones([N_tot, d]) * x_0
            x = np.array([BernsteinSplines.a_to_matrix(x_flat[:, i], self.n_eval) for i in range(d)])

        one_conv = False
        tic = time.time()

        if periodic:
            delta = np.zeros(d)
            N_it_max = N_it_max * 2

        n_tot_it = 0
        for i in range(N_it_max):
            n_tot_it+=1
            x_prev = x.copy()

            f_a_vals = f(self.t_eval_vals_ord, x, f_params, bernstein=True)
            
            for k in range(d):
                int_vals = self.I_a(f_a_vals[k, :, :], alpha = alpha)
                x[k, :, :] = x_0[k] + int_vals

                if periodic and one_conv:
                    delta[k] = int_vals[-1, -1] # t_k
                    x[k, :, :] -= (self.t_eval_vals_ord/T)**alpha * delta[k]

            if video_plot:
                plt.title("Iteration + "+str (n_tot_it))
                plt.xlabel('x')
                plt.ylabel('y')
                plt.plot(x[:,0], x[:,1])
                plt.savefig(video_path+str(int(n_tot_it))+'.png', dpi = 300)
                plt.close()

            if norm == 'sup':
                it_norm = np.max(np.abs(x-x_prev))
            else:
                it_norm = np.linalg.norm(x-x_prev)
                # it_norm = np.sqrt(np.inner(x-x_prev, x-x_prev))
            
            if verbose:
                print('Norm change', it_norm)
            if it_norm < tol:
                if periodic and one_conv == False:
                    print("Converged IVP, running again for periodicity...")
                    one_conv = True
                else:
                    print("Tolerance break!")
                    break
            elif it_norm > div_treshold:
                print("Diverging with norm", it_norm, ", aborted!")
                break

        toc = time.time()
        total_time = toc - tic
        time_per_it = total_time / n_tot_it
        print(n_tot_it, 'iterations,', np.round(total_time,4),'seconds,', np.round(time_per_it, 6), 's/it')

        if remember_x:
            self.x_storage = x

        output_dict = dict()

        output_dict['t'] = self.t_eval_vals_list
        output_dict['x'] = np.array([BernsteinSplines.a_to_vector(x[i, :, :]) for i in range(d)]).T
        output_dict['a'] = x
        output_dict['norm'] = {norm : it_norm}
        output_dict['n_it'] = n_tot_it
        output_dict['time'] = total_time
        output_dict['f_0'] = f_a_vals[:, 0, 0]

        if periodic:
            output_dict['delta'] = delta

        if video_plot:
            BernsteinSplines.write_video(video_path)
        
        return output_dict
    
    def solve_ivp_local(self, f, x_0, alpha, T, periodic = False, tol = 1e-12, div_treshold = 5e2,  N_it_max = 500, norm = 'sup', video = None, remember_x = True, f_params = None, verbose = True, sin_forcing_params = None):
        if video is not None:
            video_plot = True
            video_path = 'video_path/video_'+ str(int(time.time()))+'/'
        else:
            video_plot = False

        d = len(x_0)

        if np.array(alpha).size == 1:
            alpha = np.ones(d)*alpha
        elif np.array(alpha).size != d:
            print("ERROR! Either give one alpha or d (dimensions) alpha's!")
            return False
        
        N_tot = len(self.t_eval_vals_list)

        if remember_x and self.x_storage is not None:
            x = self.x_storage
        else:
            x_flat = np.ones([N_tot, d]) * x_0
            x = np.array([BernsteinSplines.a_to_matrix(x_flat[:, i], self.n_eval) for i in range(d)])

        tic = time.time()
        n_tot_it = 0
        f_a_vals_tot = np.zeros([d, self.t_tot_vals_ord.shape[0], self.t_tot_vals_ord.shape[1]])

        sin_forcing_enabled = False
        if sin_forcing_params is not None:
            if sin_forcing_params['A'] != 0:
                sin_forcing_enabled = True
                alpha_forcing = alpha[sin_forcing_params['dim']]

                sin_forcing_storage_key = (alpha_forcing, sin_forcing_params['A'], sin_forcing_params['omega'])
                if sin_forcing_storage_key not in self.sin_forcing_storage.keys():
                # forcing_vals = self.I_a(f_params['A']*np.sin(self.t_tot_vals_ord*f_params['omega']), alpha=alpha[-1])
                    forcing_vals = sin_forcing_params['A']* self.a_to_matrix(np.array([self.sin_I_a(t, alpha_forcing, sin_forcing_params['omega']) for t in self.t_eval_vals_list]), self.n_eval)
                else:
                    forcing_vals = self.sin_forcing_storage[sin_forcing_storage_key]

        for i_knot in tqdm(range(self.t_eval_vals_ord.shape[0]), desc='Iterating IVP for knots'):
            # breakpoint()
            int_vals_base = np.zeros([d, self.t_eval_vals_ord.shape[1]])
            if i_knot > 0:
                f_no_last = f_a_vals_tot[:, 0:(i_knot+1), :]
                f_no_last[:, -1, :] = np.zeros(f_no_last[:, -1, :].shape)
                for k in range(d):
                    # breakpoint()
                    int_vals_base[k, :] = self.I_a(f_no_last[k, :, :], alpha = alpha[k], knot_sel = ('to', i_knot))
                x[:, i_knot, :] = x[:, i_knot-1, :]
            for conv_it in range(N_it_max):
                n_tot_it+=1
                x_prev = x.copy()
                # breakpoint()
                f_a_vals_tot[:, i_knot:(i_knot+1), :] = f(self.t_eval_vals_ord[i_knot, :], x[:, i_knot:(i_knot+1), :], f_params, bernstein=True)
                
                for k in range(d):
                    # Saves some time
                    int_vals = self.I_a(f_a_vals_tot[k, i_knot, :], alpha =  alpha[k], knot_sel = ('at', i_knot))
                    x[k, i_knot, :] = x_0[k] + int_vals_base[k, :]+ int_vals

                if sin_forcing_enabled:
                    # breakpoint()
                    # forcing_vals[i_knot, :] = -self.sin_I_a(self.t_eval_vals_ord[i_knot, :], alpha[1], f_params)
                    x[sin_forcing_params['dim'], i_knot, :] += - forcing_vals[i_knot, :]
                    # int_vals = self.I_a(f_a_vals_tot[k, :, :], alpha = alpha, knot_sel = ('to', i_knot))
                    # x[k, i_knot, :] = x_0[k] + int_vals #+int_vals_base[k, :]

                if video_plot:
                    plt.title("Iteration + "+str (n_tot_it))
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.plot(x[:,0], x[:,1])
                    plt.savefig(video_path+str(int(n_tot_it))+'.png', dpi = 300)
                    plt.close()

                if norm == 'sup':
                    it_norm = np.max(np.abs(x-x_prev))
                else:
                    it_norm = np.linalg.norm(x-x_prev)
                    # it_norm = np.sqrt(np.inner(x-x_prev, x-x_prev))
                
                if verbose:
                    print('Norm change', it_norm)
                if it_norm < tol:
                    if periodic and one_conv == False:
                        print("Converged IVP, running again for periodicity...")
                        one_conv = True
                    else:
                        if verbose:
                            print("Tolerance break!")
                        break
                elif it_norm > div_treshold:
                    print("Diverging with norm", it_norm, ", aborted!")
                    break

        toc = time.time()
        total_time = toc - tic
        time_per_it = total_time / n_tot_it
        print('~', n_tot_it, 'iterations,', np.round(total_time,4),'seconds,', np.round(time_per_it, 6), 's/it')
        print('~', str(np.round((n_tot_it/N_tot),2)), 'it/knot avg.,', np.round(time_per_it*(n_tot_it/N_tot), 6), 's/knot')

        if remember_x:
            self.x_storage = x

        output_dict = dict()

        output_dict['t'] = self.t_eval_vals_list
        output_dict['x'] = np.array([BernsteinSplines.a_to_vector(x[i, :, :]) for i in range(d)]).T
        output_dict['a'] = x
        output_dict['norm'] = {norm : it_norm}
        output_dict['n_it'] = n_tot_it
        output_dict['time'] = total_time
        output_dict['f_0'] = f_a_vals_tot[:, 0, 0]

        if video_plot:
            BernsteinSplines.write_video(video_path)
        
        return output_dict
    
    ### TODO: IMPLEMENT METHOD FOR CALCULATING HIGH RES FUNCTION VALUES!!! ###

    ### TODO: IMPLEMENT METHOD FOR CALCULATING DERIVATIVE VALUES!!! ###

    # def 

    @staticmethod
    def write_video(video_path, fps = 10):
        print('Generating video...')
        
        image_folder = video_path
        video_name = video_path+'/output_video.mp4'

        n_images = len([img for img in os.listdir(image_folder) if img.endswith(".png")])
        images = [str(i)+'.png' for i in range(1,n_images+1)]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()