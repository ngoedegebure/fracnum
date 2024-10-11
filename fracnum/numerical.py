import numpy as np
import math
from scipy.special import comb
from scipy.optimize import root as scipy_root
from scipy.special import binom
from scipy.special import gamma as gammafunction
from scipy.special import gamma, hyp2f1

from scipy.sparse import csr_array

def ivp_int(f, alpha, u_0, t_limits, dt, method = "CN"):
    N = int(np.abs(t_limits[1] -  t_limits[0])/dt)
    t_vals = np.linspace(t_limits[0], t_limits[1], num = N)
    u_sols = np.zeros([N, len(u_0)])
    u_sols[0] = u_0
    I_old = 0
    kernel_new = np.array([0])
    for i in range(1, N):
        if method == "FE":
            kernel = (t_vals[i-1] + dt/2 - t_vals[:i])**(alpha-1)

            u_sols_avg = np.roll((u_sols[:i].T + np.roll(u_sols[:i].T, shift=-1))/2, shift=1)
            u_sols_avg[:, 0] = u_0

            I_i = f(t_vals[:i]+dt/2, u_sols_avg)@kernel*dt
            u_sols[i] = u_sols[i-1] + (I_i-I_old)
            I_old = I_i
        elif method == "BE":
            kernel = kernel_new
            
            u_sols_avg = np.roll((u_sols[:i].T + np.roll(u_sols[:i].T, shift=-1))/2, shift=1)
            u_sols_avg[:, 0] = u_0

            I_i = f(t_vals[:i]+dt/2, u_sols_avg)@kernel*dt
            
            kernel_new = (t_vals[i] + dt/2 - t_vals[:(i+1)])**(alpha-1)

            def BE_root_problem(u_new):
                return u_new - (u_sols[i-1] + (f(t_vals[:(i+1)]+dt/2, np.r_[u_sols_avg.T, [(u_new + u_sols_avg[:, -1])/2]].T)@kernel_new*dt - I_i))
            
            root_res = scipy_root(BE_root_problem, u_sols[i-1])

            if root_res.success == False:
                print("! ROOT FINDING FAILED FOR alpha = ", alpha)
                print(root_res)
                break

            u_sols[i] = root_res.x
        elif method == "RK4":
            # TODO: IMPLEMENT PROPER RK4 SCHEME HERE!
            k_1 = f(0, u_sols[i-1])
            k_2 = f(0, u_sols[i-1] + dt*k_1/2)
            k_3 = f(0, u_sols[i-1] + dt*k_2/2)
            k_4 = f(0, u_sols[i-1] + dt*k_3)
            
            u_sols[i] = u_sols[i-1] + dt/6 * (k_1 + 2*k_2+2*k_3+k_4)
        elif method == "CN":
            # kernel = kernel_new
            
            # u_sols_avg = np.roll((u_sols[:i].T + np.roll(u_sols[:i].T, shift=-1))/2, shift=1)
            # u_sols_avg[:, 0] = u_0

            # I_i = f(t_vals[:i]+dt/2, u_sols_avg)@kernel*dt
            
            # kernel_new = (t_vals[i] + dt/2 - t_vals[:(i+1)])**(alpha-1)

            # def CN_root_problem(u_new):
            #     return u_new - u_sols[i-1] - 0.5* ((I_i-I_old) + (f(t_vals[:(i+1)]+dt/2, np.r_[u_sols_avg.T, [(u_new + u_sols_avg[:, -1])/2]].T)@kernel_new*dt - I_i))

            kernel = kernel_new
            I_i = f(t_vals[:i]+dt/2, u_sols[:i].T)@kernel*dt
            kernel_new = (t_vals[i] + dt/2 - t_vals[:(i+1)])**(alpha-1)

            def CN_root_problem(u_new):
                u_sols_with_new = np.c_[u_sols[:i].T, u_new].T
                return u_new - u_sols[i-1] - 0.5* ((I_i-I_old) + (f(t_vals[:(i+1)]+dt/2, u_sols_with_new.T)@kernel_new*dt - I_i))

            root_res = scipy_root(CN_root_problem, u_sols[i-1])

            if root_res.success == False:
                print("! INTEGRATION ROOT FINDING FAILED FOR alpha = ", alpha)
                print(root_res)
                break

            u_sols[i] = root_res.x

            I_old = I_i
        elif method == "CN_mid":
            kernel = kernel_new
            
            u_sols_avg = np.roll((u_sols[:i].T + np.roll(u_sols[:i].T, shift=-1))/2, shift=1)
            u_sols_avg[:, 0] = u_0

            I_i = f(t_vals[:i]+dt/2, u_sols_avg)@kernel*dt
            
            kernel_new = (t_vals[i] + dt/2 - t_vals[:(i+1)])**(alpha-1)

            def CN_root_problem(u_new):
                return u_new - u_sols[i-1] - 0.5* ((I_i-I_old) + (f(t_vals[:(i+1)]+dt/2, np.r_[u_sols_avg.T, [(u_new + u_sols_avg[:, -1])/2]].T)@kernel_new*dt - I_i))
            
            root_res = scipy_root(CN_root_problem, u_sols[i-1])

            if root_res.success == False:
                print("! INTEGRATION ROOT FINDING FAILED FOR alpha = ", alpha)
                print(root_res)
                break

            u_sols[i] = root_res.x

            I_old = I_i
        elif method == "L1":
            # NOTE: FROM CHATGPT
            # Initialize parameters
            T = t_limits[1]          # Final time
            N = int(T / dt) + 1      # Number of time steps
            d = len(u_0)             # Dimension of the system
            t = np.linspace(0, T, N) # Time discretization

            # Initialize solution array
            u = np.zeros((N, d))
            u[0] = u_0               # Set initial condition

            # Precompute the weights w_n for the L1 scheme
            w = np.zeros(N - 1)
            for n in range(1, N):
                w[n - 1] = (n ** (1 - alpha) - (n - 1) ** (1 - alpha))

            # Solve the fractional differential equation using the L1 scheme
            for n in range(1, N):
                # Calculate the Caputo fractional derivative approximation
                frac_deriv = np.zeros(d)
                for k in range(n):
                    frac_deriv += w[k] * (u[n - k] - u[n - k - 1]) / dt

                frac_deriv *= (1 / gammafunction(2 - alpha))  # Apply the scaling factor for Caputo derivative

                # Update the solution using the fractional derivative
                u[n] = u[n - 1] + dt * f(t[n - 1], u[n - 1]) + dt ** alpha * frac_deriv

            u_sols = u
    return u_sols

def ivp_diethelm(f, alpha, u_0, t_limits, dt, f_params = None):
    #### TODO! MAKE FOR TWO ALPHA's!
    T = t_limits[1]
    h = dt #
    N = int(T/h)
    m = np.ceil(alpha)

    d = np.array(u_0).size

    b = np.zeros(N+1)
    a = np.zeros(N+1)
    for k in range(1, N+1):
        b[k] = k**alpha - (k-1)**alpha
        a[k] = (k+1)**(alpha+1) - 2 * k**(alpha+1) + (k-1)**(alpha+1)

    y = np.zeros([N+1, len(u_0)])
    y[0] = u_0

    for j in range(1, N+1):
        # breakpoint()
        p = u_0 + h**alpha / gammafunction(alpha+1) * np.flip(b[1:j+1]) @ f(0, y[:j, :], params = f_params)
        y[j] = u_0 + h**alpha / gammafunction(alpha+2) * (
            f(0, np.array([p]), params = f_params) + ((j-1)**(alpha+1)-(j-1-alpha)*j**alpha)*f(0,np.array([u_0]), params = f_params)
            +
            np.flip(a[2:j+1]) @ f(0, y[1:j, :], params=f_params)
        )

    return y

# def I_bernstein(t, N, a, alpha = 0):
    t_span = np.abs(np.max(t) - np.min(t))
    t = t / t_span
    n = N - 1
    bernstein_alpha = np.array([binom(n, nu) * sum([gammafunction(k+nu+1) / gammafunction(alpha + k + nu + 1) * (-1)**(k)*binom(n-nu, k)*t**(k+nu + alpha) for k in range(n-nu+1)]) for nu in range(N)])
    # print('a.shape', a.shape)
    # print('bernstein_alpha.shape', bernstein_alpha.shape)
    # print('np.sum(bernstein_alpha[:,-1])', np.sum(bernstein_alpha[:,-1]))
    # print('np.sum(bernstein_alpha[:,0])', np.sum(bernstein_alpha[:,0]))

    return a@bernstein_alpha

def I_trap(u, alpha, t_limits, dt):
    return dt**(1-alpha)/(gammafunction(2-alpha) * 2) * np.sum(u[1:] + u[:-1])

def I_simple_functions(u, alpha, dt):
    N = u.shape[0]
    if N == 0:
        return 0
    t_vals = np.linspace(0, dt*N, num = N)
    T = t_vals[-1]

    result = 1/(alpha*2) * ((T-t_vals[:-1])**alpha - (T-t_vals[1:])**alpha) @ (u[:-1] + u[1:])
    return result

def A_cd(N, dt):
    # Construct A #
    A = np.zeros([N, N])

    for i in range(N):
        if i == 0:
            A[0, 0]   = -1 / dt
            A[0, 1] = 1 / dt
        elif i == N - 1:
            A[-1, -2] = - 1/dt
            A[-1, -1] = 1/dt
        else:
            A[i, i-1] = -1 / (2*dt)
            A[i, i+1] =  1 / (2*dt)

    return A

def J_rl_trap(N, alpha_int, t_vals_int):
    # Construct J #
    J = np.zeros([N, N])
    T = t_vals_int[-1]

    # J[0, 0] += ((T-t_vals_int[i-1])**alpha_int - (T-t_vals_int[i])**alpha_int)

    for j in range(1,N):
        for i in range(1, j+1):
            # breakpoint()
            J[j, i-1] += ((T-t_vals_int[i-1])**alpha_int - (T-t_vals_int[i])**alpha_int)
            J[j, i] += ((T-t_vals_int[i-1])**alpha_int - (T-t_vals_int[i])**alpha_int)

    J = J * 1 / (2*alpha_int)

    return J


def J_rl_rect(N, alpha_int, t_vals_int, sparse = True):
    # Construct J #
    J = np.zeros([N, N])
    T = t_vals_int[-1]
    h = t_vals_int[1] - t_vals_int[0]

    # J[0, 0] += ((T-t_vals_int[i-1])**alpha_int - (T-t_vals_int[i])**alpha_int)

    for n in range(1,N):
        for j in range(n):
            J[n,j] = h**alpha_int / alpha_int * ((n - j)**alpha_int - (n-1-j)**alpha_int)
    J = J*1/gamma(alpha_int) # CHECK THIS GAMMA PART!!!!
    if sparse:
        J = csr_array(J)
    return J

# ALL BELOW MOVED TO BERNSTEINSPLINES CLASS

# class bernstein:
#     @staticmethod
#     def I_a_b(t, alpha, k, a, b):
#         if np.all(np.array(t-b)==0): # if t and b match
#             IBP_1 = ((a**k*(t-a)**alpha))/alpha
#             IBP_2 = (1/alpha) * ((t**(k+alpha)*hyp2f1(k, -alpha, k+1, 1)) - (a**k*(1-a/t)**(-alpha)*(t-a)**alpha*hyp2f1(k, -alpha, k+1, a/t)))
#         else:
#             IBP_1 = ((-b**k*(t-b)**alpha) + (a**k*(t-a)**alpha))/alpha
#             IBP_2 = (1/alpha) * ((b**k*(1-b/t)**(-alpha)*(t-b)**alpha*hyp2f1(k, -alpha, k+1, b/t)) - (a**k*(1-a/t)**(-alpha)*(t-a)**alpha*hyp2f1(k, -alpha, k+1, a/t)))

#         return 1/gamma(alpha) * (IBP_1 + IBP_2)

#     @staticmethod
#     def I_loc(t_vals, alpha, k, support):
#         a, b = support
        
#         t_pre = t_vals[t_vals <= a]
#         t_in = t_vals[(t_vals > a) & (t_vals <=b)]
#         t_post = t_vals[(t_vals > b)]

#         I_pre = np.zeros(t_pre.shape)
        
#         I_in = bernstein.I_a_b(t_in, alpha, k, a, t_in)

#         I_post = bernstein.I_a_b(t_post, alpha, k, a, b) 

#         return np.concatenate([I_pre, I_in, I_post])
    
#     def I_splines_tot(a, alpha, t_knot_vals, n_p, t_eval = None):
#         N_blocks = len(t_knot_vals) - 1
#         N_tot = N_blocks * (n_p+1)-(N_blocks-1)
#         T = t_knot_vals[-1]
#         t_vals = np.linspace(0,T, N_tot)
        
#         if t_eval is None:
#             t_eval = t_vals

#         I_vals = np.zeros(t_vals.shape)

#         for i_b in range(N_blocks):
#             t_a = t_knot_vals[i_b]
#             t_b = t_knot_vals[i_b+1]
#             # print('t_a, t_b',t_a, t_b)
            
#             # get all polynomials
#             for i_p in range(n_p + 1):
#                 i_t = i_b * (n_p) + i_p  # t index
#                 a_sel = a[i_t]

#                 coeff = np.zeros(n_p+1)
#                 for l in range(i_p, n_p+1):
#                     for k in range(0,l+1):
#                         # check -2 !!!!!!!!!!!
#                         # print(n_p-k)
#                         coeff[l-k]+=(1/(t_b-t_a))**l*binom(n_p,l)*binom(l,i_p)*(-1)**(l-i_p)*binom(l,k)*(-t_a)**k

#                 for k_order in range(n_p+1):
#                     # breakpoint()
#                     if a_sel*coeff[k_order] != 0:
#                         if alpha !=0:
#                             I_vals+=a_sel*coeff[k_order]*bernstein.I_loc(t_vals, alpha, k_order, [t_a, t_b])
#                         else:
#                             if i_b == N_blocks-1:
#                                 I_vals+=a_sel*coeff[k_order]*(t_vals**k_order)*((t_a<=t_vals)&(t_vals<=t_b))
#                             else:
#                                 I_vals+=a_sel*coeff[k_order]*(t_vals**k_order)*((t_a<=t_vals)&(t_vals<t_b))

#         return t_vals, I_vals

#     @staticmethod
#     def bernstein_multiply(a, b):
#         """
#         Multiply two Bernstein polynomials f(x) and g(x) given their coefficients.

#         Args:
#         - a: List of coefficients of the first polynomial f(x) in the Bernstein basis.
#         - b: List of coefficients of the second polynomial g(x) in the Bernstein basis.

#         Returns:
#         - c: List of coefficients of the product polynomial f(x) * g(x) in the Bernstein basis.
#         """
#         # Both polynomials must have the same degree N
#         N = len(a) - 1

#         # Initialize the coefficients of the product polynomial of degree 2N
#         d = [0] * (2 * N + 1)

#         # Compute the coefficients of the product polynomial
#         for k in range(2 * N + 1):
#             d_k = 0
#             for i in range(max(0, k - N), min(N, k) + 1):
#                 binom_N_i = comb(N, i)
#                 binom_N_k_i = comb(N, k - i)
#                 binom_2N_k = comb(2 * N, k)

#                 d_k += (binom_N_i * binom_N_k_i / binom_2N_k) * a[i] * b[k - i]

#             d[k] = d_k

#         # Downscale the polynomial of degree 2N to degree N
#         c = [0] * (N + 1)
#         for i in range(N + 1):
#             c_i = 0
#             for k in range(2 * N + 1):
#                 # Compute Bernstein basis polynomial B_{k, 2N} evaluated at i / N
#                 B_k_2N = comb(2 * N, k) * (i / N) ** k * (1 - i / N) ** (2 * N - k)
#                 c_i += d[k] * B_k_2N

#             c[i] = c_i

#         return np.array(c)
