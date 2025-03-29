from .backend import np
from scipy.special import binom, comb

class BernsteinMethods:
    @staticmethod
    def build_bernstein_binom_basis(n):
        # Converts Bernstein basis polynomials into monomials
        # Basis has size (n+1 Bernstein basis functions) x (n+1 polynomial orders)
        binom_basis = np.zeros([n+1, n+1])
        for j in range(n+1):
            for l in range(j, n+1):
                binom_basis[j, l] = binom(n, l) * binom(l, j) * (-1)**(l-j)
        return binom_basis
    
    @staticmethod
    def build_C_matrix(n, m):
        # Construct the Bernstein basis multiplication matrix C
        # Results in an n+m polynomial
        C = np.zeros((n+m+1, n+1, m+1))
        # Fill in the C matrix
        for i in range(n+1):
            for j in range(m+1):
                k = i + j
                if k <= n + m:
                    C[k, i, j] = (binom(n, i) * binom(m, j)) / binom(n+m, k)
        return C
    
    @staticmethod
    def bernstein_projection_matrix(m, n):
        """
        Compute the projection matrix to downscale Bernstein coefficients
        from degree m (high) to degree n (low).
        """
        assert n < m, "Target degree n must be less than source degree m."
        P = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                P[i, j] = comb(m, j) * comb(j, i) * comb(m - j, n - i) / comb(m, n)
        return P
    
    @staticmethod
    def eval_t(A, t_knot_vals, t_eval, der=False):
        n = A.shape[1] - 1
        k_vals = np.arange(n+1)
        B = BernsteinMethods.build_bernstein_binom_basis(n)

        if np.isscalar(t_eval):
            t_eval = np.reshape(t_eval, [1])
        t_eval = np.array(t_eval)

        t_output = np.zeros(t_eval.shape)
        t_bounds = np.array([t_knot_vals[:-1].T, t_knot_vals[1:].T]).T
        i = 0
        for t in t_eval:
            mask = (t_bounds[:,0]<=t) & (t<t_bounds[:,1])
            if t == t_knot_vals[-1]:
                mask[-1] = True
            t_a, t_b = t_bounds[mask, 0], t_bounds[mask, 1]
            s = (t - t_a) / (t_b - t_a)
            
            A_knot = A[mask]
            if der == False:
                t_output[i] = A_knot @ B @ (s ** k_vals)
            else:
                s_der_vals = np.zeros(n+1)
                s_der_vals[1:] = k_vals[1:] * (s ** (k_vals[1:]-1))
                t_output[i] = A_knot @ B @ s_der_vals / (t_b-t_a)
            i+=1

        return t_output