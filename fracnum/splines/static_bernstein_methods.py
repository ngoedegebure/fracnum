import numpy as np
from scipy.special import binom

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