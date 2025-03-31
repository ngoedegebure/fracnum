from dotenv import load_dotenv
import os

load_dotenv('environment.env')  # Load variables from .env

def str_to_bool(value):
    return value.lower() in ("true", "1", "yes", "on") if value else False

BACKEND_ENGINE = os.getenv("BACKEND_ENGINE", "numpy")
NP_EINSUM = str_to_bool(os.getenv("NP_EINSUM_FOR_DOT", "0"))
OPT_EINSUM = str_to_bool(os.getenv("NP_OPT_EINSUM", "0"))
NUMBA_PARALLEL = str_to_bool(os.getenv("NUMBA_PARALLEL", "1"))

if BACKEND_ENGINE == "cupy":
    import cupy as np    
    from cupy import einsum
else:
    import numpy as np
    if OPT_EINSUM:
        from opt_einsum import contract as einsum
    else:
        from numpy import einsum

    if BACKEND_ENGINE == "numba":
        from numba import jit

        @jit(nopython=True, parallel=NUMBA_PARALLEL)
        def dot_numba(A, B):
            # l,ln->n
            if A.shape[-1] != B.shape[-2]:
                raise ValueError("Shape mismatch: last axis of `a` must match second-to-last axis of `b`")
            
            b_shape = B.shape

            # Determine the shape of the resulting tensor
            result_shape = b_shape[-1]

            # Initialize the result tensor with zeros
            result = np.zeros(result_shape)
            
            for l in range(b_shape[0]):
                for n in range(result_shape):
                    result[n]+=A[l] * B[l, n]
            
            return result

        @jit(nopython=True, parallel=NUMBA_PARALLEL)
        def tensordot_numba(A, B):
            # TODO: ACTUALLY MAKE GENERAL, NOT JUST FOR SIZE 2!
            # kl,kln->n
            if A.shape[-1] != B.shape[-2]:
                raise ValueError("Shape mismatch: last axis of `a` must match second-to-last axis of `b`")
            
            b_shape = B.shape

            # Determine the shape of the resulting tensor
            result_shape = b_shape[-1]

            # Initialize the result tensor with zeros
            result = np.zeros(result_shape)
            
            for k in range(b_shape[0]):
                for l in range(b_shape[1]):
                    for n in range(result_shape):
                        result[n]+=A[k,l] * B[k,l,n]
            
            return result

def int_contract(A, B, einsum_dims = "", tensordot_axes=""):
    if BACKEND_ENGINE == "numba":
        n_axes = len(A.shape)
        if n_axes == 1:
            return dot_numba(A,B)    
        elif n_axes == 2:
            return tensordot_numba(A,B)
        else:
            print("WARNING: no numba implementation of tensordot for size {n_axes} yet!")
            assert 0
    else:
        if NP_EINSUM:
            return einsum(einsum_dims, A, B)
        else: 
            return np.tensordot(A, B, axes = tensordot_axes)