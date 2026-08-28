import numpy as np
from numba import cuda
@cuda.jit
def fill_rows_kernel(x, y, z, row, xval, yval, zval):
    k = cuda.grid(1)
    n = x.shape[1]
    if k < n:
        x[row, k] = xval[k]
        y[row, k] = yval[k]
        z[row, k] = zval[k]


