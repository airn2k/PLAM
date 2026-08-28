from numba import cuda
from config.params import *

@cuda.jit
def fill_rows_and_reset_type_kernel(x, y, z, particle_type, particle_mass_g, mass_budget, row, xval, yval, zval):
    k = cuda.grid(1)
    n = x.shape[1]
    if k < n:
        if x[row, k] != SENTINEL:
            cuda.atomic.add(mass_budget, MASS_BUDGET_AGED_OUT, particle_mass_g[row])
        x[row, k] = xval[k]
        y[row, k] = yval[k]
        z[row, k] = zval[k]
        particle_type[row, k] = 0  # Reset to NH3
