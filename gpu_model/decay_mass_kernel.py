from numba import cuda
@cuda.jit
def decay_mass_kernel(mass, factor, n):
    """Multiply all mass entries by a decay factor (avoids host↔device round-trip)."""
    k = cuda.grid(1)
    if k < n:
        mass[k] *= factor
