from numba import cuda
@cuda.jit
def zero_3d_kernel(arr):
    """Zero a 3-D device array using a flat 1-D grid."""
    k = cuda.grid(1)
    nx_l = arr.shape[0]
    ny_l = arr.shape[1]
    nz_l = arr.shape[2]
    total = nx_l * ny_l * nz_l
    if k < total:
        iz = k % nz_l
        rem = k // nz_l
        iy = rem % ny_l
        ix = rem // ny_l
        arr[ix, iy, iz] = 0.0


