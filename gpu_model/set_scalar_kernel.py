from numba import cuda
@cuda.jit
def set_scalar_kernel(arr, idx, value):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        arr[idx] = value


