from gpu_model.zero_3d_kernel import zero_3d_kernel
from numba import cuda
import numpy as np

def test_zero_3d_kernel():
    # Test case 1:
    # Create a 3D array with non-zero values and zero it using the kernel
    threads = 1024
    nx = np.int32(10)
    ny = np.int32(10)
    nz = np.int32(10)
    d_array = cuda.to_device(np.ones((nx, ny, nz), dtype=np.float32))

    zero_3d_kernel[1, threads](d_array)

    c_array = d_array.copy_to_host()
    assert np.all(c_array == 0.0), "All elements should be zero after applying the kernel"