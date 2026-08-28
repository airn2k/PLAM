from gpu_model.set_scalar_kernel import set_scalar_kernel
import numpy as np
from numba import cuda

def test_set_scalar_kernel():
    # Test case 1:
    # Set all elements of a 1D array to a scalar value
    threads = 1024
    n = np.int32(10)
    scalar_value = np.float32(5.0)
    d_array = cuda.to_device(np.zeros(n, dtype=np.float32))

    set_scalar_kernel[1, threads](d_array, 0, scalar_value)

    c_array = d_array.copy_to_host()
    assert c_array[0] == scalar_value, "Element 0 should be set to the scalar value"