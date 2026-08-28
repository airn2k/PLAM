from gpu_model.fill_rows_kernel import fill_rows_kernel
import numpy as np


def test_fill_rows_kernel():
    # Define the input arrays
    from numba import cuda
    x = cuda.to_device(np.zeros((3, 4), dtype=np.float32))
    y = cuda.to_device(np.zeros((3, 4), dtype=np.float32))
    z = cuda.to_device(np.zeros((3, 4), dtype=np.float32))

    # Define the values to fill
    xval = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    yval = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    zval = np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32)

    # Define the row to fill
    row = np.int32(1)

    # Launch the kernel
    threads_per_block = 128
    blocks_per_grid = (x.shape[1] + (threads_per_block - 1)) // threads_per_block
    fill_rows_kernel[blocks_per_grid, threads_per_block](x, y, z, row, xval, yval, zval)

    # Copy the results back to host memory
    x_host = x.copy_to_host()
    y_host = y.copy_to_host()
    z_host = z.copy_to_host()

    # Check if the specified row has been filled correctly
    assert np.all(x_host[row] == xval)
    assert np.all(y_host[row] == yval)
    assert np.all(z_host[row] == zval)