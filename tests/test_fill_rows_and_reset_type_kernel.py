from gpu_model.fill_rows_and_reset_type_kernel import fill_rows_and_reset_type_kernel
import numpy as np
from numba import cuda

def test_fill_rows_and_reset_type_kernel():
    # Test case 1:
    threads = 1024
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    z = np.array([[9.0, 10.0, 11.0, 12.0]], dtype=np.float32)
    particle_type = np.array([[1, 1, 1, 1]], dtype=np.int32)
    particle_mass_g = np.array([1.0], dtype=np.float32)
    row = 0
    mass_budget = np.zeros(5, dtype=np.float64)
    xval = cuda.to_device(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    yval = cuda.to_device(np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32))
    zval = cuda.to_device(np.array([90.0, 100.0, 110.0, 120.0], dtype=np.float32))

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_particle_type = cuda.to_device(particle_type)
    d_particle_mass_g = cuda.to_device(particle_mass_g)
    d_mass_budget = cuda.to_device(mass_budget)


    f_step = 4
    blocks = (f_step + threads - 1) // threads
    fill_rows_and_reset_type_kernel[blocks, threads](d_x, d_y, d_z,
                                                     d_particle_type,
                                                     d_particle_mass_g,
                                                     d_mass_budget,
                                                     np.int32(row),
                                                     xval,
                                                     yval,
                                                     zval)

    c_x = d_x.copy_to_host()
    c_y = d_y.copy_to_host()
    c_z = d_z.copy_to_host()
    c_particle_type = d_particle_type.copy_to_host()
    c_mass_budget = d_mass_budget.copy_to_host()

    assert np.allclose(c_x[row], xval.copy_to_host())
    assert np.allclose(c_y[row], yval.copy_to_host())
    assert np.allclose(c_z[row], zval.copy_to_host())
    assert np.all(c_particle_type[row] == 0)
