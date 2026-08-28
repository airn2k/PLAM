from gpu_model.remove_outofbound_3d_gpu_step import remove_outofbound_3d_gpu_step
import numpy as np
from numba import cuda
from config.params import *
def test_remove_outofbound_3d_gpu_step():
    # Test case 1:
    # Particle has 4 coordinates for each step. On step 4, the particle is out of bounds in x (50.5 > nx*grid_res = 10*1.0 = 10.0) and should be removed (set to SENTINEL).
    threads = 1024
    f_step = np.int32(4)
    step = np.int32(4)
    head = np.int32(0)
    grid_res = np.int32(1)
    dz = np.float32(1.0)
    minx = np.float32(0.0)
    miny = np.float32(0.0)
    nx = np.int32(10)
    ny = np.int32(10)
    nz = np.int32(10)
    d_x = cuda.to_device(np.array([[0.5, 1.5, 2.5, 50.5]], dtype=np.float32))
    d_y = cuda.to_device(np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32))
    d_z = cuda.to_device(np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32))
    d_particle_mass_g = cuda.to_device(np.array([1.0], dtype=np.float32))
    d_dem = cuda.to_device(np.zeros((10,10), dtype=np.float32))
    d_mass_budget = cuda.to_device(np.zeros(5, dtype=np.float64))

    remove_outofbound_3d_gpu_step(
    step,
    head,
    f_step,
    grid_res,
    dz,
    minx,
    miny,
    d_x,
    d_y,
    d_z,
    d_particle_mass_g,
    nx,
    ny,
    nz,
    d_dem,
    d_mass_budget,
    threads=threads
    )

    c_particle_positions = d_x.copy_to_host()
    assert c_particle_positions[0,3] == SENTINEL, "Particle 3 should be removed (out of bounds in x)"