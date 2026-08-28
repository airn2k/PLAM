from gpu_model.update_grid_3d_gpu_step import update_grid_3d_gpu_step
import numpy as np
from numba import cuda
def test_update_grid_3d_gpu_step():
    # Test case 1:
    # Particle has 4 coordinates for each step. On step 4, the particle is out of bounds in x (50.5 > nx*grid_res = 10*1.0 = 10.0) and should be removed (set to SENTINEL).
    threads = 1024
    f_step = np.int32(4)
    step = np.int32(3)
    head = np.int32(0)
    grid_res = np.float32(1.0)
    dz = np.float32(1.0)
    minx = np.float32(0.0)
    miny = np.float32(0.0)
    nx = np.int32(10)
    ny = np.int32(10)
    nz = np.int32(10)
    d_x = cuda.to_device(np.array([[0.0, 0.0, 0.0, 1.5]], dtype=np.float32))
    d_y = cuda.to_device(np.array([[0.0, 0.0, 0.0, 1.5]], dtype=np.float32))
    d_z = cuda.to_device(np.array([[0.0, 0.0, 0.0, 1.5]], dtype=np.float32))
    d_particle_type = cuda.to_device(np.array([[0, 0, 0, 0]], dtype=np.int32))
    d_conc3d = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.float32))
    d_particle_mass_g = cuda.to_device(np.array([1.0], dtype=np.float32))
    timesteps_per_hour = np.int32(60)
    include_nh4 = True
    d_dem = cuda.to_device(np.zeros((nx, ny), dtype=np.float32))
    seasonal_factor_step = cuda.to_device(np.array([1.0], dtype=np.float32))

    update_grid_3d_gpu_step(
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
        d_particle_type,
        d_conc3d,
        d_particle_mass_g,
        timesteps_per_hour,
        nx,
        ny,
        nz,
        include_nh4,
        d_dem,
        threads=threads,
        seasonal_factor_step=seasonal_factor_step
    )

    c_conc3d = d_conc3d.copy_to_host()
    assert c_conc3d[0,0,0] == d_particle_mass_g.copy_to_host()[0], "Element 3 should remain unchanged"