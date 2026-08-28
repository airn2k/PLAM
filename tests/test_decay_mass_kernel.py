from gpu_model.decay_mass_kernel import decay_mass_kernel
import numpy as np
from numba import cuda

def test_decay_mass_kernel():
    # Test case 1:
    threads = 1024
    decay_factor = 0.5
    particle_mass_g = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_particle_mass_g = cuda.to_device(particle_mass_g)
    f_step = 4
    decay_blocks = (f_step + threads - 1) // threads
    decay_mass_kernel[decay_blocks, threads](d_particle_mass_g, decay_factor, np.int32(f_step))
    c_particle_mass_g = d_particle_mass_g.copy_to_host()
    assert np.allclose(c_particle_mass_g, particle_mass_g * decay_factor)

