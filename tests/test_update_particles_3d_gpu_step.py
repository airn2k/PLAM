from gpu_model.update_particles_3d_gpu_step import update_particles_3d_gpu_step
import numpy as np
from numba import cuda
from config.params import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

def test_update_particles_3d_gpu_step():
    f_step = np.int32(1)
    step_in_window = np.int32(0)
    head = np.int32(1)
    threads = 1024
    d_x = cuda.to_device(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    d_y = cuda.to_device(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    d_z = cuda.to_device(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    particle_type = np.array([[1, 1, 1, 1]], dtype=np.int32)
    particle_mass_g = np.array([1.0], dtype=np.float32)
    d_particle_type = cuda.to_device(particle_type)
    d_particle_mass_g = cuda.to_device(particle_mass_g)
    u_ref = np.float32(1.0)
    v_ref = np.float32(1.0)
    z_ref = np.float32(10.0)
    z0 = np.float32(0.03)
    blh= np.float32(500.0)

    # For initial testing, turbulence factors are set to zero, so the particles will only move according to the mean wind and settling velocity. This allows us to verify that the basic movement of particles is working correctly.
    Kxy = np.float32(0.0)
    Kz_low = np.float32(0.0)
    Kz_mid = np.float32(0.0)
    Kz_high = np.float32(0.0)

    dt = np.float32(1.0)
    P_dep_surf = np.float32(0.01)
    P_dep_nh4 = np.float32(0.01)
    P_conversion = np.float32(0.0)
    P_wet_dep = np.float32(0.01)
    w_mean = np.float32(0.0)
    w_settle = np.float32(0.0)
    timesteps_per_hour = np.int32(3600)
    d_depo = cuda.to_device(np.zeros((10,10), dtype=np.float32))   
    grid_res = np.float32(1.0)
    minx = np.float32(0.0)
    miny = np.float32(0.0)
    nx = np.int32(10)
    ny = np.int32(10)
    dz = np.float32(50.0)
    z_cap = np.float32(1000.0)
    blocks = (1 + threads - 1) // threads
    rng_states = create_xoroshiro128p_states(blocks * threads, seed=12345)
    d_mass_budget = cuda.to_device(np.zeros(5, dtype=np.float64))
    d_dem = cuda.to_device(np.zeros((10,10), dtype=np.float32))
    dem_nx = np.int32(10)
    dem_ny = np.int32(10)
    dem_minx = np.float32(0.0)
    dem_miny = np.float32(0.0)
    dem_dx = np.float32(1.0)
    dem_dy = np.float32(1.0)
    d_dh_dx = cuda.to_device(np.zeros((10,10), dtype=np.float32))
    d_dh_dy = cuda.to_device(np.zeros((10,10), dtype=np.float32))
    initial_release_vz = np.zeros((4,), dtype=np.float32)
    d_initial_release_vz = cuda.to_device(initial_release_vz)
    seasonal_factor_step=np.float32(1.0)
    d_seasonal_factor_step = cuda.to_device(np.array([seasonal_factor_step], dtype=np.float16))
    sshf = np.float32(100.0)  # Example value for surface sensible heat flux 


    update_particles_3d_gpu_step(
        step_in_window,
        head,
        f_step,
        d_x,
        d_y,
        d_z,
        d_particle_type,
        u_ref,
        v_ref,
        z_ref,
        z0,
        blh,
        Kxy,
        Kz_low,
        Kz_mid,
        Kz_high,
        dt,
        P_dep_surf,
        P_dep_nh4,
        P_conversion,
        P_wet_dep,
        w_mean,
        w_settle,
        d_particle_mass_g,
        timesteps_per_hour,
        d_depo,
        grid_res,
        minx,
        miny,
        nx,
        ny,
        dz,
        z_cap,
        rng_states,
        blocks,
        threads,
        sshf,
        d_mass_budget,
        d_dem,
        dem_nx,
        dem_ny,
        dem_minx,
        dem_miny,
        dem_dx,
        dem_dy,
        d_dh_dx,
        d_dh_dy,
        d_initial_release_vz,
        d_seasonal_factor_step
    )

    assert d_x is not None
    assert not np.array_equal(d_x.copy_to_host(), np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    # Particles will move 1 units in x and y direction
    assert np.allclose(d_x.copy_to_host(), np.array([[2.0, 1.0, 1.0, 1.0]], dtype=np.float32), atol=1e-5)
    assert np.allclose(d_y.copy_to_host(), np.array([[2.0, 1.0, 1.0, 1.0]], dtype=np.float32), atol=1e-5)
    assert np.allclose(d_z.copy_to_host(), np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), atol=1e-5)
    print("Test passed: update_particles_3d_gpu_step executed successfully and updated particle positions.")
    print('Initial x positions:', np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    print("Updated x positions:", d_x.copy_to_host())
    print('Initial y positions:', np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    print("Updated y positions:", d_y.copy_to_host())
    print('Initial z positions:', np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    print("Updated z positions:", d_z.copy_to_host())