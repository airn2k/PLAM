from gpu_model.remove_outofbound_3d_kernel import remove_outofbound_3d_kernel
def remove_outofbound_3d_gpu_step(
    step,
    head,
    f_step,
    GRID_RES,
    DZ,
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
    threads=256,
):
    n_part = d_x.shape[1]
    if step <= 0 or n_part <= 0:
        return
    # 2D grid: (step, ceil(n_part/threads))
    blockspergrid_x = max(1, step)
    blockspergrid_y = max(1, (n_part + threads - 1) // threads)
    remove_outofbound_3d_kernel[(blockspergrid_x, blockspergrid_y), threads](
        step, head, f_step, GRID_RES, DZ, minx, miny, d_x, d_y, d_z, d_particle_mass_g, nx, ny, nz, d_dem, d_mass_budget
    )
