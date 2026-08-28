from gpu_model.update_grid_3d_kernel import update_grid_3d_kernel
import numpy as np
def update_grid_3d_gpu_step(
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
    d_particle_type,
    d_conc3d,
    d_particle_mass_g,
    timesteps_per_hour,
    nx,
    ny,
    nz,
    include_nh4,
    d_dem,
    threads=256,
    seasonal_factor_step=None
):
    n_part = d_x.shape[1]
    if step < 0 or n_part <= 0:
        return
    blockspergrid_x = max(1, step + 1)
    blockspergrid_y = max(1, (n_part + threads - 1) // threads)
    update_grid_3d_kernel[(blockspergrid_x, blockspergrid_y), threads](
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
        d_particle_type,
        d_conc3d,
        d_particle_mass_g,
        timesteps_per_hour,
        nx,
        ny,
        nz,
        np.int32(int(include_nh4)),
        d_dem,
        seasonal_factor_step
    )