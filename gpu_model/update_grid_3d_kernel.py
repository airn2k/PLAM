from numba import cuda
@cuda.jit
def update_grid_3d_kernel(
    step,
    head,
    f_step,
    GRID_RES,
    DZ,
    minx,
    miny,
    x,
    y,
    z,
    particle_type,
    conc3d,
    particle_mass_g,
    timesteps_per_hour,
    nx,
    ny,
    nz,
    include_nh4,
    dem,
    seasonal_factor_step,
):
    j = cuda.blockIdx.x
    k = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.y
    n_part = x.shape[1]
    if j > step or k >= n_part:
        return
    idx = head - 1 - j
    if idx < 0:
        idx += f_step
    SENTINEL = -9999.0
    if x[idx, k] == SENTINEL:
        return
    # Count NH3 only, or NH3+NH4 depending on include_nh4 flag.
    if include_nh4 == 0 and particle_type[idx, k] != 0:
        return
    ix = int((x[idx, k] // GRID_RES) - int(minx / GRID_RES))
    iy = int((y[idx, k] // GRID_RES) - int(miny / GRID_RES))
    # Terrain-following: iz is AGL (z minus local DEM height)
    local_dem = 0.0
    if 0 <= ix < nx and 0 <= iy < ny:
        local_dem = float(dem[ix, iy])

    height_agl = z[idx, k] - local_dem
    iz = int(height_agl / DZ)
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        voxel_vol = GRID_RES * GRID_RES * DZ
        val = (particle_mass_g[idx]*seasonal_factor_step[k]) / voxel_vol
        cuda.atomic.add(conc3d, (ix, iy, iz), val)
