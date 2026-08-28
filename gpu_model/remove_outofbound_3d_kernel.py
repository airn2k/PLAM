from numba import cuda
from config.params import *
@cuda.jit
def remove_outofbound_3d_kernel(
    step, head, f_step, GRID_RES, DZ, minx, miny, x, y, z, particle_mass_g, nx, ny, nz, dem, mass_budget
):
    j = cuda.blockIdx.x
    k = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.y
    n_part = x.shape[1]
    if j >= step:
        return
    if k >= n_part:
        return
    idx = head - 1 - j
    if idx < 0:
        idx += f_step
    SENTINEL = -9999.0
    if x[idx, k] == SENTINEL:
        return
    ix = int((x[idx, k] // GRID_RES) - int(minx / GRID_RES))
    iy = int((y[idx, k] // GRID_RES) - int(miny / GRID_RES))
    # Terrain-following: iz is AGL
    local_dem = 0.0
    if 0 <= ix < nx and 0 <= iy < ny:
        local_dem = float(dem[ix, iy])
    iz = int((z[idx, k] - local_dem) / DZ)
    horizontal_oob = not (0 <= ix < nx and 0 <= iy < ny)
    vertical_oob = not (0 <= iz < nz)
    if horizontal_oob or vertical_oob:
        if horizontal_oob:
            cuda.atomic.add(mass_budget, MASS_BUDGET_OOB_HORIZONTAL, particle_mass_g[idx])
        if vertical_oob:
            cuda.atomic.add(mass_budget, MASS_BUDGET_OOB_VERTICAL, particle_mass_g[idx])
        x[idx, k] = SENTINEL
        y[idx, k] = SENTINEL
        z[idx, k] = SENTINEL