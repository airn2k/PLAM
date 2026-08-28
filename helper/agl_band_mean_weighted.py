import numpy as np
from numba import njit, prange

@njit(parallel=True)
def agl_band_mean_weighted(conc3d, ground_height_grid, dz, agl_top_m):
    nz = conc3d.shape[2]

    lower_nz = np.floor(ground_height_grid / dz).astype(np.int32)
    upper_nz = np.ceil((ground_height_grid + agl_top_m) / dz).astype(np.int32)

    height_avg = np.empty(conc3d.shape[:2], dtype=np.float64)

    for i in prange(conc3d.shape[0]):
        for j in range(conc3d.shape[1]):
            z0 = lower_nz[i, j]
            z1 = upper_nz[i, j]

            if z0 < 0:
                z0 = 0
            elif z0 > nz:
                z0 = nz

            if z1 < 0:
                z1 = 0
            elif z1 > nz:
                z1 = nz

            if z1 > z0:
                s = 0.0
                for k in range(z0, z1):
                    s += conc3d[i, j, k]
                height_avg[i, j] = s / (z1 - z0)
            else:
                height_avg[i, j] = conc3d[i, j, z0]

    return height_avg