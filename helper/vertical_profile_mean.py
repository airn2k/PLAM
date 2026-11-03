import numpy as np
from numba import njit, prange

def vertical_profile_mean(conc3d, thickness,x_or_y='x'):
    nz = conc3d.shape[2]
    nx = conc3d.shape[0]
    ny = conc3d.shape[1]

    if x_or_y=='x':
        vertical_profile = np.empty((conc3d.shape[0],conc3d.shape[2]), dtype=np.float64)
        y0 = int(np.floor(ny/2-thickness))
        y1 = int(np.ceil(ny/2+thickness))
        for i in prange(conc3d.shape[0]):
            for j in range(conc3d.shape[2]):
                s = 0
                for k in range(y0,y1):
                     s += conc3d[i, k, j]
                vertical_profile[i, j] = s / (y1 - y0)
    else:   
        vertical_profile = np.empty((conc3d.shape[1],conc3d.shape[2]), dtype=np.float64)
        x0 = int(np.floor(ny/2-thickness))
        x1 = int(np.ceil(ny/2+thickness))
        for i in prange(conc3d.shape[1]):
            for j in range(conc3d.shape[2]):
                s = 0
                for k in range(x0,x1):
                     s += conc3d[k, i, j]
                vertical_profile[i, j] = s / (x1 - x0)
    
                

    return vertical_profile