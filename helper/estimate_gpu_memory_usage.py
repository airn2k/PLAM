
def estimate_gpu_memory_usage(n_particles, f_step, n_arrays=5, extra_bytes=0):
    """
    Estimate GPU memory usage for main particle arrays.
    n_arrays: number of float32 arrays (x, y, z, mass, etc.)
    extra_bytes: add any extra bytes for other arrays (e.g., depo_grid, conc3d)
    Returns usage in bytes, MB, and GB.
    """
    bytes_per_array = f_step * n_particles * 4
    total_bytes = n_arrays * bytes_per_array + extra_bytes
    mb = total_bytes / (1024**2)
    gb = total_bytes / (1024**3)
    return total_bytes, mb, gb

