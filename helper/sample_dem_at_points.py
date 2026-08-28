import numpy as np

def sample_dem_at_points(xs, ys, dem_array, transform):
    """
    Fast DEM sampling using direct coordinate transformation.
    Returns array of elevations (same shape as xs/ys).
    """
    # Convert coordinates to pixel indices using transform matrix
    # transform = [pixel_width, 0, x_origin, 0, -pixel_height, y_origin]
    pixel_width = transform[0]
    pixel_height = -transform[4]  # Note: negative because y increases downward
    x_origin = transform[2]
    y_origin = transform[5]
    
    # Calculate pixel coordinates
    cols = ((xs - x_origin) / pixel_width).astype(int)
    rows = ((y_origin - ys ) / pixel_height).astype(int)
    
    # Clip to valid bounds
    rows = np.clip(rows, 0, dem_array.shape[0] - 1)
    cols = np.clip(cols, 0, dem_array.shape[1] - 1)
    
    return dem_array[rows, cols]