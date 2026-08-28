import numpy as np
import geopandas as gpd
def sample_points_in_polygon(polygon, num_points):
    """Regular grid sampling for fast, systematic point generation."""
    minx, miny, maxx, maxy = polygon.bounds
    
    # Calculate grid spacing for systematic sampling
    area = (maxx - minx) * (maxy - miny)
    grid_spacing = np.sqrt(area / (num_points * 1.5))  # 1.5x oversample for filtering
    
    # Generate regular grid
    x_coords = np.arange(minx + grid_spacing/2, maxx, grid_spacing)
    y_coords = np.arange(miny + grid_spacing/2, maxy, grid_spacing)
    
    # Create meshgrid and flatten
    X, Y = np.meshgrid(x_coords, y_coords)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Filter points inside polygon (vectorized)
    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_flat, y_flat))
    inside_mask = points_gdf.within(polygon)
    
    valid_x = x_flat[inside_mask]
    valid_y = y_flat[inside_mask]
    
    # Take exactly num_points (repeat if needed, truncate if too many)
    if len(valid_x) == 0:
        # Fallback to centroid if no points found
        centroid = polygon.centroid
        return np.array([[centroid.x, centroid.y]] * num_points)
    elif len(valid_x) >= num_points:
        # Take first num_points
        return np.column_stack([valid_x[:num_points], valid_y[:num_points]])
    else:
        # Repeat pattern to get num_points
        repeat_factor = (num_points // len(valid_x)) + 1
        repeated_x = np.tile(valid_x, repeat_factor)[:num_points]
        repeated_y = np.tile(valid_y, repeat_factor)[:num_points]
        return np.column_stack([repeated_x, repeated_y])
        if len(points) < num_points:
            centroid = polygon.centroid
            while len(points) < num_points:
                points.append([centroid.x, centroid.y])
        
        return np.array(points)
