#!/usr/bin/env python3
"""
Create a study rectangle shapefile for a new location.
Usage: python create_study_area.py --lat LAT --lon LON --size SIZE_KM
"""

import argparse
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import os
from plam_sites.site_config import SiteConfig

def create_study_rectangle(lat, lon, size_km, site=None):
    """
    Create a square study area rectangle centered on given coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees (WGS84)
    lon : float
        Longitude in decimal degrees (WGS84)
    size_km : float
        Size of the square in kilometers (e.g., 1.0 for 1km x 1km)
    site : SiteConfig, optional
        Site configuration to determine output directory.
        If None, uses current site from config.
    """
    # Get site configuration
    if site is None:
        site = SiteConfig.get_current_site()
    
    print(f"Creating study area for site: {site.site_name}")
    
    # Create transformer: WGS84 -> ITM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    
    # Convert center point to ITM
    center_x, center_y = transformer.transform(lon, lat)
    
    print(f"Center point (WGS84): {lat:.6f}°N, {lon:.6f}°E")
    print(f"Center point (ITM): {center_x:.2f}, {center_y:.2f}")
    
    # Calculate half-size in meters
    half_size = (size_km * 1000) / 2
    
    # Create bounding box
    minx = center_x - half_size
    maxx = center_x + half_size
    miny = center_y - half_size
    maxy = center_y + half_size
    
    print(f"\nStudy area bounds (ITM):")
    print(f"  West: {minx:.2f}")
    print(f"  East: {maxx:.2f}")
    print(f"  South: {miny:.2f}")
    print(f"  North: {maxy:.2f}")
    print(f"  Size: {size_km}km x {size_km}km")
    
    # Create polygon
    polygon = box(minx, miny, maxx, maxy)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2157")
    
    # Get output path from site configuration
    output_path = site.study_rectangle
    output_dir = os.path.dirname(output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save shapefile
    gdf.to_file(output_path)
    
    print(f"\n✓ Study rectangle saved to: {output_path}")
    
    return gdf, (center_x, center_y)

def parse_dms(dms_str):
    """
    Parse DMS (Degrees, Minutes, Seconds) string to decimal degrees.
    Examples: "53°47'35.3\"N", "6°57'55.3\"W"
    """
    import re
    
    # Extract components
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    
    degrees, minutes, seconds, direction = match.groups()
    
    # Convert to decimal
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    
    # Apply sign for South/West
    if direction in ['S', 'W']:
        decimal = -decimal
    
    return decimal

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create study rectangle for a new location')
    parser.add_argument('--lat', type=str, required=True, 
                       help='Latitude (decimal or DMS format like "53°47\'35.3\\"N")')
    parser.add_argument('--lon', type=str, required=True,
                       help='Longitude (decimal or DMS format like "6°57\'55.3\\"W")')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Size of square in kilometers (default: 1.0)')
    
    args = parser.parse_args()
    
    # Parse coordinates (handle both decimal and DMS formats)
    try:
        lat = float(args.lat)
    except ValueError:
        lat = parse_dms(args.lat)
    
    try:
        lon = float(args.lon)
    except ValueError:
        lon = parse_dms(args.lon)
    
    # Create study rectangle (uses current site from config.ini)
    gdf, center = create_study_rectangle(lat, lon, args.size)
    
    print(f"\n✓ Ready for source detection!")
    print(f"  Next step: Fetch aerial imagery for this area")
