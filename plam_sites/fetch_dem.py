# This script fetches DEM data for all sites in the sites/ directory. Not essential to run, but useful for warming up the cache
import os
import requests
import rasterio
from plam_sites.site_config import SiteConfig
import geopandas as gpd
from pyproj import Transformer
from osgeo import gdal
from bmi_topography import Topography
from bmi_topography import api_key
import sys


def fetch_and_load_dem(minx, miny, maxx, maxy, crs_src, dem_dir="dem_map", output_filename=None):
    """
    Fetch and load DEM for the given extents and CRS.
    Returns (dem_array, transform) for sampling heights.
    """
    # Convert extents to WGS84
    transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(minx, miny)
    lon_max, lat_max = transformer.transform(maxx, maxy)

    # Prepare DEM output paths
    os.makedirs(dem_dir, exist_ok=True)
    cache_dir = os.path.join(dem_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    temp_path = os.path.join(dem_dir, "dem.tiff")
    if output_filename:
        output_path = output_filename
    else:
        output_path = os.path.join(dem_dir, "dem_trans.tiff")

    # Fetch DEM
    params = Topography.DEFAULT.copy()
    params["south"] = lat_min
    params["north"] = lat_max
    params["west"] = lon_min
    params["east"] = lon_max
    params["cache_dir"] = cache_dir
    params["dem_type"] = 'AW3D30'  # Use SRTMGL1 for higher resolution.must be one of ('SRTMGL3', 'SRTMGL1', 'SRTMGL1_E', 'AW3D30', 'AW3D30_E', 'SRTM15Plus', 'NASADEM', 'COP30', 'COP90', 'EU_DTM', 'GEDI_L3', 'GEBCOIceTopo', 'GEBCOSubIceTopo', 'CA_MRDEM_DSM', 'CA_MRDEM_DTM', 'USGS30m', 'USGS10m', 'USGS1m').
    topo = Topography(**params)
    topo.fetch()

    # Uncompress and translate DEM
    files = os.listdir(cache_dir)
    gdal.Translate(temp_path, os.path.join(cache_dir, files[0]), creationOptions=['COMPRESS=NONE'])
    
    # Translate to final output with proper CRS
    gdal.Translate(output_path, temp_path)

    # Load DEM with rasterio
    with rasterio.open(output_path) as dem:
        dem_array = dem.read(1)
        transform = dem.transform

    # Clean up cache and temp file
    for f in files:
        os.remove(os.path.join(cache_dir, f))
    if os.path.exists(temp_path):
        os.remove(temp_path)

ITM_EPSG = "EPSG:2157"
crs_src=ITM_EPSG

def fetch_dem():
# Loop through each site directory and fetch the DEM data
    site = SiteConfig.get_current_site()
    # Replace with the actual DEM URL pattern
    gdf = gpd.read_file(site.study_rectangle)
    bounds = gdf.total_bounds
    # Get study area bounds
    dem_hash = hash((int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])))
    dem_path = os.path.join(site.dem_dir, f"dem_trans_{abs(dem_hash)}.tiff")

    fetch_and_load_dem(
        bounds[0], bounds[1], bounds[2], bounds[3],
        crs_src=ITM_EPSG,
        dem_dir=site.dem_dir,
        output_filename=dem_path
    )
