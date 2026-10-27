#!/usr/bin/env python3
"""
ERA5 Weather Data Fetcher with Automatic Chunking
Handles long time periods by splitting into monthly requests
"""

import cdsapi
import xarray as xr
import pandas as pd
import os
import configparser
from datetime import datetime, timedelta
import sys
from plam_sites.site_config import SiteConfig
import geopandas as gpd
import math
import glob


def load_config(config_file="config.ini"):
    """Load configuration from config.ini file."""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"{config_file} not found. Copy config.ini.example to config.ini and add your API key."
        )
    
    config.read(config_file)
    return config


def get_api_key(config_file="config.ini"):
    """Get CDS API key from config file."""
    config = load_config(config_file)
    
    if 'CDS_API' not in config:
        raise ValueError("CDS_API section not found in config.ini")
    
    api_key = config['CDS_API'].get('api_key', '').strip()
    
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        raise ValueError(
            "API key not set in config.ini. "
            "Get your key from https://cds.climate.copernicus.eu/ and add it to config.ini"
        )
    
    return api_key


def setup_cdsapi_config(api_key):
    """Setup CDS API configuration file."""
    config_path = os.path.expanduser("~/.cdsapirc")
    
    with open(config_path, 'w') as f:
        f.write(f"url: https://cds.climate.copernicus.eu/api\n")
        f.write(f"key: {api_key}\n")
    
    print(f"✓ CDS API config saved to {config_path}")


def generate_monthly_chunks(start_date, end_date):
    """
    Split date range into monthly chunks.
    
    Args:
        start_date: Start date string "YYYY-MM-DD"
        end_date: End date string "YYYY-MM-DD"
        
    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    chunks = []
    current = start_dt
    
    while current <= end_dt:
        # Find end of current month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        
        chunk_end = min(next_month - timedelta(days=1), end_dt)
        
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d %H:%M:%S") if chunk_end == end_dt else chunk_end.strftime("%Y-%m-%d 23:00:00")
        ))
        
        current = next_month
    
    return chunks


def fetch_single_chunk(start_date, end_date, 
                       lat_min, lat_max, lon_min, lon_max, client):
    """
    Fetch a single chunk of ERA5 data.
    
    Returns:
        DataFrame with weather data
    """
    # Parse dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate date components
    date_range = pd.date_range(start_dt, end_dt, freq='D')
    years = sorted(list(set([d.year for d in date_range])))
    months = sorted(list(set([f"{d.month:02d}" for d in date_range])))
    days = sorted(list(set([f"{d.day:02d}" for d in date_range])))
    hours = [f"{h:02d}:00" for h in range(24)]
    
    print(f"  → {start_date} to {end_date}")
    
    # Temporary GRIB file
    temp_grib = f"era5_dataset/temp_era5_{start_dt.year}_{start_dt.month:02d}.grib"
    
    # Request data
    # Log the exact area sent to the CDS/MARS server for debugging
    request_area = [lat_max, lon_min, lat_min, lon_max]  # N, W, S, E
    print(f"    Request submitted to the MARS server with area: {request_area}")

    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'boundary_layer_height',
                'surface_sensible_heat_flux',
                'total_precipitation',
                '2m_temperature',
                '2m_dewpoint_temperature',
                'surface_solar_radiation_downwards',
                'total_cloud_cover',
            ],
            'year': [str(y) for y in years],
            'month': months,
            'day': days,
            'time': hours,
            'area': [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
            'data_format': 'grib',
        },
        temp_grib
    )



def fetch_era5_chunked(start_date, end_date,
                       lat_min, lat_max,
                       lon_min, lon_max,
                       buffer_km=None,
                       api_key=None,
                       config_file="config.ini",
                       chunk_by="month"):
    """
    Fetch ERA5 weather data with automatic chunking for long periods.
    
    Args:
        center_lat: Center latitude (degrees N)
        center_lon: Center longitude (degrees E, use negative for W)
        start_date: Start date string "YYYY-MM-DD"
        end_date: End date string "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        buffer_km: Buffer around center point in km
        output_csv: Output CSV filename
        api_key: CDS API key (if None, reads from config.ini)
        config_file: Path to config file
        chunk_by: Chunking strategy ("month" or "auto")
        lat_min: Minimum latitude of the area (degrees N)
        lat_max: Maximum latitude of the area (degrees N)
        lon_min: Minimum longitude of the area (degrees E, use negative for W)
        lon_max: Maximum longitude of the area (degrees E, use negative for W)
        
    Returns:
        DataFrame with weather data
    """
    
    # Load API key from config if not provided
    if api_key is None:
        api_key = get_api_key(config_file)
    
    # Load default buffer from config if not provided
    if buffer_km is None:
        config = load_config(config_file)
        buffer_km = float(config['CDS_API'].get('weather_buffer_km', 5.0))
    
    # Setup CDS API credentials
    setup_cdsapi_config(api_key)



    # Ensure bbox spans at least one ERA5 grid cell (approx 0.25°)
    min_span_deg = 0.25
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span < min_span_deg:
        half = min_span_deg / 2.0
        lat_min = max(lat_min    - half, -90.0)
        lat_max = min(lat_max + half, 90.0)
        lat_span = lat_max - lat_min
    if lon_span < min_span_deg:
        half = min_span_deg / 2.0
        lon_min = lon_min - half
        lon_max = lon_max + half
        lon_min = max(lon_min, -180.0)
        lon_max = min(lon_max, 180.0)
        lon_span = lon_max - lon_min

    # Validate bounding box
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError(
            f"Invalid bounding box computed: lat_min={lat_min}, lat_max={lat_max}, "
            f"lon_min={lon_min}, lon_max={lon_max}. Check center coords and buffer_km={buffer_km}."
        )
    
    print(f"  Bounding box: {lat_min:.4f} to {lat_max:.4f}°N, {lon_min:.4f} to {lon_max:.4f}°E")
    print(f"  Date range: {start_date} to {end_date}")
    
    # Split into chunks
    chunks = generate_monthly_chunks(start_date, end_date)
    print(f"  Splitting into {len(chunks)} monthly chunks...")
    
    # Initialize CDS API client
    client = cdsapi.Client()
    
    # Fetch each chunk and combine
    all_data = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"\nChunk {i}/{len(chunks)}:")
        
        fetch_single_chunk(
            chunk_start, chunk_end,
            lat_min, lat_max, lon_min, lon_max,
            client
        )

    
    # Combine all chunks
    print("\n✓ Combining all chunks...")
    # Combine all the grib files into a single grib file
    combined_grib_path = "era5_dataset/combined.grib"
    os.makedirs(os.path.dirname(combined_grib_path), exist_ok=True)
    with open(combined_grib_path, 'wb') as combined_file:
        for grib_file in sorted(glob.glob("era5_dataset/*.grib")):
            with open(grib_file, 'rb') as f:
                combined_file.write(f.read())
    
    print(f"\n✓ Complete! Weather data saved to {combined_grib_path}")

    return 


def fetch_era5_weather_chunked(start_date, end_date, lat_min=None, lat_max=None, lon_min=None, lon_max=None, output_path="era5_dataset/combined.grib"):   
    os.makedirs('era5_dataset', exist_ok=True)
    df = fetch_era5_chunked(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        start_date=start_date,
        end_date=end_date,
        config_file="./plam_sites/config.ini",
    )
