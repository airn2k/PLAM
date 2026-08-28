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


def fetch_single_chunk(center_lat, center_lon, start_date, end_date, 
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
    temp_grib = f"temp_era5_{start_dt.year}_{start_dt.month:02d}.grib"
    
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
    
    # Load and process data
    ds = xr.open_dataset(temp_grib, engine="cfgrib", 
                         backend_kwargs={'filter_by_keys': {'stepType': 'instant'}})
    
    # Subset to exact date range
    ds = ds.sel(time=slice(start_date, end_date))
    
    # Spatially average over the box
    u10 = ds['u10'].mean(dim=['latitude', 'longitude'])
    v10 = ds['v10'].mean(dim=['latitude', 'longitude'])
    blh = ds['blh'].mean(dim=['latitude', 'longitude'])
    t2m = ds['t2m'].mean(dim=['latitude', 'longitude'])
    d2m = ds['d2m'].mean(dim=['latitude', 'longitude'])
    tcc = ds['tcc'].mean(dim=['latitude', 'longitude'])
    
    # Load accumulated variables
    ds_accum = xr.open_dataset(temp_grib, engine="cfgrib",
                              backend_kwargs={'filter_by_keys': {'stepType': 'accum'}})
    ds_accum = ds_accum.sel(time=slice(start_date, end_date))
    sshf = ds_accum['sshf'].mean(dim=['latitude', 'longitude'])
    tp = ds_accum['tp'].mean(dim=['latitude', 'longitude'])
    ssrd = ds_accum['ssrd'].mean(dim=['latitude', 'longitude'])
    
    # Convert to DataFrames
    df_u10 = u10.to_dataframe(name='u10').reset_index()
    df_v10 = v10.to_dataframe(name='v10').reset_index()
    df_blh = blh.to_dataframe(name='blh').reset_index()
    df_t2m = t2m.to_dataframe(name='t2m').reset_index()
    df_d2m = d2m.to_dataframe(name='d2m').reset_index()
    df_tcc = tcc.to_dataframe(name='tcc').reset_index()
    df_sshf = sshf.to_dataframe(name='sshf').reset_index()
    df_tp = tp.to_dataframe(name='tp').reset_index()
    df_ssrd = ssrd.to_dataframe(name='ssrd').reset_index()
    
    # Combine into single DataFrame
    weather_df = df_u10[['time']].copy()
    weather_df['u10'] = df_u10['u10']
    weather_df['v10'] = df_v10['v10']
    weather_df['blh'] = df_blh['blh']
    weather_df['temp_celsius'] = df_t2m['t2m'] - 273.15
    weather_df['dewpoint_celsius'] = df_d2m['d2m'] - 273.15
    weather_df['cloud_cover'] = df_tcc['tcc']
    
    # Calculate relative humidity
    import numpy as np
    T = weather_df['temp_celsius']
    Td = weather_df['dewpoint_celsius']
    weather_df['relative_humidity'] = 100.0 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
    
    weather_df['sshf'] = df_sshf['sshf'] / 3600.0  # J/m² to W/m²
    weather_df['precipitation_mm_hr'] = df_tp['tp'] * 1000.0  # m to mm
    weather_df['solar_radiation_w_m2'] = df_ssrd['ssrd'] / 3600.0  # J/m² to W/m²
    
    # Add metadata columns
    weather_df.insert(1, 'number', 0)
    weather_df.insert(2, 'step', '0 days')
    weather_df.insert(3, 'surface', 0)
    weather_df.insert(4, 'valid_time', weather_df['time'])
    
    # Cleanup
    os.remove(temp_grib)
    
    print(f"    ✓ {len(weather_df)} records")
    
    return weather_df


def fetch_era5_chunked(center_lat, center_lon, 
                       start_date, end_date,
                       buffer_km=None,
                       output_csv="weather_data.csv",
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
    
    # Calculate bounding box
    lat_buffer = buffer_km / 111.0

    # Convert latitude to radians and compute longitude degree width at that latitude
    lat_rad = math.radians(center_lat)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        # At or extremely close to the poles, longitude degrees are undefined; use full globe
        lon_buffer = 180.0
    else:
        # 1 degree longitude ~= 111.320*cos(lat) km
        lon_buffer = buffer_km / (111.320 * cos_lat)

    # Clamp buffers to valid ranges
    lon_buffer = min(abs(lon_buffer), 180.0)
    lat_buffer = min(abs(lat_buffer), 90.0)

    lat_min = max(center_lat - lat_buffer, -90.0)
    lat_max = min(center_lat + lat_buffer, 90.0)
    lon_min = center_lon - lon_buffer
    lon_max = center_lon + lon_buffer

    # Ensure longitudes are within [-180, 180]
    lon_min = max(lon_min, -180.0)
    lon_max = min(lon_max, 180.0)

    # Ensure bbox spans at least one ERA5 grid cell (approx 0.25°)
    min_span_deg = 0.25
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span < min_span_deg:
        half = min_span_deg / 2.0
        lat_min = max(center_lat - half, -90.0)
        lat_max = min(center_lat + half, 90.0)
        lat_span = lat_max - lat_min
    if lon_span < min_span_deg:
        half = min_span_deg / 2.0
        lon_min = center_lon - half
        lon_max = center_lon + half
        lon_min = max(lon_min, -180.0)
        lon_max = min(lon_max, 180.0)
        lon_span = lon_max - lon_min

    # Validate bounding box
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError(
            f"Invalid bounding box computed: lat_min={lat_min}, lat_max={lat_max}, "
            f"lon_min={lon_min}, lon_max={lon_max}. Check center coords and buffer_km={buffer_km}."
        )
    
    print(f"Fetching ERA5 data for location ({center_lat:.4f}, {center_lon:.4f})")
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
        try:
            chunk_df = fetch_single_chunk(
                center_lat, center_lon, 
                chunk_start, chunk_end,
                lat_min, lat_max, lon_min, lon_max,
                client
            )
            all_data.append(chunk_df)
        except Exception as e:
            print(f"    ❌ Error fetching chunk: {e}")
            print(f"    Continuing with next chunk...")
            continue
    
    if not all_data:
        raise RuntimeError("Failed to fetch any data chunks")
    
    # Combine all chunks
    print("\n✓ Combining all chunks...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by time and remove duplicates
    combined_df = combined_df.sort_values('time').drop_duplicates(subset=['time'])
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))
    
    print(f"\n✓ Complete! Weather data saved to {output_csv}")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    
    return combined_df


def fetch_era5_weather_chunked(start_date, end_date):
    site = SiteConfig.get_current_site()
    output_path = site.weather_data
    # read study rectangle bounds to get center point
    study_rect_path = site.study_rectangle
    gdf = gpd.read_file(study_rect_path)
    # Convert ITM to WGS84
    gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    lat = (bounds[1] + bounds[3]) / 2  # Use center latitude from bounds
    lon = (bounds[0] + bounds[2]) / 2  # Use center longitude from bounds  


    
    
    try:
        df = fetch_era5_chunked(
            center_lat=lat,
            center_lon=lon,
            start_date=start_date,
            end_date=end_date,
            config_file="./plam_sites/config.ini",
            output_csv=output_path
        )
        
        print("\nSample data (first 5 records):")
        print(df.head())
        print("\nSample data (last 5 records):")
        print(df.tail())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Created config.ini from config.ini.example")
        print("  2. Added your CDS API key to config.ini")
        print("  3. Registered at https://cds.climate.copernicus.eu/")
        sys.exit(1)
