
################################################################
################Imports#########################################
################################################################

import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
from rasterio.transform import Affine
from datetime import datetime, timedelta
import math
import time
import os
import glob
import sys
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from plam_sites.fetch_era5_weather_chunked import fetch_era5_chunked
from dateutil.relativedelta import relativedelta
from plam_sites.site_config import SiteConfig
from plam_sites.fetch_dem import fetch_and_load_dem
from scipy.ndimage import zoom
import rasterio
import scipy.ndimage
from rasterio.transform import rowcol
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states)

# Import helper functions
from helper.sample_dem_at_points import sample_dem_at_points
from helper.estimate_gpu_memory_usage import estimate_gpu_memory_usage
from helper.sample_points_in_polygon import sample_points_in_polygon
from helper.save_monthly_agl_plots import save_monthly_agl_plots
from helper.save_monthly_depo_plots import save_monthly_depo_plots
from helper.save_annual_agl_plots import save_annual_agl_plots
from helper.save_annual_depo_plots import save_annual_depo_plots
from helper.agl_band_mean_weighted import agl_band_mean_weighted
from helper.hillshade import hillshade

# Import atmospheric calculators
from atmosphere.calculate_kxy_with_stability import calculate_kxy_with_stability
from atmosphere.calculate_ustar_with_stability import calculate_ustar_with_stability
from atmosphere.calculate_kz_profile_with_heatflux import calculate_kz_profile_with_heatflux

# Import GPU kernels
from gpu_model.set_scalar_kernel import set_scalar_kernel
from gpu_model.decay_mass_kernel import decay_mass_kernel
from gpu_model.update_particles_3d_gpu_step import update_particles_3d_gpu_step
from gpu_model.update_grid_3d_gpu_step import update_grid_3d_gpu_step
from gpu_model.remove_outofbound_3d_gpu_step import remove_outofbound_3d_gpu_step
from gpu_model.zero_3d_kernel import zero_3d_kernel
from gpu_model.fill_rows_and_reset_type_kernel import fill_rows_and_reset_type_kernel


# Import parameters 
from config.params import *

################################################################

CONC_SPECIES_MODE = os.environ.get("CONC_SPECIES_MODE", "total").strip().lower()
if CONC_SPECIES_MODE not in {"total", "nh3"}:
    print(f"WARNING: unknown CONC_SPECIES_MODE='{CONC_SPECIES_MODE}', defaulting to 'total'")
    CONC_SPECIES_MODE = "total"


def _find_source_column(gdf, name):
    columns_lower = {str(c).lower(): c for c in gdf.columns}
    return columns_lower.get(str(name).lower())


def _source_attr_or_default(gdf, attr_names, default):
    for attr_name in attr_names:
        col = _find_source_column(gdf, attr_name)
        if col is not None:
            values = pd.to_numeric(gdf[col], errors="coerce").astype(np.float32)
            values[np.isnan(values)] = float(default)
            return values
    return np.full(len(gdf), float(default), dtype=np.float32)

def _source_str_or_default(gdf, attr_names, default):
    for attr_name in attr_names:
        col = _find_source_column(gdf, attr_name)
        if col is not None:
            values = gdf[col].fillna(default).astype(str)
            return values
    return np.full(len(gdf), str(default), dtype=object)



def _source_attr_bool_or_default(gdf, attr_names, default=False):
    for attr_name in attr_names:
        col = _find_source_column(gdf, attr_name)
        if col is not None:
            values = gdf[col]
            if pd.api.types.is_bool_dtype(values):
                return values.fillna(default).astype(bool).to_numpy()
            if pd.api.types.is_numeric_dtype(values):
                return values.fillna(default).astype(bool).to_numpy()
            normalized = values.fillna("").astype(str).str.strip().str.lower()
            true_values = {"1", "true", "t", "yes", "y", "on"}
            return normalized.isin(true_values).to_numpy()
    return np.full(len(gdf), bool(default), dtype=bool)

CONC_INCLUDE_NH4 = CONC_SPECIES_MODE == "total"

################################################################
################User Params###############################################
################################################################

# choose thread size
threads = 1024
timestep_cfl_fraction = 0.5  # Target max displacement per step as fraction of GRID_RES
min_timesteps_per_hour = MIN_TIMESTEPS_PER_HOUR
max_timesteps_per_hour = MAX_TIMESTEPS_PER_HOUR
window_minutes = WINDOW_MINUTES  # Ring-buffer window length for updates
max_window_steps = 30  # Hard cap to limit per-step GPU work
sample_interval = SAMPLE_INTERVAL  # Steps between concentration snapshots / sampler readings
w_mean = 0.000  # m/s, weak convective uplift for area sources
w_settle = 0.0000
v_d = 0.03  # m/s surface deposition velocity (typical for NH3 over grassland/crops)
z0 = 0.03  # m, roughness length for wind profile (0.01=smooth, 0.03=open terrain, 0.1=crops, 0.5=forest)
decay_rate = 0.0  # DISABLED - chemical conversion now handled explicitly via NH3→NH4+ in kernel
# Wet deposition parameters for ammonia (highly water-soluble)


wet_dep_a = 1.0e-4  # Scavenging coefficient base (1/s)
wet_dep_b = 0.64    # Power law exponent for rain intensity

# NH3 → NH4+ conversion chemistry parameters
# Conversion rate depends on SO2, NOx, solar radiation, humidity, and temperature
# Base rates are modulated by meteorological conditions in the main loop
k_conversion_so2 = 5.0e-6  # Sulfate pathway (µg/m³)⁻¹ s⁻¹ - fast, irreversible
k_conversion_nox = 1.0e-6  # Nitrate pathway (µg/m³)⁻¹ s⁻¹ - equilibrium, slower
k_photo_nh3 = 1.0e-5  # Direct photochemical oxidation rate coefficient (s⁻¹) at full sun
v_d_nh4 = 0.01  # Deposition velocity for NH4+ particles (m/s) - slower than NH3

if SPECIES == "ODOR":
    species_label = "odour"
    species_unit = "OU/m³"
    DEPOSITION_ENABLED = 0

if SPECIES == "NH3":
    species_label = "ammonia"
    species_unit = "µg/m³"
    species_unit_depo = "kg/ha/yr"



sampler_x_env = os.environ.get("SAMPLER_X")
sampler_y_env = os.environ.get("SAMPLER_Y")

if not DEPOSITION_ENABLED:
    print("Deposition disabled (DEPOSITION_ENABLED=0): dry/wet deposition and deposition outputs are skipped.")

################################################################
################Load Files##############################################
################################################################

# Timing: Start
init_start = time.time()

# Load site configuration
site = SiteConfig.get_current_site()
print(f"Using study site: {site.site_name}")

# Load study area
t0 = time.time()
gdf = gpd.read_file(site.study_rectangle)
s_bounds = gdf.total_bounds
print(f"Loaded study area in {time.time()-t0:.2f} s")
print(f"  Study area bounds: X=[{s_bounds[0]:.1f}, {s_bounds[2]:.1f}], Y=[{s_bounds[1]:.1f}, {s_bounds[3]:.1f}]")

emission_sources = gpd.read_file(site.sources_shapefile)
h_bounds = emission_sources.total_bounds

# Get study area bounds
bounds = gdf.total_bounds  # minx, miny, maxx, maxy in ITM

# Simulation time period - can be overridden by command line arguments
if len(sys.argv) >= 3:
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    print(f"\n✓ Using command-line dates: {start_date} to {end_date}")
else:
    # Default dates if not provided
    start_date = "2024-04-02 12:00:00"
    end_date = "2025-07-01 12:00:00"
    print(f"\n✓ Using default dates: {start_date} to {end_date}")


if FLAT_SITE == 0:
    print("\n" + "="*70)
    print("FETCHING DEM DATA")
    print("="*70)


    # Check for existing DEM or fetch new one
    # Create unique DEM filename based on study area bounds
    dem_hash = hash((int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])))
    dem_path = os.path.join(site.dem_dir, f"dem_trans_{abs(dem_hash)}.tiff")
    t0 = time.time()

    # Check if this specific DEM exists, or use any existing DEM in the directory
    if os.path.exists(dem_path):
        print(f"✓ Using cached DEM: {dem_path}")
    else:
        # Check for any existing DEM file in the site's DEM directory
        existing_dems = glob.glob(os.path.join(site.dem_dir, "dem_trans_*.tiff"))
        if existing_dems:
            dem_path = existing_dems[0]  # Use the first available DEM
            print(f"✓ Using existing DEM from site: {dem_path}")
        else:
            print(f"Fetching DEM for study area bounds...")
            print(f"  Bounds (ITM): {bounds}")
            dem_array, dem_transform = fetch_and_load_dem(
                bounds[0], bounds[1], bounds[2], bounds[3],
                crs_src=STUDY_CRS,
                dem_dir=site.dem_dir,
                output_filename=dem_path
            )
            print(f"✓ DEM fetched and saved to {dem_path}")
            
    print(f"DEM setup time: {time.time()-t0:.2f} s")
else:
    dem_path = None
    # Generate a flat DEM at 0 m elevation for flat site with the correct shape and transform
    nx = int((bounds[2] - bounds[0]) // GRID_RES)
    ny = int((bounds[3] - bounds[1]) // GRID_RES)
    dem_array = np.zeros((ny, nx), dtype=np.float32)
    dem_transform = Affine.translation(bounds[0], bounds[3]) * Affine.scale(GRID_RES, -GRID_RES)


# Fetch weather data for study area
print("\n" + "="*70)
print("FETCHING WEATHER DATA")
print("="*70)

# Calculate center of study area for weather data request
transformer = Transformer.from_crs(STUDY_CRS, "EPSG:4326", always_xy=True)
center_x = (bounds[0] + bounds[2]) / 2
center_y = (bounds[1] + bounds[3]) / 2
center_lon, center_lat = transformer.transform(center_x, center_y)

weather_csv = site.weather_data
t0 = time.time()
if os.path.exists(weather_csv):
    print(f"✓ Using cached weather data: {weather_csv}")
    wind_df = pd.read_csv(weather_csv)
else:
    print(f"Fetching weather data for study area center: {center_lat:.4f}°N, {center_lon:.4f}°E")
    print(f"  (Note: Long time periods will be split into monthly chunks)")
    wind_df = fetch_era5_chunked(
        center_lat=center_lat,
        center_lon=center_lon,
        start_date=start_date,
        end_date=end_date,
        output_csv=weather_csv
    )
print(f"Weather data loading time: {time.time()-t0:.2f} s")

time_col = None
if "time" in wind_df.columns:
    time_col = "time"
elif "valid_time" in wind_df.columns:
    time_col = "valid_time"

if time_col is not None:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    wind_df[time_col] = pd.to_datetime(wind_df[time_col], errors="coerce")
    wind_df = wind_df[(wind_df[time_col] >= start_dt) & (wind_df[time_col] <= end_dt)].copy()
    wind_df = wind_df.reset_index(drop=True)
    if wind_df.empty:
        raise ValueError(f"No weather records in {weather_csv} for {start_dt} to {end_dt}.")
else:
    print("WARNING: No time column found in weather data; using full file.")

total_hours = len(wind_df)
print(f"✓ Loaded {total_hours} hours of weather data")


u10 = wind_df["u10"].to_numpy(dtype=np.float64, copy=False)
v10 = wind_df["v10"].to_numpy(dtype=np.float64, copy=False)
speed = np.sqrt(u10 * u10 + v10 * v10)
max_speed = float(np.nanmax(speed)) if speed.size > 0 else 0.0
if not np.isfinite(max_speed) or max_speed <= 0.0:
    max_speed = 1.0
dt_cfl = timestep_cfl_fraction * GRID_RES / max_speed
timesteps_per_hour = int(np.ceil(3600.0 / dt_cfl))
timesteps_per_hour = max(min_timesteps_per_hour, min(max_timesteps_per_hour, timesteps_per_hour))
print(
    f"✓ Objective timestep: max_wind={max_speed:.2f} m/s, "
    f"dt≈{3600.0 / timesteps_per_hour:.1f} s, "
    f"timesteps_per_hour={timesteps_per_hour}"
)
################################################################




t0 = time.time()
# Sources (area polygons)
emission_sources["area_m2"] = emission_sources.geometry.area

emission_sources["is_point_source"] = _source_attr_bool_or_default(
    emission_sources, ["Point", "point", "is_point", "is_point_source"], False
)
emission_sources["emission_rate_gps_m2"] = _source_attr_or_default(emission_sources, ["e_r"], E_GPS_M2_BASE)
emission_sources["release_height_m"] = _source_attr_or_default(emission_sources, ["r_h"], RELEASE_HEIGHT_DEFAULT)
emission_sources["release_velocity_m_s"] = _source_attr_or_default(emission_sources, ["release_velocity", "r_v"], RELEASE_VELOCITY_DEFAULT)
emission_sources["seasonal_factor"] = _source_str_or_default(emission_sources, ["seasonal_factor", "seasonal_f"], "1.0 1.0 1.0 1.0")

# Convert seasonal_factor strings to lists of floats
emission_sources["seasonal_factor"] = emission_sources["seasonal_factor"].apply(
    lambda x: [float(s.strip()) for s in str(x).split(" ")]
)

# Determine number of unique seasonal factors (1-4) based on non-default values
unique_seasonal_factor_n = emission_sources["seasonal_factor"].apply(lambda x: tuple(x)).nunique()
print(f"Detected {unique_seasonal_factor_n} unique seasonal factors (1-4) based on non-default values.")

total_emission_ou_s = 0.0
emission_scale = float(os.environ.get("EMISSION_SCALE", "1.0"))

# Fixed baseline rates (voxel-compatible), then modulated by seasonal/temperature factors
E_gps_m2_base = E_GPS_M2_BASE
base_effective_rate = E_gps_m2_base

dt = 3600.0 / timesteps_per_hour

print(f"Base emission rate (default): {E_gps_m2_base:.6e} g/m²/s")
print(f"Default release height: {RELEASE_HEIGHT_DEFAULT:.1f} m")
print(f"Release velocity default: {RELEASE_VELOCITY_DEFAULT:.2f} m/s")

# Fetch ambient SOx and NOx concentrations for NH3→NH4+ conversion
print("\n" + "="*70)
print("FETCHING SOx AND NOx CONCENTRATIONS (AGRICULTURAL SOURCES)")
print("="*70)

emission_sources["total_emission_gps"] = np.where(
    emission_sources["is_point_source"].to_numpy(),
    emission_sources["emission_rate_gps_m2"].to_numpy(),
    emission_sources["area_m2"].to_numpy() * emission_sources["emission_rate_gps_m2"].to_numpy(),
)

# There shouldn't be any nans in emission_sources["total_emission_gps"]


total_emission = float(emission_sources["total_emission_gps"].sum())
point_source_count = int(np.count_nonzero(emission_sources["is_point_source"]))
if point_source_count:
    print(f"Detected {point_source_count} point-source entries via Point; treating e_r as g/s for those sources.")
print(f"Source baseline total: housing={total_emission:.6f} g/s")



print(f"\nTotal emission sources: {len(emission_sources)}")
print(f"Source combination time: {time.time()-t0:.2f} s")


# Domain rectangle
print("Original CRS:", gdf.crs)
gdf_itm = gdf.to_crs(STUDY_CRS)
minx, miny, maxx, maxy = gdf_itm.total_bounds

# Web Mercator extents for basemap
transformer = Transformer.from_crs(STUDY_CRS, "EPSG:3857", always_xy=True)
minx_3857, miny_3857 = transformer.transform(minx, miny)
maxx_3857, maxy_3857 = transformer.transform(maxx, maxy)

t0 = time.time()


# Wind/BLH
u_wind_series = wind_df["u10"].values
v_wind_series = wind_df["v10"].values
blh_series = wind_df["blh"].values
sshf_series = wind_df["sshf"].values
precip_series = wind_df["precipitation_mm_hr"].values
temp_series = wind_df["temp_celsius"].values  # Temperature for deposition calculation
print(f"Setup after CRS/particle: {time.time()-t0:.2f} s")
rh_series = wind_df["relative_humidity"].values  # Relative humidity for aerosol partitioning
cloud_series = wind_df["cloud_cover"].values  # Cloud cover (0-1)
solar_series = wind_df["solar_radiation_w_m2"].values  # Solar radiation for photochemistry


# =========================
# Simulation settings
# =========================
total_steps = np.int64(total_hours * timesteps_per_hour)

t_hourly = np.arange(total_hours)
t_fine = np.linspace(0, total_hours - 1, total_steps)
blh_fine = np.interp(t_fine, t_hourly, blh_series[0:total_hours])
v_wind_fine = np.interp(t_fine, t_hourly, v_wind_series[0:total_hours])
u_wind_fine = np.interp(t_fine, t_hourly, u_wind_series[0:total_hours])
sshf_fine = np.interp(t_fine, t_hourly, sshf_series[0:total_hours])
precip_fine = np.interp(t_fine, t_hourly, precip_series[0:total_hours])
temp_fine = np.interp(t_fine, t_hourly, temp_series[0:total_hours])
rh_fine = np.interp(t_fine, t_hourly, rh_series[0:total_hours])
cloud_fine = np.interp(t_fine, t_hourly, cloud_series[0:total_hours])
solar_fine = np.interp(t_fine, t_hourly, solar_series[0:total_hours])

nx = int((maxx - minx) // GRID_RES)
ny = int((maxy - miny) // GRID_RES)

# Adaptive particle mass based on an objective target: particles per grid cell per step
total_emission = emission_sources["total_emission_gps"].sum()
particles_per_cell_target = 5.0
target_particles = max(1, int(particles_per_cell_target * nx * ny))
divisor = target_particles

actual_mass_per_particle = E_GPS_MULTIPLIER * total_emission / divisor
# Convert to mass per particle for this timestep (g per particle per step)
mass_per_particle = actual_mass_per_particle * dt
print(
    f"Total emission: {total_emission:.3f} g/s, target particles/step: {target_particles:,} "
    f"({particles_per_cell_target:.2f} per cell), mass per particle: {mass_per_particle:.2e} OU"
)
window_steps = max(1, int(timesteps_per_hour * (window_minutes / 60.0)))
f_step = window_steps
print(f"Ring buffer window: {f_step} steps (~{f_step * 60.0 / timesteps_per_hour:.1f} minutes)")


# 3D vertical grid
nz = int(np.ceil(Z_MAX / DZ))

# Accumulators
depo_grid = np.zeros((nx, ny), dtype=np.float32)
conc3d = np.zeros((nx, ny, nz))

# =========================
# Particles allocation with source-specific density
# =========================
t0 = time.time()
# Use emission rate for all sources with probabilistic rounding to avoid losing small sources
np.random.seed(42)  # Reproducible results

def calc_particles_debug(row):
    # Calculate exact fractional particles
    exact_particles = row["total_emission_gps"] * dt / mass_per_particle
    # Use floor + probabilistic rounding for the fractional part
    base_particles = int(exact_particles)
    fractional_part = exact_particles - base_particles
    # Emit extra particle with probability = fractional_part
    if np.random.random() < fractional_part:
        return base_particles + 1
    return base_particles

emission_sources["particles_per_step"] = emission_sources.apply(calc_particles_debug, axis=1)

print(f"Particle calculation time: {time.time()-t0:.2f} s")

# =========================
# Particles allocation
# =========================
# DEM for particle initialization (use existing dem.tiff)

# --- DEM for surface interaction ---

if FLAT_SITE == 0:
    with rasterio.open(dem_path) as dem_src:
        print(f"Opened DEM with rasterio in {time.time()-t0:.2f} s")
        t1 = time.time()
        dem_array = dem_src.read(1)
        print(f"Read DEM array in {time.time()-t1:.2f} s")
        dem_transform = dem_src.transform

    print(dem_transform)

dem_array = np.float32(dem_array)
# Downscale DEM to (nx, ny) matching simulation grid (easting, northing)
# Rasterio gives dem_array as (rows=N→S, cols=W→E).
# Zoom to (ny, nx), flip rows so axis-0 goes S→N, then transpose to (easting, northing).
t0 = time.time()
dem_zoomed = scipy.ndimage.zoom(dem_array, (ny / dem_array.shape[0], nx / dem_array.shape[1]), order=1)
dem_resized = dem_zoomed[::-1, :].T.copy()  # shape (nx, ny): [ix_east, iy_north]
print(f"DEM resize (scipy.ndimage.zoom) time: {time.time()-t0:.2f} s")

monthly_agl_top_m = 10.0
surface_layer_max = max(1, int(math.ceil(monthly_agl_top_m / DZ)))
ground_height_grid = dem_resized.astype(np.float32)
# conc3d grid is now terrain-following (AGL), so ground is always at k=0
ground_k_grid = np.zeros((nx, ny), dtype=np.int32)

# DEM metadata for kernel
dem_nx = np.int32(nx)
dem_ny = np.int32(ny)
dem_minx = np.float32(minx)
dem_miny = np.float32(miny)
dem_dx = np.float32(GRID_RES)
dem_dy = np.float32(GRID_RES)

# Terrain gradient grids for channeling and orographic lifting
# dh_dx[i,j] = dh/dx (easting slope), dh_dy[i,j] = dh/dy (northing slope)
dh_dy_grid, dh_dx_grid = np.gradient(dem_resized, GRID_RES)
dh_dx_grid = dh_dx_grid.astype(np.float32)
dh_dy_grid = dh_dy_grid.astype(np.float32)
print(f"Terrain gradients: max |dh/dx|={np.abs(dh_dx_grid).max():.4f}, max |dh/dy|={np.abs(dh_dy_grid).max():.4f}")

initial_release_vz = None


d_dem = cuda.to_device(dem_resized.astype(np.float32))
d_dh_dx = cuda.to_device(dh_dx_grid)
d_dh_dy = cuda.to_device(dh_dy_grid)

z_offset = 15.0  # meters above ground to start particles

total_particles = emission_sources["particles_per_step"].sum()
print(f"Initializing {total_particles} particles...")
t0 = time.time()

x = np.full((f_step, total_particles), -9999.0, dtype=np.float32)
y = np.full((f_step, total_particles), -9999.0, dtype=np.float32)
z = np.full((f_step, total_particles), -9999.0, dtype=np.float32)

if unique_seasonal_factor_n > 1:
    seasonal_factor_n = np.ones((total_particles,4), dtype=np.float16)
elif unique_seasonal_factor_n == 1:
    seasonal_factor = emission_sources["seasonal_factor"].apply(lambda x: float(x[0])).to_numpy(dtype=np.float16)

initial_release_vz = np.zeros((total_particles,), dtype=np.float32)
particle_type = np.zeros((f_step, total_particles), dtype=np.int32)  # 0=NH3 (gas), 1=NH4+ (particle)

d_initial_release_vz = None

print(f"Particle array allocation time: {time.time()-t0:.2f} s")
                     
t0 = time.time()
offset = 0
for i, (_, row) in enumerate(emission_sources.iterrows()):
    count = int(row["particles_per_step"])
    if count <= 0:
        continue

    print(f"  Source {i+1}/{len(emission_sources)}: {count} particles (type: {row.get('source_type', 'unknown')}, area: {row.get('area_m2', 0)/10000:.1f} ha)... ", end="", flush=True)
    
    t1 = time.time()
    geom = row.geometry
    is_point = row.get("Point", False)
    if is_point:
        base_pts = np.array([[geom.centroid.x, geom.centroid.y]], dtype=np.float32)
        print('Point source detected, using single point for sampling.')
    else:
        base_pts = sample_points_in_polygon(geom, count).astype(np.float32, copy=False)

    if base_pts.shape[0] >= count:
        pts = base_pts[:count]
    else:
        repeat_factor = (count // base_pts.shape[0]) + 1
        pts = np.tile(base_pts, (repeat_factor, 1))[:count]

    print(f"sampled in {time.time()-t1:.2f}s, ", end="", flush=True)
    if unique_seasonal_factor_n > 1:
        # Populate seasonal_factor_n for this source's particles
        seasonal_factor_n[offset : offset + count, :] = np.array(row["seasonal_factor"], dtype=np.float16)[None, :].repeat(count, axis=0)

    x[:, offset : offset + count] = pts[:, 0]
    y[:, offset : offset + count] = pts[:, 1]
    initial_release_vz[offset : offset + count] = float(row["release_velocity_m_s"])
    
    t1 = time.time()
    # Sample DEM at (x, y) for each particle
    ground_z = sample_dem_at_points(pts[:, 0], pts[:, 1], dem_array, dem_transform)
    print(f"DEM in {time.time()-t1:.2f}s")
    
    # Release height from source geometry attribute, defaulting to RELEASE_HEIGHT_DEFAULT
    release_height = float(row["release_height_m"])
    z[:, offset : offset + count] = (
        ground_z[None, :] + release_height
    )
    offset += count

print(f"Particle position initialization time: {time.time()-t0:.2f} s")



x0 = x[0].copy()
y0 = y[0].copy()
z_init = z[0].copy()
print(z_init)


d_x0 = cuda.to_device(x0.astype(np.float32, copy=False))
d_y0 = cuda.to_device(y0.astype(np.float32, copy=False))
d_z0 = cuda.to_device(z_init.astype(np.float32, copy=False))
d_initial_release_vz = cuda.to_device(initial_release_vz)


particle_mass_g = np.zeros(f_step, dtype=np.float32)
mass_budget = np.zeros(5, dtype=np.float64)
emitted_mass_total = 0.0

# Ring buffer pointer
head = 0
step_counter = 0
step_in_window = -1  # how many valid rows minus 1


# Shapes & dtypes (must be float32 / int32)
f_step, n_part = x.shape



# --- GPU memory usage estimator ---
main_arrays = 5  # x, y, z, particle_mass_g, plus a buffer
extra_arrays_bytes = depo_grid.nbytes + conc3d.nbytes  # add more if needed
total_bytes, mb, gb = estimate_gpu_memory_usage(
    n_part, f_step, n_arrays=main_arrays, extra_bytes=extra_arrays_bytes
)
print(
    f"[INFO] Estimated GPU memory usage for {n_part} particles, f_step={f_step}, {main_arrays} arrays: {mb:.1f} MB ({gb:.2f} GB)"
)

# Allocate device mirrors and copy initial state
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_z = cuda.to_device(z)
d_particle_type = cuda.to_device(particle_type)  # particle type tracking
d_particle_mass_g = cuda.to_device(particle_mass_g.astype(np.float32, copy=False))
d_depo = cuda.to_device(depo_grid)
d_mass_budget = cuda.to_device(np.zeros(5, dtype=np.float64))
blocks = (n_part + threads - 1) // threads
print(blocks)
rng_states = create_xoroshiro128p_states(blocks * threads, seed=12345)

# Pre-allocate persistent device conc3d buffer (zeroed each sample interval)
d_conc3d = cuda.device_array((nx, ny, nz), dtype=np.float32)
conc3d_total = nx * ny * nz
conc3d_blocks = (conc3d_total + threads - 1) // threads

# Pre-compute decay-kernel grid (covers f_step mass entries)
decay_blocks = (f_step + threads - 1) // threads

# Average temperature during the study for reference
temp_ref = temp_fine.mean()

# Sampler location: use center of study area
if sampler_x_env is not None and sampler_y_env is not None:
    sampler_loc_itm = (float(sampler_x_env), float(sampler_y_env))
    sampler_location_mode = "env-itm"
else:
    sampler_loc_itm = (center_x, center_y)
    sampler_location_mode = "study-area-center"
sampler_loc_wgs84 = (center_lon, center_lat)  # For reference
sampler_z = sample_dem_at_points(np.array([sampler_loc_wgs84[0]]), np.array([sampler_loc_wgs84[1]]), dem_array, dem_transform)[0] + 1.5  #2m above ground
if sampler_location_mode == "env-itm":
    sampler_transformer_wgs84 = Transformer.from_crs(STUDY_CRS, "EPSG:4326", always_xy=True)
    sampler_lon, sampler_lat = sampler_transformer_wgs84.transform(sampler_loc_itm[0], sampler_loc_itm[1])
    sampler_loc_wgs84 = (center_x, center_y)
print(f"Sampler location (WGS84): {sampler_loc_wgs84[0]:.6f}°N, {sampler_loc_wgs84[1]:.6f}°E")
print(f"Sampler location (ITM): {sampler_loc_itm[0]:.1f}, {sampler_loc_itm[1]:.1f}")
print(f"Sampler mode: {sampler_location_mode}")
print(f"Sampler height: {sampler_z:.1f} m above sea level")
sampler_idx = (
    int((sampler_loc_itm[0] - minx) // GRID_RES),
    int((sampler_loc_itm[1] - miny) // GRID_RES),
    int(sampler_z // DZ),
)
sampler_ground_k = None
if 0 <= sampler_idx[0] < nx and 0 <= sampler_idx[1] < ny:
    sampler_ground_k = int(ground_height_grid[sampler_idx[0], sampler_idx[1]]/DZ)
    k1 = min(nz, sampler_ground_k + surface_layer_max)
    print(
        f"Sampler uses DEM-aware mean: idx={sampler_idx}, ground_k={sampler_ground_k}, layer_max={surface_layer_max}, k_range=[{sampler_ground_k},{k1})"
    )
else:
    print(f"WARNING: Sampler index out of bounds: {sampler_idx}")

peak_flat_idx = int(np.argmax(ground_height_grid))
peak_ix, peak_iy = np.unravel_index(peak_flat_idx, ground_height_grid.shape)
peak_sampler_loc_itm = (
    minx + (peak_ix + 0.5) * GRID_RES,
    miny + (peak_iy + 0.5) * GRID_RES,
)
peak_sampler_ground_asl = float(ground_height_grid[peak_ix, peak_iy])
print(
    f"Peak-terrain diagnostic cell: idx=({peak_ix}, {peak_iy}), "
    f"ITM=({peak_sampler_loc_itm[0]:.1f}, {peak_sampler_loc_itm[1]:.1f}), "
    f"ground={peak_sampler_ground_asl:.1f} m ASL"
)
print(f"Monthly concentration plots use DEM-aware weighted 0-{monthly_agl_top_m:.0f} m AGL mean.")
print(
    f"Concentration species mode: {'NH3+NH4 (total)' if CONC_INCLUDE_NH4 else 'NH3 only'} "
    f"CONC_SPECIES_MODE={CONC_SPECIES_MODE})"
)
sampler_conc = []
peak_terrain_conc = []
domain_max_conc = []


def _sanitize_receptor_name(name, index):
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")
    return slug or f"receptor_{index + 1}"


def _sample_concentration_for_receptor(conc3d, receptor, nx, ny, nz, ground_height_grid, DZ):
    ix = int((float(receptor["x"]) - minx) // GRID_RES)
    iy = int((float(receptor["y"]) - miny) // GRID_RES)
    if not (0 <= ix < nx and 0 <= iy < ny):
        return 0.0

    receptor_height_m_asl = float(ground_height_grid[ix, iy]) + float(receptor["height_m_agl"])
    receptor_k = int(receptor_height_m_asl / DZ)
    if 0 <= receptor_k < nz:
        return float(conc3d[ix, iy, receptor_k])
    return 0.0


receptor_specs = site.load_receptors()
if receptor_specs:
    receptor_transformer = Transformer.from_crs("EPSG:4326", STUDY_CRS, always_xy=True)
    receptors = []
    for idx, item in enumerate(receptor_specs):
        x_val = float(item.get("x", 0.0))
        y_val = float(item.get("y", 0.0))
        if -180.0 <= x_val <= 180.0 and -90.0 <= y_val <= 90.0:
            x_val, y_val = receptor_transformer.transform(x_val, y_val)
        receptors.append(
            {
                "name": str(item.get("name") or f"receptor_{idx + 1}"),
                "x": x_val,
                "y": y_val,
                "height_m_agl": float(item.get("height_m_agl", 5.0)),
            }
        )
elif sampler_x_env is not None and sampler_y_env is not None:
    receptors = [{"name": "sampler", "x": float(sampler_x_env), "y": float(sampler_y_env), "height_m_agl": 5.0}]
else:
    receptors = [{"name": "center", "x": center_x, "y": center_y, "height_m_agl": 5.0}]

receptor_names = [receptor["name"] for receptor in receptors]
receptor_output_columns = {
    name: f"{_sanitize_receptor_name(name, idx)}_concentration_{species_unit}"
    for idx, name in enumerate(receptor_names)
}
receptor_conc_by_name = {name: [] for name in receptor_names}

for receptor in receptors:
    print(
        f"Receptor '{receptor['name']}': x={receptor['x']:.3f}, y={receptor['y']:.3f}, "
        f"height_m_agl={receptor['height_m_agl']:.3f}"
    )

# =========================
# OPTIMIZATION: Precompute all meteorological parameters
# =========================
print("\n" + "="*70)
print("PRECOMPUTING METEOROLOGICAL PARAMETERS (BATCH OPTIMIZATION)")
print("="*70)
t0_precomp = time.time()

# Preallocate arrays for all timesteps
u_array = np.zeros(total_steps, dtype=np.float32)
v_array = np.zeros(total_steps, dtype=np.float32)
Kxy_array = np.zeros(total_steps, dtype=np.float32)
Kz_low_array = np.zeros(total_steps, dtype=np.float32)
Kz_mid_array = np.zeros(total_steps, dtype=np.float32)
Kz_high_array = np.zeros(total_steps, dtype=np.float32)
P_dep_surf_array = np.zeros(total_steps, dtype=np.float32)
P_dep_nh4_array = np.zeros(total_steps, dtype=np.float32)
P_conversion_array = np.zeros(total_steps, dtype=np.float32)
P_wet_dep_array = np.zeros(total_steps, dtype=np.float32)
z_cap_array = np.zeros(total_steps, dtype=np.float32)
thermal_wind_scale_array = np.zeros(total_steps, dtype=np.float32)
emission_factor_array = np.zeros(total_steps, dtype=np.float32)
mean_temp = temp_fine.mean()
print(f"Computing parameters for {total_steps} timesteps...")

# Compute all parameters in advance
for step_counter in range(total_steps):
    if step_counter % (24 * timesteps_per_hour) == 0:
        print(f"  Precomputing day {step_counter//(24*timesteps_per_hour)+1}...")
    
    # Wind
    u = np.float32(u_wind_fine[step_counter])
    v = np.float32(v_wind_fine[step_counter])
    u_array[step_counter] = u
    v_array[step_counter] = v
    
    # Turbulence
    s = float(np.sqrt(u * u + v * v))
    ustar = calculate_ustar_with_stability(s, sshf=sshf_fine[step_counter])
    Kxy_array[step_counter] = np.float32(
        calculate_kxy_with_stability(ustar, float(blh_fine[step_counter]), sshf=sshf_fine[step_counter], kxy_scale=KXY_SCALE)
    )
    
    # Height-dependent Kz (update every hour)
    if (step_counter % timesteps_per_hour) == 0 or step_counter == 0:
        z_heights, kz_profile = calculate_kz_profile_with_heatflux(
            ustar, float(blh_fine[step_counter]), 
            sshf=sshf_fine[step_counter], 
            step_counter=step_counter,
            z_levels=[50.0, 300.0, 800.0]
        )
        Kz_low = max(float(kz_profile[0]), 0.1)
        Kz_mid = max(float(kz_profile[1]), 0.5)
        Kz_high = max(float(kz_profile[2]), 1.0)
    
    Kz_low_array[step_counter] = np.float32(Kz_low)*KZ_FACTOR   
    Kz_mid_array[step_counter] = np.float32(Kz_mid)*KZ_FACTOR
    Kz_high_array[step_counter] = np.float32(Kz_high)*KZ_FACTOR
    
    # Deposition velocities / scavenging
    temp_celsius = float(temp_fine[step_counter])
    if DEPOSITION_ENABLED:
        # Dry deposition is taken and adapted from Webster, H.N. and D.J. Thomson, Dry deposition modelling in a Lagrangian dispersion model. International Journal of Environment and Pollution, 2011. 47(1-4): p. 1–9. The term f was ommited to allow for pre-computation of deposition. 
        v_d_temp = v_d * (1.0 + 0.04 * (temp_celsius - 15.0))
        v_d_temp = max(0.01, min(0.05, v_d_temp))
        v_d_nh4_temp = v_d_nh4
        H_raw = 2*np.sqrt(Kz_low * dt)  # rough estimate of surface layer height
        H_min = 0.2*blh_fine[step_counter]
        H_max = 120.0
        H = min(H_max, max(H_min, H_raw))*H_FACTOR
        P_dep_surf_array[step_counter] = np.float32(min(1.0, (v_d_temp * dt)/H))
        P_dep_nh4_array[step_counter] = np.float32(min(1.0, (v_d_nh4_temp * dt)/H))

        # Wet deposition is a simple first order differential equation, with a constant scavenging coefficient based on rain intensity derived from the classic Jylhä, ‘Empirical Scavenging Coefficients of Radioactive Substances Released from Chernobyl’. ADMS uses the same approach
        rain_intensity = float(precip_fine[step_counter])
        if rain_intensity > 0.01:
            lambda_wet = 1.0e-4 * (rain_intensity ** 0.64)
            P_wet_dep_array[step_counter] = np.float32(min(1.0, lambda_wet * dt))
        else:
            P_wet_dep_array[step_counter] = np.float32(0.0)
    else:
        P_dep_surf_array[step_counter] = np.float32(0.0)
        P_dep_nh4_array[step_counter] = np.float32(0.0)
        P_wet_dep_array[step_counter] = np.float32(0.0)
    
    # Chemistry conversion
    # if SPECIES == "NH3":
    #     rh_current = float(rh_fine[step_counter])
    #     cloud_current = float(cloud_fine[step_counter])
    #     solar_current = float(solar_fine[step_counter])
        
    #     temp_factor_nitrate = np.exp(0.05 * (10.0 - temp_celsius))
    #     temp_factor_nitrate = max(0.2, min(3.0, temp_factor_nitrate))
        
    #     if rh_current > 70.0:
    #         rh_factor = 1.0 + 2.0 * (rh_current - 70.0) / 30.0
    #     else:
    #         rh_factor = 0.5 + 0.5 * (rh_current / 70.0)
        
    #     solar_factor = min(1.0, solar_current / 400.0)
    #     solar_factor *= (1.0 - 0.7 * cloud_current)

    #     if (SO2_CONCENTRATION is not None):
    #         so2_concentration = float(SO2_CONCENTRATION)
    #     else:
    #         so2_concentration = 1.2

    #     if (NOX_CONCENTRATION is not None):
    #         nox_concentration = float(NOX_CONCENTRATION)
    #     else:
    #         nox_concentration = 40

        
    #     lambda_so4 = k_conversion_so2 * so2_concentration * rh_factor
    #     lambda_no3 = k_conversion_nox * nox_concentration * temp_factor_nitrate * rh_factor
    #     lambda_photo = k_photo_nh3 * solar_factor
    #     lambda_conversion = lambda_so4 + lambda_no3 + lambda_photo
    #     P_conversion_array[step_counter] = np.float32(min(1.0, lambda_conversion * dt))
    
    # Boundary layer cap (AGL)
    z_cap = float(min(float(blh_fine[step_counter]), Z_MAX))
    if z_cap < 2.0 * DZ:
        z_cap = 2.0 * DZ
    z_cap_array[step_counter] = np.float32(z_cap)
    
    # Thermal (katabatic/anabatic) slope-wind scale from surface heat flux
    # Prandtl-like: velocity ~ |Q_H/(rho*cp)|^(1/3), sign follows sshf
    # ERA5 sshf is in J/m^2 (accumulated per hour); convert to W/m^2
    sshf_val = float(sshf_fine[step_counter]) / 3600.0  # W/m^2
    if abs(sshf_val) > 1.0:
        thermal_wind_scale_array[step_counter] = np.float32(
            THERMAL_WIND_ALPHA * np.sign(sshf_val) * abs(sshf_val / 1200.0) ** (1.0 / 3.0)
        )
    else:
        thermal_wind_scale_array[step_counter] = np.float32(0.0)
    
    current_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") + timedelta(
        hours=step_counter / timesteps_per_hour
    )
    month = current_datetime.month
    # There must be a better way to do this. Right now the priors are only guesswork. But they can be site specific with modification factors. 
    # It seems to me that the factors follow a sinusoidal pattern, doing a full revolution over the year, with the peaks at the start and the end of the year.
    # So it's probably better to replace it with a sinusoidal function, and modify the amplitude based on site-specific data, maybe year-around average temperature
    # So let's do that. But coming to think of it, a damping sinusoid is more appropriate, because the 2nd peak is actually dampened
    def exp(t, A, lbda):
        r"""y(t) = A \cdot \exp(-\lambda t)"""
        return A * np.exp(-lbda * t)

    def sine(t, omega, phi):
        r"""y(t) = \sin(\omega \cdot t + phi)"""
        return np.sin(omega * t + phi)

    def damped_sine(t, A, lbda, omega, phi):
        r"""y(t) = A \cdot \exp(-\lambda t) \cdot \left( \sin \left( \omega t + \phi ) \right)"""
        return exp(t, A, lbda) * sine(t, omega, phi)


    def _temperature_multiplier(sensitivity, temp_celsius, temp_reference, min_val=0.25, max_val=4.0):
        multiplier = float(np.exp(float(sensitivity) * (float(temp_celsius) - float(temp_reference))))
        return float(np.clip(multiplier, min_val, max_val))


    hour = (step_counter // timesteps_per_hour) % 24

    if SEASONAL_PROFILE_MODE == "temperature":
        temp_factor = _temperature_multiplier(
            TEMP_SENSITIVITY,
            temp_celsius,
            temp_ref,
            min_val=0.25,
            max_val=5.0,
        )
        emission_factor = temp_factor

    else:
        emission_factor = 1.0




    total_active_emission = emission_sources["total_emission_gps"].sum() * emission_factor
    
    if total_active_emission > 0:
        emission_factor_array[step_counter] = np.float32(total_active_emission / emission_sources["total_emission_gps"].sum())
    else:
        emission_factor_array[step_counter] = np.float32(0.5)
    
advective_step_array = np.sqrt(u_array.astype(np.float64) ** 2 + v_array.astype(np.float64) ** 2) * dt
diffusive_step_std_array = np.sqrt(2.0 * Kxy_array.astype(np.float64) * dt)

def _summarize_transport_scale(values, label):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        print(f"  {label}: no finite values")
        return
    print(
        f"  {label}: mean={np.mean(finite_values):.1f} m, "
        f"p50={np.percentile(finite_values, 50):.1f} m, "
        f"p90={np.percentile(finite_values, 90):.1f} m, "
        f"max={np.max(finite_values):.1f} m"
    )

print(f"✓ Precomputation complete in {time.time()-t0_precomp:.2f} seconds")
print(f"  Memory overhead: {(u_array.nbytes * 11) / 1024**2:.1f} MB for met parameters")
print("  Transport diagnostics for one model step:")
_summarize_transport_scale(advective_step_array, "advective displacement")
_summarize_transport_scale(diffusive_step_std_array, "diffusive random-walk std")
print("\n" + "="*70)
print("STARTING OPTIMIZED SIMULATION LOOP")
print("="*70)

# =========================
# Run (OPTIMIZED WITH PRECOMPUTED DATA)
sampler_avg = 0.0

# Parameters for height-dependent wind
z_ref = 10.0
z0_global = z0  # Store in local variable for reuse

output_dir = os.environ.get("OUTPUT_DIR")
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
else:
    dirs = glob.glob(os.path.join(site.outputs_dir, "*"))
    numeric_dirs = [d for d in dirs if os.path.isdir(d) and os.path.basename(d).isdigit()]
    if numeric_dirs:
        max_dir = max(numeric_dirs, key=lambda x: int(os.path.basename(x)))
        output_dir = os.path.join(site.outputs_dir, f"{os.path.basename(max_dir)}")
    else:
        output_dir = site.outputs_dir

os.makedirs(os.path.join(output_dir, 'simulation_frames3d'), exist_ok=True)

monthly_surface_sum = {}
monthly_surface_counts = {}
monthly_surface_sum_depo = {} if DEPOSITION_ENABLED else None
hourly_agl_collection = [] if HOURLY_MAPS == 1 else None

for global_step in range(total_steps):
    start = time.time()

    # Determine which season we are in for the current step
    current_month = (datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") + timedelta(hours=global_step / timesteps_per_hour)).month
    current_season = current_month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
    if unique_seasonal_factor_n > 1:
        seasonal_factor_step = seasonal_factor_n[:, current_season - 1]
    elif unique_seasonal_factor_n == 1 and seasonal_factor is not None:
        seasonal_factor_step = np.full((total_particles,), seasonal_factor[current_season - 1], dtype=np.float16)

    # Copy the seasonal factor for the current step to the device
    d_seasonal_factor_step = cuda.to_device(seasonal_factor_step.astype(np.float16))
    # Insert new row — positions and reset particle type to NH3
    fill_rows_and_reset_type_kernel[blocks, threads](
        d_x,
        d_y,
        d_z,
        d_particle_type,
        d_particle_mass_g,
        d_mass_budget,
        np.int32(head),
        d_x0,
        d_y0,
        d_z0,
    )
    set_scalar_kernel[1, 1](d_particle_mass_g, np.int32(head), np.float32(0.0))

    head = (head + 1) % f_step
    step_in_window = min(step_in_window + 1, f_step - 1)
    
    # Set emission factor from precomputed array
    particle_mass_value = np.float32(mass_per_particle * emission_factor_array[global_step])
    emitted_mass_total += float(particle_mass_value) * float(total_particles)

    set_scalar_kernel[1, 1](
        d_particle_mass_g,
        np.int32((head - 1) % f_step),
        particle_mass_value,
    )

    
    # Retrieve precomputed meteorological parameters
    u = u_array[global_step]
    v = v_array[global_step]
    Kxy = Kxy_array[global_step]
    Kz_low = Kz_low_array[global_step]
    Kz_mid = Kz_mid_array[global_step]
    Kz_high = Kz_high_array[global_step]
    P_dep_surf = P_dep_surf_array[global_step]
    P_dep_nh4 = P_dep_nh4_array[global_step]
    P_conversion = 0.0  # P_conversion_array[global_step]  # Currently disabled
    P_wet_dep = P_wet_dep_array[global_step]
    z_cap = z_cap_array[global_step]
    current_blh = float(blh_fine[global_step])
    sshf = float(sshf_fine[global_step])


    # Particle update (runs over j = 0..step_in_window)
    update_particles_3d_gpu_step(
        step_in_window,
        head,
        f_step,
        d_x,
        d_y,
        d_z,
        d_particle_type,
        u,
        v,
        z_ref,
        z0_global,
        current_blh,
        Kxy,
        Kz_low,
        Kz_mid,
        Kz_high,
        np.float32(dt),
        P_dep_surf,
        P_dep_nh4,
        P_conversion,
        P_wet_dep,
        np.float32(w_mean),
        np.float32(w_settle),
        d_particle_mass_g,
        np.float32(timesteps_per_hour),
        d_depo,
        np.float32(GRID_RES),
        np.float32(minx),
        np.float32(miny),
        np.int32(nx),
        np.int32(ny),
        np.float32(DZ),
        z_cap,
        rng_states,
        blocks,
        threads,
        np.float32(sshf),
        d_mass_budget,
        d_dem,
        dem_nx,
        dem_ny,
        dem_minx,
        dem_miny,
        dem_dx,
        dem_dy,
        d_dh_dx,
        d_dh_dy,
        d_initial_release_vz,
        d_seasonal_factor_step
    )

    # Apply exponential decay to all particle masses
    decay_factor = np.float32(np.exp(-decay_rate * dt))
    decay_mass_kernel[decay_blocks, threads](d_particle_mass_g, decay_factor, np.int32(f_step))


    
    
    #col_avg = col_avg_monthly
    #conc3d_swapped = conc3d.transpose(1,2,0)
    #th = int(conc3d_swapped.shape[2]/2)
    #col_avg = conc3d_swapped[:,1:200,th-10:th+10].mean(axis=2)
    


    if (global_step % sample_interval) == 0:
        print("Step", global_step, "of", total_steps, '...')

        # Zero persistent device conc3d buffer in-place
        zero_3d_kernel[conc3d_blocks, threads](d_conc3d)
        update_grid_3d_gpu_step(
            step_in_window,
            head,
            f_step,
            GRID_RES,
            DZ,
            minx,
            miny,
            d_x,
            d_y,
            d_z,
            d_particle_type,
            d_conc3d,
            d_particle_mass_g,
            timesteps_per_hour,
            nx,
            ny,
            nz,
            CONC_INCLUDE_NH4,
            d_dem,
            threads=threads,
            seasonal_factor_step=d_seasonal_factor_step
        )

        depo_grid = d_depo.copy_to_host()
        if SPECIES == "NH3":
            conc3d = d_conc3d.copy_to_host()
        elif SPECIES == "ODOR":
            conc3d = d_conc3d.copy_to_host()*100
            
        minutes_per_step = 60.0 / timesteps_per_hour
        total_minutes = global_step * minutes_per_step
        hours_i = int(total_minutes // 60)
        minutes_i = int(total_minutes % 60)
        # Take minute for hourly average
        if HOURLY_MAPS == 1:
            # Get hourly average AGL concentration and save it to geotiff
            hourly_average_agl = agl_band_mean_weighted(conc3d, ground_height_grid, DZ, monthly_agl_top_m)
            hourly_average_agl = np.clip(hourly_average_agl, 0, None)  # Ensure no negative values
            hourly_average_agl = hourly_average_agl.astype(np.float32)
            hourly_agl_collection.append(hourly_average_agl)
            
        
        scale_factors = (1, 1, 1)  # Downscale by half in each dimension

        sample_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') + timedelta(
            minutes=global_step * (60.0 / timesteps_per_hour)
        )
        month_key = sample_datetime.strftime('%Y-%m')
        previous_month_key = (sample_datetime - relativedelta(months=1)).strftime("%Y-%m")

        col_avg_monthly = agl_band_mean_weighted(conc3d, ground_height_grid, DZ, monthly_agl_top_m)
        if month_key not in monthly_surface_sum:
            monthly_surface_sum[month_key] = np.zeros((nx, ny), dtype=np.float64)
            if DEPOSITION_ENABLED:
                monthly_surface_sum_depo[month_key] = depo_grid.astype(np.float64)
            monthly_surface_counts[month_key] = 0
        monthly_surface_sum[month_key] += col_avg_monthly.astype(np.float64)
        if DEPOSITION_ENABLED:
            try:
                monthly_surface_sum_depo[month_key] = monthly_surface_sum_depo[previous_month_key] - depo_grid.astype(np.float64)
            except Exception:
                monthly_surface_sum_depo[month_key] = depo_grid.astype(np.float64)

        monthly_surface_counts[month_key] += 1
        
        receptor_values = {}
        for receptor in receptors:
            receptor_value = _sample_concentration_for_receptor(
                conc3d, receptor, nx, ny, nz, ground_height_grid, DZ
            )
            receptor_values[receptor["name"]] = receptor_value
            receptor_conc_by_name[receptor["name"]].append(receptor_value)

        primary_receptor_name = receptor_names[0]
        sampler = receptor_values[primary_receptor_name]
        sampler_avg += sampler
        num_samples = (global_step // sample_interval) + 1
        print(f"Sampler concentration for '{primary_receptor_name}' '{species_unit}':", sampler_avg / num_samples)
        sampler_conc.append(sampler)  # store in {species_unit}
        peak_terrain_value = float(col_avg_monthly[peak_ix, peak_iy])
        domain_max_value = float(np.max(col_avg_monthly))
        peak_terrain_conc.append(peak_terrain_value)
        domain_max_conc.append(domain_max_value)
        print(f"Peak-terrain concentration ({species_unit}):", peak_terrain_value)
        print(f"Domain max concentration ({species_unit}):", domain_max_value)


        if (global_step % 1) == 0:
            #Calculate frame's date and time 

            start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            frame_date = start_datetime + timedelta(hours=hours_i, minutes=minutes_i)
            
            # Create DataFrame with matching timestamps for each concentration value
            timestamps = []
            for i, conc in enumerate(sampler_conc):
                step_time = start_datetime + timedelta(minutes=i * sample_interval * (60.0 / timesteps_per_hour))
                timestamps.append(step_time.strftime('%Y-%m-%d %H:%M'))
            
            # Clean filename (remove spaces and colons)
            start_clean = start_date.replace(' ', '_').replace(':', '-')
            end_clean = end_date.replace(' ', '_').replace(':', '-')


            data = {
                'date_time': timestamps,
                f'peak_terrain_concentration_{species_unit}': peak_terrain_conc,
                f'domain_max_concentration_{species_unit}': domain_max_conc,
            }
            for receptor_name in receptor_names:
                data[receptor_output_columns[receptor_name]] = receptor_conc_by_name[receptor_name]
            if len(receptor_names) == 1:
                data[f'concentration_{species_unit}'] = receptor_conc_by_name[receptor_names[0]]
            df = pd.DataFrame(data)
            filename = os.path.join(output_dir, f'sampler_timeseries_{start_clean}_{end_clean}.csv')

            df.to_csv(filename, index_label='time_step_100s')
            print(f"✓ Saved sampler timeseries: {filename}")
            if PRODUCE_MAPS == 1:
                conc_agl = agl_band_mean_weighted(
                    conc3d,
                    ground_height_grid,
                    DZ,
                    50,
                )
           
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.style.use("dark_background")
                masked_grid = np.ma.masked_where(
                    conc_agl.T < 1e-9, conc_agl.T  # Lower threshold to show very small values
                )

                # Create a DEM background using hillshade
                dem_hillshade = hillshade(dem_array, azimuth=315, angle_altitude=45)

                ax.imshow(
                    dem_hillshade, 
                    cmap="gray", 
                    extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
                    zorder=0,
                    alpha=1,
                )

                im = ax.imshow(
                    masked_grid, 
                    origin="lower",
                    cmap="hot",
                    extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
                    zorder=1,
                    alpha=0.9,
                    vmin=0,
                    vmax=10
                )

                # Add date and time annnotation as text in the top-left corner
                ax.text(0.02, 0.95, frame_date.strftime('%Y-%m-%d %H:%M'), transform=ax.transAxes, fontsize=12, color="white", verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

                cax = inset_axes(ax, width="5%", height="50%", loc="lower right", borderpad=1.0)
                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.set_label(f"Avg ambient {species_label} ({species_unit})\n(0-50 m AGL mean)", fontsize=12, color="white", bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                cbar.ax.tick_params(labelsize=12, colors="white")
                
                cbar.ax.yaxis.set_ticks_position("left")
                cbar.ax.yaxis.set_label_position("left")

                # Tick-label backgrounds
                for ticklabel in cbar.ax.get_yticklabels():
                    ticklabel.set_bbox(dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
)

                
                ax.axis("off")
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                # Save to site outputs directory with timestamp
                fig.savefig(os.path.join(output_dir, f"fig{global_step}.png"), dpi=300, bbox_inches='tight',pad_inches=0)
                plt.close(fig)

 

    remove_outofbound_3d_gpu_step(
        step_in_window,
        head,
        f_step,
        GRID_RES,
        DZ,
        minx,
        miny,
        d_x,
        d_y,
        d_z,
        d_particle_mass_g,
        nx,
        ny,
        nz,
        d_dem,
        d_mass_budget,
        threads=threads,
    )


    end = time.time()
    #print(end - start, "seconds for step", global_step)

# Save final receptor timeseries if not already written
if receptor_names and receptor_conc_by_name[receptor_names[0]]:
    timestamps = []
    for i, _ in enumerate(receptor_conc_by_name[receptor_names[0]]):
        step_time = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=i * sample_interval * (60.0 / timesteps_per_hour))
        timestamps.append(step_time.strftime('%Y-%m-%d %H:%M'))

    data = {
        'date_time': timestamps,
        f'peak_terrain_concentration_{species_unit}': peak_terrain_conc,
        f'domain_max_concentration_{species_unit}': domain_max_conc,
    }
    for receptor_name in receptor_names:
        data[receptor_output_columns[receptor_name]] = receptor_conc_by_name[receptor_name]
    if len(receptor_names) == 1:
        data[f'concentration_{species_unit}'] = receptor_conc_by_name[receptor_names[0]]
    df = pd.DataFrame(data)
    filename = os.path.join(output_dir, f'sampler_timeseries_{start_clean}_{end_clean}.csv')
    df.to_csv(filename, index_label='time_step_100s')
    print(f"✓ Saved final sampler timeseries: {filename}")

# =========================
# Copy final results from device
# =========================
print("\nCopying final deposition grid from device...")
depo_grid = d_depo.copy_to_host()
x = d_x.copy_to_host()
particle_mass_g = d_particle_mass_g.copy_to_host()
mass_budget = d_mass_budget.copy_to_host().astype(np.float64)

start_clean = start_date.replace(' ', '_').replace(':', '-')
end_clean = end_date.replace(' ', '_').replace(':', '-')




active_counts = np.count_nonzero(x != SENTINEL, axis=1).astype(np.float64)
airborne_mass_total = float(np.sum(active_counts * particle_mass_g.astype(np.float64)))

dry_dep_mass_total = float(mass_budget[MASS_BUDGET_DRY_DEP])
wet_dep_mass_total = float(mass_budget[MASS_BUDGET_WET_DEP])
oob_horizontal_mass_total = float(mass_budget[MASS_BUDGET_OOB_HORIZONTAL])
oob_vertical_mass_total = float(mass_budget[MASS_BUDGET_OOB_VERTICAL])
oob_mass_total = oob_horizontal_mass_total + oob_vertical_mass_total
aged_out_mass_total = float(mass_budget[MASS_BUDGET_AGED_OUT])
accounted_mass_total = dry_dep_mass_total + wet_dep_mass_total + oob_mass_total + aged_out_mass_total + airborne_mass_total
unaccounted_mass_total = emitted_mass_total - accounted_mass_total

print("Mass budget diagnostics (emission units over simulation period):")
print(f"  Emitted mass: {emitted_mass_total:.6e}")
print(f"  Dry deposition loss: {dry_dep_mass_total:.6e}")
print(f"  Wet deposition loss: {wet_dep_mass_total:.6e}")
print(f"  Out-of-bounds loss: {oob_mass_total:.6e}")
print(f"    Horizontal OOB loss: {oob_horizontal_mass_total:.6e}")
print(f"    Vertical OOB loss: {oob_vertical_mass_total:.6e}")
print(f"  Aged-out ring-buffer loss: {aged_out_mass_total:.6e}")
print(f"  Final airborne mass: {airborne_mass_total:.6e}")
print(f"  Accounted mass: {accounted_mass_total:.6e}")
print(f"  Unaccounted residual: {unaccounted_mass_total:.6e}")

mass_budget_file = os.path.join(output_dir, f'mass_budget_{start_clean}_{end_clean}.json')
with open(mass_budget_file, 'w') as mbf:
    json.dump({
        'emitted_mass_g': emitted_mass_total,
        'dry_deposition_mass_g': dry_dep_mass_total,
        'wet_deposition_mass_g': wet_dep_mass_total,
        'out_of_bounds_mass_g': oob_mass_total,
        'out_of_bounds_horizontal_mass_g': oob_horizontal_mass_total,
        'out_of_bounds_vertical_mass_g': oob_vertical_mass_total,
        'aged_out_mass_g': aged_out_mass_total,
        'final_airborne_mass_g': airborne_mass_total,
        'accounted_mass_g': accounted_mass_total,
        'unaccounted_residual_g': unaccounted_mass_total,
    }, mbf, indent=2)
print(f"✓ Saved mass budget: {mass_budget_file}")

# Convert depo_grid from grams-NH3 per cell (simulation period) to kg N / yr / ha
# NH3→N mass ratio: 14.007/17.031, g→kg: /1000, cell→ha: *10000/GRID_RES², period→year: *8766/total_hours
if DEPOSITION_ENABLED:
    cell_area_m2 = GRID_RES * GRID_RES
    hours_per_year = 365.25 * 24
    nh3_to_n = 14.007 / 17.031
    depo_grid = (depo_grid
                 * nh3_to_n            # g NH3 → g N
                 / 1000.0              # g → kg
                 * (10000.0 / cell_area_m2)  # per cell → per ha
                 * (hours_per_year / total_hours))  # sim period → year

    total_deposition = np.sum(depo_grid)
    max_deposition = np.max(depo_grid)
    min_deposition = np.min(depo_grid)
    nonzero_cells = np.count_nonzero(depo_grid)
    print(f"Deposition statistics (kg N/yr/ha):")
    print(f"  Total deposition (sum): {total_deposition:.4f} kg N/yr/ha")
    print(f"  Max deposition: {max_deposition:.6f} kg N/yr/ha")
    print(f"  Min deposition: {min_deposition:.6f} kg N/yr/ha")
    print(f"  Non-zero cells: {nonzero_cells}/{depo_grid.size} ({nonzero_cells/depo_grid.size*100:.1f}%)")

    # Save deposition array
    depo_array_file = os.path.join(output_dir, f'deposition_array_{start_clean}_{end_clean}.npy')
    np.save(depo_array_file, depo_grid)
    print(f"✓ Saved deposition array: {depo_array_file}")
else:
    print("Deposition disabled: skipping deposition conversion and deposition array output.")

# Save grid metadata for combined plotting in run_parallel_seasons.py
grid_meta = {
    'minx_3857': float(minx_3857), 'maxx_3857': float(maxx_3857),
    'miny_3857': float(miny_3857), 'maxy_3857': float(maxy_3857),
    'nx': int(nx), 'ny': int(ny),
    'GRID_RES': float(GRID_RES),
    'total_hours': int(total_hours),
}
meta_file = os.path.join(output_dir, f'grid_metadata_{start_clean}_{end_clean}.json')
with open(meta_file, 'w') as mf:
    json.dump(grid_meta, mf)

# =========================
# Final outputs
# =========================

if monthly_surface_sum:
    save_monthly_agl_plots(
        monthly_surface_sum,
        monthly_surface_counts,
        output_dir,
        minx_3857,
        maxx_3857,
        miny_3857,
        maxy_3857,
        monthly_agl_top_m,
        species_label,
        species_unit
    )
    if DEPOSITION_ENABLED:
        save_monthly_depo_plots(
            monthly_surface_sum_depo,
            monthly_surface_counts,
            output_dir,
            minx_3857,
            maxx_3857,
            miny_3857,
            maxy_3857,
            species_label,
            species_unit_depo
        )
else:
    print("No monthly concentration samples were collected; monthly AGL plots were not generated.")


if DEPOSITION_ENABLED:
    save_annual_depo_plots(
        depo_grid,
        output_dir,
        minx_3857,
        maxx_3857,
        miny_3857,
        maxy_3857,
        species_label,
        species_unit_depo
    )

if HOURLY_MAPS == 1:
        tif_path = os.path.join(output_dir, f"hourly_98th_{sample_datetime.strftime('%Y-%m')}.tif")
        # Take the 98 percentile of the hourly averge AGL concentration to avoid extreme outliers
        hourly_average_agl = np.percentile(np.array(hourly_agl_collection), 98, axis=0)
        nx, ny = hourly_average_agl.shape
        pixel_width = (maxx - minx) / nx
        pixel_height = (maxy - miny) / ny
        transform = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=nx,
            width=ny,
            count=1,
            dtype=hourly_average_agl.dtype,
            crs='EPSG:2157',
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(hourly_average_agl.T[::-1, :], 1)