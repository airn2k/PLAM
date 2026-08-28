#!/usr/bin/env python3
"""
PLAM Site Management CLI

Simple command-line tool for managing study sites.

Usage:
    python site_manager.py list                    # List all sites
    python site_manager.py current                 # Show current site
    python site_manager.py switch <site_name>      # Switch to a site
    python site_manager.py create <site_name>      # Create new site
    python site_manager.py info [site_name]        # Show site info
    python site_manager.py paths [site_name]       # Show site paths
"""

import sys
import numpy as np
import argparse
from pathlib import Path
from plam_sites.site_config import SiteConfig
import geopandas as gpd
from plam_sites.create_study_area import create_study_rectangle


def _get_site(site_name: str = None):
    if site_name is None:
        return SiteConfig.get_current_site()
    return SiteConfig(site_name)


def list_sites():
    """List all available sites."""
    sites = SiteConfig.list_sites()
    current = SiteConfig.get_current_site().site_name
    
    print("\nAvailable sites:")
    print("=" * 50)
    
    if not sites:
        print("  No sites found. Create one with: site_manager.py create <name>")
        return
    
    for site_name in sorted(sites):
        marker = "→" if site_name == current else " "
        print(f"  {marker} {site_name}")
    
    print(f"\nCurrent site: {current}")
    print("=" * 50)


def show_current():
    """Show current site details."""
    site = SiteConfig.get_current_site()
    print(f"\nCurrent site: {site.site_name}")
    print(f"Directory: {site.site_dir}")
    
    # Check which files exist
    print("\nData availability:")
    checks = [
        ("Study area", site.study_rectangle),
        ("Animal housing", site.housing_shapefile),
        ("Weather data", site.weather_data),
        ("Google imagery", site.full_area_google),
        ("Bing imagery", site.full_area_bing),
        ("Building detections", site.buildings_detected),
        ("Detection cache", site.detection_cache),
    ]
    
    for name, path in checks:
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")


def switch_site(site_name: str):
    """Switch to a different site."""
    sites = SiteConfig.list_sites()
    
    if site_name not in sites:
        print(f"Error: Site '{site_name}' does not exist.")
        print(f"Available sites: {', '.join(sites)}")
        create_site(site_name)
        return
    
    SiteConfig.set_current_site(site_name)
    print(f"✓ Switched to site: {site_name}")


def create_site(site_name: str):
    """Create a new site."""
    sites = SiteConfig.list_sites()
    
    if site_name in sites:
        print(f"Error: Site '{site_name}' already exists.")
        return
    
    # Get optional description
    print(f"Creating new site: {site_name}")
    description = input("Enter description (optional): ").strip()
    
    site = SiteConfig.create_site(site_name, description or None)
    print(f"\nNew site created at: {site.site_dir}")
    print("\nSwitching:")
    switch_site(site_name)
    lat = np.float64(input("Enter site's centre latitude in degrees:\n"))
    lon = np.float64(input("Enter site's centre longitude in degrees:\n "))
    size = np.float64(input("Enter study domain diameter in km:\n"))
    create_study_rectangle(lat, lon, size, site)
    start = input("Enter the start date of the study in YYYY-MM-DD format:\n")
    end = input("Enter the end date of the study in YYYY-MM-DD format:\n")
    print(f"  3. Fetch ERA5 weather data: python fetch_era5_weather_chunked.py  --START YYYY-MM-DD --END YYYY-MM-DD")
 


def show_info(site_name: str = None):
    """Show detailed site information."""
    if site_name is None:
        site = SiteConfig.get_current_site()
    else:
        sites = SiteConfig.list_sites()
        if site_name not in sites:
            print(f"Error: Site '{site_name}' does not exist.")
            return
        site = SiteConfig(site_name)
    
    print(f"\nSite: {site.site_name}")
    print("=" * 50)
    print(f"Directory: {site.site_dir}")
    
    # Read metadata if available
    metadata_file = site.site_dir / "site_metadata.txt"
    if metadata_file.exists():
        print("\nMetadata:")
        with open(metadata_file) as f:
            for line in f:
                print(f"  {line.rstrip()}")
    
    # Count files in each subdirectory
    print("\nData summary:")
    subdirs = ['study_area', 'imagery', 'sources', 'weather', 'dem', 'outputs', 'cache']
    for subdir in subdirs:
        subdir_path = site.site_dir / subdir
        if subdir_path.exists():
            file_count = len([f for f in subdir_path.iterdir() if f.is_file()])
            print(f"  {subdir:15} {file_count:3} files")
    
    # Try to get study area bounds if shapefile exists
    study_rect = Path(site.study_rectangle)
    if study_rect.exists():
        try:
            gdf = gpd.read_file(site.study_rectangle)
            bounds = gdf.total_bounds
            centroid = gdf.geometry.centroid.iloc[0]
            print(f"\nStudy area:")
            print(f"  Bounds: {bounds}")
            print(f"  Center: ({centroid.y:.4f}°N, {centroid.x:.4f}°E)")
            print(f"  CRS: {gdf.crs}")
        except Exception as e:
            print(f"\n  (Could not read study area: {e})")


def show_paths(site_name: str = None):
    """Show all paths for a site."""
    if site_name is None:
        site = SiteConfig.get_current_site()
    else:
        sites = SiteConfig.list_sites()
        if site_name not in sites:
            print(f"Error: Site '{site_name}' does not exist.")
            return
        site = SiteConfig(site_name)
    
    print(f"\nPaths for site: {site.site_name}")
    print("=" * 50)
    
    paths = [
        ("Study rectangle", site.study_rectangle),
        ("Housing shapefile", site.housing_shapefile),
        ("Fields shapefile", site.fields_shapefile),
        ("Weather data", site.weather_data),
        ("Google imagery", site.full_area_google),
        ("Bing imagery", site.full_area_bing),
        ("Google bounds", site.google_bounds),
        ("Bing bounds", site.bing_bounds),
        ("Buildings detected", site.buildings_detected),
        ("Detection cache", site.detection_cache),
        ("Review progress", site.review_progress),
        ("DEM directory", site.dem_dir),
        ("Outputs directory", site.outputs_dir),
        ("Cache directory", site.cache_dir),
    ]
    
    for name, path in paths:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"{exists} {name:20} {path}")


def show_receptors(site_name: str = None):
    """List configured receptors for a site."""
    site = _get_site(site_name)
    receptors = site.load_receptors()

    print(f"\nReceptors for site: {site.site_name}")
    print("=" * 50)
    if not receptors:
        print("  No receptors configured.")
        print("  Add one with: python site_manager.py add-receptor <name> <x> <y> [height_m_agl]")
        return

    for receptor in receptors:
        print(
            f"  - {receptor['name']}: x={receptor['x']}, y={receptor['y']}, "
            f"height_m_agl={receptor['height_m_agl']}"
        )


def add_receptor(name: str, x: float, y: float, height_m_agl: float = 5.0, site_name: str = None):
    """Add or update a receptor definition for the current site."""
    site = _get_site(site_name)
    receptors = site.load_receptors()

    existing = next((r for r in receptors if str(r.get("name", "")).lower() == name.lower()), None)
    if existing is None:
        receptors.append({"name": name, "x": x, "y": y, "height_m_agl": height_m_agl})
    else:
        existing["x"] = x
        existing["y"] = y
        existing["height_m_agl"] = height_m_agl

    site.save_receptors(receptors)
    print(f"✓ Saved receptor '{name}' for site: {site.site_name}")


def remove_receptor(name: str, site_name: str = None):
    """Remove a receptor definition from the current site."""
    site = _get_site(site_name)
    receptors = site.load_receptors()
    filtered = [r for r in receptors if str(r.get("name", "")).lower() != name.lower()]

    if len(filtered) == len(receptors):
        print(f"Receptor '{name}' was not found.")
        return

    site.save_receptors(filtered)
    print(f"✓ Removed receptor '{name}' from site: {site.site_name}")


def main():
    parser = argparse.ArgumentParser(
        description="PLAM Site Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    # List all sites
  %(prog)s current                 # Show current site
  %(prog)s switch clara_bog        # Switch to Clara Bog site
  %(prog)s create new_site         # Create new site
  %(prog)s info killycony          # Show info for Killycony site
  %(prog)s paths                   # Show paths for current site
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # list command
    subparsers.add_parser('list', help='List all available sites')
    
    # current command
    subparsers.add_parser('current', help='Show current site')
    
    # switch command
    switch_parser = subparsers.add_parser('switch', help='Switch to a different site')
    switch_parser.add_argument('site_name', help='Name of site to switch to')
    
    # create command
    create_parser = subparsers.add_parser('create', help='Create a new site')
    create_parser.add_argument('site_name', help='Name for the new site')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show detailed site information')
    info_parser.add_argument('site_name', nargs='?', help='Site name (default: current)')
    
    # paths command
    paths_parser = subparsers.add_parser('paths', help='Show all site paths')
    paths_parser.add_argument('site_name', nargs='?', help='Site name (default: current)')

    # receptors command
    receptors_parser = subparsers.add_parser('receptors', help='List configured receptors')
    receptors_parser.add_argument('site_name', nargs='?', help='Site name (default: current)')

    # add-receptor command
    add_receptor_parser = subparsers.add_parser('add-receptor', help='Add a receptor location')
    add_receptor_parser.add_argument('name', help='Receptor name')
    add_receptor_parser.add_argument('x', type=float, help='Receptor x-coordinate in site CRS (ITM)')
    add_receptor_parser.add_argument('y', type=float, help='Receptor y-coordinate in site CRS (ITM)')
    add_receptor_parser.add_argument('height_m_agl', nargs='?', type=float, default=5.0, help='Height above ground in meters (default: 5.0)')

    # remove-receptor command
    remove_receptor_parser = subparsers.add_parser('remove-receptor', help='Remove a receptor location')
    remove_receptor_parser.add_argument('name', help='Receptor name')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_sites()
    elif args.command == 'current':
        show_current()
    elif args.command == 'switch':
        switch_site(args.site_name)
    elif args.command == 'create':
        create_site(args.site_name)
    elif args.command == 'info':
        show_info(args.site_name)
    elif args.command == 'paths':
        show_paths(args.site_name)
    elif args.command == 'receptors':
        show_receptors(args.site_name)
    elif args.command == 'add-receptor':
        add_receptor(args.name, args.x, args.y, args.height_m_agl)
    elif args.command == 'remove-receptor':
        remove_receptor(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
