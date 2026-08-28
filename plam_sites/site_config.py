"""
Site Configuration Management for PLAM

This module provides centralized management of study site paths and configurations.
Each study site has its own directory with all necessary data files.

Usage:
    from site_config import SiteConfig
    
    # Get current site configuration
    site = SiteConfig.get_current_site()
    
    # Access site-specific paths
    housing_path = site.housing_shapefile
    weather_path = site.weather_data
    
    # Switch to a different site
    SiteConfig.set_current_site("killycony")
    
    # List all available sites
    sites = SiteConfig.list_sites()
"""

import os
import csv
import json
import configparser
from pathlib import Path
from typing import Dict, Optional, List
from pathlib import Path


class SiteConfig:
    """Manages study site configurations and file paths."""
    
    # Root directory for all site data
    SITES_ROOT = Path.cwd() / "sites"
    SITES_ROOT.mkdir(exist_ok=True)
    # Configuration file path
    CONFIG_FILE = Path(__file__).parent / "config.ini"
    
    def __init__(self, site_name: str):
        """
        Initialize site configuration.
        
        Args:
            site_name: Name of the study site (e.g., 'clara_bog', 'killycony')
        """
        self.site_name = site_name
        self.site_dir = self.SITES_ROOT / site_name
        
        # Ensure site directory exists
        self.site_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self._create_subdirectories()
    
    def _create_subdirectories(self):
        """Create standard subdirectory structure for the site."""
        subdirs = [
            'study_area',      # Study rectangle shapefiles
            'imagery',         # Satellite imagery and bounds
            'sources',         # Animal housing source data
            'weather',         # ERA5 weather data
            'dem',             # Digital elevation models
            'outputs',         # Simulation results
            'cache',           # Cached data (detections, etc.)
        ]
        
        for subdir in subdirs:
            (self.site_dir / subdir).mkdir(exist_ok=True)
    
    # ==================== Path Properties ====================
    
    @property
    def study_rectangle(self) -> str:
        """Path to study rectangle shapefile."""
        return str(self.site_dir / "study_area" / "study_rectangle.shp")
    
    @property
    def housing_shapefile(self) -> str:
        """Path to animal housing shapefile."""
        return str(self.site_dir / "sources" / "animal_housing_reviewed.shp")
    
    @property
    def weather_data(self) -> str:
        """Path to weather data CSV."""
        return str(self.site_dir / "weather" / "weather_data.csv")
    
    @property
    def imagery_dir(self) -> str:
        """Directory for satellite imagery."""
        return str(self.site_dir / "imagery")
    
    @property
    def full_area_google(self) -> str:
        """Path to Google satellite imagery."""
        return str(self.site_dir / "imagery" / "full_area_z19.png")
    
    @property
    def full_area_bing(self) -> str:
        """Path to Bing/ESRI satellite imagery."""
        return str(self.site_dir / "imagery" / "full_area_bing_z19.png")
    
    @property
    def google_bounds(self) -> str:
        """Path to Google imagery bounds file."""
        return str(self.site_dir / "imagery" / "full_area_z19_bounds.txt")
    
    @property
    def bing_bounds(self) -> str:
        """Path to Bing imagery bounds file."""
        return str(self.site_dir / "imagery" / "full_area_bing_z19_bounds.txt")
    
    @property
    def sources_shapefile(self) -> str:
        """Path to detected buildings shapefile."""
        return str(self.site_dir / "sources" / "sources.shp")
    
    @property
    def detection_cache(self) -> str:
        """Path to building detection cache."""
        return str(self.site_dir / "cache" / "detected_buildings_cache.pkl")
    
    @property
    def review_progress(self) -> str:
        """Path to review progress cache."""
        return str(self.site_dir / "cache" / "review_progress.pkl")
    
    @property
    def dem_dir(self) -> str:
        """Directory for DEM files."""
        return str(self.site_dir / "dem")
    
    @property
    def outputs_dir(self) -> str:
        """Directory for simulation outputs."""
        return str(self.site_dir / "outputs")

    @property
    def receptors_file(self) -> str:
        """Path to the JSON file storing receptor definitions."""
        return str(self.site_dir / "receptors.json")

    @property
    def receptors_csv(self) -> str:
        """Path to the CSV file storing receptor definitions."""
        return str(self.site_dir / "receptors.csv")
    
    @property
    def cache_dir(self) -> str:
        """Directory for cached files."""
        return str(self.site_dir / "cache")

    def load_receptors(self) -> List[Dict[str, object]]:
        """Load receptor definitions for this site from JSON or CSV files."""
        json_path = Path(self.receptors_file)
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except Exception:
                raw = None

            if isinstance(raw, dict):
                raw_items = raw.get("receptors", [])
            elif isinstance(raw, list):
                raw_items = raw
            else:
                raw_items = None

            if raw_items is not None:
                receptors = self._parse_receptor_items(raw_items)
                if receptors:
                    return receptors

        csv_path = Path(self.receptors_csv)
        if csv_path.exists():
            return self._parse_receptor_csv(csv_path)

        return []

    def _parse_receptor_items(self, raw_items) -> List[Dict[str, object]]:
        receptors: List[Dict[str, object]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or f"receptor_{idx + 1}").strip()
            if not name:
                name = f"receptor_{idx + 1}"

            x = item.get("x", item.get("easting"))
            y = item.get("y", item.get("northing"))
            if x is None or y is None:
                continue

            try:
                x_val = float(x)
                y_val = float(y)
            except (TypeError, ValueError):
                continue

            height_m_agl = item.get("height_m_agl", item.get("height_m", 5.0))
            try:
                height_val = float(height_m_agl)
            except (TypeError, ValueError):
                height_val = 5.0

            receptors.append({
                "name": name,
                "x": x_val,
                "y": y_val,
                "height_m_agl": height_val,
            })

        return receptors

    def _parse_receptor_csv(self, path: Path) -> List[Dict[str, object]]:
        receptors: List[Dict[str, object]] = []
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames:
                    return receptors

                field_map = {}
                for field in reader.fieldnames:
                    normalized = (field or "").strip().lower()
                    if normalized in {"name", "receptor", "receptor_name", "id"}:
                        field_map["name"] = field
                    elif normalized in {"x", "longitude", "lon", "easting", "east"}:
                        field_map["x"] = field
                    elif normalized in {"y", "latitude", "lat", "northing", "north"}:
                        field_map["y"] = field
                    elif normalized in {"height", "height_m_agl", "height_m", "z", "z_m"}:
                        field_map["height_m_agl"] = field

                for idx, row in enumerate(reader):
                    name = (row.get(field_map.get("name", "")) or "").strip()
                    if not name:
                        name = f"receptor_{idx + 1}"

                    x_val = row.get(field_map.get("x", ""))
                    y_val = row.get(field_map.get("y", ""))
                    if x_val is None or y_val is None:
                        continue

                    try:
                        x_float = float(x_val)
                        y_float = float(y_val)
                    except (TypeError, ValueError):
                        continue

                    height_val = 5.0
                    height_cell = row.get(field_map.get("height_m_agl", ""))
                    if height_cell is not None and str(height_cell).strip() != "":
                        try:
                            height_val = float(height_cell)
                        except (TypeError, ValueError):
                            height_val = 5.0

                    receptors.append({
                        "name": name,
                        "x": x_float,
                        "y": y_float,
                        "height_m_agl": height_val,
                    })
        except Exception:
            return []

        return receptors

    def save_receptors(self, receptors: List[Dict[str, object]]):
        """Persist receptor definitions for this site to receptors.json."""
        path = Path(self.receptors_file)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(receptors, fh, indent=2)
        return path
    
    # ==================== Static Methods ====================
    
    @staticmethod
    def get_current_site() -> 'SiteConfig':
        """
        Get the current active site from config.ini.
        
        Returns:
            SiteConfig instance for the current site
        """
        config = configparser.ConfigParser()
        config.read(SiteConfig.CONFIG_FILE)
        
        # Get current site name, default to 'clara_bog'
        if 'SITE' in config and 'current_site' in config['SITE']:
            site_name = config['SITE']['current_site']
        else:
            site_name = 'clara_bog'
            # Save default to config
            SiteConfig.set_current_site(site_name)
        
        return SiteConfig(site_name)
    
    @staticmethod
    def set_current_site(site_name: str):
        """
        Set the current active site in config.ini.
        
        Args:
            site_name: Name of the site to activate
        """
        config = configparser.ConfigParser()
        config.read(SiteConfig.CONFIG_FILE)
        
        # Ensure SITE section exists
        if 'SITE' not in config:
            config['SITE'] = {}
        
        config['SITE']['current_site'] = site_name
        
        # Write back to config file
        with open(SiteConfig.CONFIG_FILE, 'w') as f:
            config.write(f)
        
        print(f"✓ Switched to site: {site_name}")
    
    @staticmethod
    def list_sites() -> List[str]:
        """
        List all available study sites.
        
        Returns:
            List of site names
        """
        if not SiteConfig.SITES_ROOT.exists():
            return []
        
        return [d.name for d in SiteConfig.SITES_ROOT.iterdir() if d.is_dir()]
    
    @staticmethod
    def create_site(site_name: str, description: Optional[str] = None) -> 'SiteConfig':
        """
        Create a new study site.
        
        Args:
            site_name: Name for the new site
            description: Optional description of the site
        
        Returns:
            SiteConfig instance for the new site
        """
        site = SiteConfig(site_name)
        
        # Create a metadata file
        metadata_file = site.site_dir / "site_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Site Name: {site_name}\n")
            if description:
                f.write(f"Description: {description}\n")
            from datetime import datetime
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✓ Created new site: {site_name}")
        print(f"  Directory: {site.site_dir}")
        
        return site
    
    def __repr__(self) -> str:
        return f"SiteConfig(site_name='{self.site_name}', site_dir='{self.site_dir}')"


# Convenience function for backward compatibility
def get_site_paths() -> Dict[str, str]:
    """
    Get all site-specific paths as a dictionary.
    
    Returns:
        Dictionary with all site paths
    """
    site = SiteConfig.get_current_site()
    return {
        'site_name': site.site_name,
        'study_rectangle': site.study_rectangle,
        'housing_shapefile': site.housing_shapefile,
        'fields_shapefile': site.fields_shapefile,
        'weather_data': site.weather_data,
        'full_area_google': site.full_area_google,
        'full_area_bing': site.full_area_bing,
        'google_bounds': site.google_bounds,
        'bing_bounds': site.bing_bounds,
        'buildings_detected': site.buildings_detected,
        'detection_cache': site.detection_cache,
        'review_progress': site.review_progress,
        'dem_dir': site.dem_dir,
        'outputs_dir': site.outputs_dir,
        'cache_dir': site.cache_dir,
    }


if __name__ == "__main__":
    # Demo usage
    print("Available sites:", SiteConfig.list_sites())
    
    site = SiteConfig.get_current_site()
    print(f"\nCurrent site: {site.site_name}")
    print(f"Study rectangle: {site.study_rectangle}")
    print(f"Housing shapefile: {site.housing_shapefile}")
    print(f"Weather data: {site.weather_data}")
