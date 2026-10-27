import sys

from plam_sites import site_manager
from plam_sites.fetch_era5_entire_area import fetch_era5_weather_chunked
from plam_sites.fetch_dem import fetch_dem
if __name__ == "__main__":
    action = input('Welcome to the PLAM Bulk Weather Manager. If you wish to bulk download weather data for an entire region, press Enter to continue...\n')
    if action == '':
        start_date = input('Please enter the start date of the data you want to download (YYYY-MM-DD):\n')
        end_date = input('Please enter the end date of the data you want to download (YYYY-MM-DD):\n')
        lat_max = float(input('Please enter the maximum latitude of the area:\n'))
        lat_min = float(input('Please enter the minimum latitude of the area:\n'))
        lon_max = float(input('Please enter the maximum longitude of the area:\n'))
        lon_min = float(input('Please enter the minimum longitude of the area:\n'))
        fetch_era5_weather_chunked(start_date=start_date, end_date=end_date,
                                   lat_min=lat_min, lat_max=lat_max,
                                   lon_min=lon_min, lon_max=lon_max)

    else:
        print('Exiting.')
        sys.exit(1)