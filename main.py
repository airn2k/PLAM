import sys

from plam_sites import site_manager
from plam_sites.fetch_era5_weather_chunked import fetch_era5_weather_chunked
from plam_sites.fetch_dem import fetch_dem
if __name__ == "__main__":
    input('Welcome to the PLAM Sites Manager. Press Enter to continue...\n')
    site_name = input('Please enter the name of an existing  site or a new site.\n')
    site_manager.switch_site(site_name)
    action = input('Please select an action from the following options:\n' + ''.join(['1. Download ERA5 weather data\n', 
                                                                                       '2. Fetch DEM data\n',
                                                                                       '3. Run simulation\n']) + '\n\n')
    if action == '1':
        start_date = input('Please enter the start date of the data you want to download (YYYY-MM-DD):\n')
        end_date = input('Please enter the end date of the data you want to download (YYYY-MM-DD):\n')
        fetch_era5_weather_chunked(start_date=start_date, end_date=end_date)

    if action == '2':
        fetch_dem_res = input('Do you want to fetch DEM data for the site? (y/n):\n')
        if fetch_dem_res.lower() == 'y':
            fetch_dem()

    if action == '3':
        start_date = input('Please enter the start date of the simulation (YYYY-MM-DD):\n')
        end_date = input('Please enter the end date of the simulation (YYYY-MM-DD):\n')
        num_batches = input('Please enter the number of batches for the simulation:\n')
        gpu_count = input('Please enter the number of GPUs to use for the simulation:\n')
        sys.argv = ['run_parallel.py', start_date, end_date, '--num-batches', num_batches, '--num-gpus', gpu_count]

    else:
        print('Invalid action selected. Exiting.')
        sys.exit(1)