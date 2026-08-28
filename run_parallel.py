#!/usr/bin/env python3
"""
Adaptive Parallel Simulation Runner for AIRN2K Model

Automatically determines optimal parallelization based on:
- Available GPU capacity (target 80-90% utilization)
- Requested time period
- Per-process GPU usage (~5%)

Usage:
    python run_parallel_seasons.py 2024-01-01 2024-12-31
    python run_parallel_seasons.py 2024-01-01 2024-12-31 --num-gpus 8  # Multi-GPU
"""

import subprocess
import sys
import argparse
from datetime import datetime, timedelta
import time
import os
import math
import numpy as np 
import pandas as pd
import glob
from plam_sites.site_config  import SiteConfig

def parse_date(date_str):
    """Parse date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(date_str, '%Y-%m-%d')

def combine_results(chunks, timestamp, output_dir, site_dir):
    """Combine CSV results from all chunks into a single file."""

    csv_files = glob.glob(os.path.join(output_dir, '**', 'sampler_timeseries_*.csv'), recursive=True)

    if not csv_files:
        print("⚠️  No CSV files found to combine")
        return

    print(f"\nFound {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        print(f"  {f}")

    try:
        dfs = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                print(f"⚠️  Skipping unreadable CSV {csv_file}: {exc}")
                continue

            if 'date_time' not in df.columns:
                print(f"⚠️  Skipping {csv_file}: missing date_time column")
                continue

            concentration_cols = [
                col for col in df.columns
                if col == 'concentration_OU/m³' or col.endswith('_concentration_OU/m³')
            ]
            if not concentration_cols:
                concentration_cols = [col for col in df.columns if 'concentration' in col.lower()]

            if not concentration_cols:
                print(f"⚠️  Skipping {csv_file}: no concentration columns found")
                continue

            dfs.append(df)

        if not dfs:
            print("⚠️  No valid CSV files with expected columns")
            return

        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        combined_df['date_time'] = pd.to_datetime(combined_df['date_time'], errors='coerce')
        combined_df = combined_df.dropna(subset=['date_time'])

        sort_cols = ['date_time']
        if 'time_step_100s' in combined_df.columns:
            sort_cols.append('time_step_100s')
        combined_df = combined_df.sort_values(sort_cols)

        if 'time_step_100s' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['date_time', 'time_step_100s'], keep='first')
        else:
            combined_df = combined_df.drop_duplicates(subset=['date_time'], keep='first')

        combined_df['date_time'] = combined_df['date_time'].dt.strftime('%Y-%m-%d %H:%M')

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'sampler_timeseries_combined_{timestamp}.csv')
        combined_df.to_csv(output_file, index=False)

        print(f"\n✓ Combined results saved to: {output_file}")
        print(f"  Total data points: {len(combined_df)}")
        if not combined_df.empty:
            print(f"  Date range: {combined_df['date_time'].iloc[0]} to {combined_df['date_time'].iloc[-1]}")

        concentration_cols = [
            col for col in combined_df.columns
            if col == 'concentration_OU/m³' or col.endswith('_concentration_OU/m³')
        ]
        if not concentration_cols:
            concentration_cols = [col for col in combined_df.columns if 'concentration' in col.lower()]

        if concentration_cols:
            numeric_conc = combined_df[concentration_cols].apply(pd.to_numeric, errors='coerce')
            mean_conc = numeric_conc.to_numpy(dtype=float).mean()
            max_conc = numeric_conc.to_numpy(dtype=float).max()
            print(f"  Mean concentration: {mean_conc:.2f} OU/m³")
            print(f"  Max concentration: {max_conc:.2f} OU/m³")

    except Exception as e:
        print(f"⚠️  Error combining results: {e}")


def combine_maps(output_dir, timestamp):
    """Combine deposition and concentration arrays from all chunks into weighted-average maps."""
    import numpy as np
    import json

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import rasterio
        from rasterio.transform import Affine
        import contextily as ctx
    except ImportError as e:
        print(f"⚠️  Cannot create combined plots (missing library): {e}")
        return

    def save_combined_annual_agl_plots(conc_array, out_dir, minx, maxx, miny, maxy):
        annual_dir = os.path.join(out_dir, 'annual_agl_plots')
        os.makedirs(annual_dir, exist_ok=True)

        conc_plot = np.asarray(conc_array, dtype=np.float64)
        vmax = float(np.nanpercentile(conc_plot, 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        masked = np.ma.masked_where(conc_plot.T < 1e-9, conc_plot.T)
        im = ax.imshow(
            masked,
            origin='lower',
            cmap='hot',
            extent=[minx, maxx, miny, maxy],
            zorder=1,
            alpha=0.8,
            vmin=0.0,
            vmax=vmax,
        )

        cax = inset_axes(ax, width='5%', height='50%', loc='lower right', borderpad=1.0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Avg ambient odour (OU/m³)\n(0-100 m AGL mean)', fontsize=9, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        png_path = os.path.join(annual_dir, 'annual_agl_0_100m.png')
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"✓ Saved combined annual AGL plot: {png_path}")

        tif_path = os.path.join(annual_dir, 'annual_agl_0_100m.tif')
        nx, ny = conc_plot.shape
        pixel_width = (maxx - minx) / nx
        pixel_height = (maxy - miny) / ny
        transform = Affine.translation(minx, maxy) * Affine.scale(pixel_width, -pixel_height)
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=1,
            dtype=conc_plot.dtype,
            crs='EPSG:3857',
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(conc_plot.T[::-1, :], 1)
        print(f"✓ Saved combined annual AGL GeoTIFF: {tif_path}")

    # Discover all chunk metadata + arrays
    meta_files = sorted(glob.glob(os.path.join(output_dir, '**', 'grid_metadata_*.json'), recursive=True))
    if not meta_files:
        print("⚠️  No grid_metadata JSON files found — cannot create combined maps")
        return

    # Load extent from first metadata (all chunks share the same grid)
    with open(meta_files[0]) as f:
        meta = json.load(f)
    minx_3857 = meta['minx_3857']
    maxx_3857 = meta['maxx_3857']
    miny_3857 = meta['miny_3857']
    maxy_3857 = meta['maxy_3857']
    nx, ny = meta['nx'], meta['ny']

    # --- Deposition: time-weighted average of kg N/yr/ha ---
    depo_files = sorted(glob.glob(os.path.join(output_dir, '**', 'deposition_array_*.npy'), recursive=True))
    if depo_files:
        total_hours = 0
        depo_accum = np.zeros((nx, ny), dtype=np.float64)
        for df in depo_files:
            # Find matching metadata in the same directory
            d = os.path.dirname(df)
            mf = glob.glob(os.path.join(d, 'grid_metadata_*.json'))
            hours = 1
            if mf:
                with open(mf[0]) as f:
                    hours = json.load(f).get('total_hours', 1)
            arr = np.load(df)
            depo_accum += arr.astype(np.float64) * hours
            total_hours += hours
        depo_combined = (depo_accum / max(total_hours, 1)).astype(np.float32)

        # Save combined array
        np.save(os.path.join(output_dir, f'deposition_combined_{timestamp}.npy'), depo_combined)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use("dark_background")
        masked = np.ma.masked_where(depo_combined.T < 1e-3, depo_combined.T)
        im = ax.imshow(masked, origin='lower', cmap='hot',
                       extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
                       zorder=1, alpha=0.8)
        
        cax = inset_axes(ax, width='5%', height='50%', loc='lower right', borderpad=1.0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('N deposition (kg N/yr/ha)', fontsize=9, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        depo_png = os.path.join(output_dir, f'deposition_combined_{timestamp}.png')
        fig.savefig(depo_png, dpi=150)
        plt.close(fig)
        print(f"✓ Combined deposition map: {depo_png}")
        print(f"  Max: {depo_combined.max():.4f} kg N/yr/ha  Mean (nonzero): {depo_combined[depo_combined>0].mean():.4f} kg N/yr/ha")
    else:
        print("⚠️  No deposition_array .npy files found")

    # --- Concentration: time-weighted average of µg/m³ ---
    conc_files = sorted(glob.glob(os.path.join(output_dir, '**', 'concentration_array_*.npy'), recursive=True))
    if conc_files:
        total_hours = 0
        conc_accum = np.zeros((nx, ny), dtype=np.float64)
        for cf in conc_files:
            d = os.path.dirname(cf)
            mf = glob.glob(os.path.join(d, 'grid_metadata_*.json'))
            hours = 1
            if mf:
                with open(mf[0]) as f:
                    hours = json.load(f).get('total_hours', 1)
            arr = np.load(cf)
            conc_accum += arr.astype(np.float64) * hours
            total_hours += hours
        conc_combined = (conc_accum / max(total_hours, 1)).astype(np.float32)

        np.save(os.path.join(output_dir, f'concentration_combined_{timestamp}.npy'), conc_combined)
        save_combined_annual_agl_plots(conc_combined, output_dir, minx_3857, maxx_3857, miny_3857, maxy_3857)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use("dark_background")
        masked = np.ma.masked_where(conc_combined.T < 1e-9, conc_combined.T)
        im = ax.imshow(masked, origin='lower', cmap='hot',
                       extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
                       zorder=1, alpha=0.8)

        cax = inset_axes(ax, width='5%', height='50%', loc='lower right', borderpad=1.0)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('Avg ambient odour (OU/m³)\n(0-100 m mean)', fontsize=9, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        conc_png = os.path.join(output_dir, f'concentration_combined_{timestamp}.png')
        fig.savefig(conc_png, dpi=150)
        plt.close(fig)
        print(f"✓ Combined concentration map: {conc_png}")
        print(f"  Max: {conc_combined.max():.4f} OU/m³  Mean (nonzero): {conc_combined[conc_combined>0].mean():.4f} OU/m³")
    else:
        print("⚠️  No concentration_array .npy files found")

def calculate_optimal_chunks(start_date, end_date):
    """Calculate optimal chunking based on the number of days in the period."""
    total_days = (end_date - start_date).days
    
    num_chunks = int(np.ceil((total_days / 365) * 12))  # Default: 1 chunk per month

    


    chunks = []
    current_start = start_date
    
    for i in range(num_chunks):

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if (current_start.year % 4 == 0 and current_start.year % 100 != 0) or (current_start.year % 400 == 0):
            days_in_month[1] = 29  # Leap year adjustment
            
        month_index = current_start.month - 1
        if i == 0:
            # Figure out which month is the  start date in. The first chunk is from the start of the month to the end of the month, and subsequent chunks are full months.
            chunk_end = current_start + timedelta(days=days_in_month[month_index] - current_start.day)  # End of the month
        elif i == num_chunks - 1:
            # Figure out what the last chunk should be. It should end at the end date.
            chunk_end = end_date - timedelta(days=1)  # Ensure last chunk ends exactly at end_date
        else:
            # For intermediate chunks, just add the number of days in the month
            chunk_end = current_start + timedelta(days=days_in_month[month_index]) - timedelta(days=1)  # End of the month

        chunk_name = current_start.strftime('%Y%m')
        
        chunks.append((chunk_name, current_start, chunk_end))
        current_start = chunk_end + timedelta(days=1)
    
    return chunks, num_chunks

def run_simulation(chunk_name, start_date, end_date, log_file, output_dir, gpu_id=0):
    """Run a single simulation chunk on a specific GPU."""
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"  {chunk_name}: {start_str} to {end_str} [GPU {gpu_id}]")
    
    # Export USE_CUDA environment variable for the subprocess
    env = os.environ.copy()
    env['USE_CUDA'] = str(int(gpu_id is not None))
    
    cmd = [
        'python', 'run_model.py',
        start_str, end_str
    ]
    
    # Set CUDA_VISIBLE_DEVICES to assign specific GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['OUTPUT_DIR'] = output_dir
    
    with open(log_file, 'w') as log:
        log.write(f"Simulation: {chunk_name}\n")
        log.write(f"Period: {start_str} to {end_str}\n")
        log.write(f"GPU: {gpu_id}\n")
        log.write(f"Started: {datetime.now()}\n")
        log.write("="*70 + "\n\n")
        log.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
    
    return process, log_file

def main():
    site_dir = SiteConfig.get_current_site().site_dir
    parser = argparse.ArgumentParser(
        description='Run PLAM simulations in parallel with adaptive chunking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-optimize parallelization for full year (single GPU)
  python run_parallel_seasons.py 2024-01-01 2024-12-31
  
  # Use 8 GPUs for massive parallelization
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --num-gpus 8
  # Use 15-day chunks
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --chunk-days 15
  
  # Split into exactly 12 batches
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --num-batches 12
  
  # Use 3-month chunks across 4 GPUs
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --chunk-months 3 --num-gpus 4
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --max-parallel 4
  
  # Use 3-month chunks across 4 GPUs
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --chunk-months 3 --num-gpus 4
  
  # Target 90% GPU utilization per GPU with 8 GPUs
  python run_parallel_seasons.py 2024-01-01 2024-12-31 --target-util 0.90 --num-gpus 8
  
  # Custom per-process GPU usage with multi-GPU
        """
    )
    parser.add_argument('start_date', help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('end_date', help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs available (default: 1)')
    parser.add_argument('--max-parallel', type=int,
                        help='Maximum number of parallel processes (overrides auto-calculation)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without actually running')

    # Create output folder (allow override for orchestrated runs)
    env_output_dir = os.environ.get("OUTPUT_DIR", "").strip()
    if env_output_dir:
        output_dir = env_output_dir
    else:
        output_dir = os.path.join(site_dir, 'outputs', '1')
        dirs = glob.glob(os.path.join(site_dir, 'outputs', '*'))
        if dirs:
            numeric_dirs = [d for d in dirs if os.path.isdir(d) and os.path.basename(d).isdigit()]
            if numeric_dirs:
                max_dir = max(numeric_dirs, key=lambda x: int(os.path.basename(x)))
                new_dir_num = int(os.path.basename(max_dir)) + 1
                output_dir = os.path.join(site_dir, 'outputs', f'{new_dir_num}')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'simulation_frames3d'), exist_ok=True)
        
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Use format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
        sys.exit(1)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    # Calculate optimal chunks
    chunks, num_chunks = calculate_optimal_chunks(
        start_date, end_date
    )
    
    total_days = (end_date - start_date).days


    
    # Create logs directory
    os.makedirs(os.path.join(output_dir, 'simulation_logs'), exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*70)
    print("ADAPTIVE PARALLEL SIMULATION")
    print("="*70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_days} days)")
    print(f"GPUs: {args.num_gpus}")
    print("="*70)
    print("\nChunks to run:")
    
    for i, (chunk_name, chunk_start, chunk_end) in enumerate(chunks):
        days = (chunk_end - chunk_start).days
        if args.num_gpus > 0:
            gpu_id = i % args.num_gpus
            print(f"{chunk_name}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')} ({days} days) [GPU {gpu_id}]")
        else:
            print(f"{chunk_name}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')} ({days} days) [CPU]")
    
    if args.dry_run:
        print("\n✓ Dry run complete (no simulations started)")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("STARTING SIMULATIONS")
    print("="*70 + "\n")
    
    processes = []
    
    for i, (chunk_name, chunk_start, chunk_end) in enumerate(chunks):
        log_file = os.path.join(output_dir, f"simulation_logs/{chunk_name}_{timestamp}.log")
        if args.num_gpus > 0:
            gpu_id = i % args.num_gpus  # Round-robin GPU assignment
        else:
            gpu_id = None
        chunk_output_dir = os.path.join(output_dir, chunk_name)
        os.makedirs(chunk_output_dir, exist_ok=True)
        process, log = run_simulation(chunk_name, chunk_start, chunk_end, log_file, chunk_output_dir, gpu_id)
        processes.append((chunk_name, process, log, gpu_id))
        
        # Small delay to avoid startup conflicts
        time.sleep(2)
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS STARTED")
    print("="*70)
    print("\nMonitoring progress (use Ctrl+C to stop all)...")
    print("\nLog files:")
    for name, _, log, gpu_id in processes:
        print(f"  tail -f {log}  # {name} on GPU {gpu_id}")
    print()
    
    try:
        # Wait for all processes to complete
        start_time = time.time()
        last_update = 0
        last_gpu_update = 0
        
        while any(p.poll() is None for _, p, _, _ in processes):
            time.sleep(5)
            
            current_time = time.time()
            
            # Show status every 10 seconds
            if current_time - last_update >= 10:
                running = sum(1 for _, p, _, _ in processes if p.poll() is None)
                completed = len(processes) - running
                elapsed = current_time - start_time
                print(f"[{elapsed/60:.1f}m] Progress: {completed}/{len(processes)} complete, {running} running...")
                last_update = current_time
                
            # Show GPU status every 60 seconds
            if current_time - last_gpu_update >= 60:
                if args.num_gpus > 0:
                    print("\nGPU Status:")
                    try:
                        # Also print the simulation progress by comparing the last recorded time in the csv output with the end of the chunk. Datetime is in YYYY-MM-DD HH:MM format
                        
                        # Replace spaces and colons in chunk_start and chunk_end for file naming
                        start_clean = chunk_start.strftime('%Y-%m-%d_%H-%M-%S')
                        end_clean = chunk_end.strftime('%Y-%m-%d_%H-%M-%S')
                        if os.path.exists(os.path.join(chunk_output_dir, f'sampler_timeseries_{start_clean}_{end_clean}.csv')):
                            latest_sample_date = pd.read_csv(os.path.join(chunk_output_dir, f'sampler_timeseries_{start_clean}_{end_clean}.csv'), parse_dates=['date_time']).date_time.max()
                            progress = (latest_sample_date - chunk_start).total_seconds() / (chunk_end - chunk_start).total_seconds() * 100
                            print(f"  Simulation progress: {progress:.1f}%")
                            print(f" Estimated time remaining: {((100/progress * elapsed/60)-elapsed/60):.1f} minutes")

                        nvidia_output = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                                        capture_output=True, text=True, timeout=5).stdout
                        for line in nvidia_output.strip().split('\n'):
                            if line:
                                parts = line.split(',')
                                if len(parts) >= 4:
                                    gpu_id, util, mem_used, mem_total = [p.strip() for p in parts]
                                    print(f"  GPU {gpu_id}: {util}% util, {mem_used}/{mem_total} MB")
                    except Exception as e:
                        print(f"  (nvidia-smi error: {e})")
                if args.num_gpus == 0:
                    print("  No GPUs detected (running on CPU)")

                print()
                last_gpu_update = current_time
        
        total_time = time.time() - start_time
        
        print("\n\n" + "="*70)
        print("ALL SIMULATIONS COMPLETE")
        print("="*70)
        print(f"Total wall-clock time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print()
        
        # Show results grouped by GPU
        print("Results by GPU:")
        for gpu_id in range(args.num_gpus):
            gpu_processes = [(n, p, l) for n, p, l, g in processes if g == gpu_id]
            if gpu_processes:
                print(f"\n  GPU {gpu_id}:")
                for name, process, log in gpu_processes:
                    if process.returncode == 0:
                        print(f"    ✓ {name}: SUCCESS")
                    else:
                        print(f"    ✗ {name}: FAILED (code {process.returncode})")
                    print(f"      Log: {log}")
        
        # Overall summary
        failed_count = sum(1 for _, p, _, _ in processes if p.returncode != 0)
        if failed_count > 0:
            print(f"\n⚠️  {failed_count}/{len(processes)} simulations failed. Check log files for details.")
            sys.exit(1)
        else:
            effective_speedup = num_chunks if args.num_gpus == 1 else f"{num_chunks}x (across {args.num_gpus} GPUs)"
            print(f"\n✓ All {len(processes)} simulations completed successfully!")
            print(f"  Speedup: ~{effective_speedup} faster than sequential")
            
            # Combine CSV results
            print("\n" + "="*70)
            print("COMBINING RESULTS")
            print("="*70)
            combine_results(chunks, timestamp, output_dir,site_dir)
            combine_maps(output_dir, timestamp)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Terminating all processes...")
        for name, process, _, gpu_id in processes:
            if process.poll() is None:
                process.terminate()
                print(f"  Terminated: {name} (GPU {gpu_id})")
        
        time.sleep(2)
        # Force kill if still running
        for name, process, _, gpu_id in processes:
            if process.poll() is None:
                process.kill()
                print(f"  Force killed: {name} (GPU {gpu_id})")
        
        sys.exit(1)

if __name__ == '__main__':
    main()
