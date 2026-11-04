import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio

def save_monthly_depo_plots(
    monthly_surface_sum,
    monthly_surface_counts,
    output_dir,
    minx_3857,
    maxx_3857,
    miny_3857,
    maxy_3857,
    species_label,
    species_unit_depo,
    grid_res,
    total_hours
):
    """Save monthly mean concentration plots for the near-surface AGL layer."""
    monthly_dir = os.path.join(output_dir, "monthly_depo_plots")
    os.makedirs(monthly_dir, exist_ok=True)

    for month_key in sorted(monthly_surface_sum.keys()):
        depo = monthly_surface_sum[month_key] 
        nh3_to_n = 14.007 / 17.031
        cell_area_m2 = grid_res*grid_res
        depo = (depo
                    * nh3_to_n            # microg NH3 → microg N
                    / 1000000000.0              # microg → kg
                    * (10000.0 / cell_area_m2)  # per cell → per ha
                    * (365.25*24/total_hours)  # sim period → year
        vmax = float(np.nanpercentile(depo, 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        masked_grid = np.ma.masked_where(depo.T <= 1e-6, depo.T)
        im = ax.imshow(
            masked_grid,
            origin="lower",
            cmap="winter",
            extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
            zorder=1,
            alpha=0.8,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"Monthly Deposition Rate ({species_label}) - {month_key}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Deposition Rate ({species_unit_depo})")


        out_tif = os.path.join(monthly_dir, f"monthly_depo_{species_label}_{month_key}.tif")
        nx, ny = depo.shape
        out_png = os.path.join(monthly_dir, f"monthly_depo_{species_label}_{month_key}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"✓ Saved monthly depo: {out_png}")
        pixel_width = (maxx_3857 - minx_3857) / nx
        pixel_height = (maxy_3857 - miny_3857) / ny

        from rasterio.transform import Affine
        # conc_ug is shaped [x, y], so transpose to [y, x] for GeoTIFF row/col order.
        # Use a top-left origin transform so the first row in the raster corresponds to maxy.
        transform = Affine.translation(minx_3857, maxy_3857) * Affine.scale(pixel_width, -pixel_height)

        with rasterio.open(
            out_tif,
            'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=1,
            dtype=depo.dtype,
            crs='EPSG:3857',
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(depo.T[::-1, :], 1)
        print(f"✓ Saved monthly depo GeoTIFF: {out_tif}")
