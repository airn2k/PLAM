import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def save_agl_plots(
    conc3d,
    total_steps,
    dz,
    output_dir,
    minx_3857,
    maxx_3857,
    miny_3857,
    maxy_3857,
    agl_top_m,
    species_label,
    species_unit,
    start_date,
    end_date,
):
    """Save mean concentration plots for the near-surface AGL layer."""
    agl_dir = os.path.join(output_dir, "agl_plots")
    os.makedirs(agl_dir, exist_ok=True)

    conc_plot = conc3d/total_steps

    vmax = float(np.nanpercentile(conc_plot, 98.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 10.0
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use("dark_background")
    masked_grid = np.ma.masked_where(conc_plot.T <= 1e-6, conc_plot.T)
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
    ax.set_title(
        f"Average Concentration (0-{agl_top_m} m AGL) - {start_date[0:7]} to {end_date[0:7]} ({species_label})"
    )
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label=f"Concentration ({species_unit})"
    )

    out_png = os.path.join(
        agl_dir, f"agl_0_{agl_top_m}m_{start_date[0:7]}_{end_date[0:7]}.png"
    )
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved average 0-{agl_top_m} m AGL plot: {out_png}")

    # Save as GeoTIFF
    out_tif = os.path.join(
        agl_dir, f"agl_0_{agl_top_m}m_{start_date[0:7]}_{end_date[0:7]}.tif"
    )
    nx, ny = conc_plot.shape
    pixel_width = (maxx_3857 - minx_3857) / nx
    pixel_height = (maxy_3857 - miny_3857) / ny

    from rasterio.transform import Affine

    # conc_ug is shaped [x, y], so transpose to [y, x] for GeoTIFF row/col order.
    # Use a top-left origin transform so the first row in the raster corresponds to maxy.
    transform = Affine.translation(minx_3857, maxy_3857) * Affine.scale(
        pixel_width, -pixel_height
    )

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=conc_plot.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(conc_plot.T[::-1, :], 1)
    print(f"✓ Saved average 0-{agl_top_m} m AGL GeoTIFF: {out_tif}")
