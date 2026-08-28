
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def save_annual_agl_plots(
    annual_surface_mean,
    output_dir,
    minx_3857,
    maxx_3857,
    miny_3857,
    maxy_3857,
    monthly_agl_top_m,
    species_label,
    species_unit
):
    """Save annual mean concentration plots and GeoTIFFs for the near-surface AGL layer."""
    annual_dir = os.path.join(output_dir, "annual_agl_plots")
    os.makedirs(annual_dir, exist_ok=True)

    conc_plot = np.asarray(annual_surface_mean, dtype=np.float64)
    vmax = float(np.nanpercentile(conc_plot, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 10.0

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use("dark_background")
    masked_grid = np.ma.masked_where(conc_plot.T <= 1e-6, conc_plot.T)
    im = ax.imshow(
        masked_grid,
        origin="lower",
        cmap="hot",
        extent=[minx_3857, maxx_3857, miny_3857, maxy_3857],
        zorder=1,
        alpha=0.8,
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(f"Annual Avg Concentration (0-{monthly_agl_top_m} m AGL) ({species_label})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Concentration ({species_unit})")

    out_png = os.path.join(annual_dir, f"annual_agl_0_{monthly_agl_top_m}m.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved annual 0-{monthly_agl_top_m} m AGL plot: {out_png}")

    out_tif = os.path.join(annual_dir, f"annual_agl_0_{monthly_agl_top_m}m.tif")
    nx, ny = conc_plot.shape
    pixel_width = (maxx_3857 - minx_3857) / nx
    pixel_height = (maxy_3857 - miny_3857) / ny
    from rasterio.transform import Affine
    transform = Affine.translation(minx_3857, maxy_3857) * Affine.scale(pixel_width, -pixel_height)
    with rasterio.open(
        out_tif,
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
    print(f"✓ Saved annual 0-{monthly_agl_top_m} m AGL GeoTIFF: {out_tif}")

