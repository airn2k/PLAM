
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def save_annual_depo_plots(
    annual_depo,
    output_dir,
    minx_3857,
    maxx_3857,
    miny_3857,
    maxy_3857,
    species_label,
    species_unit_depo
):
    """Save annual deposition plots and GeoTIFFs."""
    annual_dir = os.path.join(output_dir, "annual_depo_plots")
    os.makedirs(annual_dir, exist_ok=True)

    depo = np.asarray(annual_depo, dtype=np.float64)
    vmax = float(np.nanpercentile(depo, 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use("dark_background")
    masked_grid = np.ma.masked_where(depo.T <= 1e-6, depo.T)
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
    ax.set_title(f"Annual Deposition Rate ({species_label})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Deposition Rate ({species_unit_depo})")

    out_png = os.path.join(annual_dir, f"annual_depo_{species_label}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved annual depo plot: {out_png}")

    out_tif = os.path.join(annual_dir, f"annual_depo_{species_label}.tif")
    nx, ny = depo.shape
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
        dtype=depo.dtype,
        crs='EPSG:3857',
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(depo.T[::-1, :], 1)
    print(f"✓ Saved annual depo GeoTIFF: {out_tif}")