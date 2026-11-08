import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def save_vertical_plots(
    conc3d,
    total_steps,
    output_dir,
    minx_3857,
    maxx_3857,
    dz,
    top_z,
    z_of_interest,
    receptor_vertical_rad,
    species_label,
    species_unit,
    grid_res,
    ground_height_grid,
    start_clean,
    end_clean,
):
    """Save vertical mean concentration plots for the near-surface AGL layer."""
    vertical_dir = os.path.join(output_dir, "vertical_plots")
    os.makedirs(vertical_dir, exist_ok=True)

    conc3d_mean = conc3d/total_steps
    minnz = int(np.min(ground_height_grid) / dz)
    maxnz = int((z_of_interest + np.min(ground_height_grid)) / dz)
    dnx = int(receptor_vertical_rad / grid_res)
    conc_plot = conc3d_mean[
        int(conc3d_mean.shape[0] / 2 - dnx) : int(conc3d_mean.shape[0] / 2 + dnx),
        minnz:maxnz
    ]
    conc_plot = conc_plot.mean(axis=0)
    vmax = float(np.nanpercentile(conc_plot, 98.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 10.0

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use("dark_background")
    masked_grid = np.ma.masked_where(conc_plot.T <= 1e-6, conc_plot.T)
    im = ax.imshow(
        masked_grid,
        origin="lower",
        cmap="winter",
        extent=[
            -receptor_vertical_rad,
            +receptor_vertical_rad,
            minnz * dz,
            maxnz * dz,
        ],
        zorder=1,
        alpha=0.8,
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(f"Average Vertical Concentration {start_clean[0:7]} - {end_clean[0:7]}")
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label=f"Concentration ({species_unit})"
    )
    ax.set_xlabel("Horizontal Distance from the Sampler (m)")
    ax.set_ylabel("Height Above the Sea Level (m)")

    # 1. Get the shape of the array
    rows, cols = ground_height_grid.shape
    # 2. Calculate the center indices using integer division
    mid_row = rows // 2
    mid_col = cols // 2
    plt.text(
        0,
        ground_height_grid[mid_row, mid_col] + 2,
        "X ALPHA sampler",
        color="red",
        verticalalignment="center",
    )

    out_png = os.path.join(vertical_dir, f"vertical_{start_clean[0:7]}_{end_clean[0:7]}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved vertical plot: {out_png}")

    # Save as GeoTIFF
    out_tif = os.path.join(vertical_dir, f"vertical_{start_clean[0:7]}_{end_clean[0:7]}.tif")
    nx, ny = conc_plot.shape
    pixel_width = grid_res
    pixel_height = dz

    from rasterio.transform import Affine

    # conc_ug is shaped [x, y], so transpose to [y, x] for GeoTIFF row/col order.
    # Use a top-left origin transform so the first row in the raster corresponds to maxy.
    transform = Affine.translation(minx_3857, top_z) * Affine.scale(
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
    print(f"✓ Saved vertical GeoTIFF: {out_tif}")
