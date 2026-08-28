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
    species_unit_depo
):
    """Save monthly mean concentration plots for the near-surface AGL layer."""
    monthly_dir = os.path.join(output_dir, "monthly_depo_plots")
    os.makedirs(monthly_dir, exist_ok=True)

    for month_key in sorted(monthly_surface_sum.keys()):
        count = max(1, int(monthly_surface_counts.get(month_key, 0)))
        monthly_mean = monthly_surface_sum[month_key] / count
        depo = monthly_mean

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
        ax.set_title(f"Monthly Deposition Rate ({species_label}) - {month_key}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Deposition Rate ({species_unit_depo})")

        out_png = os.path.join(monthly_dir, f"monthly_depo_{species_label}_{month_key}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"✓ Saved monthly depo: {out_png}")
