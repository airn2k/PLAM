# Parallelised Lagrangian Atmospheric Model (PLAM)

A GPU-accelerated Lagrangian atmospheric dispersion model for high-resolution simulation of ammonia and other atmospheric emissions.

---

## Credits

This software was developed as part of the **Ammonia Impact Reduction on Natura 2000 Sites (AIRN2K)** project, funded by the **Environmental Protection Agency (EPA) Ireland**, Grant No. **2022-NE-1125**.

- **Creator & Maintainer:** Dr. Shayan Kabiri, University College Dublin, Ireland, shayan.kabiri@ucd.ie, shaikabiri13@gmail.com
- **Project Supervisor:** Prof. Tom Curran, University College Dublin, Ireland, tom.curran@ucd.ie 

---

## Overview

PLAM is a parallelised Lagrangian particle dispersion model designed for high-resolution atmospheric dispersion simulations using CUDA-enabled GPUs. The model supports multiple emission source types, terrain-aware transport, ERA5 meteorological forcing, and scalable multi-GPU execution.

This README is currently a work in progress and will be expanded with additional documentation and examples.

---

## System Requirements

PLAM requires:

- **NVIDIA GPU with CUDA 12.x support**
- **Python 3.12** (recommended)
- **8-16 GB of GPU memory (VRAM)** for typical simulations
  - Higher spatial or temporal resolution may require additional VRAM.

---

## Installation

Clone the repository and create a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or
.venv\\Scripts\\activate       # Windows

pip install -r requirements.txt
```

---

## First-Time Setup

Run `main.py` to create a new simulation site or select an existing one.

```bash
python main.py
```

Simulation parameters can be configured in:

```text
config/params.py
```

Model outputs are written to the `outputs/` directory within the selected site.

---

## Diagnostic figures

`plot_diagnostic_summary_figures.py` exports the combined monthly and annual
predicted-versus-measured plot as both PNG and A4 landscape PDF. Add
`--scatter-grayscale` to export a grayscale version in both formats.

---

## Protected-area results

With the site outputs, reference measurements, diagnostic metric CSVs, and SAC
and SPA shapefiles available locally, run:

```bash
python summarize_protected_area_results.py
```

The script uses the highest-numbered run for each site. It produces a LaTeX
performance table, a deposition-summary CSV, and A4 PDF and PNG maps in
`diagnostic_figures/` (or `--output-dir`). Deposition is averaged over the union
of SAC and SPA polygons inside each model grid, with overlapping polygons
counted once and partial cells approximated using an 8-by-8 subgrid. Monthly
annualised deposition rates and concentration maps are weighted by simulated
hours. The CSV records the contributing protected areas and the fraction of
their union inside the grid. Site averages cover all available run intervals;
these periods differ by site and may contain gaps.

By default, the script slides a 12-calendar-month window through each monitoring
site's latest run, finds the window with the highest duration-weighted
protected-area mean deposition at each site, and maps the three highest sites.
It writes all sites' peak windows and rates to
`diagnostic_figures/protected_area_peak_window_ranking.csv`. Windows differ
between sites and deliberately select high-deposition periods rather than
typical years. All twelve monthly rasters are required, but simulations can
still have gaps or partial months; panel titles show dates, and the ranking CSV
reports simulated and window hours.
Maps show the central 8 km by 8 km of each model grid, reprojected to Irish
Transverse Mercator (EPSG:2157) with nearest-neighbour sampling. Site ranking
and table deposition values still use the full protected-area portions within
each original grid. Each map row uses a shared grayscale upper limit at the
98th percentile pooled across its three displayed grids; threshold highlights
use unclipped values.
The additional grayscale map uses dots for concentration above 1
and up to 3 µg/m³, and vertical stripes for concentration above 3 µg/m³ or
deposition above 5 kg N/ha/year in bog SACs.
Use `--map-sites SITE SITE SITE` to choose three sites or `--auto-map-sites`
to select the highest measured means among sites with twelve outputs in
`--year` (2024 by default); both alternatives use each site's selected
diagnostic window instead. The concentration rasters represent the 0–3 m AGL
layer, not 0–4 m.
Concentration map highlights show values above 1 and 3 µg/m³; deposition values
above 5 kg N/ha/year are highlighted within the mapped Clara, Killyconny, and
Raheenmore bog SAC boundaries only. SAC boundaries are not habitat maps, and
the deposition rasters represent modeled ammonia-derived nitrogen, not total
nitrogen deposition. These visual references do not establish critical-load
exceedance or compliance.

For a separate three-panel Raheenmore map, run:

```bash
python plot_raheenmore_thresholds.py
```

It maps the site's highest-deposition 12-month window over the same central
8 km square, with continuous grayscale concentration capped at the displayed
grid's 98th percentile and white, grey and black categories. Concentration classes are
≤1, >1–3 and >3 µg/m³; deposition classes are ≤1, >1–5 and >5 kg N/ha/year
across the whole map. Deposition breakpoints are visual references, not proof
of ecological critical-load exceedance; the modeled load is ammonia-derived
nitrogen only.

For the three highest-deposition sites in a 3-by-3 figure, run
`python plot_peak_site_thresholds.py`. Rows show sites and columns show
continuous concentration, concentration classes and deposition classes.
The continuous scale uses the 98th percentile pooled across the three displayed
grids; class boundaries use the unclipped values. The same deposition
breakpoints apply across each whole map and are not site-specific critical
loads.

---

## Running a Simulation

The primary execution script is `run_parallel.py`.

Example: run a one-day simulation using one GPU and one processing batch.

```bash
python run_parallel.py 2019-01-01 2019-01-02 --num-batches 1 --num-gpus 1
```

Display all available options.

```bash
python run_parallel.py --help
```

---

## Emission Sources

Emission sources are read from:

```text
sources/sources.shp
```

Create a polygon shapefile containing the emission sources (e.g., livestock buildings, industrial facilities, lagoons, or other emitting areas) and place it in the `sources/` directory of the site.

### Required Attribute

| Attribute | Type | Description |
|----------|------|-------------|
| `e_v` | float | Emission rate |

- **Area sources:** `e_v` is specified in **g/m²/s**
- **Point sources:** `e_v` is specified in **g/s**

### Point Sources

PLAM treats all sources as area sources by default. To represent a point source:

1. Create a small polygon surrounding the point location.
2. Add a boolean attribute named `Point`.
3. Set `Point = True`.

The model will use the polygon centroid as the emission location.

### Optional Attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `r_v` | float | Exit velocity |
| `r_h` | float | Release height above ground |

If these attributes are absent, default values from `config/params.py` are used.

### Seasonal Emission Factors

An optional string attribute named `seasonal_f` may be added.

Example:

```text
"1.0, 1.0, 1.0, 1.0"
```

The four values correspond to:

1. Winter
2. Spring
3. Summer
4. Autumn

These factors are applied multiplicatively to the emission rate during each season.

---

## Weather Data (ERA5)

Meteorological data are downloaded automatically from the **Copernicus ERA5 reanalysis dataset** through `main.py`.

### Configure API Access

1. Create an account on the **Copernicus Climate Data Store (CDS)**.
2. Generate a CDS API key.
3. Add the key to:

```text
plam_sites/config.ini
```

Format:

```ini
[CDS_API]
api_key = YOUR_API_KEY
```

---

## Digital Elevation Model (DEM)

Terrain data are downloaded automatically from the **Copernicus DEM dataset** via `main.py`.

The DEM is used to determine terrain elevation and particle height above ground level (AGL). Required tiles are downloaded and mosaicked automatically for the selected site.

### Configure API Access

1. Create an account with **OpenTopography**.
2. Generate an API key.
3. Create a file named:

```text
.opentopography.txt
```

Place it either:

- in your home directory, or
- in the root directory of the project.

The file should contain only:

```text
YOUR_API_KEY
```

---

## License

PLAM is licensed under the GNU General Public License v3.0 (GPL-3.0).

You are free to use, modify, and redistribute this software under the terms of the GPLv3 license. See the LICENSE file in this repository for the full license text.

---

## Citation

If you use PLAM in academic work, please cite the repository's DOI. 