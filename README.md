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