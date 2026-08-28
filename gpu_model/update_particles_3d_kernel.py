from numba import cuda
from config.params import *
import numpy as np
import math
from numba.cuda.random import (create_xoroshiro128p_states,
                                       xoroshiro128p_normal_float32,
                                       xoroshiro128p_uniform_float32)

@cuda.jit
def update_particles_3d_kernel(
    x,
    y,
    z,  # (f_step, n_part) float32
    particle_type,  # (f_step, n_part) int32, 0=NH3, 1=NH4+
    u_ref,
    v_ref,
    z_ref,       # Reference height for wind (typically 10m)
    z0,          # Roughness length
    blh,         # Boundary layer height
    Kxy,
    Kz_low,      # Kz for near-surface (0-100m)
    Kz_mid,      # Kz for mid-level (100-500m) 
    Kz_high,     # Kz for upper level (>500m)
    dt,  # scalars
    P_dep_surf,  # Deposition probability for NH3
    P_dep_nh4,   # Deposition probability for NH4+
    P_conversion,  # NH3→NH4+ conversion probability
    P_wet_dep,   # Wet deposition probability
    w_mean,
    w_settle,  # scalars
    particle_mass_g,  # (f_step,) float32
    timesteps_per_hour,  # scalar
    depo_grid,  # (nx, ny) float32
    GRID_RES,
    minx,
    miny,  # scalars
    nx,
    ny,  # ints
    z_cap,  # scalar
    rng_states,  # RNG state per thread
    step_in_window,  # number of valid time-slices minus 1
    head,            # ring-buffer head pointer
    f_step_param,    # ring-buffer length
    dem,  # (nx, ny) float32, DEM heights
    dem_nx,
    dem_ny,
    dem_minx,
    dem_miny,
    dem_dx,
    dem_dy,
    dh_dx_grid,  # (nx, ny) float32, terrain gradient in easting
    dh_dy_grid,  # (nx, ny) float32, terrain gradient in northing
    mass_budget,  # [dry_dep, wet_dep, oob_horizontal, oob_vertical, aged_out] mass totals (g)
    chan_alpha,   # channeling strength (0=off)
    chan_z_top,   # channeling height ceiling AGL (m)
    chan_slope_ref,  # reference slope for channeling weight
    oro_alpha,    # orographic lifting strength (0=off)
    oro_z_top,    # orographic lifting height ceiling AGL (m)
    sshf,         # surface sensible heat flux (W/m²)
    release_vz,         # (n_part,) float32 initial vertical velocity for newly emitted particles
    seasonal_factor_step  # (n_part,) float32 seasonal factor for each particle
):
    j = cuda.blockIdx.x
    k = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.y
    n_part = x.shape[1]
    if j > step_in_window or k >= n_part:
        return

    idx = head - 1 - j
    if idx < 0:
        idx += f_step_param

    if x[idx, k] == SENTINEL:
        return

    ix_dem = int((x[idx, k] - dem_minx) / dem_dx)
    iy_dem = int((y[idx, k] - dem_miny) / dem_dy)
    local_ground_z = 0.0
    if 0 <= ix_dem < dem_nx and 0 <= iy_dem < dem_ny:
        local_ground_z = float(dem[ix_dem, iy_dem])

    particle_height_agl = float(z[idx, k] - local_ground_z)
    particle_height_agl = max(particle_height_agl, z0 * 1.01)   

    # Derived from Paulson, C. A. ‘The Mathematical Representation of Wind Speed and Temperature Profiles in the Unstable Atmospheric Surface Layer’.
    kappa = np.float32(0.4)  # von Karman constant
    g = np.float32(9.81)
    
    # Calculate Monin-Obukhov length for stability
    wind_speed_ref = math.sqrt(u_ref*u_ref + v_ref*v_ref)
    ustar = (kappa * wind_speed_ref) / math.log(z_ref/z0)
    
    H_kinematic = sshf/ (1.225 * 1005)  # Convert to kinematic units
    if math.fabs(H_kinematic) > 1e-6:
        L_MO = -(ustar*ustar*ustar) / (kappa * g * H_kinematic)
        L_MO = max(min(L_MO, 10000), -10000)
    else:
        L_MO = 1e6  # Neutral
    
    # Stability functions
    def psi_m_profile(zeta):
        zeta = max(min(zeta, 2.0), -2.0)
        if zeta > 0:  # Stable
            return -5 * zeta
        elif zeta < 0:  # Unstable
            x = (1 - 16 * zeta)**0.25
            return 2*math.log((1+x)/2) + math.log((1+x**2)/2) - 2*math.atan(x) + math.pi/2
        return 0.0
    
    # Calculate wind speed ratio using log profile with stability
    if particle_height_agl < blh:
        zeta_z = particle_height_agl / L_MO
        zeta_ref = z_ref / L_MO

        log_term = math.log(particle_height_agl / z_ref)
        stability_correction = psi_m_profile(zeta_z) - psi_m_profile(zeta_ref)
        
        # Wind speed ratio
        wind_ratio = (math.log(particle_height_agl / z0) + stability_correction) / (math.log(z_ref / z0) + psi_m_profile(zeta_ref))
        wind_ratio = max(min(wind_ratio, 3.0), 0.1)  # Prevent extreme values
    else:
        # Above BLH, assume constant wind
        wind_ratio = (math.log(blh / z0)) / (math.log(z_ref / z0))
        wind_ratio = max(min(wind_ratio, 2.5), 0.8)

    wind_ratio = 1.0  # For testing, override wind ratio to 1.0 (no change in wind speed)
    # Apply ratio to both components
    u_z = u_ref * wind_ratio
    v_z = v_ref * wind_ratio


    # --- Terrain slope lookup ---
    slope_x = 0.0
    slope_y = 0.0
    if 0 <= ix_dem < dem_nx and 0 <= iy_dem < dem_ny:
        slope_x = float(dh_dx_grid[ix_dem, iy_dem])
        slope_y = float(dh_dy_grid[ix_dem, iy_dem])

    # --- Wind channeling: reduce cross-slope flow without fully removing it ---
    # A heuristic terrain-channeling parameterization in which the resolved wind is decomposed into components parallel and normal to local terrain contours, and the cross-slope component is reduced by an empirical factor that depends on terrain slope and height above ground.
    w_oro = 0.0
    if chan_alpha > 0.0 and particle_height_agl < chan_z_top:
        slope_mag = math.sqrt(slope_x * slope_x + slope_y * slope_y)
        if slope_mag > 1.0e-6:
            height_weight = 1.0 - particle_height_agl / chan_z_top
            if height_weight < 0.0:
                height_weight = 0.0
            slope_weight = slope_mag / (slope_mag + chan_slope_ref)
            alpha_eff = chan_alpha * height_weight * slope_weight
            # Contour-tangent unit vector (perpendicular to gradient)
            tx = -slope_y / slope_mag
            ty =  slope_x / slope_mag
            nx_slope = slope_x / slope_mag
            ny_slope = slope_y / slope_mag
            wind_dot_tangent = u_z * tx + v_z * ty
            wind_dot_normal = u_z * nx_slope + v_z * ny_slope
            normal_scale = 1.0 - alpha_eff
            if normal_scale < CHANNELING_CROSS_SLOPE_MIN:
                normal_scale = CHANNELING_CROSS_SLOPE_MIN
            u_z = wind_dot_tangent * tx + wind_dot_normal * normal_scale * nx_slope
            v_z = wind_dot_tangent * ty + wind_dot_normal * normal_scale * ny_slope

    # --- Orographic lifting ---
    # Mountain waves theory solution for boundary layer, derived from Smith, Ronald B. ‘The Influence of Mountains on the Atmosphere’..
    if oro_alpha > 0.0 and particle_height_agl < oro_z_top:
        height_weight_oro = 1.0 - particle_height_agl / oro_z_top
        if height_weight_oro < 0.0:
            height_weight_oro = 0.0
        if OROGRAPHIC_USE_PRECHANNEL_WIND:
            w_oro = oro_alpha * height_weight_oro * (u_ref * wind_ratio * slope_x + v_ref * wind_ratio * slope_y)
        else:
            w_oro = oro_alpha * height_weight_oro * (u_z * slope_x + v_z * slope_y)


    x[idx, k] += u_z * dt
    y[idx, k] += v_z * dt
    if j == 0:
        z[idx, k] += release_vz[k]/2
    z[idx, k] += (w_mean - w_settle + w_oro) * dt

    if Kxy > 0.0:
        std_xy = math.sqrt(2.0 * Kxy * dt)
        x[idx, k] += xoroshiro128p_normal_float32(rng_states, k) * std_xy
        y[idx, k] += xoroshiro128p_normal_float32(rng_states, k) * std_xy

    ix_dem = int((x[idx, k] - dem_minx) / dem_dx)
    iy_dem = int((y[idx, k] - dem_miny) / dem_dy)
    z_dem = 0.0
    if 0 <= ix_dem < dem_nx and 0 <= iy_dem < dem_ny:
        z_dem = float(dem[ix_dem, iy_dem])

    particle_height_agl = z[idx, k] - z_dem
    # Smooth Kz profile (tanh transitions) + Thomson drift correction
    # Transition heights and smoothing half-width
    z1 = 100.0   # low ↔ mid transition (AGL)
    z2 = 500.0   # mid ↔ high transition (AGL)
    delta = 20.0  # smoothing half-width (m)
    h = max(particle_height_agl, 0.1)
    t1 = math.tanh((h - z1) / delta)
    t2 = math.tanh((h - z2) / delta)
    s1 = 0.5 * (1.0 + t1)
    s2 = 0.5 * (1.0 + t2)
    current_kz = Kz_low + (Kz_mid - Kz_low) * s1 + (Kz_high - Kz_mid) * s2
    # dKz/DZ (analytical derivative of tanh smoothing)
    ds1 = 0.5 / delta * (1.0 - t1 * t1)
    ds2 = 0.5 / delta * (1.0 - t2 * t2)
    dKz_DZ = (Kz_mid - Kz_low) * ds1 + (Kz_high - Kz_mid) * ds2

    if current_kz > 0.0:
        std_z = math.sqrt(2.0 * current_kz * dt)
        z[idx, k] += dKz_DZ * dt + xoroshiro128p_normal_float32(rng_states, k) * std_z

    # # Chemical conversion: NH3 (type=0) → NH4+ (type=1)
    if SPECIES == "NH3" and particle_type[idx, k] == 0 and P_conversion > 0.0:
        if xoroshiro128p_uniform_float32(rng_states, k) <= P_conversion:
            particle_type[idx, k] = 1  # Convert to NH4+ particle
            
    ix_dem = int((x[idx, k] - dem_minx) / dem_dx)
    iy_dem = int((y[idx, k] - dem_miny) / dem_dy)
    
    if DEPOSITION_ENABLED == 1:
        if 0 <= ix_dem < dem_nx and 0 <= iy_dem < dem_ny:
            z_dem = dem[ix_dem, iy_dem]
            if z[idx, k] < z_dem:
                P_dep = P_dep_surf if particle_type[idx, k] == 0 else P_dep_nh4
                if xoroshiro128p_uniform_float32(rng_states, k) <= P_dep:
                    ix = int(math.floor(x[idx, k] / GRID_RES) - math.floor(minx / GRID_RES))
                    iy = int(math.floor(y[idx, k] / GRID_RES) - math.floor(miny / GRID_RES))
                    add_mass = particle_mass_g[idx]*seasonal_factor_step[k]
                    if 0 <= ix < nx and 0 <= iy < ny:
                        cuda.atomic.add(depo_grid, (ix, iy), add_mass)
                    cuda.atomic.add(mass_budget, MASS_BUDGET_DRY_DEP, add_mass)
                    x[idx, k] = SENTINEL
                    y[idx, k] = SENTINEL
                    z[idx, k] = SENTINEL
                    return
                else:
                    # Inelastic rebound from penetration depth (momentum proxy from this step)
                    penetration = z_dem - z[idx, k]
                    if penetration < 0.0:
                        penetration = 0.0

                    # e=1 mirrors depth exactly; e<1 damps rebound near the surface.
                    restitution = 0.35
                    z_new = z_dem + restitution * penetration
                    z_floor = z_dem + z0
                    if z_new < z_floor:
                        z_new = z_floor
                    z[idx, k] = z_new

        if P_wet_dep > 0.0:
            if xoroshiro128p_uniform_float32(rng_states, k) <= P_wet_dep:
                ix = int(math.floor(x[idx, k] / GRID_RES) - math.floor(minx / GRID_RES))
                iy = int(math.floor(y[idx, k] / GRID_RES) - math.floor(miny / GRID_RES))
                add_mass = particle_mass_g[idx]
                if 0 <= ix < nx and 0 <= iy < ny:
                    cuda.atomic.add(depo_grid, (ix, iy), add_mass)
                cuda.atomic.add(mass_budget, MASS_BUDGET_WET_DEP, add_mass)
                x[idx, k] = SENTINEL
                y[idx, k] = SENTINEL
                z[idx, k] = SENTINEL
                return
    else:
        z_dem = dem[ix_dem, iy_dem]
        if z[idx, k] < z_dem:
            # If deposition is not enabled, jusut check for rebound
            penetration = z_dem - z[idx, k]
            if penetration < 0.0:
                penetration = 0.0

            # e=1 mirrors depth exactly; e<1 damps rebound near the surface.
            restitution = 0.35
            z_new = z_dem + restitution * penetration
            z_floor = z_dem + z0
            if z_new < z_floor:
                z_new = z_floor
            z[idx, k] = z_new

    if 0 <= ix_dem < dem_nx and 0 <= iy_dem < dem_ny:
        z_cap_abs = z_dem + z_cap
        if z[idx, k] >= z_cap_abs:
            z_new = 2.0 * z_cap_abs - z[idx, k]
            z_floor = z_dem + z0
            if z_new < z_floor:
                z_new = z_floor
            z[idx, k] = z_new
