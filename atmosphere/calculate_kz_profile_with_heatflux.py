import numpy as np

def calculate_kz_profile_with_heatflux(u_star, blh, sshf, step_counter, z_levels=None):
    """
    Calculate height-dependent Kz profile using Monin-Obukhov similarity with surface heat flux
    
    Returns:
        z_heights: array of height levels (m)
        kz_profile: array of Kz values at each height level (m²/s)
    """

    # Similar to calculate_kxy_with_stability, but for z profile, using Monin-Obukhov length and stability functions to compute Kz at different heights.
    # Based on Deardorff, Businger-Dyer, and Tennekes & Lumley's work.
    # Constants
    kappa = 0.4  # von Karman constant
    g = 9.81     # gravity
    cp = 1005    # specific heat of air at constant pressure
    rho = 1.225  # air density at sea level
    
    # Validate inputs
    u_star = max(u_star, 0.01)  # Ensure minimum friction velocity
    blh = max(blh, 100.0)       # Ensure minimum boundary layer height
    
    # Calculate surface heat flux in kinematic units (K⋅m/s)
    H_kinematic = sshf / (rho * cp)  # Convert from W/m² to K⋅m/s
    
    # Monin-Obukhov length with bounds checking
    if abs(H_kinematic) > 1e-6:  # Avoid division by zero
        L_MO = -(u_star**3) / (kappa * g * H_kinematic)
        # Limit L_MO to reasonable bounds to avoid numerical issues
        L_MO = np.clip(L_MO, -10000, 10000)
    else:
        L_MO = 1e6  # Very stable/neutral conditions
    
    # Height-dependent mixing length
    if z_levels is None:
        z_heights = np.linspace(10, blh, 50)  # More levels for better resolution
    else:
        z_heights = np.array(z_levels)
    
    # Ensure heights are within reasonable bounds
    z_heights = np.clip(z_heights, 10.0, blh)
    
    # Stability functions based on L_MO with numerical stability
    if L_MO > 0:  # Stable conditions
        zeta = z_heights / L_MO
        zeta = np.clip(zeta, 0, 2.0)  # Limit stability parameter
        phi_m = 1 + 4.7 * zeta
        l_mix = kappa * z_heights / phi_m
    elif L_MO < 0:  # Unstable conditions
        zeta = z_heights / abs(L_MO)
        zeta = np.clip(zeta, 0, 2.0)  # Limit stability parameter
        # Ensure we don't get negative values under the root
        arg = np.maximum(1 - 15 * zeta, 0.01)
        phi_m = arg**(-0.25)
        l_mix = kappa * z_heights / phi_m
    else:  # Neutral
        l_mix = kappa * z_heights
    
    # Limit mixing length by boundary layer height and ensure positive values
    l_mix = np.minimum(l_mix, 0.1 * blh)
    l_mix = np.maximum(l_mix, 0.01 * z_heights)  # Minimum mixing length
    
    # Calculate Kz profile with bounds
    kz_profile = l_mix * u_star
    kz_profile = np.clip(kz_profile, 0.1, 1000.0)  # Reasonable Kz bounds
    
    # Check for NaN values and replace if found
    if np.any(np.isnan(kz_profile)):
        print(f"Warning: NaN detected in Kz calculation. u*={u_star}, BLH={blh}, SSHF={sshf}, L_MO={L_MO}")
        # Use simple fallback calculation
        kz_profile = np.full_like(z_heights, kappa * z_heights * u_star / blh * 100)
        kz_profile = np.clip(kz_profile, 1.0, 100.0)
    
    return z_heights.astype(np.float32), kz_profile.astype(np.float32)