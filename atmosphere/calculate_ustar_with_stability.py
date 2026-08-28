import numpy as np
from atmosphere.psi_m import psi_m
def calculate_ustar_with_stability(wind_speed, sshf, z_ref=10.0, z0=0.03):
    """Calculate u* with atmospheric stability corrections"""
    kappa = 0.4
    g = 9.81
    cp = 1005
    rho = 1.225

    # Derived from Stull, Roland B. ‘Similarity Theory’. In An Introduction to Boundary Layer Meteorology
    # Validate inputs
    wind_speed = max(wind_speed, 0.1)  # Minimum wind speed
    z0 = max(z0, 0.001)              # Minimum roughness length
    
    # Basic friction velocity (neutral conditions)
    ustar_neutral = (kappa * wind_speed) / np.log(z_ref / z0)
    
    # Stability correction
    H_kinematic = sshf / (rho * cp)
    
    if abs(H_kinematic) > 1e-6:
        L_MO = -(ustar_neutral**3) / (kappa * g * H_kinematic)
        # Limit L_MO to reasonable bounds
        L_MO = np.clip(L_MO, -10000, 10000)
        zeta = z_ref / L_MO
        
        if zeta > 0:  # Stable
            zeta = min(zeta, 2.0)  # Limit stability parameter
            phi_m = 1 + 5 * zeta
        elif zeta < 0:  # Unstable
            zeta = max(zeta, -2.0)  # Limit stability parameter
            arg = max(1 - 16 * zeta, 0.01)  # Ensure positive argument
            phi_m = arg**(-0.25)
        else:  # Neutral
            phi_m = 1.0
        
        # Corrected friction velocity with bounds checking
        denominator = np.log(z_ref / z0) + psi_m(zeta)
        if abs(denominator) < 1e-6:
            ustar = ustar_neutral
        else:
            ustar = (kappa * wind_speed) / denominator
    else:
        ustar = ustar_neutral
    
    # Check for NaN and apply physical bounds
    if np.isnan(ustar) or ustar <= 0:
        ustar = ustar_neutral
    
    return np.clip(ustar, 0.01, 2.0)  # Physical bounds