import numpy as np

def calculate_kxy_with_stability(ustar, blh, sshf, slhf=None, kxy_scale=2.0):
    # Calculate Kxy with stability-dependent scaling
    # Monin-Obukhov length for stability:
    if abs(sshf) > 1e-6:
        L_MO = -(ustar**3) / (0.4 * 9.81 * (sshf / (1.225 * 1005)))  # Convert to kinematic units
        L_MO = np.clip(L_MO, -10000, 10000)
    else:
        L_MO = 1e6  # Neutral

    # Convective enhancement for very unstable conditions
    # Deardorff (1970) suggests that for convective conditions, horizontal diffusion can be enhanced due to large eddies, that scales with the convective velocity scale w* and boundary layer height.
    if sshf > 0 and slhf is not None:
        # Convective velocity scale
        w_star = ((9.81 * blh * (sshf + 0.61 * 273 * slhf/2.45e6)) / 273)**0.333
        K_xy = 0.1 * w_star * blh

    else:
        # From Businger-Dyer stability functions 
        zeta = blh / L_MO

        # Kxy base is derived from K ~ u* x L from Tennekes and Lumley, A First Course in Turbulence. 
        Kxy_base = kxy_scale * ustar * blh

        if zeta < 0:
            stability_factor = (1 - 15*zeta)**0.25
        else:
            stability_factor = 1.0 / (1 + 4.7 *zeta)
        
        Kxy = Kxy_base * stability_factor
            
