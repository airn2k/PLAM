import numpy as np

def psi_m(zeta):
    """Stability function for momentum"""
    # Derived from Paulson, C. A. ‘The Mathematical Representation of Wind Speed and Temperature Profiles in the Unstable Atmospheric Surface Layer’.

    # Clip zeta to reasonable bounds to prevent numerical issues
    zeta = np.clip(zeta, -2.0, 2.0)
    
    if zeta > 0:  # Stable
        return -5 * zeta
    elif zeta < 0:  # Unstable
        # Ensure argument is positive
        arg = max(1 - 16 * zeta, 0.01)
        x = arg**0.25
        return 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi/2
    else:
        return 0.0
    