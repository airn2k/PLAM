from helper.agl_band_mean_weighted import agl_band_mean_weighted
import numpy as np
def test_agl_band_mean_weighted():
    # Test case 1:
    # Create a 3D concentration array with nx = 2, ny = 2, nz = 3 and concentration for nz = 0, 1, 2, are 1.0, 2.0, 3.0 for all nx and ny
    conc3d = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]], dtype=np.float64)
    # Ground height grid has 0 value for nx and ny, so the lower_nz will be 0 and upper_nz will be 2 (0 + 2.0 / 1.0 = 2)
    ground_height_grid = np.zeros((2, 2), dtype=np.float64)
    dz = 1.0
    agl_top_m = 2.0

    expected_output = np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float64)

    output = agl_band_mean_weighted(conc3d, ground_height_grid, dz, agl_top_m)

    assert np.allclose(output, expected_output), "Test case 1 failed"