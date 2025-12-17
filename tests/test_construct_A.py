"""
Tests for construct_A module.
"""
import sys
sys.path.insert(0, '../src')
import config

import numpy as np
import pytest


from grid_setup import construct_grid
from construct_A import construct_A
import jax.numpy as jnp


class TestConstructA:
    """Tests for construct_A function."""
    grid = construct_grid(P=10, Q=8, W=1.0, H=1.0)
    A = construct_A(grid)

    def test_shape(self):
        """Test that A has correct dimensions."""
        assert self.A.shape == (self.grid.P * self.grid.Q, 2 * self.grid.P * self.grid.Q)
    
    def test_sparsity(self):
        """Test that A is sparse (at most 4 non-zeros per row)."""
        
        
        for row in range(self.grid.P * self.grid.Q):
            nnz = np.count_nonzero(self.A[row, :])
            assert nnz <= 4, f"Row {row} has {nnz} non-zeros, expected <= 4"
    
    def test_interior_has_4_entries(self):
        """Test that interior points have exactly 4 non-zero entries."""    
        for m in self.grid.indices()['interior']:
            nnz = np.count_nonzero(self.A[m, :])
            assert nnz == 4, f"Interior point {m} has {nnz} non-zeros, expected 4"
    
    def test_corners_have_2_entries(self):
        """Test that corner points have exactly 2 non-zero entries."""
        for m in self.grid.indices()['corners']:
            nnz = np.count_nonzero(self.A[m, :])
            assert nnz == 2, f"Corner point {m} has {nnz} non-zeros, expected 2"
    
    def test_side_boundaries_have_3_entries(self):
        """Test that side boundary points have exactly 3 non-zero entries."""
        for boundary in [self.grid.indices()['left_b'], self.grid.indices()['right_b'], 
                         self.grid.indices()['bottom_b'], self.grid.indices()['top_b']]:
            for m in boundary:
                nnz = np.count_nonzero(self.A[m, :])
                assert nnz == 3, f"Boundary point {m} has {nnz} non-zeros, expected 3"
    
    def test_divergence_free_field_gives_zero(self):
        """Test that A @ U_dot = 0 for a divergence-free velocity field."""
        # Create a simple divergence-free field: solid body rotation
        # u = -y, v = x (around center)
        U_dot = np.zeros(2 * self.grid.P * self.grid.Q)
        for m in range(self.grid.P * self.grid.Q):
            col = m % self.grid.P
            row = m // self.grid.P
            x = (col + 1) * self.grid.dx  # +1 because grid points are interior
            y = (row + 1) * self.grid.dy
            x_c, y_c = 0.5, 0.5  # center
            U_dot[2*m] = -(y - y_c)     # u = -(y - y_c)
            U_dot[2*m + 1] = (x - x_c)  # v = (x - x_c)
        
        result = self.A @ U_dot
        # Should be approximately zero (not exactly due to boundary effects)
        # Check interior points only
        for m in self.grid.indices()['interior']:
            assert abs(result[m]) < 1e-10, f"Interior point {m}: div = {result[m]}"
    
    def test_uniform_flow_divergence(self):
        """Test divergence of uniform flow is zero."""       
        # Uniform flow: u = 1, v = 0 everywhere
        U_dot = np.zeros(2 * self.grid.P * self.grid.Q)
        for m in range(self.grid.P * self.grid.Q):
            U_dot[2*m] = 1.0      # u = 1
            U_dot[2*m + 1] = 0.0  # v = 0
        
        result = self.A @ U_dot
        
        # For interior points, du/dx = 0 (uniform), dv/dy = 0
        for m in self.grid.indices()['interior']:
            assert abs(result[m]) < 1e-10, f"Interior point {m}: div = {result[m]}"
    
    def test_coefficients_scale_with_grid(self):
        """Test that coefficients scale correctly with dx, dy."""
        grid1 = construct_grid(P=self.grid.P, Q=self.grid.Q, W=1.0, H=1.0)
        grid2 = construct_grid(P=self.grid.P, Q=self.grid.Q, W=2.0, H=2.0)
        
        A1 = construct_A(grid1)
        A2 = construct_A(grid2)
        
        # A2 should have half the coefficient magnitudes of A1
        # (since inv_dx halves when dx doubles)
        ratio = np.abs(A1[A1 != 0]) / np.abs(A2[A2 != 0])
        assert np.allclose(ratio, 2.0), "Coefficients should scale as 1/dx"
    
    def test_small_grid_manual_check(self):
        """Manually verify A for a small 3x3 grid."""        
        # Grid layout (0-based):
        #   6  7  8   (top)
        #   3  4  5
        #   0  1  2   (bottom)
        
        # Point 4 is the only interior point
        # Its divergence uses: left=3, right=5, below=1, above=7
        # A[4, 2*3] = -1 (left u)
        # A[4, 2*5] = +1 (right u)
        # A[4, 2*1+1] = -1 (below v)
        # A[4, 2*7+1] = +1 (above v)
        grid = construct_grid(P=3, Q=3, W=4.0, H=4.0)
        A = construct_A(grid)

        assert A[4, 6] == -1.0   # 2*3 = 6, left u
        assert A[4, 10] == 1.0   # 2*5 = 10, right u
        assert A[4, 3] == -1.0   # 2*1+1 = 3, below v
        assert A[4, 15] == 1.0   # 2*7+1 = 15, above v
        
        # Check that only these 4 entries are non-zero in row 4
        row4_nnz = jnp.nonzero(A[4, :], size=4)[0]
        assert set(np.asarray(row4_nnz)) == {3, 6, 10, 15}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])