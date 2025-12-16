"""
Tests for construct_C module.
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '../src')
from grid_setup import create_grid_indices
from construct_C import construct_C


class TestConstructC:
    """Tests for construct_C function."""
    
    def test_shape(self):
        """Test that C has correct dimensions."""
        P, Q = 10, 8
        grid = create_grid_indices(P, Q)
        U = np.zeros(2 * P * Q)
        C = construct_C(grid, dx=0.1, dy=0.1, U=U, rho=1.0, nu=0.1, U_lid=1.0)
        assert C.shape == (2 * P * Q,)
    
    def test_zero_velocity_gives_zero_C(self):
        """Test that C=0 when U=0 (no convection, no diffusion)."""
        P, Q = 5, 5
        grid = create_grid_indices(P, Q)
        U = np.zeros(2 * P * Q)
        C = construct_C(grid, dx=0.1, dy=0.1, U=U, rho=1.0, nu=0.1, U_lid=0.0)
        
        assert np.allclose(C, 0.0), "C should be zero when U=0 and U_lid=0"
    
    def test_uniform_flow_no_convection(self):
        """Test that uniform flow has no convective contribution."""
        P, Q = 6, 6
        grid = create_grid_indices(P, Q)
        dx = dy = 0.1
        
        # Uniform flow: u=1, v=0 everywhere
        U = np.zeros(2 * P * Q)
        for m in range(P * Q):
            U[2*m] = 1.0
            U[2*m + 1] = 0.0
        
        # With nu=0, C should be zero for interior (no diffusion, no convection)
        C = construct_C(grid, dx=dx, dy=dy, U=U, rho=1.0, nu=0.0, U_lid=1.0)
        
        # Interior points should have C=0 (uniform flow, no viscosity)
        for m in grid['interior']:
            assert abs(C[2*m]) < 1e-10, f"C_u at interior {m} = {C[2*m]}"
            assert abs(C[2*m + 1]) < 1e-10, f"C_v at interior {m} = {C[2*m + 1]}"
    
    def test_lid_boundary_effect(self):
        """Test that lid velocity affects top boundary C values."""
        P, Q = 5, 5
        grid = create_grid_indices(P, Q)
        dx = dy = 0.1
        rho = 1.0
        nu = 0.1
        
        # Zero interior velocity
        U = np.zeros(2 * P * Q)
        
        # Compare C with different lid velocities
        C1 = construct_C(grid, dx, dy, U, rho, nu, U_lid=0.0)
        C2 = construct_C(grid, dx, dy, U, rho, nu, U_lid=1.0)
        
        # Top boundary and top corners should differ
        top_points = list(grid['top_b']) + [grid['corners'][2], grid['corners'][3]]
        
        differs = False
        for m in top_points:
            if not np.allclose(C1[2*m:2*m+2], C2[2*m:2*m+2]):
                differs = True
                break
        
        assert differs, "Lid velocity should affect C at top boundary"
    
    def test_viscous_diffusion(self):
        """Test viscous term with a known velocity profile."""
        P, Q = 5, 5
        grid = create_grid_indices(P, Q)
        dx = dy = 0.2
        rho = 1.0
        nu = 0.1
        dm = rho * dx * dy
        
        # Set up a parabolic profile u = y^2 (only the center point)
        # At interior point, laplacian(u) = d2u/dy2 = 2
        U = np.zeros(2 * P * Q)
        
        # For the center point m=12 (row 2, col 2 in 5x5)
        # Set velocities to create u = y^2 pattern locally
        # y positions: below=0.2, center=0.4, above=0.6
        # u values: below=0.04, center=0.16, above=0.36
        
        # Center point (2,2) -> m = 2*5 + 2 = 12
        m_center = 12
        m_below = m_center - P  # 7
        m_above = m_center + P  # 17
        m_left = m_center - 1   # 11
        m_right = m_center + 1  # 13
        
        # Set u values for parabolic profile
        U[2*m_center] = 0.16      # u at center
        U[2*m_below] = 0.04       # u below
        U[2*m_above] = 0.36       # u above
        U[2*m_left] = 0.16        # u left (same y)
        U[2*m_right] = 0.16       # u right (same y)
        
        C = construct_C(grid, dx, dy, U, rho, nu, U_lid=0.0)
        
        # Expected: d2u/dy2 = (0.36 - 2*0.16 + 0.04)/0.04 = 2
        # d2u/dx2 = 0 (same values left/right)
        # alpha = nu * 2 = 0.2
        # Convection A ≈ 0 (v=0, u differences cancel in x)
        # C_u = dm * (A - alpha) = dm * (-0.2) = -0.008
        
        expected_C_u = -dm * nu * 2.0
        assert abs(C[2*m_center] - expected_C_u) < 1e-10, \
            f"C_u = {C[2*m_center]}, expected {expected_C_u}"
    
    def test_convective_term(self):
        """Test convective term with known velocity gradient."""
        P, Q = 5, 5
        grid = create_grid_indices(P, Q)
        dx = dy = 0.2
        rho = 1.0
        nu = 0.0  # No viscosity to isolate convection
        dm = rho * dx * dy
        
        U = np.zeros(2 * P * Q)
        
        # Center point m=12
        m_center = 12
        m_below = m_center - P
        m_above = m_center + P
        m_left = m_center - 1
        m_right = m_center + 1
        
        # Set u0=1, v0=0 at center
        U[2*m_center] = 1.0
        U[2*m_center + 1] = 0.0
        
        # Set du/dx = (u2-u1)/(2*dx) = (0.4 - 0.0)/(2*0.2) = 1.0
        U[2*m_left] = 0.0
        U[2*m_right] = 0.4
        
        # Set neighbors for above/below (same as center for no y-gradient)
        U[2*m_below] = 1.0
        U[2*m_above] = 1.0
        
        C = construct_C(grid, dx, dy, U, rho, nu, U_lid=0.0)
        
        # A = 0.5*u0*(u2-u1)/dx = 0.5*1.0*(0.4-0.0)/0.2 = 1.0
        # C_u = dm * A = 0.04 * 1.0 = 0.04
        expected_C_u = dm * 0.5 * 1.0 * (0.4 - 0.0) / dx
        
        assert abs(C[2*m_center] - expected_C_u) < 1e-10, \
            f"C_u = {C[2*m_center]}, expected {expected_C_u}"
    
    def test_scaling_with_rho(self):
        """Test that C scales linearly with rho."""
        P, Q = 4, 4
        grid = create_grid_indices(P, Q)
        
        U = np.random.rand(2 * P * Q)
        
        C1 = construct_C(grid, 0.1, 0.1, U, rho=1.0, nu=0.1, U_lid=1.0)
        C2 = construct_C(grid, 0.1, 0.1, U, rho=2.0, nu=0.1, U_lid=1.0)
        
        # C should scale with rho (through dm = rho * dx * dy)
        assert np.allclose(C2, 2.0 * C1), "C should scale linearly with rho"
    
    def test_symmetry_with_symmetric_input(self):
        """Test C has expected symmetry for symmetric velocity field."""
        P, Q = 5, 5
        grid = create_grid_indices(P, Q)
        dx = dy = 0.2
        
        # Create left-right symmetric velocity field
        U = np.zeros(2 * P * Q)
        for m in range(P * Q):
            col = m % P
            row = m // P
            # u symmetric about center column
            U[2*m] = abs(col - 2) * 0.1
            U[2*m + 1] = 0.0
        
        C = construct_C(grid, dx, dy, U, rho=1.0, nu=0.1, U_lid=0.0)
        
        # Check that points symmetric about center have related C values
        # Point (1,2) and (3,2) should have C_v equal (by symmetry)
        m1 = 2 * P + 1  # col=1, row=2
        m2 = 2 * P + 3  # col=3, row=2
        
        assert abs(C[2*m1 + 1] - C[2*m2 + 1]) < 1e-10, \
            "Symmetric points should have equal C_v"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])