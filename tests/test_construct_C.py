"""
Tests for construct_C module.
"""

import sys
import numpy as np
import pytest


sys.path.insert(0, '../src')
from grid_setup import construct_grid
from construct_C import construct_C
import jax
jax.config.update("jax_enable_x64", True)

class TestConstructC:
    """Tests for construct_C function."""
    
    def test_shape(self):
        """Test that C has correct dimensions."""
        P, Q = 10, 8
        grid = construct_grid(P, Q)
        U = np.zeros(2 * P * Q)
        C = construct_C(grid, U, rho=1.0, nu=0.1, U_lid=1.0)
        assert C.shape == (2 * P * Q,)
    
    def test_zero_velocity_gives_zero_C(self):
        """Test that C=0 when U=0 (no convection, no diffusion)."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        U = np.zeros(2 * P * Q)
        C = construct_C(grid, U, rho=1.0, nu=0.1, U_lid=0.0)
        
        assert np.allclose(C, 0.0), "C should be zero when U=0 and U_lid=0"
    
    def test_uniform_flow_no_convection(self):
        """Test that uniform flow has no convective contribution."""
        P, Q = 6, 6
        grid = construct_grid(P, Q)
        
        # Uniform flow: u=1, v=0 everywhere
        U = np.zeros(2 * P * Q)
        for m in range(P * Q):
            U[2*m] = 1.0
            U[2*m + 1] = 0.0
        
        # With nu=0, C should be zero for interior (no diffusion, no convection)
        C = construct_C(grid, U, rho=1.0, nu=0.0, U_lid=1.0)
        
        # Interior points should have C=0 (uniform flow, no viscosity)
        for m in grid.indices()['interior']:
            assert abs(C[2*m]) < 1e-10, f"C_u at interior {m} = {C[2*m]}"
            assert abs(C[2*m + 1]) < 1e-10, f"C_v at interior {m} = {C[2*m + 1]}"
    
    def test_lid_boundary_effect(self):
        """Test that lid velocity affects top boundary C values."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        dx = dy = 0.1
        rho = 1.0
        nu = 0.1
        
        # Zero interior velocity
        U = np.zeros(2 * P * Q)
        
        # Compare C with different lid velocities
        C1 = construct_C(grid, U, rho, nu, U_lid=0.0)
        C2 = construct_C(grid, U, rho, nu, U_lid=1.0)
        
        # Top boundary and top corners should differ
        top_points = list(grid.indices()['top_b']) + [grid.indices()['corners'][2], grid.indices()['corners'][3]]
        
        differs = False
        for m in top_points:
            if not np.allclose(C1[2*m:2*m+2], C2[2*m:2*m+2]):
                differs = True
                break
        
        assert differs, "Lid velocity should affect C at top boundary"
    
    def test_viscous_diffusion(self):
        """Test viscous term with a known velocity profile."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        rho = 1.0
        nu = 0.1
        dx = grid.dx
        dy = grid.dy
        dm = rho * dx * dy
        
        U = np.zeros(2 * P * Q)
        
        # Center point (2,2) -> m = 2*5 + 2 = 12
        m_center = 12
        m_below = m_center - P  # 7
        m_above = m_center + P  # 17
        m_left = m_center - 1   # 11
        m_right = m_center + 1  # 13
        
        # Set u values such that d2u/dy2 = 2
        # (u_above - 2*u_center + u_below) / dy^2 = 2
        # Choose u_center = 1.0, and symmetric difference
        # u_above - 2*u_center + u_below = 2 * dy^2
        # Let u_below = u_center - delta, u_above = u_center + delta
        # Then: (u_center + delta) - 2*u_center + (u_center - delta) = 0  # symmetric doesn't work
        # Instead: u_above = u_center + dy^2, u_below = u_center - dy^2 doesn't work either
        # 
        # Simpler: set u_below = 0, u_center = dy^2, u_above = 4*dy^2
        # Then: (4*dy^2 - 2*dy^2 + 0) / dy^2 = 2  ✓
        
        u_below = 0.0
        u_center = dy**2
        u_above = 4 * dy**2
        
        U[2*m_center] = u_center
        U[2*m_below] = u_below
        U[2*m_above] = u_above
        U[2*m_left] = u_center   # same as center (no x-gradient)
        U[2*m_right] = u_center  # same as center (no x-gradient)
        
        C = construct_C(grid, U, rho, nu, U_lid=0.0)
        
        # d2u/dy2 = (4*dy^2 - 2*dy^2 + 0) / dy^2 = 2
        # d2u/dx2 = 0
        # alpha = nu * 2 = 0.2
        # C_u = dm * (0 - alpha) = -dm * 0.2
        
        expected_C_u = -dm * nu * 2.0
        assert abs(C[2*m_center] - expected_C_u) < 1e-10, \
            f"C_u = {C[2*m_center]}, expected {expected_C_u}"
    
    def test_convective_term(self):
        """Test convective term with known velocity gradient."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        #dx = dy = 0.2
        dx = grid.dx
        dy = grid.dy
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
        
        C = construct_C(grid, U, rho, nu, U_lid=0.0)
        
        # A = 0.5*u0*(u2-u1)/dx = 0.5*1.0*(0.4-0.0)/0.2 = 1.0
        # C_u = dm * A = 0.04 * 1.0 = 0.04
        expected_C_u = dm * 0.5 * 1.0 * (0.4 - 0.0) / dx
        
        assert abs(C[2*m_center] - expected_C_u) < 1e-10, \
            f"C_u = {C[2*m_center]}, expected {expected_C_u}"
    
    def test_scaling_with_rho(self):
        """Test that C scales linearly with rho."""
        P, Q = 4, 4
        grid = construct_grid(P, Q)
        
        U = np.random.rand(2 * P * Q)
        
        C1 = construct_C(grid, U, rho=1.0, nu=0.1, U_lid=1.0)
        C2 = construct_C(grid, U, rho=2.0, nu=0.1, U_lid=1.0)
        
        # C should scale with rho (through dm = rho * dx * dy)
        assert np.allclose(C2, 2.0 * C1), "C should scale linearly with rho"
    
    def test_symmetry_with_symmetric_input(self):
        """Test C has expected symmetry for symmetric velocity field."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        dx = dy = 0.2
        
        # Create left-right symmetric velocity field
        U = np.zeros(2 * P * Q)
        for m in range(P * Q):
            col = m % P
            row = m // P
            # u symmetric about center column
            U[2*m] = abs(col - 2) * 0.1
            U[2*m + 1] = 0.0
        
        C = construct_C(grid, U, rho=1.0, nu=0.1, U_lid=0.0)
        
        # Check that points symmetric about center have related C values
        # Point (1,2) and (3,2) should have C_v equal (by symmetry)
        m1 = 2 * P + 1  # col=1, row=2
        m2 = 2 * P + 3  # col=3, row=2
        
        assert abs(C[2*m1 + 1] - C[2*m2 + 1]) < 1e-10, \
            "Symmetric points should have equal C_v"

    def test_construct_C_vectorized(self):
        """Test that the vectorized, current version of construct_C gives the same result as the original version."""
        P, Q = 5, 5
        grid = construct_grid(P, Q)
        dx = dy = 0.2
        rho = 1.0
        nu = 0.1
        U_lid = 0.0
        for _ in range(100):
            U = np.random.rand(2 * P * Q)  # random test velocity
            C_vectorized = construct_C(grid, U, rho, nu, U_lid)
            indices = grid.indices()
            indices['P'] = grid.P
            indices['Q'] = grid.Q
            indices['dx'] = grid.dx
            indices['dy'] = grid.dy
            C_original = construct_C_original(indices, grid.dx, grid.dy, U, rho, nu, U_lid)
            assert np.allclose(C_vectorized, C_original), "construct_C_vectorized and construct_C_original give different results"


def construct_C_original(grid, dx, dy, U, rho, nu, U_lid):
    """
    Construct the C vector from current velocity field.
    
    Parameters
    ----------
    grid : dict
        Grid indices from create_indices()
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    U : ndarray
        Current velocity field, shape (2*P*Q,)
        U[2*m] = u at point m, U[2*m+1] = v at point m
    rho : float
        Fluid density
    nu : float
        Kinematic viscosity
    U_lid : float
        Lid velocity (top boundary condition)
    
    Returns
    -------
    C : ndarray
        Free acceleration vector, shape (2*P*Q,)
    """
    P = grid['P']
    Q = grid['Q']
    
    C = np.zeros(2 * P * Q)
    dm = rho * dx * dy
    
    # Corners
    _fill_corners_C(C, grid['corners'], P, dx, dy, U, dm, nu, U_lid)
    
    # Boundaries
    _fill_left_C(C, grid['left_b'], P, dx, dy, U, dm, nu)
    _fill_right_C(C, grid['right_b'], P, dx, dy, U, dm, nu)
    _fill_bottom_C(C, grid['bottom_b'], P, dx, dy, U, dm, nu)
    _fill_top_C(C, grid['top_b'], P, dx, dy, U, dm, nu, U_lid)
    
    # Interior
    _fill_interior_C(C, grid['interior'], P, dx, dy, U, dm, nu)
    
    return C


def _compute_C_point(u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu):
    """
    Compute C for a single grid point.
    
    Neighbor notation:
        u1, v1 = left neighbor
        u2, v2 = right neighbor
        ua, va = below neighbor
        ub, vb = above neighbor
        u0, v0 = current point
    """
    # Convective terms
    A = 0.5 * u0 * (u2 - u1) / dx + 0.5 * v0 * (ub - ua) / dy
    B = 0.5 * u0 * (v2 - v1) / dx + 0.5 * v0 * (vb - va) / dy
    
    # Viscous terms (Laplacian)
    alpha = nu * ((u2 - 2*u0 + u1) / dx**2 + (ub - 2*u0 + ua) / dy**2)
    beta = nu * ((v2 - 2*v0 + v1) / dx**2 + (vb - 2*v0 + va) / dy**2)
    
    # C = dm * (convection - diffusion)
    C_u = dm * (A - alpha)
    C_v = dm * (B - beta)
    
    return C_u, C_v


def _get_velocities(U, m):
    """Get u, v at point m."""
    return U[2*m], U[2*m + 1]


def _fill_interior_C(C, interior, P, dx, dy, U, dm, nu):
    """Fill C for interior points (all 4 neighbors exist)."""
    for m in interior:
        u0, v0 = _get_velocities(U, m)
        u1, v1 = _get_velocities(U, m - 1)      # left
        u2, v2 = _get_velocities(U, m + 1)      # right
        ua, va = _get_velocities(U, m - P)      # below
        ub, vb = _get_velocities(U, m + P)      # above
        
        C[2*m], C[2*m + 1] = _compute_C_point(
            u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
        )


def _fill_left_C(C, left_b, P, dx, dy, U, dm, nu):
    """Fill C for left boundary (u1=0, v1=0 at wall)."""
    for m in left_b:
        u0, v0 = _get_velocities(U, m)
        u1, v1 = 0.0, 0.0                       # wall
        u2, v2 = _get_velocities(U, m + 1)      # right
        ua, va = _get_velocities(U, m - P)      # below
        ub, vb = _get_velocities(U, m + P)      # above
        
        C[2*m], C[2*m + 1] = _compute_C_point(
            u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
        )


def _fill_right_C(C, right_b, P, dx, dy, U, dm, nu):
    """Fill C for right boundary (u2=0, v2=0 at wall)."""
    for m in right_b:
        u0, v0 = _get_velocities(U, m)
        u1, v1 = _get_velocities(U, m - 1)      # left
        u2, v2 = 0.0, 0.0                       # wall
        ua, va = _get_velocities(U, m - P)      # below
        ub, vb = _get_velocities(U, m + P)      # above
        
        C[2*m], C[2*m + 1] = _compute_C_point(
            u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
        )


def _fill_bottom_C(C, bottom_b, P, dx, dy, U, dm, nu):
    """Fill C for bottom boundary (ua=0, va=0 at wall)."""
    for m in bottom_b:
        u0, v0 = _get_velocities(U, m)
        u1, v1 = _get_velocities(U, m - 1)      # left
        u2, v2 = _get_velocities(U, m + 1)      # right
        ua, va = 0.0, 0.0                       # wall
        ub, vb = _get_velocities(U, m + P)      # above
        
        C[2*m], C[2*m + 1] = _compute_C_point(
            u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
        )


def _fill_top_C(C, top_b, P, dx, dy, U, dm, nu, U_lid):
    """Fill C for top boundary (ub=U_lid, vb=0 at lid)."""
    for m in top_b:
        u0, v0 = _get_velocities(U, m)
        u1, v1 = _get_velocities(U, m - 1)      # left
        u2, v2 = _get_velocities(U, m + 1)      # right
        ua, va = _get_velocities(U, m - P)      # below
        ub, vb = U_lid, 0.0                     # lid
        
        C[2*m], C[2*m + 1] = _compute_C_point(
            u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
        )


def _fill_corners_C(C, corners, P, dx, dy, U, dm, nu, U_lid):
    """Fill C for corner points."""
    bl, br, tl, tr = corners
    
    # Bottom-left: left=wall, below=wall
    m = bl
    u0, v0 = _get_velocities(U, m)
    u1, v1 = 0.0, 0.0                       # wall (left)
    u2, v2 = _get_velocities(U, m + 1)      # right
    ua, va = 0.0, 0.0                       # wall (below)
    ub, vb = _get_velocities(U, m + P)      # above
    C[2*m], C[2*m + 1] = _compute_C_point(
        u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
    )
    
    # Bottom-right: right=wall, below=wall
    m = br
    u0, v0 = _get_velocities(U, m)
    u1, v1 = _get_velocities(U, m - 1)      # left
    u2, v2 = 0.0, 0.0                       # wall (right)
    ua, va = 0.0, 0.0                       # wall (below)
    ub, vb = _get_velocities(U, m + P)      # above
    C[2*m], C[2*m + 1] = _compute_C_point(
        u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
    )
    
    # Top-left: left=wall, above=lid
    m = tl
    u0, v0 = _get_velocities(U, m)
    u1, v1 = 0.0, 0.0                       # wall (left)
    u2, v2 = _get_velocities(U, m + 1)      # right
    ua, va = _get_velocities(U, m - P)      # below
    ub, vb = U_lid, 0.0                     # lid (above)
    C[2*m], C[2*m + 1] = _compute_C_point(
        u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
    )
    
    # Top-right: right=wall, above=lid
    m = tr
    u0, v0 = _get_velocities(U, m)
    u1, v1 = _get_velocities(U, m - 1)      # left
    u2, v2 = 0.0, 0.0                       # wall (right)
    ua, va = _get_velocities(U, m - P)      # below
    ub, vb = U_lid, 0.0                     # lid (above)
    C[2*m], C[2*m + 1] = _compute_C_point(
        u0, v0, u1, u2, ua, ub, v1, v2, va, vb, dx, dy, dm, nu
    )

def construct_C_vectorized(P, Q, dx, dy, U, rho, nu, U_lid):
    dm = rho * dx * dy
    
    # Reshape to (Q, P, 2)
    U_grid = U.reshape(Q, P, 2)
    u = U_grid[:, :, 0]
    v = U_grid[:, :, 1]
    
    # Pad with boundary conditions
    u_padded = np.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    v_padded = np.pad(v, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    
    # Fix top boundary: u = U_lid
    u_padded[-1, 1:-1] = U_lid
    
    # Extract current points and neighbors
    u0 = u_padded[1:-1, 1:-1]
    u1 = u_padded[1:-1, :-2]    # left
    u2 = u_padded[1:-1, 2:]     # right
    ua = u_padded[:-2, 1:-1]    # below
    ub = u_padded[2:, 1:-1]     # above
    
    v0 = v_padded[1:-1, 1:-1]
    v1 = v_padded[1:-1, :-2]
    v2 = v_padded[1:-1, 2:]
    va = v_padded[:-2, 1:-1]
    vb = v_padded[2:, 1:-1]
    
    # Convective terms (all points at once)
    A = 0.5 * u0 * (u2 - u1) / dx + 0.5 * v0 * (ub - ua) / dy
    B = 0.5 * u0 * (v2 - v1) / dx + 0.5 * v0 * (vb - va) / dy
    
    # Viscous terms
    alpha = nu * ((u2 - 2*u0 + u1) / dx**2 + (ub - 2*u0 + ua) / dy**2)
    beta = nu * ((v2 - 2*v0 + v1) / dx**2 + (vb - 2*v0 + va) / dy**2)
    
    # C = dm * (convection - diffusion)
    C_u = dm * (A - alpha)
    C_v = dm * (B - beta)
    
    # Stack and flatten back to interleaved format
    C_grid = np.stack([C_u, C_v], axis=-1)  # shape (Q, P, 2)
    C = C_grid.reshape(-1)                   # shape (2*P*Q,)
    
    return C

if __name__ == "__main__":
    pytest.main([__file__, "-v"])