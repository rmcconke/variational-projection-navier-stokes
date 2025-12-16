"""
Module 3: Construct the C vector.

C contains the mass-weighted "free acceleration" terms:
    C = dm * (convection - diffusion)
    C = dm * (-U_dot_free)

where:
    convection = u * du/dx + v * du/dy  (for u-component)
    diffusion = nu * laplacian(u)       (for u-component)
    dm = rho * dx * dy                  (element mass)

The discretization uses central differences:
    du/dx ≈ (u2 - u1) / (2*dx)  ->  but code uses 0.5*(u2-u1)/dx = (u2-u1)/(2*dx)
    laplacian(u) ≈ (u2 - 2*u0 + u1)/dx^2 + (ub - 2*u0 + ua)/dy^2
"""

import numpy as np


def construct_C(grid, dx, dy, U, rho, nu, U_lid):
    """
    Construct the C vector from current velocity field.
    
    Parameters
    ----------
    grid : dict
        Grid indices from create_grid_indices()
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