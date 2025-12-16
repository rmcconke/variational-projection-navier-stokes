"""
Module 2: Construct the A matrix (discrete divergence operator).

A enforces the incompressibility constraint: [A] @ U_dot = 0
This is the discretized form of div(u_dot) = 0.

Matrix dimensions: (P*Q) rows x (2*P*Q) columns
- Each row corresponds to one grid point's divergence constraint
- Columns correspond to velocity DOFs: [u0, v0, u1, v1, ..., u_{PQ-1}, v_{PQ-1}]
"""

import numpy as np
import jax.numpy as jnp

# TODO: Check factor of 2 discrepancy with paper (Eq. 28 uses 1/(2dx), code uses 1/dx)


def construct_A(grid):
    """
    Construct the discrete divergence operator matrix A.
    
    Parameters
    ----------
    grid : dict
        Grid indices from construct_grid()
    
    Returns
    -------
    A : ndarray
        Divergence operator matrix of shape (P*Q, 2*P*Q)
    """
    P = grid['P']
    Q = grid['Q']
    dx = grid['dx']
    dy = grid['dy']
    
    A = np.zeros((P * Q, 2 * P * Q))
    
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    
    # Corners
    _fill_corners(A, grid['indices']['corners'], P, inv_dx, inv_dy)
    
    # Boundaries
    _fill_left_boundary(A, grid['indices']['left_b'], P, inv_dx, inv_dy)
    _fill_right_boundary(A, grid['indices']['right_b'], P, inv_dx, inv_dy)
    _fill_bottom_boundary(A, grid['indices']['bottom_b'], P, inv_dx, inv_dy)
    _fill_top_boundary(A, grid['indices']['top_b'], P, inv_dx, inv_dy)
    
    # Interior
    _fill_interior(A, grid['indices']['interior'], P, inv_dx, inv_dy)
    
    return A


def _fill_corners(A, corners, P, inv_dx, inv_dy):
    """Fill A matrix entries for corner points."""
    bl, br, tl, tr = corners
    
    # Bottom-left: no left neighbor, no below neighbor
    m = bl
    A[m, 2*(m+1)] = inv_dx      # right neighbor u
    A[m, 2*(m+P)+1] = inv_dy    # above neighbor v
    
    # Bottom-right: no right neighbor, no below neighbor
    m = br
    A[m, 2*(m-1)] = -inv_dx     # left neighbor u
    A[m, 2*(m+P)+1] = inv_dy    # above neighbor v
    
    # Top-left: no left neighbor, no above neighbor
    m = tl
    A[m, 2*(m+1)] = inv_dx      # right neighbor u
    A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v
    
    # Top-right: no right neighbor, no above neighbor
    m = tr
    A[m, 2*(m-1)] = -inv_dx     # left neighbor u
    A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v


def _fill_left_boundary(A, left_b, P, inv_dx, inv_dy):
    """Fill A matrix entries for left boundary (excluding corners)."""
    for m in left_b:
        # No left neighbor (wall)
        A[m, 2*(m+1)] = inv_dx      # right neighbor u
        A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v
        A[m, 2*(m+P)+1] = inv_dy    # above neighbor v


def _fill_right_boundary(A, right_b, P, inv_dx, inv_dy):
    """Fill A matrix entries for right boundary (excluding corners)."""
    for m in right_b:
        # No right neighbor (wall)
        A[m, 2*(m-1)] = -inv_dx     # left neighbor u
        A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v
        A[m, 2*(m+P)+1] = inv_dy    # above neighbor v


def _fill_bottom_boundary(A, bottom_b, P, inv_dx, inv_dy):
    """Fill A matrix entries for bottom boundary (excluding corners)."""
    for m in bottom_b:
        # No below neighbor (wall)
        A[m, 2*(m-1)] = -inv_dx     # left neighbor u
        A[m, 2*(m+1)] = inv_dx      # right neighbor u
        A[m, 2*(m+P)+1] = inv_dy    # above neighbor v


def _fill_top_boundary(A, top_b, P, inv_dx, inv_dy):
    """Fill A matrix entries for top boundary (excluding corners)."""
    for m in top_b:
        # No above neighbor (lid, but v=0 there)
        A[m, 2*(m-1)] = -inv_dx     # left neighbor u
        A[m, 2*(m+1)] = inv_dx      # right neighbor u
        A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v


def _fill_interior(A, interior, P, inv_dx, inv_dy):
    """Fill A matrix entries for interior points."""
    for m in interior:
        A[m, 2*(m-1)] = -inv_dx     # left neighbor u
        A[m, 2*(m+1)] = inv_dx      # right neighbor u
        A[m, 2*(m-P)+1] = -inv_dy   # below neighbor v
        A[m, 2*(m+P)+1] = inv_dy    # above neighbor v