"""
Module 4: Optimal U_dot solver.

Computes the optimal acceleration that minimizes the Appellian (pressure gradient cost)
while satisfying the divergence-free constraint.

From the paper (VPNS formulation, Eq. 21):
    U_dot_tilde = N @ U_dot_free_tilde
    
where:
    N = I - P                           (null space projector)
    P = (A @ M^{-1/2})^+ @ (A @ M^{-1/2})  (range space projector)
    U_dot_free_tilde = M^{-1/2} @ U_dot_free = -M^{-1/2} @ C / dm

The code uses:
    C_tilde = M^{-1/2} @ C
    U_dot_tilde_star = -N @ C_tilde
    S_star = 0.5 * C_tilde.T @ P @ C_tilde  (optimal Appellian)
    U_dot = M^{-1/2} @ U_dot_tilde_star
"""

import numpy as np
import time
from scipy.sparse import csr_matrix

import jax.numpy as jnp
from jax import jit

# TODO: sparse with jax?
def compute_projection_matrices(A, M_neg_sqrt, sparse=None):
    """
    Compute the projection matrices P and N.
    
    Parameters
    ----------
    A : jax.Array
        Divergence operator matrix, shape (P*Q, 2*P*Q)
    M_neg_sqrt : jax.Array
        M^{-1/2} as 1D array (diagonal of the matrix), shape (2*P*Q,)
    
    Returns
    -------
    P : jax.Array
        Range space projector, shape (2*P*Q, 2*P*Q)
    N : jax.Array
        Null space projector, shape (2*P*Q, 2*P*Q)
    """
    n_dof = len(M_neg_sqrt)
    
    # A_tilde = A @ diag(M_neg_sqrt)
    A_tilde = A * M_neg_sqrt  # broadcasting
    
    # Moore-Penrose pseudoinverse
    A_tilde_pinv = jnp.linalg.pinv(A_tilde)
    
    # P = A_tilde^+ @ A_tilde
    P = A_tilde_pinv @ A_tilde
    
    # N = I - P
    N = jnp.eye(n_dof) - P
    
    return P, N


@jit
def compute_optimal_Udot(M_neg_sqrt, C, P, N):
    """
    Compute the optimal acceleration and Appellian.
    
    Parameters
    ----------
    M_neg_sqrt : jax.Array
        M^{-1/2} as 1D array, shape (2*P*Q,)
    C : jax.Array
        Free acceleration vector (mass-weighted), shape (2*P*Q,)
    P : jax.Array
        Range space projector, shape (2*P*Q, 2*P*Q)
    N : jax.Array
        Null space projector, shape (2*P*Q, 2*P*Q)
    
    Returns
    -------
    U_dot : jax.Array
        Optimal acceleration, shape (2*P*Q,)
    S_star : float
        Optimal Appellian value
    """
    # C_tilde = M^{-1/2} @ C (element-wise since M is diagonal)
    C_tilde = M_neg_sqrt * C
    
    # U_dot_tilde_star = -N @ C_tilde
    U_dot_tilde_star = -N @ C_tilde
    
    # S_star = 0.5 * C_tilde.T @ P @ C_tilde
    S_star = 0.5 * jnp.dot(C_tilde, P @ C_tilde)
    
    # U_dot = M^{-1/2} @ U_dot_tilde_star
    U_dot = M_neg_sqrt * U_dot_tilde_star
    
    return U_dot, S_star


def compute_M_neg_sqrt(grid, rho):
    """
    Compute M^{-1/2} as a 1D array (diagonal elements).
    
    M = rho * dx * dy * I, so M^{-1/2} = (rho * dx * dy)^{-1/2} * I
    
    Parameters
    ----------
    grid : Grid
        Grid namedtuple with P, Q, dx, dy
    rho : float
        Fluid density
    
    Returns
    -------
    M_neg_sqrt : jax.Array
        Diagonal of M^{-1/2}, shape (2*P*Q,)
    """
    dm = rho * grid.dx * grid.dy
    m_neg_sqrt_val = dm ** (-0.5)
    return jnp.full(2 * grid.P * grid.Q, m_neg_sqrt_val)