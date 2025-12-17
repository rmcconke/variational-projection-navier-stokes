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
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnames=['grid'])
def construct_C(grid, U, rho, nu, U_lid):
   
    dm = rho * grid.dx * grid.dy
    
    # Reshape to (Q, P, 2)
    U_grid = U.reshape(grid.Q, grid.P, 2)
    u = U_grid[:, :, 0]
    v = U_grid[:, :, 1]
    
    # Pad with boundary conditions
    u_padded = jnp.pad(u, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    v_padded = jnp.pad(v, ((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    
    # Fix top boundary: u = U_lid (JAX functional update)
    u_padded = u_padded.at[-1, 1:-1].set(U_lid)
    
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
    A = 0.5 * u0 * (u2 - u1) / grid.dx + 0.5 * v0 * (ub - ua) / grid.dy
    B = 0.5 * u0 * (v2 - v1) / grid.dx + 0.5 * v0 * (vb - va) / grid.dy
    
    # Viscous terms
    alpha = nu * ((u2 - 2*u0 + u1) / grid.dx**2 + (ub - 2*u0 + ua) / grid.dy**2)
    beta = nu * ((v2 - 2*v0 + v1) / grid.dx**2 + (vb - 2*v0 + va) / grid.dy**2)
    
    # C = dm * (convection - diffusion)
    C_u = dm * (A - alpha)
    C_v = dm * (B - beta)
    
    # Stack and flatten back to interleaved format
    C_grid = jnp.stack([C_u, C_v], axis=-1)  # shape (Q, P, 2)
    C = C_grid.reshape(-1)                    # shape (2*P*Q,)
    
    return C


def construct_C_old(grid, U, rho, nu, U_lid):
    P = grid['P']
    Q = grid['Q']
    dx = grid['dx']
    dy = grid['dy']
    
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