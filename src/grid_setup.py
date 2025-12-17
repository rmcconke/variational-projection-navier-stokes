"""
Module 1: Grid setup and boundary index calculations for lid-driven cavity flow.

Grid layout (0-based indexing):
- P points along width (x-direction)
- Q points along height (y-direction)
- Points numbered in column-major order: m = j*P + i
  where i = column (0 to P-1), j = row (0 to Q-1)
"""

import numpy as np
import jax.numpy as jnp
from typing import NamedTuple


class GridIndices(NamedTuple):
    corners: np.ndarray
    left_b: np.ndarray
    right_b: np.ndarray
    bottom_b: np.ndarray
    top_b: np.ndarray
    interior: np.ndarray

class Grid(NamedTuple):
    P: int
    Q: int
    dx: float
    dy: float
    W: float=1.0
    H: float=1.0

    def indices(self):
        return create_grid_indices(self.P, self.Q)

def construct_grid(P, Q, W=1.0, H=1.0):
    dx = W / (P + 1)
    dy = H / (Q + 1)
    return Grid(P=P, Q=Q, dx=dx, dy=dy, W=W, H=H)


def create_grid_indices(P, Q):
    """
    Create all boundary and interior point index sets.
    Only used for one-time setup of the A matrix.
    """
    # Corners: bottom-left, bottom-right, top-left, top-right
    corners = np.array([
        0,              # bottom-left
        P - 1,          # bottom-right
        P * (Q - 1),    # top-left
        P * Q - 1       # top-right
    ], dtype=np.int32)
    
    # Left boundary: column 0, rows 1 to Q-2
    left_b = np.array([j * P for j in range(1, Q - 1)], dtype=np.int32)
    
    # Right boundary: column P-1, rows 1 to Q-2
    right_b = np.array([j * P + (P - 1) for j in range(1, Q - 1)], dtype=np.int32)
    
    # Bottom boundary: row 0, columns 1 to P-2
    bottom_b = np.array([i for i in range(1, P - 1)], dtype=np.int32)
    
    # Top boundary: row Q-1, columns 1 to P-2
    top_b = np.array([P * (Q - 1) + i for i in range(1, P - 1)], dtype=np.int32)
    
    # Interior points: rows 1 to Q-2, columns 1 to P-2
    interior = np.array([
        j * P + i 
        for j in range(1, Q - 1) 
        for i in range(1, P - 1)
    ], dtype=np.int32)
    
    return {
        'corners': corners,
        'left_b': left_b,
        'right_b': right_b,
        'bottom_b': bottom_b,
        'top_b': top_b,
        'interior': interior,
    }