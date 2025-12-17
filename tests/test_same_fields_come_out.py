import numpy as np
import sys
sys.path.insert(0, '../src')
from grid_setup import construct_grid
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt
import pytest


def test_same_fields_come_out():
    """Very simple test to check that the same fields come out as when the solver was validated.
    The solver was validated at 50x50 resolution, and then I ran the solver with just a 5x5 resolution, 100 steps
    The fields were saved to a file, and now we check that the fields are the same.
    """
    U_ref = np.load('U_5x5_Re100_100steps_CFL0.05_conv1e-8.npy')

    P, Q = 5,5
    W, H = 1.0, 1.0
    rho = 999.8
    mu = 0.9
    U_lid = 0.0180036*5
    nu = mu / rho

    n_steps = 100
    CFL_max = 0.05

    dx = W / (P + 1)
    dy = H / (Q + 1)
    dt = CFL_max * min(dx, dy) / U_lid

    grid = construct_grid(P, Q, W, H)
    A = construct_A(grid)
    M_neg_sqrt = compute_M_neg_sqrt(grid, rho)
    P_proj, N_proj = compute_projection_matrices(A, M_neg_sqrt)

    U = np.zeros(2 * P * Q)  # Initial velocity

    for step in range(1, n_steps + 1):
        C = construct_C(grid, U, rho, nu, U_lid)
        
        U_dot, _ = compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj)
        
        U = U + dt * U_dot

    assert np.allclose(U, U_ref)
    print(f"Example fields: U = {U[:10]}, U_ref = {U_ref[:10]}")

if __name__ == '__main__':
    pytest.main([__file__, "-v"])