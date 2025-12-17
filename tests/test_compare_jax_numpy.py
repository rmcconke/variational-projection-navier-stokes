import numpy as np
import jax.numpy as jnp
from jax import jit, lax
import time
import sys
sys.path.insert(0, '../src')

import jax
jax.config.update("jax_enable_x64", True)

from grid_setup import construct_grid
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt

from numpy_implementation.grid_setup_numpy import construct_grid as construct_grid_numpy
from numpy_implementation.construct_A_numpy import construct_A as construct_A_numpy
from numpy_implementation.construct_C_numpy import construct_C as construct_C_numpy
from numpy_implementation.optimal_Udot_numpy import compute_projection_matrices as compute_projection_matrices_numpy
from numpy_implementation.optimal_Udot_numpy import compute_optimal_Udot as compute_optimal_Udot_numpy
from numpy_implementation.optimal_Udot_numpy import compute_M_neg_sqrt as compute_M_neg_sqrt_numpy


P, Q = 50, 50
W, H = 1.0, 1.0
rho = 999.8
mu = 0.9
U_lid = 0.0180036 * 5
nu = mu / rho

n_steps = 1000
CFL_max = 0.1


def test_jax_vs_numpy():
    """
    Test that JAX is faster than NumPy, and returns the same results. 
    Note: this is with FP64 JAX, not FP32.
    The numpy implementation here is from the last main branch that only used numpy.
    It is vectorized, and uses sparse matrix multiplication.
    I ran this test in a loop 10x:
    Mean speedup: 3.59 ± 0.63x
    TODO: When I have a more concrete/consistent API for running the simulations, this will need to be updated.
    """
    # ----------------------
    # NumPy implementation
    # ----------------------
    grid_np = construct_grid_numpy(P, Q, W, H)
    dt = CFL_max * min(grid_np['dx'], grid_np['dy']) / U_lid

    A_np = construct_A_numpy(grid_np)
    M_neg_sqrt_np = compute_M_neg_sqrt_numpy(grid_np, rho)
    print('Computing projection matrices (NumPy)...')
    P_proj_np, N_proj_np = compute_projection_matrices_numpy(A_np, M_neg_sqrt_np)
    
    U_numpy = np.zeros(2 * P * Q)

    # Warmup step
    C = construct_C_numpy(grid_np, U_numpy, rho, nu, U_lid)
    U_dot, _ = compute_optimal_Udot_numpy(M_neg_sqrt_np, C, P_proj_np, N_proj_np)
    U_numpy = U_numpy + dt * U_dot

    print(f"Running NumPy implementation ({n_steps} steps)...")
    start_time_numpy = time.perf_counter()
    for step in range(n_steps):
        C = construct_C_numpy(grid_np, U_numpy, rho, nu, U_lid)
        U_dot, _ = compute_optimal_Udot_numpy(M_neg_sqrt_np, C, P_proj_np, N_proj_np)
        U_numpy = U_numpy + dt * U_dot
    elapsed_time_numpy = time.perf_counter() - start_time_numpy
    print(f"NumPy: {elapsed_time_numpy:.4f} seconds")

    # ----------------------
    # JAX implementation
    # ----------------------
    grid_jax = construct_grid(P, Q, W, H)

    A_jax = construct_A(grid_jax)
    M_neg_sqrt_jax = compute_M_neg_sqrt(grid_jax, rho)
    print('Computing projection matrices (JAX)...')
    P_proj_jax, N_proj_jax = compute_projection_matrices(A_jax, M_neg_sqrt_jax)

    U_jax = jnp.zeros(2 * P * Q)

    @jit
    def run_n_steps(U, n):
        def body(i, U):
            C = construct_C(grid_jax, U, rho, nu, U_lid)
            U_dot, _ = compute_optimal_Udot(M_neg_sqrt_jax, C, P_proj_jax, N_proj_jax)
            return U + dt * U_dot
        return lax.fori_loop(0, n, body, U)

    # Warmup (triggers JIT compilation)
    U_jax = run_n_steps(U_jax, 1)
    U_jax.block_until_ready()

    print(f"Running JAX implementation ({n_steps} steps)...")
    start_time_jax = time.perf_counter()
    U_jax = run_n_steps(U_jax, n_steps)
    U_jax.block_until_ready()
    elapsed_time_jax = time.perf_counter() - start_time_jax
    print(f"JAX: {elapsed_time_jax:.4f} seconds")

    print(f"JAX speedup: {elapsed_time_numpy / elapsed_time_jax:.2f}x")

    assert elapsed_time_jax < elapsed_time_numpy, "JAX should be faster than NumPy"
    assert np.allclose(U_jax, U_numpy), "Results should match"
    return elapsed_time_numpy / elapsed_time_jax # return speedup


if __name__ == "__main__":
    test_jax_vs_numpy()