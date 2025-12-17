import config  # Must be first!
import jax.numpy as jnp
from jax import jit, lax
import jax

from grid_setup import construct_grid
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt
P, Q = 50, 50
W, H = 1.0, 1.0
rho = 999.8
mu = 0.9
U_lid = 0.0180036 * 5
nu = mu / rho

n_steps = 100000000
CFL_max = 0.1
conv_tol = 1e-5
log_every = 100  # Controls lax.scan block size

grid = construct_grid(P, Q, W, H)
dx = grid.dx
dy = grid.dy

def run_simulation():
    # Setup grid
    grid = construct_grid(P, Q, W, H)
    dt = CFL_max * min(grid.dx, grid.dy) / U_lid

    print(f"CFL max: {CFL_max:.6f}")
    print(f"Pe: {U_lid * grid.dx / nu:.6f}")
    print(f"Re: {rho * U_lid * W / mu:.6f}")
    print(f"dt: {dt:.6f}")

    # Build matrices (one-time setup)
    A = construct_A(grid)
    M_neg_sqrt = compute_M_neg_sqrt(grid, rho)
    P_proj, N_proj = compute_projection_matrices(A, M_neg_sqrt)

    # Initial velocity
    U = jnp.zeros(2 * P * Q)

    # JIT-compile a block of steps using lax.scan
    @jit
    def run_block(U):
        def body(U, _):
            C = construct_C(grid, U, rho, nu, U_lid)
            U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj)
            U_new = U + dt * U_dot
            return U_new, (U_dot, S_star)
        
        U_final, (U_dot_history, S_star_history) = lax.scan(body, U, None, length=log_every)
        return U_final, U_dot_history[-1], S_star_history[-1]

    # Time-stepping loop
    step = 0
    while step < n_steps:
        U_old = U
        U, U_dot, S_star = run_block(U)
        step += log_every

        # Convergence monitoring
        dU = jnp.linalg.norm(U - U_old)
        U_norm = jnp.linalg.norm(U)
        rel_change = dU / U_norm if U_norm > 1e-12 else dU
        
        print(f"Step {step:10d} dU/|U|: {rel_change:.6e} S_star: {S_star:.6e} |U_dot|: {jnp.linalg.norm(U_dot):.6e}")

        if rel_change < conv_tol:
            print("Converged!")
            break

    # Save results
    #
    return U


if __name__ == "__main__":
    U = run_simulation()
    jnp.save('results/Re100_50x50_U.npy', U)