import numpy as np
import plotext as plt
from grid_setup import construct_grid
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt

P, Q = 50,50
W, H = 1.0, 1.0
rho = 999.8
mu = 0.9
U_lid = 0.0180036*5
nu = mu / rho

n_steps = 100000000 # We're just going to run until convergence
CFL_max = 0.1
conv_tol = 1e-8

log_every = 10  # Update plot every N steps

if __name__ == "__main__":
    grid = construct_grid(P, Q, W, H)
    dt = CFL_max * min(grid['dx'], grid['dy']) / U_lid
    print(f"CFL max: {CFL_max:.6f}")
    print(f"Pe: {U_lid * grid['dx'] / nu:.6f}")
    print(f"Re: {rho * U_lid * W / mu:.6f}")
    print(f"dt: {dt:.6f}")

    A = construct_A(grid)
    M_neg_sqrt = compute_M_neg_sqrt(grid, rho)

    P_proj, N_proj = compute_projection_matrices(A, M_neg_sqrt)

    U = np.zeros(2 * P * Q)  # Initial velocity
    U_old = U.copy()

    for step in range(1, n_steps + 1):
        # Compute free acceleration terms
        C = construct_C(grid, U, rho, nu, U_lid)
        
        # Compute optimal divergence-free acceleration
        U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj)
        
        # Euler update
        U_old = U.copy()
        U = U + dt * U_dot
        
        # Convergence monitoring
        dU = np.linalg.norm(U - U_old)
        U_norm = np.linalg.norm(U)
        rel_change = dU / U_norm if U_norm > 1e-12 else dU
            
        # Live plot update
        if step % log_every == 0 or step == 1:
            print(f"Step {step:10d} dU/|U|: {rel_change:.6e} S_star: {S_star:.6e} |U_dot|: {np.linalg.norm(U_dot):.6e}")

        if rel_change < conv_tol:
            break

    np.save('results/Re100_50x50_U.npy', U)



