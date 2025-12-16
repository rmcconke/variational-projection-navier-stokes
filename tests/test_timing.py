import sys
sys.path.insert(0, '../src')

from grid_setup import construct_grid
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt
import numpy as np
import timeit
import scipy

P, Q = 30, 30
rho = 999.8
mu = 0.9
U_lid = 0.0180036*5
nu = mu / rho

CFL_max = 0.05
dx = 1.0 / (1.0 + 1) # set implicitly L, W = 1.0 (for timing code)
dy = 1.0 / (1.0 + 1)
dt = CFL_max * min(dx, dy) / U_lid


def timer(func, repeat=10, number=1):
    """Run func multiple times and return statistics"""
    times = timeit.repeat(lambda: func(), repeat=repeat, number=number)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Or use helper
def sparse_size(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

# Timing construct_grid
grid = construct_grid(P, Q)
stats = timer(lambda: construct_grid(P, Q), repeat=10)
print(f"construct_grid: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

## Timing construct_A
A = construct_A(grid)
stats = timer(lambda: construct_A(grid), repeat=10)
print(f"construct_A: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

## Timing compute_M_neg_sqrt
M_neg_sqrt = compute_M_neg_sqrt(grid, rho)
stats = timer(lambda: compute_M_neg_sqrt(grid, rho), repeat=10)
print(f"compute_M_neg_sqrt: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

## Timing pseudoinverses
A_tilde = A * M_neg_sqrt  # broadcasting: each column j multiplied by M_neg_sqrt[j]    
stats = timer(lambda: np.linalg.pinv(A_tilde), repeat=1)
print(f"np.linalg.pinv(A_tilde): {stats['mean']:.4f}s ± {stats['std']:.4f}s")
stats= timer(lambda: scipy.linalg.pinv(A_tilde), repeat=1)
print(f"scipy.linalg.pinv(A_tilde): {stats['mean']:.4f}s ± {stats['std']:.4f}s")

## Timing compute_projection_matrices
P_proj, N_proj = compute_projection_matrices(A, M_neg_sqrt)
stats = timer(lambda: compute_projection_matrices(A, M_neg_sqrt), repeat=1)
print(f"compute_projection_matrices: {stats['mean']:.4f}s ± {stats['std']:.4f}s")
print(f"sparse P_proj size: {sparse_size(P_proj)/1E6} Mbytes")
print(f"sparse N_proj size: {sparse_size(N_proj)/1E6} Mbytes")

P_proj_dense, N_proj_dense = compute_projection_matrices(A, M_neg_sqrt, sparse=False)
print(f"dense P_proj size: {P_proj_dense.nbytes/1E6} Mbytes")
print(f"dense N_proj size: {N_proj_dense.nbytes/1E6} Mbytes")

U = np.zeros(2 * P * Q)  # Initial velocity
U_old = U.copy()

# Run first 5 steps then time
print("Running first 5 steps...")
for step in range(5):
    C = construct_C(grid, U, rho, nu, U_lid)
    U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj)
    # Euler update
    U_old = U.copy()
    U = U + dt * U_dot
print("Done. timing!")

# Timing construct_C
stats = timer(lambda: construct_C(grid, U, rho, nu, U_lid), repeat=10)
print(f"construct_C: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

# Timing compute_optimal_Udot
stats = timer(lambda: compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj), repeat=10)
print(f"compute_optimal_Udot (sparse): {stats['mean']:.4f}s ± {stats['std']:.4f}s")

# Timing compute_optimal_Udot
stats = timer(lambda: compute_optimal_Udot(M_neg_sqrt, C, P_proj_dense, N_proj_dense), repeat=10)
print(f"compute_optimal_Udot (dense): {stats['mean']:.4f}s ± {stats['std']:.4f}s")

print(f'Grid: {P}x{Q} = {P*Q}')
print(f'P_proj shape: {P_proj.shape}')
print(f'N_proj shape: {N_proj.shape}')
print(f'U shape: {U.shape}')


from test_construct_C import construct_C_original
grid['indices']['P'] = grid['P']
grid['indices']['Q'] = grid['Q']
grid['indices']['dx'] = grid['dx']
grid['indices']['dy'] = grid['dy']
stats = timer(lambda: construct_C_original(grid['indices'], grid['dx'], grid['dy'], U, rho, nu, U_lid), repeat=10)
print(f"construct_C_original: {stats['mean']:.4f}s ± {stats['std']:.4f}s")

from construct_C import construct_C
stats = timer(lambda: construct_C(grid, U, rho, nu, U_lid), repeat=10)
print(f"construct_C (vectorized): {stats['mean']:.4f}s ± {stats['std']:.4f}s")



