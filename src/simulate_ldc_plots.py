import numpy as np
import plotext as plt
from grid_setup import create_grid_indices
from construct_A import construct_A
from construct_C import construct_C
from optimal_Udot import compute_projection_matrices, compute_optimal_Udot, compute_M_neg_sqrt

P, Q = 5,5
W, H = 1.0, 1.0
rho = 999.8
mu = 0.9
U_lid = 0.0180036*5
nu = mu / rho

n_steps = 100
CFL_max = 0.05
conv_tol = 1e-8

plot_every = 10  # Update plot every N steps

# =============================================================================
# Setup (done once)
# =============================================================================
dx = W / (P + 1)
dy = H / (Q + 1)
dt = CFL_max * min(dx, dy) / U_lid

print(f"CFL max: {CFL_max:.6f}")
print(f"Pe: {U_lid * dx / nu:.6f}")
print(f"Re: {rho * U_lid * W / mu:.6f}")
print(f"dt: {dt:.6f}")

grid = create_grid_indices(P, Q)
A = construct_A(grid, dx, dy)
M_neg_sqrt = compute_M_neg_sqrt(P, Q, rho, dx, dy)

P_proj, N_proj = compute_projection_matrices(A, M_neg_sqrt)

U = np.zeros(2 * P * Q)  # Initial velocity
U_old = U.copy()

rel_history = []
S_star_history = []

# =============================================================================
# Main time-stepping loop
# =============================================================================
for step in range(1, n_steps + 1):
    # Compute free acceleration terms
    C = construct_C(grid, dx, dy, U, rho, nu, U_lid)
    
    # Compute optimal divergence-free acceleration
    U_dot, S_star = compute_optimal_Udot(M_neg_sqrt, C, P_proj, N_proj)
    
    # Euler update
    U_old = U.copy()
    U = U + dt * U_dot
    
    # Convergence monitoring
    dU = np.linalg.norm(U - U_old)
    U_norm = np.linalg.norm(U)
    rel_change = dU / U_norm if U_norm > 1e-12 else dU
    
    rel_history.append(rel_change)
    S_star_history.append(S_star)
    
    # Live plot update
    if step % plot_every == 0 or step == 1:
        print(f"Step {step:.10d} S_star: {S_star:.6e} |U_dot|: {np.linalg.norm(U_dot):.6e}")
        plt.clear_terminal()
        plt.clear_figure()
        log_rel = [np.log10(r) if r > 0 else -12 for r in rel_history]
        plt.plot(log_rel)
        plt.ylim(-18, 0)
        plt.title(f"Convergence (step {step}/{n_steps})")
        plt.ylabel("log10(|dU|/|U|)")
        plt.xlabel("step")
        plt.show()

    if rel_change < conv_tol:
        break

np.save('results/U.npy', U)
print(f"\nDone! Max velocity: {np.max(np.abs(U/U_lid)):.6e}")
print(f"Max divergence: {np.max(np.abs(A @ U)):.6e}")

# =============================================================================
# Results
# =============================================================================

# =============================================================================
# Extract and plot centerline data
# =============================================================================
import matplotlib.pyplot as plt
from reference_data import U_CENTERLINE_RE100, V_CENTERLINE_RE100

# Extract velocity field as 2D arrays
u = np.zeros((Q, P))
v = np.zeros((Q, P))
for m in range(P * Q):
    col = m % P
    row = m // P
    u[row, col] = U[2*m]
    v[row, col] = U[2*m + 1]

# Grid coordinates
x = np.array([(i + 1) * dx for i in range(P)])
y = np.array([(j + 1) * dy for j in range(Q)])
X, Y = np.meshgrid(x, y)

# Velocity magnitude
vel_mag = np.sqrt(u**2 + v**2)

# Vertical centerline (x ≈ 0.5): middle column
mid_col = P // 2
y_vert = Y[:, mid_col]
u_vert = u[:, mid_col] / U_lid

# Horizontal centerline (y ≈ 0.5): middle row
mid_row = Q // 2
x_horz = X[mid_row, :]
v_horz = v[mid_row, :] / U_lid

# Get reference data
y_ref = U_CENTERLINE_RE100[:,0]
u_ref = U_CENTERLINE_RE100[:,1]
x_ref = V_CENTERLINE_RE100[:,0]
v_ref = V_CENTERLINE_RE100[:,1]

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Centerline plots
ax1 = axes[0, 0]
ax1.plot(u_ref, y_ref, 'k-', linewidth=2, label='OpenFOAM (ref)')
ax1.plot(u_vert, y_vert, 'ro--', markersize=4, linewidth=1, markevery=2, label=f'VPNS {P}×{Q}')
ax1.set_xlabel(r'$u/U_{lid}$')
ax1.set_ylabel('y')
ax1.set_title('u-velocity along vertical centerline (x=0.5)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(x_ref, v_ref, 'k-', linewidth=2, label='OpenFOAM (ref)')
ax2.plot(x_horz, v_horz, 'ro--', markersize=4, linewidth=1, markevery=2, label=f'VPNS {P}×{Q}')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$v/U_{lid}$')
ax2.set_title('v-velocity along horizontal centerline (y=0.5)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Velocity magnitude contours
ax3 = axes[1, 0]
contour = ax3.contourf(X, Y, vel_mag / U_lid, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax3, label=r'$|V|/U_{lid}$')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Velocity magnitude')
ax3.set_aspect('equal')

# Streamlines
ax4 = axes[1, 1]
ax4.streamplot(X, Y, u, v, density=1.5, linewidth=0.8, color=vel_mag, cmap='viridis')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Streamlines')
ax4.set_aspect('equal')
ax4.set_xlim(x.min(), x.max())
ax4.set_ylim(y.min(), y.max())

plt.tight_layout()
plt.savefig('ldc_results.png', dpi=150)
print("\nFigure saved to ldc_results.png")