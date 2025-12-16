import matplotlib.pyplot as plt
from reference_data import U_CENTERLINE_RE100, V_CENTERLINE_RE100
from ldc_Re100_simulation import P, Q, U_lid, dx, dy

import numpy as np

U = np.load('results/Re100_50x50_U.npy')


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
ax1.scatter(u_ref, y_ref, c='k',label='Ghia et. al (ref)')
ax1.plot(u_vert, y_vert, 'b-', markersize=4, linewidth=1, markevery=2, label=f'VPNS {P}×{Q}')
ax1.set_xlabel(r'$u/U_{lid}$')
ax1.set_ylabel('y')
ax1.set_title('u-velocity along vertical centerline (x=0.5)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.scatter(x_ref, v_ref, c='k',label='Ghia et. al (ref)')
ax2.plot(x_horz, v_horz, 'b-', markersize=4, linewidth=1, markevery=2, label=f'VPNS {P}×{Q}')
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
plt.savefig('results/ldc_Re100_validation.png', dpi=150)
print("\nFigure saved to ldc_results.png")