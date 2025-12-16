"""
Reference data for Re=20 lid-driven cavity validation.

Digitized from Figure 1 in Taha & Anand (2025):
"Variational Projection of Navier-Stokes"

Data is from OpenFOAM 125x125 simulation (red line/asterisks).

IMPORTANT: The paper uses coordinates from -1 to 1.
We convert to 0 to 1 for our solver:
    x_ours = (x_paper + 1) / 2
    y_ours = (y_paper + 1) / 2
"""

import numpy as np


# =============================================================================
# Figure 1(a): u/U_lid vs y along vertical centerline (x = 0.5)
# Data provided as (u/U_lid, y_paper) pairs
# y_paper is in [-1, 1] range, convert to [0, 1]: y = (y_paper + 1) / 2
# =============================================================================

U_CENTERLINE_RE100 = np.array([
    [1.00000, 1.00000],
    [0.9766, 0.84123],
    [0.9688, 0.78871],
    [0.9609, 0.73722],
    [0.9531, 0.68717],
    [0.8516, 0.23151],
    [0.7344, 0.00332],
    [0.6172, -0.13641],
    [0.5000, -0.20581],
    [0.4531, -0.21090],
    [0.2813, -0.15662],
    [0.1719, -0.10150],
    [0.1016, -0.06434],
    [0.0703, -0.04775],
    [0.0625, -0.04192],
    [0.0547, -0.03717],
    [0.0000, 0.00000],
])

# =============================================================================
# Table 2: v-velocity along horizontal centerline (y = 0.5)
# Data provided as (x, v/U_lid) pairs for Re=100
# =============================================================================

# (x, v/U_lid) pairs for Re=100
V_CENTERLINE_RE100 = np.array([
    [1.0000, 0.00000],
    [0.9688, -0.05906],
    [0.9609, -0.07391],
    [0.9531, -0.08864],
    [0.9453, -0.10313],
    [0.9063, -0.16914],
    [0.8594, -0.22445],
    [0.8047, -0.24533],
    [0.5000, 0.05454],
    [0.2344, 0.17527],
    [0.2266, 0.17507],
    [0.1563, 0.16077],
    [0.0938, 0.12317],
    [0.0781, 0.10890],
    [0.0703, 0.10091],
    [0.0625, 0.09233],
    [0.0000, 0.00000],
])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    ax[0].scatter(U_CENTERLINE_RE100[:,1], U_CENTERLINE_RE100[:,0])
    ax[1].scatter(V_CENTERLINE_RE100[:,0], V_CENTERLINE_RE100[:,1])
    plt.savefig('reference_data.png', dpi=150, bbox_inches='tight')
    plt.show()