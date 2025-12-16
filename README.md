# Python implementation of Variational Navier-Stokes Solver
Based on the paper by Taha and Anand: [Variational Projection of Navier-Stokes: Fluid Mechanics as a Quadratic Programming Problem](https://arxiv.org/abs/2511.03896). Adapted from their provided MATLAB source code.

# Current state
Code is validated for Re=100 lid driven cavity (see `results/` folder). 

```
# Runs simulation on 50x50 grid (takes a couple minutes)
python3 src/ldc_Re100_simulation.py 

# Plots against reference data (Ghia et. al LDC data https://doi.org/10.1016/0021-9991(82)90058-4)
python3 src/ldc_Re100_validation.py
```

(I'm using uv, so you can also do `uv sync`, `uv run src/ldc_Re100_simulation.py`, and `uv run src/ldc_Re100_validation.py`)

Hard-coded:
- 2nd order central difference scheme for all terms
- Lid driven cavity boundary conditions

# To-do's:
- Change to jax implementation for GPU acceleration
- Switch to generic BC's to allow different flows to be simulated
- Allow other numerical schemes