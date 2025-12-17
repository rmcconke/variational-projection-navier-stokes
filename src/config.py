import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") # CHANGE TO SET THE CUDA DEVICE

import jax
jax.config.update("jax_enable_x64", True)

# Optional: expose for other modules
DEVICE = jax.devices()[0]
BACKEND = jax.default_backend()

print(f"JAX device: {DEVICE}")
print(f"JAX backend: {BACKEND}")
print(f"x64 enabled: {jax.config.x64_enabled}")