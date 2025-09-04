import orbax.checkpoint as ocp
import jax
import numpy as np
from jax import numpy as jnp
from pathlib import Path
from attack import keep_indices
import matplotlib.pyplot as plt

from utils import plot_raster_simple
base_dir = Path("save_ckpts_big").resolve()
base_dir.mkdir(parents=True, exist_ok=True)
step = 0
with ocp.CheckpointManager(base_dir) as mngr:

    restored = mngr.restore(step)
    print(restored.arrays.keys())

base_S_hist = restored.arrays['base_S_hist']
pruned_S_hist_batch = restored.arrays['pruned_S_hist_batch']
removed_ids = restored.arrays['removed_ids']

print(removed_ids[0])
take_idx = keep_indices(removed_ids[0], 100)
print(take_idx)
print(base_S_hist[:, take_idx].shape)

print(pruned_S_hist_batch[0][ :, take_idx].shape)

plot_raster_simple(base_S_hist[:, take_idx], 5000)
plt.show()
plot_raster_simple(pruned_S_hist_batch[0][ :, take_idx],  5000)
plt.show()
