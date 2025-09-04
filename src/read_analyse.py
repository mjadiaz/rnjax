import orbax.checkpoint as ocp
import jax
import numpy as np
from jax import numpy as jnp
from pathlib import Path
from attack import keep_indices
import matplotlib.pyplot as plt
from measures import entropic_measures,lz_complexity_measures, sample_entropy_measures
from utils import plot_raster_simple
from email.mime import base
base_dir = Path("save_test_intermediate").resolve()
base_dir.mkdir(parents=True, exist_ok=True)
step = 3
with ocp.CheckpointManager(base_dir) as mngr:

    restored = mngr.restore(step)
    print(restored.arrays.keys())

base_S_hist = restored.arrays['base_S_hist']
pruned_S_hist_batch = restored.arrays['pruned_S_hist_batch']
removed_ids = restored.arrays['removed_ids']
neuron_type = restored.arrays['neuron_type']

take_idx = keep_indices(removed_ids[step], 200)
# print(base_S_hist[:, take_idx].shape)
# print(pruned_S_hist_batch[step].shape)

# Needs to be numpy for measure libs infomeasure and Antropy
pre_attack_S = np.array(base_S_hist)
post_attack_S = np.array(pruned_S_hist_batch[step])


emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
print(emsrs)
lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
print(lz)
sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)
print(sp)
