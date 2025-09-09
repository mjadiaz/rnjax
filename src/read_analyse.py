from networkx.readwrite.json_graph import adjacency
import orbax.checkpoint as ocp
import jax
import numpy as np
from jax import numpy as jnp
from pathlib import Path
from attack import keep_indices
import matplotlib.pyplot as plt
from measures import entropic_measures,lz_complexity_measures, sample_entropy_measures
from utils import plot_raster_simple

import time
from importlib import metadata

base_dir = Path("save_test_intermediate").resolve()
base_dir.mkdir(parents=True, exist_ok=True)

n_of_directories =len([d for d in base_dir.iterdir() if d.is_dir()])
# step = 3
total_time_start = time.time()

all_emsrs = []
all_lz = []
all_sp = []
with ocp.CheckpointManager(base_dir) as mngr:
    for step in range(n_of_directories):
        time_start = time.time()
        restored = mngr.restore(step)
        print(restored.arrays.keys())

        base_S_hist = restored.arrays['base_S_hist']
        pruned_S_hist_batch = restored.arrays['pruned_S_hist_batch']
        adjacency_matrix = np.array(restored.arrays['W0'])
        removed_ids = restored.arrays['removed_ids']
        neuron_type = restored.arrays['neuron_type']
        n_nodes = restored.metadata['parameters']['N']

        take_idx = keep_indices(removed_ids[step], n_nodes)

        # Needs to be numpy for measure libs infomeasure and Antropy
        pre_attack_S = np.array(base_S_hist)
        post_attack_S = np.array(pruned_S_hist_batch[step])
        # Initialize lists to collect metrics if it's the first step

        emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
        all_emsrs.append(emsrs)
        lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
        all_lz.append(lz)
        sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)
        all_sp.append(sp)

        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

total_time_end = time.time()
print(f"Total time taken for all directories: {total_time_end - total_time_start} seconds")
