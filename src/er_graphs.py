import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from utils import plot_adjacency_matrix, plot_raster
from networks import create_random_network, run_single_network
import matplotlib.pyplot as plt
import time

from jax import lax, random
import jax
import jax.numpy as jnp

from attack import run_attack_batch_base, run_attack_batch_stdp
from run_modules import generate_graphs_and_neurons, test_attack_pipeline_seq_base, test_attack_pipeline_seq_stdp


def generate_erdos_renyi(n_nodes: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    if rng is None:
        rng = np.random.default_rng()
    p = rng.uniform(*kwargs.get("p", (0.01, 0.1)))

    generated_seed = int(rng.integers(0, 2**32 - 1))
    graph = nx.erdos_renyi_graph(n_nodes, p=p, directed=True, seed=generated_seed)

    params = {'p': p}
    return graph, params


structures = {
    # Erdős-Rényi configurations
    'ER_sparse': {
        'kwargs':{'p': (0.001, 0.01)},
        'graph_category': 'sparse',
        'weight_bounds': (1., 50.),
    },
    'ER_intermediate': {
        'kwargs':{'p': (0.01, 0.05)},
        'graph_category': 'intermediate',
        'weight_bounds': (1., 10.),
    },
    'ER_dense': {
        'kwargs':{'p': (0.05, 0.3)},
        'graph_category': 'dense',
        'weight_bounds': (1., 3.),
    }
}

if __name__ == "__main__":
    # Define the structure types and their corresponding keys
    # Generate random seeds for experiments
    rng = np.random.default_rng()
    experiment_configs = [
        ('ER_sparse', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('ER_intermediate', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('ER_dense', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1)))
    ]

    # Run each experiment
    for strtr, attack_key, graphs_key in experiment_configs:
        save_name = 'save/'  + strtr

        attack_params = {
            'n_nodes': 220,
            'T_global': 1000, #ms
            'batch_size': 20,
            'attack_fraction': 0.1,
            'attack_key': attack_key,
            'graphs_key': graphs_key,
            'number_of_graphs': 2
        }

        p_ggn = lambda structure, attack_params: generate_graphs_and_neurons(structure, attack_params, generate_erdos_renyi)

        # Run base test
        test_attack_pipeline_seq_base(structures[strtr], save_name+'_base', attack_params, p_ggn)

        # Increment keys for STDP test
        attack_params['attack_key'] = attack_params['attack_key'] + 1
        attack_params['graphs_key'] = attack_params['graphs_key'] + 1

        # Run STDP test
        test_attack_pipeline_seq_stdp(structures[strtr], save_name+'_stdp', attack_params, p_ggn)
