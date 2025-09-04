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
def generate_erdos_renyi(n_nodes: int, nx_seed: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    """Generate Erdős–Rényi random graph."""

    # Parameter ranges for different categories
    param_ranges = {
        "sparse": (0.001, 0.01),
        "intermediate": (0.01, 0.1),
        "dense": (0.1, 0.3)
    }

    # Use provided probability or sample from category
    if 'p' in kwargs:
        p = kwargs['p']
    else:
        category = kwargs.get('category', 'intermediate')
        if category not in param_ranges:
            raise ValueError(f"ER category must be one of: {list(param_ranges.keys())}")
        p = rng.uniform(*param_ranges[category])

    graph = nx.erdos_renyi_graph(n_nodes, p=p, directed=True, seed=nx_seed)

    params = {
        'p': p,
        'category': kwargs.get('category', 'custom' if 'p' in kwargs else 'intermediate')
    }

    return graph, params

def test_one_graph():
    # Example usage
    n_nodes = 500
    nx_seed = 42
    rng = np.random.default_rng(3)
    T_global = 5000 #ms
    graph, params = generate_erdos_renyi(n_nodes, nx_seed, rng, p=0.1)
    plot_adjacency_matrix(graph, weights_range=(0,1), cmap='Blues')


    neurons, graph = create_random_network(graph, n_nodes, weight_bounds= (1., 10.))

    plot_adjacency_matrix(graph, weights_range=(-10., 10.), cmap='PuOr_r')
    print(f"Generated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Parameters: {params}")


    V_hist, S_hist, I_syn_hist, I_ext, final_state, final_stdp_state = run_single_network(
        neurons, graph, T=T_global, dt=0.25, I_ext=None, I_ext_val=10, network_type='stdp_syn', nkey=42)

    plot_raster(S_hist, neurons, final_state.W, T_global, title="Spike raster", apply_style=True)

    syn_strength = jnp.abs(I_syn_hist).mean(axis=1)
    # Average absolute external input per neuron
    if I_ext.ndim == 1:  # Same for all neurons
        ext_strength = jnp.abs(I_ext)  # Shape: (time_steps,)
    else:  # Neuron-specific
        ext_strength = jnp.abs(I_ext).mean(axis=1)  # Shape: (time_steps,)

    denom = syn_strength + ext_strength
    R_t = jnp.where(denom > 0, syn_strength / denom, 0)

    plt.figure()
    plt.hist(R_t)
    plt.title('Relative Synaptic Strength')
    plt.xlabel('Time (ms)')
    plt.ylabel('Relative Strength')
    plt.show()


def test_attack_pipeline_seq_base(structure, save_path, attack_params):
    t0_global = time.perf_counter()

    start_time = time.time()

    n_nodes = attack_params['n_nodes']
    T_global = attack_params['T_global'] #ms
    batch_size = attack_params['batch_size']
    attack_fraction = attack_params['attack_fraction']
    attack_key = attack_params['attack_key']
    graphs_key = attack_params['graphs_key']
    number_of_graphs = attack_params['number_of_graphs']

    G_list = []
    neurons_list = []
    weight_bounds = structure['weight_bounds']
    p_bounds = structure['p']

    for i in range(number_of_graphs):
        key = random.PRNGKey(graphs_key + i)

        rng_key, nx_key = random.split(key)
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        rng = np.random.default_rng(int(random.randint(rng_key, (), 0, 2**30 - 1)))

        p = rng.uniform(p_bounds[0], p_bounds[1])
        rng_key, nx_key = random.split(key)
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        rng = np.random.default_rng(int(random.randint(rng_key, (), 0, 2**30 - 1)))
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        nx_seed = int(random.randint(nx_key, (), 0, 2**30 - 1))

        # GRAPH GENRATION
        graph, params = generate_erdos_renyi(n_nodes, nx_seed, rng, p=p)
        neurons, graph = create_random_network(graph, n_nodes, weight_bounds= weight_bounds, key=key)

        G_list.append(graph)
        neurons_list.append(neurons)


    I_ext = jnp.ones((T_global,n_nodes))*10

    end_time = time.time()
    print(f"Creating graphs and neurons total {end_time - start_time:.4f} seconds")
    run_attack_batch_base(
        G_list,
        neurons_list,
        I_ext,
        batch_size,
        attack_fraction,
        attack_key,
        save_path
        )
    t1_global = time.perf_counter()
    print(f"ALL TOOK {t1_global- t0_global:.3f}s")


def test_attack_pipeline_seq_stdp(structure, save_path, attack_params):
    t0_global = time.perf_counter()

    start_time = time.time()
    n_nodes = attack_params['n_nodes']
    T_global = attack_params['T_global'] #ms
    batch_size = attack_params['batch_size']
    attack_fraction = attack_params['attack_fraction']
    attack_key = attack_params['attack_key']
    graphs_key = attack_params['graphs_key']
    number_of_graphs = attack_params['number_of_graphs']

    G_list = []
    neurons_list = []
    weight_bounds = structure['weight_bounds']
    p_bounds = structure['p']
    for i in range(number_of_graphs):
        key = random.PRNGKey(graphs_key + i)

        rng_key, nx_key = random.split(key)
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        rng = np.random.default_rng(int(random.randint(rng_key, (), 0, 2**30 - 1)))

        p = rng.uniform(p_bounds[0], p_bounds[1])
        rng_key, nx_key = random.split(key)
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        rng = np.random.default_rng(int(random.randint(rng_key, (), 0, 2**30 - 1)))
        # Use a smaller upper bound (2**30 - 1 instead of 2**31)
        nx_seed = int(random.randint(nx_key, (), 0, 2**30 - 1))
        graph, params = generate_erdos_renyi(n_nodes, nx_seed, rng, p=p)
        neurons, graph = create_random_network(graph, n_nodes, weight_bounds= weight_bounds, key=key)

        G_list.append(graph)
        neurons_list.append(neurons)


    I_ext = jnp.ones((T_global,n_nodes))*10

    end_time = time.time()
    print(f"Creating graphs and neurons total {end_time - start_time:.4f} seconds")
    run_attack_batch_stdp(
        G_list,
        neurons_list,
        I_ext,
        batch_size,
        attack_fraction,
        attack_key,
        save_path = save_path
        )
    t1_global = time.perf_counter()
    print(f"ALL TOOK {t1_global- t0_global:.3f}s")

structures = {
    # Erdős-Rényi configurations
    'ER_sparse': {
        'p': (0.001, 0.01),
        'graph_category': 'sparse',
        'weight_bounds': (1., 50.),
    },
    'ER_intermediate': {
        'p': (0.01, 0.05),
        'graph_category': 'intermediate',
        'weight_bounds': (1., 10.),
    },
    'ER_dense': {
        'p': (0.05, 0.3),
        'graph_category': 'dense',
        'weight_bounds': (1., 3.),
    }
}

if __name__ == "__main__":
    attack_params = {
        'n_nodes': 100,
        'T_global': 5000, #ms
        'batch_size': 100,
        'attack_fraction': 0.1,
        'attack_key': 42,
        'graphs_key': 256,
        'number_of_graphs': 3
    }

    test_attack_pipeline_seq_base(structures['ER_intermediate'], 'save_0', attack_params)
