import numpy as np
from typing import Dict, List, Optional
import networkx as nx
from networks import create_random_network
import jax.random as jr


#from er_graphs import test_attack_pipeline_seq_base, test_attack_pipeline_seq_stdp
from run_modules import generate_graphs_and_neurons, test_attack_pipeline_seq_base, test_attack_pipeline_seq_stdp, test_one_graph

def generate_scale_free(n_nodes: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    """
    Generate a directed, (asymptotically) scale-free graph by:
      1) building an undirected Barabási–Albert graph, then
      2) orienting each edge with probability `orient_frac` using `bias`.
    """

    if rng is None:
        rng = np.random.default_rng()
    # Sample parameters from provided ranges
    m = int(rng.uniform(*kwargs.get("m", (1, 2))))
    orient_frac = kwargs.get("orient_frac", 0.5)
    bias = rng.uniform(*kwargs.get("bias", (0.0, 1.0)))
    p = rng.uniform(*kwargs.get("p", (0.0, 0.1)))

    # Calculate maximum q value such that p + q < 1
    max_q = 1.0 - p
    # Sample q from its range, but cap it at max_q to ensure p + q < 1
    q_min, q_max = kwargs.get("q", (0.0, 0.1))
    q = rng.uniform(q_min, min(q_max, max_q))


    # Validation
    if not (0.0 <= orient_frac <= 1.0):
        raise ValueError("orient_frac must be in [0, 1].")
    if not (0.0 <= bias <= 1.0):
        raise ValueError("bias must be in [0, 1].")
    if m < 1:
        raise ValueError("m must be >= 1.")
    if n_nodes <= m:
        raise ValueError("n_nodes must be > m (required by the BA model).")


    # First, generate a seed from the RNG instead of using nx_seed directly
    generated_seed = int(rng.integers(0, 2**32 - 1))
    G = nx.extended_barabasi_albert_graph(
        n_nodes, m, p=p, q=q, seed=generated_seed)

    # Convert to directed graph with orientation
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))  # keep node attributes

    for u, v, data in G.edges(data=True):
        if rng.random() < orient_frac:
            # make it one-way
            if rng.random() < bias:
                H.add_edge(u, v, **data)
            else:
                H.add_edge(v, u, **data)
        else:
            # keep it symmetric (both directions)
            H.add_edge(u, v, **data)
            H.add_edge(v, u, **data)

    # Store all parameters to return later
    params = {
        'm': m,
        'orient_frac': orient_frac,
        'bias': bias,
        'p': p,
        'q': q
    }
    return H, params
    # Scale-Free configurations
    #


def test_graph():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    G, params= generate_scale_free(200, rng, **structures['SF_receiver']['kwargs'])

    print(params)
    plt.imshow(nx.to_numpy_array(G).T)
    plt.show()



structures = {
    'SF_receiver_sparse': {
        'graph_type': 'SF',
        'graph_category': 'receiver',
        'kwargs': {
            'm': (1,3),
            'orient_frac':  1.,
            'bias': (0.01, 0.2),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 50.),
    },
    'SF_receiver_intermediate': {
        'graph_type': 'SF',
        'graph_category': 'receiver',
        'kwargs': {
            'm': (4,6),
            'orient_frac':  1.,
            'bias': (0.01, 0.2),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 30.),
    },
    'SF_receiver_dense': {
        'graph_type': 'SF',
        'graph_category': 'receiver',
        'kwargs': {
            'm': (7,10),
            'orient_frac':  1.,
            'bias': (0.01, 0.2),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 20.),
    },
    'SF_broadcaster_sparse': {
        'graph_type': 'SF',
        'graph_category': 'broadcaster',
        'kwargs': {
            'm': (1,3),
            'orient_frac':  1.,
            'bias': (0.7, 1.),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 50.),
    },
    'SF_broadcaster_intermediate': {
        'graph_type': 'SF',
        'graph_category': 'broadcaster',
        'kwargs': {
            'm': (4,6),
            'orient_frac':  1.,
            'bias': (0.7, 1.),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 30.),
    },
    'SF_broadcaster_dense': {
        'graph_type': 'SF',
        'graph_category': 'broadcaster',
        'kwargs': {
            'm': (7,10),
            'orient_frac':  1.,
            'bias': (0.7, 1.),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 20.),
    },
    'SF_balanced_sparse': {
        'graph_type': 'SF',
        'graph_category': 'balanced',
        'kwargs': {
            'm': (1, 3),
            'orient_frac':  1.,
            'bias': (0.3, 0.6),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 50.),
    },
    'SF_balanced_intermediate': {
        'graph_type': 'SF',
        'graph_category': 'balanced',
        'kwargs': {
            'm': (4, 6),
            'orient_frac':  1.,
            'bias': (0.3, 0.6),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 40.),
    },
    'SF_balanced_dense': {
        'graph_type': 'SF',
        'graph_category': 'balanced',
        'kwargs': {
            'm': (7, 10),
            'orient_frac':  1.,
            'bias': (0.3, 0.6),
            'p': (0.001, 0.1), # Keep Scale-free property
            'q': (0.001, 0.1),# Keep Scale-free property
        },
        'weight_bounds': (1., 30.),
    }
}

# HOW TO TEST ONE GRAPH
# rng = np.random.default_rng()

# n_nodes = 500
# strtr = 'SF_balanced_dense'
# G, params= generate_scale_free(n_nodes, rng, **structures[strtr]['kwargs'])
# test_one_graph(n_nodes, G, params, structures[strtr])

if __name__ == '__main__':
    """
    Study different directional properties such as receiver, broadcaster and balanced but most importantly
    density possibly with Sparse: m ∈ [1, 2], Intermediate: m ∈ [3, 5] and Dense: m ∈ [6, 9]. So in this case we have 3**2 combinations.
    In random we can have just 3 controling density.

    Remember we need to calibrate the weights so that the networks is synapse driven
    """

    # Define the structure types and their corresponding keys
    rng = np.random.default_rng()
    experiment_configs = [
        ('SF_receiver_sparse',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_receiver_intermediate', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_receiver_dense', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_broadcaster_sparse', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_broadcaster_intermediate',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_broadcaster_dense', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_balanced_sparse', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_balanced_intermediate',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SF_balanced_dense',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
    ]

    # Run each experiment
    for strtr, attack_key, graphs_key in experiment_configs:
        save_name = 'save/'  + strtr

        attack_params = {
            'n_nodes': 500,
            'T_global': 2000, #ms
            'batch_size': 10,
            'attack_fraction': 0.1,
            'attack_key': attack_key,
            'graphs_key': graphs_key,
            'number_of_graphs': 250
        }

        p_ggn = lambda structure, attack_params: generate_graphs_and_neurons(structure, attack_params, generate_scale_free)

        # Run base test
        test_attack_pipeline_seq_base(structures[strtr], save_name+'_base', attack_params, p_ggn)

        # Increment keys for STDP test
        attack_params['attack_key'] = attack_params['attack_key'] + 1
        attack_params['graphs_key'] = attack_params['graphs_key'] + 1

        # Run STDP test
        test_attack_pipeline_seq_stdp(structures[strtr], save_name+'_stdp', attack_params, p_ggn)
