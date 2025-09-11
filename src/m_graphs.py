import numpy as np
from typing import Dict, List, Optional
import networkx as nx
from networks import create_random_network
import jax.random as jr


from run_modules import generate_graphs_and_neurons, test_attack_pipeline_seq_base, test_attack_pipeline_seq_stdp, test_one_graph

def generate_stochastic_block_model(n_nodes: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    """Generate stochastic block model graph."""

    if rng is None:
        rng = np.random.default_rng()
    n_blocks = kwargs.get('n_blocks', 4)

    # Use provided block sizes or create equal-sized blocks
    if 'block_sizes' in kwargs:
        block_sizes = kwargs['block_sizes']
        if sum(block_sizes) != n_nodes:
            raise ValueError(f"Sum of block_sizes ({sum(block_sizes)}) must equal n_nodes ({n_nodes})")
    else:
        block_sizes = [n_nodes // n_blocks] * n_blocks
        # Handle remainder
        remainder = n_nodes % n_blocks
        for i in range(remainder):
            block_sizes[i] += 1

    # Get probability ranges from kwargs
    p_in_range = kwargs.get('p_in', None)
    p_out_range = kwargs.get('p_out', None)

    # Sample specific values using rng if ranges are provided
    if isinstance(p_in_range, tuple) and len(p_in_range) == 2:
        p_in = rng.uniform(p_in_range[0], p_in_range[1])
    else:
        p_in = p_in_range

    if isinstance(p_out_range, tuple) and len(p_out_range) == 2:
        p_out = rng.uniform(p_out_range[0], p_out_range[1])
    else:
        p_out = p_out_range

    # Create probability matrix
    P = np.full((n_blocks, n_blocks), p_out)
    np.fill_diagonal(P, p_in)

    generated_seed = int(rng.integers(0, 2**32 - 1))
    graph = nx.stochastic_block_model(block_sizes, P, directed=True, seed=generated_seed)

    params = {
        'n_blocks': n_blocks,
        'block_sizes': block_sizes,
        'p_in': p_in,
        'p_out': p_out,
        'P': P.tolist(),  # Convert numpy array to list for JSON serialization

    }

    return graph, params



def test_graph():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()
    G, params= generate_stochastic_block_model(200, rng, **structures['SBM_assortative']['kwargs'])

    print(params)
    plt.imshow(nx.to_numpy_array(G).T)
    plt.show()

structures = {
    'SBM_sparse_4': {
        'graph_type': 'SBM',
        'graph_category': 'sparse',
        'kwargs': {
            "n_blocks": 4 ,
            "p_in": (0.015, 0.025),
            "p_out": (0.001, 0.003),
        },
        'weight_bounds': (1., 60.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_sparse_3': {
        'graph_type': 'SBM',
        'graph_category': 'sparse',
        'kwargs': {
            "n_blocks": 3 ,
            "p_in": (0.015, 0.025),
            "p_out": (0.001, 0.003),
        },
        'weight_bounds': (1., 50.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_sparse_2': {
        'graph_type': 'SBM',
        'graph_category': 'sparse',
        'kwargs': {
            "n_blocks": 2 ,
            "p_in": (0.015, 0.025),
            "p_out": (0.001, 0.003),
        },
        'weight_bounds': (1., 40.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_intermediate_4': {
        'graph_type': 'SBM',
        'graph_category': 'intermediate',
        'kwargs': {
            "n_blocks": 4 ,
            "p_in": (0.03, 0.05),
            "p_out": (0.003, 0.007),
        },
        'weight_bounds': (1., 40.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_intermediate_3': {
        'graph_type': 'SBM',
        'graph_category': 'intermediate',
        'kwargs': {
            "n_blocks": 3 ,
            "p_in": (0.03, 0.05),
            "p_out": (0.003, 0.007),
        },
        'weight_bounds': (1., 30.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_intermediate_2': {
        'graph_type': 'SBM',
        'graph_category': 'intermediate',
        'kwargs': {
            "n_blocks": 2 ,
            "p_in": (0.03, 0.05),
            "p_out": (0.003, 0.007),
        },
        'weight_bounds': (1., 20.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_dense_4': {
        'graph_type': 'SBM',
        'graph_category': 'dense',
        'kwargs': {
            "n_blocks": 4 ,
            "p_in": (0.06, 0.09),
            "p_out": (0.006, 0.012),
        },
        'weight_bounds': (1., 10.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_dense_3': {
        'graph_type': 'SBM',
        'graph_category': 'dense',
        'kwargs': {
            "n_blocks": 3 ,
            "p_in": (0.06, 0.09),
            "p_out": (0.006, 0.012),
        },
        'weight_bounds': (1., 10.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    },
    'SBM_dense_2': {
        'graph_type': 'SBM',
        'graph_category': 'dense',
        'kwargs': {
            "n_blocks": 2 ,
            "p_in": (0.06, 0.09),
            "p_out": (0.006, 0.012),
        },
        'weight_bounds': (1., 5.),
        #   'description': 'SBM Assortative (high within-block, low between-block)'
    }}
if __name__ == '__main__':
    """
    For modular graphs we can vary p_in and p_out to control density, and also different block sizes and of course number of modules = [2,3,4].

    For density we can vary with the following bounds.
    Sparse:
        p_in = (0.015, 0.025)    # 1.5% to 2.5% intra-block connectivity
        p_out = (0.001, 0.003)   # 0.1% to 0.3% inter-block connectivity
        # Ratio: ~5-25x stronger intra-block connections

    Intermediate:
        p_in = (0.03, 0.05)      # 3% to 5% intra-block connectivity
        p_out = (0.003, 0.007)   # 0.3% to 0.7% inter-block connectivity
        # Ratio: ~5-15x stronger intra-block connections

    Dense:
        p_in = (0.06, 0.09)      # 6% to 9% intra-block connectivity
        p_out = (0.006, 0.012)   # 0.6% to 1.2% inter-block connectivity
        # Ratio: ~5-15x stronger intra-block connections

    Remember we need to calibrate the weights so that the networks is synapse driven
    """

    # Define the structure types and their corresponding keys
    # Generate random seeds for experiments
    rng = np.random.default_rng()
    experiment_configs = [
        ('SBM_sparse_4',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_sparse_3', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_sparse_2',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_intermediate_4',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_intermediate_3',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_intermediate_2',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_dense_4', int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_dense_3',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1))),
        ('SBM_dense_2',int(rng.integers(0, 2**32 - 1)), int(rng.integers(0, 2**32 - 1)))
    ]

    # Run each experiment
    for strtr, attack_key, graphs_key in experiment_configs:
        save_name = 'save/'  + strtr

        attack_params = {
            'n_nodes': 200,  # Changed from 500 to match other experiments
            'T_global': 2000, #ms
            'batch_size': 20,
            'attack_fraction': 0.1,
            'attack_key': attack_key,
            'graphs_key': graphs_key,
            'number_of_graphs': 20  # Changed from 1 to match other experiments
        }

        p_ggn = lambda structure, attack_params: generate_graphs_and_neurons(structure, attack_params, generate_stochastic_block_model)

        # Run base test
        test_attack_pipeline_seq_base(structures[strtr], save_name+'_base', attack_params, p_ggn)

        # Increment keys for STDP test
        attack_params['attack_key'] = attack_params['attack_key'] + 1
        attack_params['graphs_key'] = attack_params['graphs_key'] + 1

        # Run STDP test
        test_attack_pipeline_seq_stdp(structures[strtr], save_name+'_stdp', attack_params, p_ggn)
