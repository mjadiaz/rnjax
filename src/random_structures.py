import numpy as np
import networkx as nx
import random
from typing import Dict, List, Optional



def get_structure_configurations():
    """
    Define all network structure configurations to test.

    Returns:
        dict: Dictionary with structure configurations
    """
    structures = {
        # Erdős-Rényi configurations
        'ER_sparse': {
            'graph_type': 'ER',
            'graph_category': 'sparse',
            'weight_bounds': (1., 50.),
       #     'description': 'Erdős-Rényi Sparse (p=0.001-0.01)'
        },
        'ER_intermediate': {
            'graph_type': 'ER',
            'graph_category': 'intermediate',
            'weight_bounds': (1., 10.),
         #   'description': 'Erdős-Rényi Intermediate (p=0.01-0.05)'
        },
        'ER_dense': {
            'graph_type': 'ER',
            'graph_category': 'dense',
            'weight_bounds': (1., 3.),
        #    'description': 'Erdős-Rényi Dense (p=0.05-0.3)'
        },

        # Scale-Free configurations
        'SF_receiver': {
            'graph_type': 'SF',
            'graph_category': 'receiver',
            'weight_bounds': (1., 50.),
        #    'description': 'Scale-Free Receiver Model (high α, low γ)'
        },
        'SF_broadcaster': {
            'graph_type': 'SF',
            'graph_category': 'broadcaster',
            'weight_bounds': (1., 20.),
          #  'description': 'Scale-Free Broadcaster (low α, high γ)'
        },
        'SF_balanced': {
            'graph_type': 'SF',
            'graph_category': 'balanced',
            'weight_bounds': (1., 30.),
          #  'description': 'Scale-Free Balanced (moderate α, β, γ)'
        },

        # Stochastic Block Model configurations
        'SBM_assortative': {
            'graph_type': 'SBM',
            'graph_category': 'assortative',
            'weight_bounds': (1., 20.),
         #   'description': 'SBM Assortative (high within-block, low between-block)'
        },
        'SBM_disassortative': {
            'graph_type': 'SBM',
            'graph_category': 'disassortative',
            'weight_bounds': (1., 5.),
          #  'description': 'SBM Disassortative (low within-block, high between-block)'
        },
        #'SBM_mixed': {
       #     'graph_type': 'SBM',
        #    'graph_category': 'mixed',
       #     'description': 'SBM Mixed (moderate within/between-block connections)'
        }

    return structures

def generate_structure(graph_type: str, n_nodes: int = 200, seed: Optional[int] = None, **graph_args) -> tuple[nx.DiGraph, Dict]:
    """
    Generate a directed graph structure of the specified type.

    Parameters:
    -----------
    graph_type : str
        Type of graph to generate. Options: 'ER', 'SF', 'SBM'
    n_nodes : int, default=200
        Number of nodes in the graph
    seed : int, optional
        Random seed for reproducibility
    **graph_args : dict
        Additional arguments specific to each graph type

    Returns:
    --------
    tuple[nx.DiGraph, Dict]
        Generated directed graph and dictionary of parameters used

    Graph-specific arguments:
    ------------------------
    ER (Erdős–Rényi):
        - category: str, options: 'sparse', 'intermediate', 'dense'
        - p: float, edge probability (overrides category if provided)

    SF (Scale-Free):
        - category: str, options: 'receiver', 'broadcaster', 'balanced'
        - m: int, number of edges to attach from a new node (overrides category if provided)
        - orient_frac: float, probability of orienting each edge (0.0-1.0)
        - bias: float, probability that oriented edge goes u->v instead of v->u (0.0-1.0)

    SBM (Stochastic Block Model):
        - category: str, options: 'assortative', 'disassortative', 'mixed'
        - n_blocks: int, default=4
        - P: np.ndarray, probability matrix (overrides category if provided)
        - block_sizes: List[int], sizes of each block
    """
    if seed is None:
        seed = 42

    rng = np.random.default_rng(seed=seed)
    nx_seed = int(rng.integers(0, 2**32 - 1))

    graph_type = graph_type.upper()

    if graph_type == 'ER':
        graph, params = _generate_erdos_renyi(n_nodes, nx_seed, rng, **graph_args)
    elif graph_type == 'SF':
        graph, params = _generate_scale_free(n_nodes, nx_seed, rng, **graph_args)
    elif graph_type == 'SBM':
        graph, params = _generate_stochastic_block_model(n_nodes, nx_seed, rng, **graph_args)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Options are 'ER', 'SF', 'SBM'")

    # Add common parameters
    params.update({
        'graph_type': graph_type,
        'n_nodes': n_nodes,
        'seed': seed,
        'nx_seed': nx_seed
    })

    return graph, params


def _generate_erdos_renyi(n_nodes: int, nx_seed: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
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


def _generate_scale_free(n_nodes: int, nx_seed: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    """
    Generate a directed, (asymptotically) scale-free graph by:
      1) building an undirected Barabási–Albert graph, then
      2) orienting each edge with probability `orient_frac` using `bias`.
    """

    # Parameter ranges for different categories
    param_ranges = {
        "m":           {"receiver": (4, 10),     "broadcaster": (4, 10),     "balanced": (4, 10)},
        "orient_frac": {"receiver": 1., "broadcaster": 1., "balanced": 1.},
        "bias":        {"receiver": (0.01, 0.2), "broadcaster":(0.7, 1.) , "balanced": (0.3, 0.6)}
    }

    # Use provided parameters or sample from category
    if all(param in kwargs for param in ['m', 'orient_frac', 'bias']):
        m = kwargs['m']
        orient_frac = kwargs['orient_frac']
        bias = kwargs['bias']
    else:
        category = kwargs.get('category', 'balanced')
        if category not in param_ranges["m"]:
            raise ValueError(f"SF category must be one of: {list(param_ranges['m'].keys())}")
        m = int(rng.uniform(*param_ranges["m"][category]))
        orient_frac = param_ranges["orient_frac"][category]
        bias = rng.uniform(*param_ranges["bias"][category])

    # Validation
    if not (0.0 <= orient_frac <= 1.0):
        raise ValueError("orient_frac must be in [0, 1].")
    if not (0.0 <= bias <= 1.0):
        raise ValueError("bias must be in [0, 1].")
    if m < 1:
        raise ValueError("m must be >= 1.")
    if n_nodes <= m:
        raise ValueError("n_nodes must be > m (required by the BA model).")

    # Use the provided nx_seed to create a Python random generator
    py_rng = random.Random(nx_seed)

    # Create initial complete graph for BA model
    #start_graph = nx.erdos_renyi_graph(m, 0.5, seed=py_rng.randint(0, 2**32 - 1))
    #G = nx.barabasi_albert_graph(
    #    n_nodes, m,
    #    seed=py_rng.randint(0, 2**32 - 1),
    #    #initial_graph=start_graph
    #)
    G = nx.extended_barabasi_albert_graph(
        n_nodes, m, p=0.3, q=0.3)

    # Convert to directed graph with orientation
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))  # keep node attributes

    for u, v, data in G.edges(data=True):
        if py_rng.random() < orient_frac:
            # make it one-way
            if py_rng.random() < bias:
                H.add_edge(u, v, **data)
            else:
                H.add_edge(v, u, **data)
        else:
            # keep it symmetric (both directions)
            H.add_edge(u, v, **data)
            H.add_edge(v, u, **data)

    params = {
        'm': m,
        'orient_frac': orient_frac,
        'bias': bias,
        'category': kwargs.get('category', 'custom' if all(param in kwargs for param in ['m', 'orient_frac', 'bias']) else 'balanced')
    }

    return H, params


def _generate_stochastic_block_model(n_nodes: int, nx_seed: int, rng: np.random.Generator, **kwargs) -> tuple[nx.DiGraph, Dict]:
    """Generate stochastic block model graph."""

    # Parameter ranges for different categories
    param_ranges = {
        "assortative": {
            "p_in": (0.05, 0.2),
            "p_out": (0.001, 0.01)
        },
        "disassortative": {
            "p_in": (0.001, 0.01),
            "p_out": (0.05, 0.2)
        },
        "mixed": {
            "p_in": (0.02, 0.1),
            "p_out": (0.02, 0.1)
        }
    }

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

    # Use provided probability matrix or sample from category
    if 'P' in kwargs:
        P = kwargs['P']
    else:
        category = kwargs.get('category', 'mixed')
        if category not in param_ranges:
            raise ValueError(f"SBM category must be one of: {list(param_ranges.keys())}")
        P = _sample_sbm_params(category, n_blocks, param_ranges, rng)

    graph = nx.stochastic_block_model(block_sizes, P, directed=True, seed=nx_seed)

    params = {
        'n_blocks': n_blocks,
        'block_sizes': block_sizes,
        'P': P.tolist(),  # Convert numpy array to list for JSON serialization
        'category': kwargs.get('category', 'custom' if 'P' in kwargs else 'mixed')
    }

    return graph, params





def _sample_sbm_params(category: str, n_blocks: int, param_ranges: Dict, rng: np.random.Generator) -> np.ndarray:
    """Sample SBM probability matrix for a given category."""
    p_in = rng.uniform(*param_ranges[category]["p_in"])
    p_out = rng.uniform(*param_ranges[category]["p_out"])

    # Create probability matrix
    P = np.full((n_blocks, n_blocks), p_out)
    np.fill_diagonal(P, p_in)
    return P


def _simplify_graph(G: nx.Graph) -> nx.DiGraph:
    """Convert to simple DiGraph by removing self-loops."""
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# Convenience functions for backward compatibility and easy access to predefined types
def generate_er_graphs(n_nodes: int = 200, seed: Optional[int] = None) -> Dict[str, tuple[nx.DiGraph, Dict]]:
    """Generate all three types of ER graphs."""
    return {
        'sparse': generate_structure('ER', n_nodes, seed, category='sparse'),
        'intermediate': generate_structure('ER', n_nodes, seed, category='intermediate'),
        'dense': generate_structure('ER', n_nodes, seed, category='dense')
    }


def generate_sf_graphs(n_nodes: int = 200, seed: Optional[int] = None) -> Dict[str, tuple[nx.DiGraph, Dict]]:
    """Generate all three types of scale-free graphs."""
    return {
        'receiver': generate_structure('SF', n_nodes, seed, category='receiver'),
        'broadcaster': generate_structure('SF', n_nodes, seed, category='broadcaster'),
        'balanced': generate_structure('SF', n_nodes, seed, category='balanced')
    }


def generate_sbm_graphs(n_nodes: int = 200, n_blocks: int = 4, seed: Optional[int] = None) -> Dict[str, tuple[nx.DiGraph, Dict]]:
    """Generate all three types of SBM graphs."""
    return {
        'assortative': generate_structure('SBM', n_nodes, seed, category='assortative', n_blocks=n_blocks),
        'disassortative': generate_structure('SBM', n_nodes, seed, category='disassortative', n_blocks=n_blocks),
        'mixed': generate_structure('SBM', n_nodes, seed, category='mixed', n_blocks=n_blocks)
    }
