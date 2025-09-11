import numpy as np
from typing import Dict, List, Optional
import networkx as nx
from networks import create_random_network
import jax.random as jr
from utils import plot_adjacency_matrix, plot_raster
from networks import create_random_network, run_single_network
import time
from attack import run_attack_batch_base, run_attack_batch_stdp

from jax import lax, random
import jax
import jax.numpy as jnp

from tqdm import tqdm

def test_attack_pipeline_seq_base(structure, save_path, attack_params, gn_fn):
    t0_global = time.perf_counter()

    n_nodes = attack_params['n_nodes']
    T_global = attack_params['T_global'] #ms
    batch_size = attack_params['batch_size']
    attack_fraction = attack_params['attack_fraction']
    attack_key = attack_params['attack_key']

    dt = 0.25
    steps = int(T_global / dt)

    G_list, neurons_list = gn_fn(structure, attack_params)
    I_ext = jnp.ones((steps, n_nodes)) * 10

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
    print(f"Total time pipeline base {t1_global- t0_global:.3f}s")



def test_attack_pipeline_seq_stdp(structure, save_path, attack_params, gn_fn):
    t0_global = time.perf_counter()

    n_nodes = attack_params['n_nodes']
    T_global = attack_params['T_global'] #ms
    batch_size = attack_params['batch_size']
    attack_fraction = attack_params['attack_fraction']
    attack_key = attack_params['attack_key']

    dt = 0.25
    steps = int(T_global / dt)

    G_list, neurons_list = gn_fn(structure, attack_params)
    I_ext = jnp.ones((steps, n_nodes)) * 10

    run_attack_batch_stdp(
        G_list,
        neurons_list,
        I_ext,
        batch_size,
        attack_fraction,
        attack_key,
        save_path=save_path
    )
    t1_global = time.perf_counter()
    print(f"Total time pipeline stdp:  {t1_global- t0_global:.3f}s")

def test_one_graph(n_nodes, graph, params, structure):
    # Example usage
    #n_nodes = 500
    #nx_seed = 42
    #rng = np.random.default_rng()
    T_global = 5000 #ms
    #if graph == None:
       # graph, params = generate_erdos_renyi(n_nodes, nx_seed, rng, p=0.1)

    plot_adjacency_matrix(graph, weights_range=(0,1), cmap='Blues')


    weight_bounds = structure['weight_bounds']
    neurons, graph = create_random_network(graph, n_nodes, weight_bounds=weight_bounds)

    plot_adjacency_matrix(graph, weights_range=(-weight_bounds[1], weight_bounds[1]), cmap='PuOr_r')
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

def generate_graphs_and_neurons(structure, attack_params, gg_fn):
    """Generate a list of graphs and neurons based on the given structure and attack parameters.

    Args:
        structure: Dictionary containing graph generation parameters with 'kwargs' key
        attack_params: Dictionary containing attack parameters

    Returns:
        tuple: (G_list, neurons_list) containing generated graphs and neurons
    """
    import time
    start_time = time.time()

    n_nodes = attack_params['n_nodes']
    graphs_key = attack_params['graphs_key']
    number_of_graphs = attack_params['number_of_graphs']

    G_list = []
    neurons_list = []
    weight_bounds = structure['weight_bounds']

    for i in tqdm(range(number_of_graphs)):
        # Create a seed for this graph
        seed = graphs_key + i
        rng = np.random.default_rng(seed)

        # Graph generation using scale-free model
        graph, params = gg_fn(n_nodes, rng, **structure['kwargs'])

        # Assuming create_random_network still exists and is needed
        # If not available, you'll need to implement or adapt this function
        key = jr.PRNGKey(seed)
        neurons, graph = create_random_network(graph, n_nodes, weight_bounds=weight_bounds, key=key)

        G_list.append(graph)
        neurons_list.append(neurons)

    end_time = time.time()
    print(f"Creating graphs and neurons total {end_time - start_time:.4f} seconds")

    return G_list, neurons_list
