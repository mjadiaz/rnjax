# Note: Imports may appear unresolved in some environments but are correct for the project
import jax
import jax.numpy as jnp
from jax import lax, random
from typing import Tuple, List, Optional
import numpy as np  # for time array in visualization
import time
from datetime import datetime
import gc
import json
import os

# Import necessary components from the simple_functional_stdp module
from networks import (
    NetworkParams, NetworkState, STDPParams, STDPState,
    run_base_network, run_stdp_network,
    run_base_network_syn, run_stdp_network_syn,
    create_network_params, create_initial_state,
    create_random_network, create_stdp_params, create_initial_stdp_state,
    IzhikevichNeuron
)
from pathlib import Path

from jax import tree_util as jtu

def move_to_host_if_needed(pytree):
    # Only move if at least one array is not on CPU
    def get_device_platform(x):
        if hasattr(x, 'device') and hasattr(x.device, 'platform'):
            return x.device.platform
        else:
            return 'cpu'  # Non-JAX arrays are considered to be on CPU

    devices = jax.tree.leaves(jax.tree.map(get_device_platform, pytree))
    if any(d != 'cpu' for d in devices):
        return jax.device_get(pytree)
    return pytree


# ---------- timing helper ----------
def barrier(tree):
    # Forces device work to finish so your timing is accurate.
    jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
    return tree

def keep_indices(remove_idx: jnp.ndarray, N: int) -> jnp.ndarray:
    """Return fixed-size (N-k,) indices to keep. O(N). JIT/vmap-friendly."""
    k = remove_idx.shape[0]  # static under jit
    # mask: True for keep, False for remove
    mask = jnp.ones((N,), dtype=bool).at[remove_idx].set(False)
    # Grab exactly N-k kept indices, preserving order (left→right).
    keep_idx = jnp.nonzero(mask, size=N - k, fill_value=0)[0]  # (N-k,)
    return keep_idx

def _take_vec(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(x, idx, axis=0)

def _take_square(M: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    M1 = jnp.take(M, idx, axis=0)
    M2 = jnp.take(M1, idx, axis=1)
    return M2

# -------- core (no STDP) --------

def prune_once_no_stdp(state, params, remove_idx: jnp.ndarray):
    N = state.v.shape[0]  # use shape -> static for XLA
    keep_idx = keep_indices(remove_idx, N)

    new_state = type(state)(
        v          = _take_vec(state.v, keep_idx),
        u          = _take_vec(state.u, keep_idx),
        last_spike = _take_vec(state.last_spike, keep_idx),
        W          = _take_square(state.W, keep_idx),
        time_step  = state.time_step,
    )
    new_params = type(params)(
        a  = _take_vec(params.a, keep_idx),
        b  = _take_vec(params.b, keep_idx),
        c  = _take_vec(params.c, keep_idx),
        d  = _take_vec(params.d, keep_idx),
        dt = params.dt,
        N  = N - remove_idx.shape[0],
    )
    return (new_state, new_params, None, None)

@jax.jit
def prune_batch_no_stdp(state, params, remove_idx_BK: jnp.ndarray):
    return jax.vmap(prune_once_no_stdp, in_axes=(None, None, 0))(state, params, remove_idx_BK)

# -------- core (with STDP) --------

def prune_once_with_stdp(state, params, remove_idx: jnp.ndarray, stdp_state, stdp_params):
    N = state.v.shape[0]
    keep_idx = keep_indices(remove_idx, N)

    new_state = type(state)(
        v          = _take_vec(state.v, keep_idx),
        u          = _take_vec(state.u, keep_idx),
        last_spike = _take_vec(state.last_spike, keep_idx),
        W          = _take_square(state.W, keep_idx),
        time_step  = state.time_step,
    )
    new_params = type(params)(
        a  = _take_vec(params.a, keep_idx),
        b  = _take_vec(params.b, keep_idx),
        c  = _take_vec(params.c, keep_idx),
        d  = _take_vec(params.d, keep_idx),
        dt = params.dt,
        N  = N - remove_idx.shape[0],
    )
    new_stdp_state = type(stdp_state)(
        pre_trace  = _take_vec(stdp_state.pre_trace,  keep_idx),
        post_trace = _take_vec(stdp_state.post_trace, keep_idx),
        delta_w    = _take_square(stdp_state.delta_w, keep_idx),
    )
    new_stdp_params = type(stdp_params)(
        A_plus          = stdp_params.A_plus,
        A_minus         = stdp_params.A_minus,
        tau             = stdp_params.tau,
        update_interval = stdp_params.update_interval,
        plastic_mask    = _take_square(stdp_params.plastic_mask, keep_idx),
        trace_decay     = stdp_params.trace_decay,
    )
    return (new_state, new_params, new_stdp_state, new_stdp_params)

@jax.jit
def prune_batch_with_stdp(state, params, remove_idx_BK: jnp.ndarray, stdp_state, stdp_params):
    return jax.vmap(prune_once_with_stdp, in_axes=(None, None, 0, None, None))(
        state, params, remove_idx_BK, stdp_state, stdp_params
    )

# -------- tiny dispatcher (not jitted) --------

def prune_batch(state, params, remove_idx_BK, stdp_state=None, stdp_params=None):
    if (stdp_state is None) or (stdp_params is None):
        return prune_batch_no_stdp(state, params, remove_idx_BK)
    else:
        return prune_batch_with_stdp(state, params, remove_idx_BK, stdp_state, stdp_params)

def prune_once(state, params, remove_idx_BK, stdp_state=None, stdp_params=None):
    if (stdp_state is None) or (stdp_params is None):
        return prune_once_no_stdp(state, params, remove_idx_BK)
    else:
        return prune_once_with_stdp(state, params, remove_idx_BK, stdp_state, stdp_params)


def run_attacked_network(params: NetworkParams, state: NetworkState, I_ext: jnp.ndarray, remove_idx: jnp.ndarray,
                         stdp_state = None, stdp_params = None):
    """Run a network after pruning some nodes."""
    # Prune the network
    result = prune_once(state, params, remove_idx, stdp_state, stdp_params)
    pruned_state, pruned_params, pruned_stdp_state, pruned_stdp_params = result

    # Adjust the input current for the pruned network
    keep_idx = keep_indices(remove_idx, params.N)
    pruned_I_ext = I_ext[:, keep_idx]

    # Run the pruned network
    if stdp_state is not None and stdp_params is not None and pruned_stdp_state is not None and pruned_stdp_params is not None:
        print(f"Running attacked STDP network with {pruned_params.N} neurons")
        return run_stdp_network(
            pruned_params, pruned_stdp_params,
            pruned_state, pruned_stdp_state,
            pruned_I_ext
        )
    else:
        print(f"Running attacked base network with {pruned_params.N} neurons")
        return run_base_network(pruned_params, pruned_state, pruned_I_ext)

def run_attack_batch_base(G_list, neurons_list, I_ext, batch_size, attack_fraction=0.1, nkey=42, save_path="save" ):

    run_base_network_jit = jax.jit(run_base_network_syn)
    run_base_network_batch = jax.jit(jax.vmap(run_base_network_syn))

    base_dir = Path(save_path).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    for i, (G, neurons) in enumerate(zip(G_list, neurons_list)):
        # Create parameters and state
        #t0 = time.perf_counter()
        key = random.PRNGKey(nkey + i)
        params = create_network_params(neurons)
        initial_state = create_initial_state(neurons, G, add_noise=True)
        N = params.N
        # E == 1, I == 0
        neuron_types = jnp.array([int(n.neuron_type == "E") for n in neurons])

        # t1 = time.perf_counter()
        # print(f"Creating networks params {t1 - t0:.4f} seconds")

        # Base run
        # t0 = time.perf_counter()
        base_out = run_base_network_jit(
            params, initial_state, I_ext
        )
        #barrier(base_out)
        # t1 = time.perf_counter()
        base_final_state, base_V_hist, base_S_hist, base_syn_hist = base_out
        # print(f"Base execution completed in {t1 - t0:.4f} seconds")

        mean_abs_syn = np.mean(np.abs(base_syn_hist))      # average across time & neurons
        base_driver_fraction = mean_abs_syn / (mean_abs_syn + np.unique(I_ext))
        # Create batch of removal indices
        # Different sets of neurons to remove for each example in the batch
        # t0 = time.perf_counter()
        batch_prune_key = random.fold_in(key, 2000)
        removed_neurons = int(np.floor(attack_fraction * N))
        remove_idx_batch = jax.vmap(
            lambda k: random.choice(k, jnp.arange(N), shape=(removed_neurons,), replace=False)
        )(random.split(batch_prune_key, batch_size))
        #barrier(remove_idx_batch)
        # t1 = time.perf_counter()
        # print(f"Batch Pruning, CHOICE completed in {t1 - t0:.4f} seconds")


        # t0 = time.perf_counter()
        # Prune the batch
        pruned_states_batch, pruned_params_batch, _, _ = prune_batch_no_stdp(
            base_final_state, params, remove_idx_batch
        )
        #barrier((pruned_states_batch, pruned_params_batch, pruned_stdp_states_batch, pruned_stdp_params_batch))
        # t1 = time.perf_counter()
        # print(f"Batch Pruning completed in {t1 - t0:.4f} seconds")

        # t0 = time.perf_counter()

        # Adjust current
        # t0 = time.perf_counter()
        Nk = N - removed_neurons
        pruned_I_ext = I_ext[:, :Nk]  # (T, N-k)
        pruned_I_ext_batch = jnp.broadcast_to(pruned_I_ext[None, :, :], (batch_size, pruned_I_ext.shape[0], pruned_I_ext.shape[1]))

        #barrier(pruned_I_ext_batch)
        # t1 = time.perf_counter()
        # print(f"Adjusting batch CURRENT completed in {t1 - t0:.4f} seconds")


        # Run batched

        # t0 = time.perf_counter()
        pruned_out = run_base_network_batch(
            pruned_params_batch, pruned_states_batch, pruned_I_ext_batch
        )
        #barrier(pruned_out)
        # t1 = time.perf_counter()
        (pruned_final_states_batch, pruned_V_hist_batch, pruned_S_hist_batch, pruned_syn_hist) = pruned_out
        # print(f"Batched execution completed in {t1 - t0:.4f} seconds")
        print(f"Batched execution completed base, i:{i}")

        mean_abs_syn = np.mean(np.abs(pruned_syn_hist), axis=(1, 2))
        # For each batch, calculate the driver fraction and then average
        # First, get unique current values (should be the same for all batches)
        unique_current = np.unique(I_ext)
        # Calculate driver fraction by properly handling batch dimension
        b_driver_fraction = mean_abs_syn / (mean_abs_syn + unique_current)
        # Save
        # t0 = time.perf_counter()
        metadata = {
            'experiment_type': "base_node_removal",
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'N': int(N),
                'batch_size': int(batch_size),
                'attack_fraction': float(attack_fraction),
            }
        }
        arrays_to_save = {
            "pruned_S_hist_batch": pruned_S_hist_batch,
            "base_S_hist": base_S_hist,
            "base_driver_fraction": base_driver_fraction,
            "batch_driver_fraction": b_driver_fraction,
            "W0": initial_state.W,
            "removed_ids": remove_idx_batch,
            "neuron_type": neuron_types
        }

        state_host = move_to_host_if_needed(arrays_to_save)

        # Create a step directory for this iteration
        step_dir = base_dir / str(i)
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save arrays using numpy
        for key, array in state_host.items():
            np.save(step_dir / f"{key}.npy", array)

        # t1 = time.perf_counter()
        # print(f"save step {i} took {t1- t0:.3f}s")

        # Now it's safe to delete everything
        del base_out, base_final_state, base_V_hist, base_S_hist, neuron_types
        del remove_idx_batch, pruned_states_batch, pruned_params_batch
        del pruned_I_ext_batch, pruned_out
        del pruned_final_states_batch, pruned_V_hist_batch, pruned_S_hist_batch
        del arrays_to_save

        # Light garbage collection
        gc.collect()
    return True

def run_attack_batch_stdp(G_list, neurons_list, I_ext, batch_size, attack_fraction=0.1, nkey=42, save_path="save" ):

    run_stdp_network_jit = jax.jit(run_stdp_network_syn)
    run_stdp_network_batch = jax.jit(jax.vmap(run_stdp_network_syn))

    base_dir = Path(save_path).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    for i, (G, neurons) in enumerate(zip(G_list, neurons_list)):
        # Create parameters and state
        #t0 = time.perf_counter()
        key = random.PRNGKey(nkey + i)
        params = create_network_params(neurons)
        initial_state = create_initial_state(neurons, G, add_noise=True)
        N = params.N
        stdp_params = create_stdp_params(params, initial_state.W)
        initial_stdp_state = create_initial_stdp_state(N)
        # E == 1, I == 0
        neuron_types = jnp.array([int(n.neuron_type == "E") for n in neurons])


        # t1 = time.perf_counter()
        # print(f"Creating networks params {t1 - t0:.4f} seconds")

        # Base run
        # t0 = time.perf_counter()
        base_out = run_stdp_network_jit(
            params, stdp_params, initial_state, initial_stdp_state, I_ext
        )
        #barrier(base_out)
        # t1 = time.perf_counter()
        base_final_state, base_final_stdp_state, base_V_hist, base_S_hist, base_syn_hist = base_out
        # print(f"Base execution completed in {t1 - t0:.4f} seconds")

        mean_abs_syn = np.mean(np.abs(base_syn_hist))      # average across time & neurons
        base_driver_fraction = mean_abs_syn / (mean_abs_syn + np.unique(I_ext))
        # Create batch of removal indices
        # Different sets of neurons to remove for each example in the batch
        # t0 = time.perf_counter()
        batch_prune_key = random.fold_in(key, 2000)
        removed_neurons = int(np.floor(attack_fraction * N))
        remove_idx_batch = jax.vmap(
            lambda k: random.choice(k, jnp.arange(N), shape=(removed_neurons,), replace=False)
        )(random.split(batch_prune_key, batch_size))
        #barrier(remove_idx_batch)
        # t1 = time.perf_counter()
        # print(f"Batch Pruning, CHOICE completed in {t1 - t0:.4f} seconds")


        # t0 = time.perf_counter()
        # Prune the batch
        pruned_states_batch, pruned_params_batch, pruned_stdp_states_batch, pruned_stdp_params_batch = prune_batch_with_stdp(
            base_final_state, params, remove_idx_batch, base_final_stdp_state, stdp_params
        )
        #barrier((pruned_states_batch, pruned_params_batch, pruned_stdp_states_batch, pruned_stdp_params_batch))
        # t1 = time.perf_counter()
        # print(f"Batch Pruning completed in {t1 - t0:.4f} seconds")

        # t0 = time.perf_counter()

        # Adjust current
        # t0 = time.perf_counter()
        Nk = N - removed_neurons
        pruned_I_ext = I_ext[:, :Nk]  # (T, N-k)
        pruned_I_ext_batch = jnp.broadcast_to(pruned_I_ext[None, :, :], (batch_size, pruned_I_ext.shape[0], pruned_I_ext.shape[1]))

        #barrier(pruned_I_ext_batch)
        # t1 = time.perf_counter()
        # print(f"Adjusting batch CURRENT completed in {t1 - t0:.4f} seconds")


        # Run batched

        # t0 = time.perf_counter()
        pruned_out = run_stdp_network_batch(
            pruned_params_batch, pruned_stdp_params_batch,
            pruned_states_batch, pruned_stdp_states_batch, pruned_I_ext_batch
        )
        #barrier(pruned_out)
        # t1 = time.perf_counter()
        (pruned_final_states_batch, pruned_final_stdp_states_batch,
        pruned_V_hist_batch, pruned_S_hist_batch, pruned_syn_hist) = pruned_out
        # print(f"Batched execution completed in {t1 - t0:.4f} seconds")
        print(f"Batched execution completed stdp, i:{i}")

        # Calculate mean across all batches, time steps, and neurons
        mean_abs_syn = np.mean(np.abs(pruned_syn_hist), axis=( 1, 2))
        # For each batch, calculate the driver fraction and then average
        # First, get unique current values (should be the same for all batches)
        unique_current = np.unique(I_ext)
        # Calculate driver fraction by properly handling batch dimension
        b_driver_fraction = mean_abs_syn / (mean_abs_syn + unique_current)
        # Save
        # t0 = time.perf_counter()
        metadata = {
            'experiment_type': "stdp_node_removal",
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'N': int(N),
                'batch_size': int(batch_size),
                'attack_fraction': float(attack_fraction),
            }
        }
        arrays_to_save = {
            "pruned_S_hist_batch": pruned_S_hist_batch,
            "base_S_hist": base_S_hist,
            "base_driver_fraction": base_driver_fraction,
            "batch_driver_fraction": b_driver_fraction,
            "W0": initial_state.W,
            "removed_ids": remove_idx_batch,
            "neuron_type": neuron_types
        }

        state_host = move_to_host_if_needed(arrays_to_save)

        # Create a step directory for this iteration
        step_dir = base_dir / str(i)
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save arrays using numpy
        for key, array in state_host.items():
            np.save(step_dir / f"{key}.npy", array)

        # t1 = time.perf_counter()
        # print(f"save step {i} took {t1- t0:.3f}s")

        # Now it's safe to delete everything
        del base_out, base_final_state, base_V_hist, base_S_hist, neuron_types
        del remove_idx_batch, pruned_states_batch, pruned_params_batch
        del pruned_I_ext_batch, pruned_out
        del pruned_final_states_batch, pruned_V_hist_batch, pruned_S_hist_batch
        del arrays_to_save
        del base_final_stdp_state, pruned_stdp_states_batch, pruned_stdp_params_batch

        # Light garbage collection
        gc.collect()
    return True

def test_attack_pipeline_seq():
    t0_global = time.perf_counter()

    start_time = time.time()
    N = 100
    number_of_graphs = 3
    G_list = []
    neurons_list = []
    for i in range(number_of_graphs):
        key = random.PRNGKey(256 + i)
        neurons, G = create_random_network(N=N, p_connect=0.1, key=key)
        G_list.append(G)
        neurons_list.append(neurons)
    time_steps = 2000

    I_ext = jnp.ones((time_steps, N))*10
    batch_size = 20
    attack_fraction = 0.1
    nkey = 42
    end_time = time.time()
    print(f"Creating graphs and neurons total {end_time - start_time:.4f} seconds")
    run_attack_batch_base(
        G_list,
        neurons_list,
        I_ext,
        batch_size,
        attack_fraction,
        nkey
        )
    t1_global = time.perf_counter()
    print(f"ALL TOOK {t1_global- t0_global:.3f}s")

def test_attack_pipeline_seq_stdp():
    t0_global = time.perf_counter()

    start_time = time.time()
    N = 100
    number_of_graphs = 3
    G_list = []
    neurons_list = []
    for i in range(number_of_graphs):
        key = random.PRNGKey(256 + i)
        neurons, G = create_random_network(N=N, p_connect=0.1, key=key)
        G_list.append(G)
        neurons_list.append(neurons)
    time_steps = 2000

    I_ext = jnp.ones((time_steps, N))*10
    batch_size = 100
    attack_fraction = 0.1
    nkey = 42
    end_time = time.time()
    print(f"Creating graphs and neurons total {end_time - start_time:.4f} seconds")
    run_attack_batch_stdp(
        G_list,
        neurons_list,
        I_ext,
        batch_size,
        attack_fraction,
        nkey
        )
    t1_global = time.perf_counter()
    print(f"ALL TOOK {t1_global- t0_global:.3f}s")
    # No need to wait for saves as we're not using Orbax's async saving


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pipe_base":
        test_attack_pipeline_seq()
    elif len(sys.argv) > 1 and sys.argv[1] == "pipe_stdp":
        test_attack_pipeline_seq_stdp()
