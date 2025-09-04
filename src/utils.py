import numpy as np
import networkx as nx
import copy
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib as mpl

def plot_style_one():
    # Single-column Nature width ≈ 90 mm → ~3.54 in
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (4.8, 1.5),   # small, compact
        "font.size": 7,                  # 7–8 pt typical
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,              # editable text in Illustrator
        "ps.fonttype": 42
    })
def nx_random(N, p=0.2, seed=None):
    G = nx.erdos_renyi_graph(N, p=p, directed=True, seed=seed)
    return G

def nx_scale_free(
    N=100,
    alpha=0.3,
    beta=0.4,
    gamma=0.3,
    delta_in=0.5,
    delta_out=0.5,
    seed=None):
    G = nx.scale_free_graph(
        N,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        seed=seed

    )
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def create_graph(graph_fn, seed, graph_args):
    G = graph_fn(seed=int(seed), **graph_args)
    return G


def plot_adjacency_matrix(G, attack_nodes=None, weights_range=None, figsize=(4,4), cmap='PuOr'):
    """
    Visualize adjacency matrix with attack nodes highlighted.

    Args:
        G: NetworkX graph
        attack_nodes: List of node indices under attack
        weights_range: Tuple of (min_weight, max_weight) for colormap
        N: Original network size (before any additions)
        title: Optional plot title
    """
    if isinstance(G, np.ndarray):
        adj_matrix = G
    else:
        adj_matrix = nx.to_numpy_array(G)

    N = len(G)

    if weights_range:
        weights_low, weights_high = weights_range
    else:
        weights_low, weights_high = None,None

    fig = plt.figure(figsize=figsize)

    # Main matrix plot (transposed for conventional orientation)
    im = plt.imshow(adj_matrix.T,
                   cmap=cmap,
                   vmin=weights_low,
                   vmax=weights_high,
                   aspect='equal')

    # Add divider line between original and added nodes
    plt.axhline(N - 0.5, color='black', linestyle='--', linewidth=1)
    plt.axvline(N - 0.5, color='black', linestyle='--', linewidth=1)

    # Highlight attack nodes
    if attack_nodes:
        for idx in attack_nodes:
            plt.axhspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1, linewidth=0)
            plt.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1, linewidth=0)

    # Add colorbar with label
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Connection Weight')
    plt.xlim(right=N-0.5)
    # Labels and title
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    #plt.title()

    plt.tight_layout()
    return fig

def plot_network_distributions(G, time, R_t):
    """
    Create a 2x2 figure with:
    1. In-degree distribution
    2. Out-degree distribution
    3. Synapse-driven ratio vs. time (scatter)
    4. Histogram of synapse-driven ratio

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph to analyze.
    time : array-like
        Time points (ms).
    R_t : array-like
        Synapse-driven ratio values corresponding to time.
    """
    # Compute degrees
    in_degrees = [deg for _, deg in G.in_degree()]
    out_degrees = [deg for _, deg in G.out_degree()]

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 1. In-degree distribution
    axs[0, 0].hist(in_degrees, bins=range(0, max(in_degrees)+1),
                   density=False, color='coral', alpha=0.7)
    axs[0, 0].set_title("In-Degree Distribution")
    axs[0, 0].set_xlabel("In-Degree")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(True)

    # 2. Out-degree distribution
    axs[0, 1].hist(out_degrees, bins=range(0, max(out_degrees)+1),
                   density=False, color='seagreen', alpha=0.7)
    axs[0, 1].set_title("Out-Degree Distribution")
    axs[0, 1].set_xlabel("Out-Degree")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].grid(True)

    # 3. Synapse-driven ratio vs time (scatter)
    axs[1, 0].scatter(time, R_t, s=1)
    axs[1, 0].set_xlabel("Time (ms)")
    axs[1, 0].set_ylabel("Synapse-driven ratio")
    axs[1, 0].set_title("Synapse-driven ratio over time")

    # 4. Histogram of synapse-driven ratio
    axs[1, 1].hist(R_t, bins=30, color='gray', alpha=0.7)
    axs[1, 1].set_xlabel("Synapse-driven ratio")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_title("Distribution of Synapse-driven ratio")

    plt.tight_layout()
    plt.show()
    return fig

def plot_raster(S_hist, neurons, W, T_total, title="Spike raster", apply_style=True):
    """
    Standalone spike raster (scatter) using the same E/I ordering as your merged plot.

    Args:
        S_hist: (T x N) binary or bool spike matrix
        neurons (list): list of neuron objects with .neuron_type attribute ('E' or 'I')
        W (array-like): (N x N) weight/connectivity matrix
        T_total (float): total simulated time in ms
        title (str): title for the plot
        apply_style (bool): whether to apply plot_style_one()

    Returns:
        fig, ax_raster, dict(info) with {'col_order','split_index'}
    """
    if apply_style:
        plot_style_one()

    col_order, split_index, _ = get_neuron_order(neurons, W)
    N = len(neurons)

    # Determine time axis from number of timesteps
    T = S_hist.shape[0]
    dt = T_total / T
    time_axis = np.arange(T) * dt

    fig, ax_raster = plt.subplots()
    spike_times = []
    spike_neurons = []

    S_hist = np.asarray(S_hist)
    for row_idx, neuron_idx in enumerate(col_order, start=1):
        spike_mask = S_hist[:, neuron_idx].astype(bool)
        spikes = time_axis[spike_mask]
        spike_times.extend(spikes)
        spike_neurons.extend([row_idx] * len(spikes))

    ax_raster.scatter(
        spike_times, spike_neurons,
        s=0.1, c='k', marker='o', linewidths=0
    )
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_ylabel('Neuron ID')
    ax_raster.set_ylim(0.5, N + 0.5)
    ax_raster.axhline(split_index + 0.5, color='k', lw=0.4)
    ax_raster.set_title(title, pad=2)

    # Compact E/I labels aligned with the split
    x0 = ax_raster.get_xlim()[0]
    ax_raster.text(x0, split_index + 2, 'I', ha='left', va='bottom', fontsize=7, fontweight='bold')
    ax_raster.text(x0, split_index - 2, 'E', ha='left', va='top', fontsize=7, fontweight='bold')

    return fig, ax_raster, {"col_order": col_order, "split_index": split_index}

def plot_raster_simple(S_hist, T_total, title="Spike raster", apply_style=True):
    """
    Simple spike raster (scatter) without any neuron ordering.

    Args:
        S_hist: (T x N) binary or bool spike matrix
        T_total (float): total simulated time in ms
        title (str): title for the plot
        apply_style (bool): whether to apply plot_style_one()

    Returns:
        fig, ax_raster
    """
    if apply_style:
        plot_style_one()

    N = S_hist.shape[1]

    # Determine time axis from number of timesteps
    T = S_hist.shape[0]
    dt = T_total / T
    time_axis = np.arange(T) * dt

    fig, ax_raster = plt.subplots()
    spike_times = []
    spike_neurons = []

    S_hist = np.asarray(S_hist)
    for neuron_idx in range(N):
        spike_mask = S_hist[:, neuron_idx].astype(bool)
        spikes = time_axis[spike_mask]
        spike_times.extend(spikes)
        spike_neurons.extend([neuron_idx + 1] * len(spikes))

    ax_raster.scatter(
        spike_times, spike_neurons,
        s=0.1, c='k', marker='o', linewidths=0
    )
    ax_raster.set_xlabel('Time (ms)')
    ax_raster.set_ylabel('Neuron ID')
    ax_raster.set_ylim(0.5, N + 0.5)
    ax_raster.set_title(title, pad=2)

    return fig, ax_raster



def get_neuron_order(neurons, W):
    """
    Returns ordering and sorted connectivity matrix based on excitatory/inhibitory (E/I) split.

    Args:
        neurons (list): list of neuron objects, each with attribute `neuron_type` ('E' or 'I')
        W (array-like): (N x N) weight/connectivity matrix

    Returns:
        col_order (np.ndarray): indices with all excitatory neurons first, then inhibitory
        split_index (int): number of excitatory neurons (boundary between E and I)
        W_sorted (np.ndarray): adjacency matrix reordered on both axes by col_order
    """
    W = np.asarray(W).copy()

    # Build excitatory mask from neuron metadata
    exc_mask = np.array(
        [getattr(n, "neuron_type", None) == "E" for n in neurons],
        dtype=bool
    )

    col_order = np.r_[np.where(exc_mask)[0], np.where(~exc_mask)[0]]
    split_index = exc_mask.sum()

    # Reorder rows and columns consistently
    W_sorted = W[np.ix_(col_order, col_order)]

    return col_order, split_index, W_sorted
