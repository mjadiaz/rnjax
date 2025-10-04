import numpy as np
import infomeasure as im
import antropy as ant
import numpy as np
import networkx as nx
import sys
import os
sys.path.append(os.path.abspath('../src'))

import matplotlib.pyplot as plt 

from attack import run_attack_batch_stdp, run_attack_batch_stdp_control
import jax.numpy as jnp
from networks import run_single_network, create_random_network
from utils import plot_raster, plot_adjacency_matrix
import numpy as np
from measures import entropic_measures


im_args = {"approach": "miller_madow", "base": 2}

def keep_indices(remove_idx: np.ndarray, N: int) -> np.ndarray:
    """Return fixed-size (N-k,) indices to keep. O(N)."""
    k = remove_idx.shape[0]
    # mask: True for keep, False for remove
    mask = np.ones((N,), dtype=bool)
    mask[remove_idx] = False
    # Grab exactly N-k kept indices, preserving order (left→right).
    keep_idx = np.nonzero(mask)[0]  # (N-k,)
    return keep_idx

def _take_vec(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take(x, idx, axis=0)

def _take_square(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    M1 = np.take(M, idx, axis=0)
    M2 = np.take(M1, idx, axis=1)
    return M2

from tqdm import tqdm
import numpy as np

def compute_post_batch_entropies(step_data, N, test_fraction, keep_indices, 
                                 step_size=500, n_batches=10, 
                                 im_args=None, T_total=4000, window_size=None, control=True):
    """
    Compute post-batch entropies and times.

    Parameters
    ----------
    step_data : dict
        Contains 'base_S_hist', 'pruned_S_hist_batch', and 'removed_ids'.
    N : int
        Total number of nodes.
    test_fraction : int
        Window size for entropy calculation.
    keep_indices : callable
        Function to compute surviving node indices given removed_ids and N.
    step_size : int, optional
        Step size for sliding window. Default is 500.
    n_batches : int, optional
        Number of batches. Default is 10.
    im_args : dict, optional
        Arguments to pass to im.entropy. Default is Miller-Madow with base 2.
    T_total : int, optional
        Total experiment duration. Default is 4000.

    Returns
    -------
    post_batch_entropies : list of list of float
    post_batch_times : list of list of float
    """

    if im_args is None:
        im_args = {"approach": "miller_madow", "base": 2}

    post_batch_entropies = []
    post_batch_times = []
    if window_size is None:
        window_size = test_fraction

    for batch in range(n_batches):
        entropies = []
        times = []

        if control:
            S_hist_R = np.array(step_data['pruned_S_hist_batch'])
        else:
            S_hist_R = np.array(step_data['pruned_S_hist_batch'][batch])

        S_hist = np.array(step_data['base_S_hist'])
        removed_ids = np.array(step_data['removed_ids'][batch])

        surviving_nodes = keep_indices(removed_ids, N)

        A = S_hist[:, surviving_nodes]
        if control:
            B = S_hist_R[:, surviving_nodes]
        else:
            B = S_hist_R

        A = np.concatenate([A, B], axis=0)

        # Build time axis
        T = A.shape[0]
        dt = T_total / T
        real_time_axis = np.arange(T) * dt

        for start in tqdm(range(0, T - window_size + 1, step_size), desc=f"Batch {batch+1}/{n_batches}"):
            window = A[start:start+window_size]
            H = im.entropy(window, **im_args)
            entropies.append(H)

            centre_idx = start + window_size // 2
            times.append(real_time_axis[centre_idx])

        post_batch_entropies.append(entropies)
        post_batch_times.append(times)

    return post_batch_entropies, post_batch_times

import numpy as np

def bootstrap_entropy_statistics(post_batch_entropies, n_boot=10000, ci=95, seed=42):
    """
    Compute mean entropy and bootstrap confidence intervals across batches.

    Parameters
    ----------
    post_batch_entropies : list of list of float
        Entropy values from multiple batches, each possibly of different length.
    n_boot : int, optional
        Number of bootstrap samples. Default is 10000.
    ci : float, optional
        Confidence interval width in percent. Default is 95.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    means : np.ndarray
        Mean entropy values at each timepoint (with NaN padding if needed).
    cis_low : np.ndarray
        Lower confidence interval bound at each timepoint.
    cis_high : np.ndarray
        Upper confidence interval bound at each timepoint.
    """

    # Pad to common length
    max_len = max(len(x) for x in post_batch_entropies)
    entropy_mat = np.full((len(post_batch_entropies), max_len), np.nan)
    for i, arr in enumerate(post_batch_entropies):
        entropy_mat[i, :len(arr)] = arr

    # Compute mean
    means = np.nanmean(entropy_mat, axis=0)

    # Bootstrap CIs
    alpha = (100 - ci) / 2
    cis_low = []
    cis_high = []

    rng = np.random.default_rng(seed=seed)
    for t in range(entropy_mat.shape[1]):
        vals = entropy_mat[:, t]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            cis_low.append(np.nan)
            cis_high.append(np.nan)
            continue

        boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
        boots_mean = np.mean(boots, axis=1)
        cis_low.append(np.percentile(boots_mean, alpha))
        cis_high.append(np.percentile(boots_mean, 100 - alpha))

    return means, np.array(cis_low), np.array(cis_high)

neurons, G = create_random_network(N=500, p_connect=0.01, weight_bounds=(0.1,100))
dt = 0.25
T_global = 2000
steps = int(T_global / dt)
I_ext = jnp.ones((steps, 500)) * 10

results = run_attack_batch_stdp_control(
        [G], 
        [neurons], 
        I_ext, 10, attack_fraction=0.2, nkey=465, save_path=False )

step_data = results[0][0]
S_hist = np.array(step_data['base_S_hist'])
fraction = 0.33
N = 500

