import infomeasure as im
import antropy as ant
import numpy as np
import networkx as nx
import quantities as pq
from elephant.spike_train_synchrony import spike_contrast
from neo.core import SpikeTrain

def convert_to_spiketrains(S, dt=0.25, T=1000):
    """
    Convert a binary spike matrix into Elephant/Neo SpikeTrain objects.

    Parameters
    ----------
    S : np.ndarray
        Binary spike matrix of shape (time_steps, n_neurons).
        Each entry is 0 (no spike) or 1 (spike).
    dt : float
        Time step in ms (default 0.25).
    T : float
        Total duration in ms (default 1000).

    Returns
    -------
    list of neo.core.SpikeTrain
        One SpikeTrain per neuron.
    """
    n_neurons = S.shape[1]
    spiketrains = []

    for n in range(n_neurons):
        spike_indices = np.where(S[:, n] == 1)[0]
        spike_times = spike_indices * dt * pq.ms
        st = SpikeTrain(times=spike_times,
                        t_start=0*pq.ms,
                        t_stop=T*pq.ms)
        spiketrains.append(st)

    return spiketrains

def kuramoto_from_binary(S):
    """
    Compute a time-resolved Kuramoto synchrony index from a binary spike matrix.

    Parameters
    ----------
    S : array-like, shape (T, N)
        Binary spike matrix: T time bins by N neurons. S[t, n] is 1 if neuron n
        spikes in bin t, otherwise 0.

    Returns
    -------
    r_t : ndarray, shape (T,)
        Kuramoto order parameter over time; NaN where insufficient phase data.
    r_mean : float
        Time-average of r_t, ignoring NaNs (overall synchrony score).
    theta : ndarray, shape (T, N)
        Interpolated phases for each neuron over time in [0, 2π); NaN where
        phase is undefined (e.g., before first spike or after last spike).
    """

    # Unpack dimensions: T time bins, N neurons.
    T, N = S.shape

    # For each neuron, find the indices (time bins) where spikes occur.
    # spike_idx[n] is an array like [t0, t1, ...] for neuron n.
    spike_idx = [np.flatnonzero(S[:, n]) for n in range(N)]

    # Allocate the phase matrix theta (T x N) and fill with NaN to indicate
    # "phase not defined" (e.g., outside inter-spike intervals).
    theta = np.full((T, N), np.nan, float)

    # Build phases neuron by neuron.
    for n in range(N):
        idx = spike_idx[n]

        # We need at least two spikes to define a phase that increases
        # between spikes; otherwise skip this neuron.
        if len(idx) < 2:
            continue

        # For each consecutive spike pair (t0 -> t1), linearly interpolate
        # phase from 0 to 2π across the bins [t0, t1).
        for k in range(len(idx) - 1):
            t0, t1 = idx[k], idx[k+1]
            span = t1 - t0  # number of bins in this inter-spike interval

            # Guard against pathological cases (shouldn't happen with strictly
            # increasing spike indices, but we’re cautious).
            if span <= 0:
                continue

            # np.arange(span) produces 0, 1, ..., span-1.
            # Divide by span to get ramp in [0, 1), then scale by 2π.
            theta[t0:t1, n] = 2*np.pi * (np.arange(span) / span)

        # Define the phase exactly at the last spike time as 0 (equivalently 2π).
        # After the last spike, phase remains NaN (undefined) by design.
        theta[idx[-1], n] = 0.0

    # Allocate the time-resolved Kuramoto order parameter r(t).
    # It will be NaN at times where no neuron has a defined phase.
    r_t = np.full(T, np.nan)

    # For each time bin, compute the circular mean length over neurons
    # that have a valid phase at that time.
    for t in range(T):
        valid = ~np.isnan(theta[t])  # neurons with defined phase at time t
        if valid.any():
            # Convert phases to unit phasors e^{iθ}, average across neurons,
            # and take the magnitude |·| to get r(t) ∈ [0, 1].
            r_t[t] = np.abs(np.mean(np.exp(1j * theta[t, valid])))

    # Global (time-averaged) synchrony, ignoring times where r(t) is NaN.
    r_mean = np.nanmean(r_t)

    # Return the time series r(t), its mean, and the full phase matrix.
    return r_t, r_mean, theta

def synchrony_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33, T=2000):

    test_fraction = int(len(S_hist) * fraction)
    T_conversion = int(T * fraction)
    A = S_hist[-test_fraction:, surviving_nodes]
    B = S_hist_R[:test_fraction]
    C = S_hist_R[-test_fraction:]

    _, r_mean_a, _ = kuramoto_from_binary(A)
    _, r_mean_b, _ = kuramoto_from_binary(B)
    _, r_mean_c, _ = kuramoto_from_binary(C)


    # el_syn_a = spike_contrast(
    #     convert_to_spiketrains(A, T=T_conversion),
    #     return_trace=False, min_bin=0.25*pq.ms)
    # el_syn_b = spike_contrast(
    #     convert_to_spiketrains(B, T=T_conversion),
    #     return_trace=False, min_bin=0.25*pq.ms)
    # el_syn_c = spike_contrast(
    #     convert_to_spiketrains(C, T=T_conversion),
    #     return_trace=False, min_bin=0.25*pq.ms)


    return {
        'r_mean_a': r_mean_a,
        'r_mean_b': r_mean_b,
        'r_mean_c': r_mean_c,
        # 'el_syn_a': el_syn_a,
        # 'el_syn_b': el_syn_b,
        # 'el_syn_c': el_syn_c,
    }

# Resilience assesment
def entropic_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33, im_args=None):
    """
    Calculate entropy-based measures for system resilience assessment.

    For better readability, time series are labeled as:
    A = pre-attack state (st_pre_attack)
    B = post-attack initial state (st_post_attack_init)
    C = post-attack final state (st_post_attack_final)

    Returns a dictionary with:
        - Entropy of A, B, C (joint entropy of network microstates)
        - Entropy of A.T, B.T, C.T (joint entropy across neuron spike trains)
        - Mutual information between A, B, C (microstate perspective)
        - Mutual information between A.T, B.T, C.T (neuron spike train perspective)
        - Maximum possible entropy for each (for normalization)
    """

    if im_args is None:
        im_args = {"approach": "miller_madow", "base": 2}

    test_fraction = int(len(S_hist) * fraction)

    A = S_hist[-test_fraction:, surviving_nodes]
    B = S_hist_R[:test_fraction]
    C = S_hist_R[-test_fraction:]

    # Entropy of network microstates (rows = time, columns = neurons)
    h_A = im.entropy(A, **im_args)
    h_B = im.entropy(B, **im_args)
    h_C = im.entropy(C, **im_args)

    # Entropy across neuron spike trains (rows = neurons, columns = time)
    h_A_T = im.entropy(A.T, **im_args)
    h_B_T = im.entropy(B.T, **im_args)
    h_C_T = im.entropy(C.T, **im_args)

    # Mutual information between microstate sequences
    mi_AB = im.mutual_information(A, B, **im_args)
    mi_AC = im.mutual_information(A, C, **im_args)
    mi_AA = im.mutual_information(A, A, **im_args)

    # Mutual information between neuron spike trains
    mi_AB_T = im.mutual_information(A.T, B.T, **im_args)
    mi_AC_T = im.mutual_information(A.T, C.T, **im_args)
    mi_AA_T = im.mutual_information(A.T, A.T, **im_args)

    return {
        'h_A': h_A,
        'h_B': h_B,
        'h_C': h_C,
        'h_A_T': h_A_T,
        'h_B_T': h_B_T,
        'h_C_T': h_C_T,
        'mi_AB': mi_AB,
        'mi_AC': mi_AC,
        'mi_AA': mi_AA,
        'mi_AB_T': mi_AB_T,
        'mi_AC_T': mi_AC_T,
        'mi_AA_T': mi_AA_T,
    }

def _entropic_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33, im_args=None):
    """
    Calculate entropy-based measures for system resilience assessment.

    For better readability, time series are labeled as:
    A = pre-attack state (st_pre_attack)
    B = post-attack initial state (st_post_attack_init)
    C = post-attack final state (st_post_attack_final)
    """

    if im_args is None:
        im_args = {"approach": "miller_madow", "base": 2}

    test_fraction = int(len(S_hist) * fraction)

    A = S_hist[-test_fraction:, surviving_nodes]
    B = S_hist_R[:test_fraction]
    C = S_hist_R[-test_fraction:]

    h_A = im.entropy(A, **im_args)
    h_B = im.entropy(B, **im_args)
    h_C = im.entropy(C, **im_args)

    mi_AB = im.mutual_information(A,B,**im_args)
    mi_AC = im.mutual_information(A,C,**im_args)
    mi_AA = im.mutual_information(A,A,**im_args)

    return {
        'h_A': h_A,
        'h_B': h_B,
        'h_C': h_C,
        'mi_AB': mi_AB,
        'mi_AC': mi_AC,
        'mi_AA': mi_AA
    }


def lz_complexity_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33):
    test_fraction = int(len(S_hist)*fraction)

    st_pre_attack = S_hist[-test_fraction:, surviving_nodes]
    st_post_attack_init = S_hist_R[:test_fraction]
    st_post_attack_final = S_hist_R[-test_fraction:]

    lz_pre_list = []
    lz_posti_list = []
    lz_postf_list = []
    for j in range(len(surviving_nodes)):
        lz_pre = ant.lziv_complexity(st_pre_attack[:, j], normalize=True)
        lz_posti = ant.lziv_complexity(st_post_attack_init[:, j], normalize=True)
        lz_postf = ant.lziv_complexity(st_post_attack_final[:, j], normalize=True)

        lz_pre_list.append(lz_pre)
        lz_posti_list.append(lz_posti)
        lz_postf_list.append(lz_postf)

    avg_lz_pre=  np.mean(lz_pre_list)
    avg_lz_posti = np.mean(lz_posti_list)
    avg_lz_postf = np.mean(lz_postf_list)
    # Instead of the print statement, create a dictionary:
    results_dict = {
        "avg_lz_pre": avg_lz_pre,
        "avg_lz_post_init": avg_lz_posti,
        "avg_lz_post_final": avg_lz_postf
    }
    return results_dict

def sample_entropy_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33, order=2):
    test_fraction = int(len(S_hist)*fraction)

    st_pre_attack = S_hist[-test_fraction:, surviving_nodes]
    st_post_attack_init = S_hist_R[:test_fraction]
    st_post_attack_final = S_hist_R[-test_fraction:]

    se_pre_list = []
    se_posti_list = []
    se_postf_list = []
    for j in range(len(surviving_nodes)):
        se_pre = ant.sample_entropy(st_pre_attack[:, j], order=order)
        se_posti = ant.sample_entropy(st_post_attack_init[:, j], order=order)
        se_postf = ant.sample_entropy(st_post_attack_final[:, j], order=order)

        se_pre_list.append(se_pre)
        se_posti_list.append(se_posti)
        se_postf_list.append(se_postf)

    avg_se_pre = np.mean(se_pre_list)
    avg_se_posti = np.mean(se_posti_list)
    avg_se_postf = np.mean(se_postf_list)
    # Instead of the print statement, create a dictionary:
    results_dict = {
        "avg_se_pre": avg_se_pre,
        "avg_se_post_init": avg_se_posti,
        "avg_se_post_final": avg_se_postf
    }
    return results_dict
# Graph measures

def global_metrics_directed(G, weight='weight'):
    """
    Computes various global metrics for a directed graph G.
    """

    def global_efficiency_directed(G):
        n = len(G)
        if n < 2:
            return 0.0
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        total = 0
        for u in G:
            for v in G:
                if u != v:
                    try:
                        d = path_lengths[u][v]
                        total += 1 / d
                    except KeyError:
                        continue  # No path from u to v
        return total / (n * (n - 1))

    # 1. Global efficiency
    global_eff = global_efficiency_directed(G)

    # global clustering coefficient
    C = nx.average_clustering(G.to_undirected())
    # average_shortest_path_length
    if nx.is_strongly_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        # Use largest strongly connected component
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        L = nx.average_shortest_path_length(G.subgraph(largest_scc))

    # Small-worldness
    n = G.number_of_nodes()
    p = nx.density(G)  # edge probability
    G_rand =  nx.erdos_renyi_graph(n, p, directed=True)

    C_rand = nx.average_clustering(G_rand.to_undirected())
    if nx.is_strongly_connected(G_rand):
        L_rand = nx.average_shortest_path_length(G_rand)
    else:
        largest_scc_rand = max(nx.strongly_connected_components(G_rand), key=len)
        L_rand = nx.average_shortest_path_length(G_rand.subgraph(largest_scc_rand))

    # Small-worldness with error handling for division by zero
    if C_rand == 0 or L == 0 or L_rand == 0:
        S = None
    else:
        S = (C / C_rand) / (L / L_rand)

    # 2. Average betweenness centrality
    bet = nx.betweenness_centrality(G, normalized=True, weight=weight)
    avg_bet = float(np.mean(list(bet.values())))

    # 3. Average closeness centrality
    close = nx.closeness_centrality(G)
    avg_close = float(np.mean(list(close.values())))

    # 4. Average clustering coefficient (note: for directed, nx.clustering treats it as undirected)
    local_clust = nx.clustering(G.to_undirected())  # convert to undirected for local clustering
    C_avg = sum(local_clust.values()) / G.number_of_nodes()

    # 5. Transitivity (global clustering)
    trans = nx.transitivity(G.to_undirected())  # same note as above

    # density
    density = nx.density(G)

    return {
        "density": density,
        "global_efficiency": global_eff,
        "avg_betweenness": avg_bet,
        "avg_closeness": avg_close,
        "avg_clustering": C_avg,
        "transitivity": trans,
        "global_cc": C,
        "avg_spl": L,
        "swnss": S
    }
