import infomeasure as im
import antropy as ant
import numpy as np
import networkx as nx

# Resilience assesment

def entropic_measures(S_hist, S_hist_R, surviving_nodes, fraction=0.33, im_args=None):
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

    return {
        "global_efficiency": global_eff,
        "avg_betweenness": avg_bet,
        "avg_closeness": avg_close,
        "avg_clustering": C_avg,
        "transitivity": trans,
        "global_cc": C,
        "avg_spl": L,
        "swnss": S
    }
