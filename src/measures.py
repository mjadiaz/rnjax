import infomeasure as im
import antropy as ant
import numpy as np
import networkx as nx

# Resilience assesment

def entropic_measures(S_hist, S_hist_R, fraction, surviving_nodes, im_args=None):

    if im_args is None:
        im_args = {"approach": "miller_madow", "base": 2}

    test_fraction = int(len(S_hist) * fraction)

    st_pre_attack = S_hist[-test_fraction:, surviving_nodes]
    st_post_attack_init = S_hist_R[:test_fraction]
    st_post_attack_final = S_hist_R[-test_fraction:]

    je_pre = im.entropy(st_pre_attack, **im_args)
    je_post_init = im.entropy(st_post_attack_init, **im_args)
    je_post_final = im.entropy(st_post_attack_final, **im_args)

    mi_post_init = im.mutual_information(
        st_pre_attack,
        st_post_attack_init,
        **im_args)

    mi_post_final = im.mutual_information(
        st_pre_attack,
        st_post_attack_final,
        **im_args)

    self_mi = im.mutual_information(
        st_pre_attack,
        st_pre_attack,
        **im_args)

    return {
        'je_pre': je_pre,
        'je_post_init': je_post_init,
        'je_post_final': je_post_final,
        'mi_post_init': mi_post_init,
        'mi_post_final': mi_post_final,
        'self_mi': self_mi
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
