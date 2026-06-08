import random
import networkx as nx
import numpy as np
import igraph as ig
import pandas as pd
import scipy.sparse as sp


def cal_eff_igraph(G, n0, weight):
    n = len(G.nodes())
    if not weight:
        g0 = ig.Graph.from_networkx(G)
        SP0 = g0.distances()
    else:
        nodes = list(G.nodes())
        node_mapping = {node: i for i, node in enumerate(nodes)}
        edges = [(node_mapping[e[0]], node_mapping[e[1]]) for e in G.edges()]
        edge_weights = [G[e[0]][e[1]]['weight'] for e in G.edges()]
        g0 = ig.Graph(n=n, edges=edges, directed=True)
        g0.es['weight'] = edge_weights
        SP0 = g0.distances(weights='weight')

    eff_list = [1.0 / SP0[i][j] for i in range(n) for j in range(n) if i != j and SP0[i][j] != np.inf]
    av_eff = sum(eff_list) / (n0 * (n0 - 1)) if eff_list else 0
    return av_eff


def analyze_network(G, df, n0):
    sorted_nodes = df.sort_values(by='importance', ascending=False)['node']
    sizes = [1]
    efficiencies = [cal_eff_igraph(G, n0, True)]
    for node in sorted_nodes:
        G.remove_node(node)
        if len(G) == 0:
            size = 0
            efficiency = 0
            break
        else:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            size = len(largest_cc) / n0
            efficiency = cal_eff_igraph(G, n0, True)
        sizes.append(size)
        efficiencies.append(efficiency)
    return sizes, efficiencies


def intentional_attack_initial_network(G, rn, importance_sequence):
    order_nodes = list(importance_sequence.keys())
    G.remove_nodes_from(order_nodes[:rn])
    return G


def intentional_attack_current_network(G, step, weight, method):
    if weight:
        if method == 'DCA':
            DC = dict(G.degree(weight='weight'))
            importance_sequence = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'BCA':
            BC = nx.betweenness_centrality(G, weight='weight')
            importance_sequence = dict(sorted(BC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'CCA':
            CC = nx.closeness_centrality(G, distance='weight')
            importance_sequence = dict(sorted(CC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'ECA':

            if G.number_of_edges() == 0:
                EC = {n: 0.0 for n in G.nodes()}
            else:
                try:

                    EC = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                except Exception:

                    EC = {n: 0.0 for n in G.nodes()}
            importance_sequence = dict(sorted(EC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'CHA':
            PR = nx.pagerank(G, weight='weight')
            importance_sequence = dict(sorted(PR.items(), key=lambda x: x[1], reverse=True))
    else:
        if method == 'DCA':
            DC = dict(G.degree())
            importance_sequence = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'BCA':
            BC = nx.betweenness_centrality(G)
            importance_sequence = dict(sorted(BC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'CCA':
            CC = nx.closeness_centrality(G)
            importance_sequence = dict(sorted(CC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'ECA':
            if G.number_of_edges() == 0:
                EC = {n: 0.0 for n in G.nodes()}
            else:
                try:
                    EC = nx.eigenvector_centrality(G, max_iter=1000)
                except Exception:
                    EC = {n: 0.0 for n in G.nodes()}
            importance_sequence = dict(sorted(EC.items(), key=lambda x: x[1], reverse=True))
        elif method == 'CHA':
            PR = nx.pagerank(G)
            importance_sequence = dict(sorted(PR.items(), key=lambda x: x[1], reverse=True))

    order_nodes = list(importance_sequence.keys())
    G.remove_nodes_from(order_nodes[:step])
    return G


def get_Sq_network(G, n, step, nums, Sq0, Eq0, weight, mode, method, func):
    S_q = np.zeros(nums)
    E_q = np.zeros(nums)
    c = 0
    S_q[0] = Sq0
    E_q[0] = Eq0

    if mode == 'initial_network':
        if weight:
            if method == 'DCA':
                DC0 = dict(G.degree(weight='weight'))
                importance_sequence = dict(sorted(DC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'BCA':
                BC0 = nx.betweenness_centrality(G, weight='weight')
                importance_sequence = dict(sorted(BC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'CCA':
                CC0 = nx.closeness_centrality(G, distance='weight')
                importance_sequence = dict(sorted(CC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'ECA':

                try:
                    EC0 = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                except Exception:
                    EC0 = {node: 0.0 for node in G.nodes()}
                importance_sequence = dict(sorted(EC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'CHA':
                PR0 = nx.pagerank(G, weight='weight')
                importance_sequence = dict(sorted(PR0.items(), key=lambda x: x[1], reverse=True))
        else:
            if method == 'DCA':
                DC0 = dict(G.degree())
                importance_sequence = dict(sorted(DC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'BCA':
                BC0 = nx.betweenness_centrality(G)
                importance_sequence = dict(sorted(BC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'CCA':
                CC0 = nx.closeness_centrality(G)
                importance_sequence = dict(sorted(CC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'ECA':
                try:
                    EC0 = nx.eigenvector_centrality(G, max_iter=1000)
                except Exception:
                    EC0 = {node: 0.0 for node in G.nodes()}
                importance_sequence = dict(sorted(EC0.items(), key=lambda x: x[1], reverse=True))
            elif method == 'CHA':
                PR0 = nx.pagerank(G)
                importance_sequence = dict(sorted(PR0.items(), key=lambda x: x[1], reverse=True))

        if method == 'random':
            labels = list(G.nodes())
            random.shuffle(labels)
            importance_sequence = dict(zip(labels, labels))

    while True:
        c = c + 1
        if c == nums:
            break

        if mode == 'initial_network':
            G0 = G.copy()
            IA_G = func(G0, step * c, importance_sequence)
        elif mode == 'current_network':
            IA_G = func(G, step, weight, method)
            G = IA_G.copy()

        if len(IA_G.nodes()) == 0:
            break

        Gcc = sorted(nx.weakly_connected_components(IA_G), key=len, reverse=True)
        if len(Gcc) == 0:
            break
        LCC = IA_G.subgraph(Gcc[0])
        n_LCC = len(LCC.nodes)
        S_q[c] = n_LCC / n
        E_q[c] = cal_eff_igraph(IA_G, n, weight)

    return S_q, E_q