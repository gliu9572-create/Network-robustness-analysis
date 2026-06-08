import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig
from utils import *
import scipy as sp
plt.rc('font', family='Times New Roman')


def find_zero_proportion(q, values, metric_name, method_name):

    zero_idx = np.atleast_1d(values == 0).nonzero()[0]
    if len(zero_idx) > 0:
        q_zero = q[zero_idx[0]]
        print(f"{method_name} - {metric_name} The proportion of nodes removed for the first time is 0: {q_zero:.4f}")
        return q_zero
    else:
        print(f"{method_name} - {metric_name} Did not reach 0 throughout the entire process 0")
        return None

if __name__ == '__main__':
    f_size = 16
    excel_path = 'importance.xlsx'
    df = pd.read_excel(excel_path)
    excel_path_edges = 'network_data1.xlsx'
    edges_df = pd.read_excel(excel_path_edges)

    print(edges_df.head())
    weight = True
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    n0, m0 = nx.number_of_nodes(G), nx.number_of_edges(G)
    print("The initial number of nodes and edges in the network are:", n0, m0)

    Eq0 = cal_eff_igraph(G, n0, weight)
    print("The global efficiency of network G is:", Eq0)

    n_end = n0
    step = 1
    if n_end % step == 0:
        nums = int(n_end / step)
    else:
        nums = int(n_end / step) + 1

    q = np.linspace(0, n_end, nums) / n0
    Sq0 = 1.0

    deliberate_sizes, deliberate_efficiencies = analyze_network(G.copy(), df, n0)
    print("Static Analysis - NE:", deliberate_efficiencies)
    print("Static analysis - RS:", deliberate_sizes)

    samples = 10
    S_q1 = np.zeros(nums)
    E_q1 = np.zeros(nums)

    for i in range(samples):
        S_q, E_q = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, 'initial_network', 'random', intentional_attack_initial_network)
        S_q1 += S_q
        E_q1 += E_q
    S_q1 /= samples
    E_q1 /= samples

    print("\n=== Static robustness analysis (initial_network) ===")
    mode = 'initial_network'
    S_q2_static, E_q2_static = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'DCA', intentional_attack_initial_network)
    S_q3_static, E_q3_static = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'BCA', intentional_attack_initial_network)
    S_q4_static, E_q4_static = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'CCA', intentional_attack_initial_network)
    S_q5_static, E_q5_static = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'ECA', intentional_attack_initial_network)
    S_q6_static, E_q6_static = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'CHA', intentional_attack_initial_network)

    methods_static = {
        'RA': (S_q1, E_q1),
        'DCA': (S_q2_static, E_q2_static),
        'BCA': (S_q3_static, E_q3_static),
        'CCA': (S_q4_static, E_q4_static),
        'ECA': (S_q5_static, E_q5_static),
        'CHA': (S_q6_static, E_q6_static),
        'CIA': (deliberate_sizes, deliberate_efficiencies)
    }
    for method, (S_q, E_q) in methods_static.items():
        find_zero_proportion(q, S_q, "RS", method)
        find_zero_proportion(q, E_q, "NE", method)

    print("\n=== Dynamic robustness analysis (current_network) ===")
    mode = 'current_network'
    S_q2_dynamic, E_q2_dynamic = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'DCA', intentional_attack_current_network)
    S_q3_dynamic, E_q3_dynamic = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'BCA', intentional_attack_current_network)
    S_q4_dynamic, E_q4_dynamic = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'CCA', intentional_attack_current_network)
    S_q5_dynamic, E_q5_dynamic = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'ECA', intentional_attack_current_network)
    S_q6_dynamic, E_q6_dynamic = get_Sq_network(G, n0, step, nums, Sq0, Eq0, weight, mode, 'CHA', intentional_attack_current_network)

    methods_dynamic = {
        'RA': (S_q1, E_q1),
        'DCA': (S_q2_dynamic, E_q2_dynamic),
        'BCA': (S_q3_dynamic, E_q3_dynamic),
        'CCA': (S_q4_dynamic, E_q4_dynamic),
        'ECA': (S_q5_dynamic, E_q5_dynamic),
        'CHA': (S_q6_dynamic, E_q6_dynamic),
        'CIA': (deliberate_sizes, deliberate_efficiencies)
    }
    for method, (S_q, E_q) in methods_dynamic.items():
        find_zero_proportion(q, S_q, "RS", method)
        find_zero_proportion(q, E_q, "NE", method)


    plt.figure(figsize=(10, 6))
    plt.plot(q, S_q1, color='orange', linestyle='-', markersize='5', marker='o', label='RA', markevery=2)
    plt.plot(q, S_q5_static, "bo-", label='DCA', markersize='5', markevery=2)
    plt.plot(q, S_q3_static, "go-", label='BCA', markersize='5', markevery=2)
    plt.plot(q, deliberate_sizes, "yo-", label='CCA', markersize='5', markevery=2)
    plt.plot(q, S_q4_static, "mo-", label='ECA', markersize='5', markevery=2)
    plt.plot(q, S_q6_static, "co-", label='CHA', markersize='5', markevery=2)
    plt.plot(q, S_q2_static, "ro-", label='CIA', markersize='5', markevery=2)
    plt.legend(loc=1, edgecolor="black", fontsize=10, fancybox=False)
    plt.xlabel("Remove node proportion", fontsize=12)
    plt.ylabel("Relative size of the largest connected component", fontsize=12)
    plt.title("Static Robustness - Largest Connected Component", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=10)
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    plt.plot(q, E_q1, color='orange', linestyle='-', marker='o', markersize='5', label='RA', markevery=2)
    plt.plot(q, E_q5_static, "bo-", label='DCA', markersize='5', markevery=2)
    plt.plot(q, E_q3_static, "go-", label='BCA', markersize='5', markevery=2)
    plt.plot(q, deliberate_efficiencies, "yo-", label='CCA', markersize='5', markevery=2)
    plt.plot(q, E_q4_static, "mo-", label='ECA', markersize='5', markevery=2)
    plt.plot(q, E_q6_static, "co-", label='CHA', markersize='5', markevery=2)
    plt.plot(q, E_q2_static, "ro-", label='CIA', markersize='5', markevery=2)
    plt.legend(loc=1, edgecolor="black", fontsize=10, fancybox=False)
    plt.xlabel("Remove node proportion", fontsize=12)
    plt.ylabel("Network efficiency", fontsize=12)
    plt.title("Static Robustness - Network Efficiency", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 0.16)
    plt.tick_params(labelsize=10)
    plt.grid(False)


    plt.figure(figsize=(10, 6))
    plt.plot(q, S_q1, color='orange', linestyle='-', markersize='5', marker='o', label='RA', markevery=2)
    plt.plot(q, S_q5_dynamic, "bo-", label='DCA', markersize='5', markevery=2)
    plt.plot(q, S_q3_dynamic, "go-", label='BCA', markersize='5', markevery=2)
    plt.plot(q, deliberate_sizes, "yo-", label='CCA', markersize='5', markevery=2)
    plt.plot(q, S_q4_dynamic, "mo-", label='ECA', markersize='5', markevery=2)
    plt.plot(q, S_q6_dynamic, "co-", label='CHA', markersize='5', markevery=2)
    plt.plot(q, S_q2_dynamic, "ro-", label='CIA', markersize='5', markevery=2)
    plt.legend(loc=1, edgecolor="black", fontsize=10, fancybox=False)
    plt.xlabel("Remove node proportion", fontsize=12)
    plt.ylabel("Relative size of the largest connected component", fontsize=12)
    plt.title("Dynamic Robustness - Largest Connected Component", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(labelsize=10)
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    plt.plot(q, E_q1, color='orange', linestyle='-', marker='o', markersize='5', label='RA', markevery=2)
    plt.plot(q, E_q5_dynamic, "bo-", label='DCA', markersize='5', markevery=2)
    plt.plot(q, E_q3_dynamic, "go-", label='BCA', markersize='5', markevery=2)
    plt.plot(q, deliberate_efficiencies, "yo-", label='CCA', markersize='5', markevery=2)
    plt.plot(q, E_q4_dynamic, "mo-", label='ECA', markersize='5', markevery=2)
    plt.plot(q, E_q6_dynamic, "co-", label='CHA', markersize='5', markevery=2)
    plt.plot(q, E_q2_dynamic, "ro-", label='CIA', markersize='5', markevery=2)
    plt.legend(loc=1, edgecolor="black", fontsize=10, fancybox=False)
    plt.xlabel("Remove node proportion", fontsize=12)
    plt.ylabel("Network efficiency", fontsize=12)
    plt.title("Dynamic Robustness - Network Efficiency", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 0.16)
    plt.tick_params(labelsize=10)
    plt.grid(False)

    plt.tight_layout()
    plt.show()