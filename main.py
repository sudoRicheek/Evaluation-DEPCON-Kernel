# Tested on:
# ---------------
# python==3.9.4
# networkx==2.5.1
# numpy==1.21.0
# scikit-learn==0.24.2
from networkx.generators.random_graphs import fast_gnp_random_graph
import networkx as nx
from sklearn import metrics
from numpy import linalg as LA
from numpy.random import default_rng
import numpy as np
import json
import random
import itertools
import argparse

from utils.seed_handler import load_seed
from utils.kmeans import plain_kmeans
from utils.poly_kern import kernel_k_means_poly
from utils.rbf_kernel_k_means import kkm_rbf
from utils.dep_con_kernel_k_means import kernel_k_means_depcon

parser = argparse.ArgumentParser(description='Run tests.')
parser.add_argument('-V', type=int, help='number of vertices')
parser.add_argument('-r', '--runs', type=int, help='number of runs')
parser.add_argument('-K', type=int, help='number of dags')
parser.add_argument('-p', type=float, help='probability of edge creation')
parser.add_argument('-S', type=int, help='sample size')
parser.add_argument('-sl', type=int, help='search range lower', default=3)
parser.add_argument('-su', type=int, help='search range upper', default=10)
parser.add_argument('--SAME_TC', dest='SAME_TC', action='store_true')
parser.add_argument('--LINCORR_ZERO', dest='LINCORR_ZERO', action='store_true')
parser.add_argument('--SAME_MEAN', dest='SAME_MEAN', action='store_true')
args = parser.parse_args()


def gen_random_weighted_dag(V, p, a=0.25, b=2) -> nx.DiGraph:
    """
    Takes in the number of vertices and probability of edge creation
    (Same terminology as:
    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html)

    Inputs: V <- Number of vertices
            p <- Probability of edge creation
            a <- Lower limit of edge weight
            b <- Upper limit of edge weight

    Output: nx.DiGraph <- A Directed Acyclic Graph
            with edge weights in [-b,-a] U [a, b]
            topo_ord <- A Topological order of the nodes
    """
    random_ug = fast_gnp_random_graph(V, p, directed=False)
    topo_ord = rng.permutation(V)
    inv_topo_ord = np.empty(V)  # Inverse of the topological permutation
    inv_topo_ord[topo_ord] = np.arange(V)

    random_dag = nx.DiGraph()
    random_dag.add_nodes_from(range(V))

    positive_weights = rng.uniform(
        a, b, size=random_ug.number_of_edges())
    pos_neg_switch = rng.choice(
        [-1, 1], size=random_ug.number_of_edges())

    weights = positive_weights * pos_neg_switch  # hadamard
    random_dag.add_weighted_edges_from([(u, v, w) if inv_topo_ord[u] < inv_topo_ord[v] else (
        v, u, w) for (u, v), w in zip(random_ug.edges(), weights)])

    return random_dag, topo_ord


def gen_rand_dags_same_tc(V=10, a=0.25, b=2, N=1):
    """
    Code to get random weighted DAGs from 
        the given transitive closure
    V: number of vertices
    a: lower bound of edge weight
    b: upper bound of edge weight
    N: Number of DAGs to generate
    """
    extra_edges = set()
    while(len(extra_edges) == 0):
        rand_dag, topo_ord = gen_random_weighted_dag(V=V, p=0.2, a=a, b=b)
        transitive_closure = nx.transitive_closure_dag(
            rand_dag, topo_order=topo_ord)
        transitive_reduc = nx.transitive_reduction(rand_dag)

        base_edges = set(transitive_reduc.edges())
        extra_edges = set(transitive_closure.edges()) - base_edges
        array_extra_edges = list(extra_edges)

    num_extra = len(array_extra_edges)

    dag_list = []
    for _ in range(N):
        # Number of edges to choose on top of the base edges
        num_choice_extra = rng.choice(num_extra+1)
        choice_indices = rng.choice(
            num_extra, size=num_choice_extra, replace=False)  # Which extra edges to choose
        choice_extra_edges = set([tuple(liss) for liss in np.array(array_extra_edges)[
                                 choice_indices].tolist()])  # Choose the extra edges

        dag_edges = choice_extra_edges | base_edges

        random_dag = nx.DiGraph()
        random_dag.add_nodes_from(range(V))

        positive_weights = rng.uniform(
            a, b, size=len(dag_edges))
        pos_neg_switch = rng.choice(
            [-1, 1], size=len(dag_edges))

        weights = positive_weights * pos_neg_switch  # hadamard
        random_dag.add_weighted_edges_from(
            [(u, v, w) for (u, v), w in zip(dag_edges, weights)])

        dag_list.append(random_dag)

    return dag_list, topo_ord


def generate_sample_data(dag_list, V, sample_size=50, p_dag=None, SAME_MEAN=False):
    """
    Samples data from the DAGs present in the list.
    Inputs: dag_list <- List of nx.DiGraphs
            V <- Number of vertices in each DAG
            sample_size <- Number of sample points required 
            p_dag <- Fraction of sample points from each component dag 
                    (List of fractions adding upto one[not checked in code])
            mu <- List of means from which the gaussian errors are sampled
            stddevs <- List of Std devs. used to sample the gaussians
    """
    K = len(dag_list)  # Number of DAGs in the list of DAGs
    if not p_dag:
        p_dag = [1/K] * K

    data = []
    for p, (dag, topo_ord) in zip(p_dag, dag_list):
        # Number of samples to take from current DAG
        num_samps = int(sample_size * p)

        if SAME_MEAN:
            mu = np.zeros(V)
            stddevs = rng.uniform(0.5, 100, V)
        else:
            mu = rng.uniform(-4, 4, V)
            stddevs = rng.uniform(0.5, 2, V)

        dag_data = rng.normal(size=(num_samps, V))
        dag_data = (dag_data * stddevs + mu).T  # Sampling error terms

        for node in topo_ord:
            for pred_node in dag.predecessors(node):
                dag_data[node] += dag_data[pred_node] * \
                    dag[pred_node][node]['weight']
        data.append(dag_data.T)

    return data


def generate_sample_data_lincorrzero(inp_dag, V, sample_size=50, SAME_MEAN=False):
    (dag, topo_ord) = inp_dag
    # Number of samples to take from current DAG
    num_samps = sample_size//2

    if SAME_MEAN:
        mu = np.zeros(V)
        stddevs = rng.uniform(0.5, 100, V)
    else:
        mu = rng.uniform(-4, 4, V)
        stddevs = rng.uniform(0.5, 2, V)

    dag_data1 = rng.normal(size=(num_samps, V))
    dag_data1 = np.vstack((dag_data1, -1 * dag_data1))
    dag_data1 = (dag_data1 * stddevs + mu).T  # Sampling error terms

    dag_data2 = rng.normal(size=(num_samps, V))
    dag_data2 = np.vstack((dag_data2, -1 * dag_data2))
    dag_data2 = (dag_data2 * stddevs + mu).T  # Sampling error terms

    # Leaf nodes with in_degree>0 and out_degree=0
    leafedges = [(src, dst) for (src, dst) in dag.edges()
                 if dag.out_degree(dst) == 0 and dag.in_degree(dst) >= 1]
    choice_size = min(3, len(leafedges))
    choice_indices = rng.choice(
        len(leafedges), size=choice_size, replace=False)
    fixed_edges = set([leafedges[i] for i in choice_indices])
    print(fixed_edges)

    for (_, dst) in fixed_edges:
        # dag_data2[dst] = np.zeros_like(dag_data2[dst])
        # dag_data1[dst] = np.zeros_like(dag_data1[dst])
        temp = rng.normal(0, 0.5, num_samps)
        dag_data2[dst] = np.append(temp, -1*temp)
        temp = rng.normal(0, 0.5, num_samps)
        dag_data1[dst] = np.append(temp, -1*temp)

    for node in topo_ord:
        for pred_node in dag.predecessors(node):
            if (pred_node, node) in fixed_edges:
                t1 = dag_data2[pred_node] - dag_data2[pred_node].mean()
                # dag_data2[node] += np.sqrt((t1*t1).max() - t1*t1) * rng.choice([-1,1], size=num_samps)
                dag_data2[node] += np.sqrt((t1*t1).max() - t1*t1)
            else:
                dag_data1[node] += dag_data1[pred_node] * \
                    dag[pred_node][node]['weight']
                dag_data2[node] += dag_data2[pred_node] * \
                    dag[pred_node][node]['weight']

    data1 = dag_data1.T
    data2 = dag_data2.T

    return data1, data2


def get_causal_dependencies(G, topo_ord):
    """
    Can optimise further by reversing the topological order,
    i.e, going bottom-up and memoizing (some sort of dynamic
    programming).
    """
    dependencies = set()
    visited_set = set()
    for node in topo_ord:
        if node in visited_set:
            continue
        descendants = nx.descendants(G, node)
        descendants.add(node)
        visited_set.update(descendants)
        dependencies.update(itertools.permutations(descendants, 2))
    return dependencies


def organise_data(sampled_data, dag_list, shuffle=True):
    """
    Input format :- 
                sampled_data <- Shape (K, num_data_points, V)
    Output format :- 
                data <- with the new shape (total_num_data_points, V)
                labels <- Marking which DAG the sample-point belongs to
    """
    K = len(sampled_data)

    newdata = np.vstack(sampled_data)
    dag_labels = [[i]*dag_data.shape[0]
                  for i, dag_data in enumerate(sampled_data)]
    dag_labels = np.fromiter(itertools.chain.from_iterable(dag_labels), int)

    causal_dependencies = [
        frozenset(get_causal_dependencies(dag, topo_ord)) for dag, topo_ord in dag_list]  # Non-mutable sets of dependencies of given DAGs

    # This part can be optimized! But since number of dags
    # will be pretty small, it works pretty fast for now.
    uci_dependencies = {}
    for i, dependency in enumerate(causal_dependencies):
        uci_dependencies[dependency] = uci_dependencies.get(
            dependency, []) + [i]

    # Stores which unconditional independencies correspond to which DAGs
    unconditional_equivalence_labels = np.empty(K, dtype=int)
    for i, (_, value) in enumerate(uci_dependencies.items()):
        unconditional_equivalence_labels[value] = i
    # print(unconditional_equivalence_labels)

    ue_labels = np.copy(dag_labels)
    for i, val in enumerate(unconditional_equivalence_labels):
        ue_labels[ue_labels == i] = val

    if shuffle:
        # Random shuffle to prevent bias
        indices = np.arange(dag_labels.shape[0])
        rng.shuffle(indices)
        newdata = newdata[indices]
        dag_labels = dag_labels[indices]
        ue_labels = ue_labels[indices]

    return newdata, dag_labels, ue_labels


def multiple_runs_comparison(V=10, K=6, p=0.1, sample_size=600, runs=50, SAME_TC=False, CLUSTER_SEARCH=False, LINCORR_ZERO=False, SAME_MEAN=False, search_range=np.arange(3, 13)):
    adj_rand = {'depcon': [], 'rbf': [], 'poly': [], 'pkmeans': []}
    v_measure = {'depcon': [], 'rbf': [], 'poly': [], 'pkmeans': []}
    calinski = {'depcon': [], 'rbf': [], 'poly': [], 'pkmeans': []}

    predictions = {'rbf': [], 'depcon': [], 'poly': [], 'pkmeans': []}

    for i in range(runs):
        print("RUN ::: ", i)

        if SAME_TC:
            dag_list, topo_ord = gen_rand_dags_same_tc(V, N=K)
            dag_list = list(zip(dag_list, [np.array(topo_ord)]*K))
        elif LINCORR_ZERO:
            dag_list = [gen_random_weighted_dag(V, p) for _ in range(K//2)]
        else:
            dag_list = [gen_random_weighted_dag(V, p) for _ in range(K)]
            # returns list of np.arrays

        if LINCORR_ZERO:
            data = []
            for i in range(K//2):
                d1, d2 = generate_sample_data_lincorrzero(
                    dag_list[i], V=V, sample_size=sample_size//K)
                data.append(d1)
                data.append(d2)

            sample_data, true_dag_labels, true_ue_labels = organise_data(
                data, dag_list*2, shuffle=False)
        else:
            data = generate_sample_data(
                dag_list, V, sample_size, SAME_MEAN=SAME_MEAN)
            print(len(data))
            sample_data, true_dag_labels, true_ue_labels = organise_data(
                data, dag_list, shuffle=False)

        cluster_result = {'depcon': 0, 'rbf': 0, 'poly': 0, 'pkmeans': 0}
        if CLUSTER_SEARCH:
            clusters_iter = {'depcon': 0, 'rbf': 0, 'poly': 0, 'pkmeans': 0}
            cluster_predict = {'rbf': -1, 'depcon': -
                               1, 'poly': -1, 'pkmeans': -1}
            best_scores = {'rbf': -1, 'depcon': -1, 'poly': -1, 'pkmeans': -1}
            for num_clus in search_range:
                for method_name, algo in [('depcon', kernel_k_means_depcon), ('rbf', kkm_rbf),
                                          ('poly', kernel_k_means_poly), ('pkmeans', plain_kmeans)]:
                    clusters_iter[method_name] = algo(
                        sample_data, num_clus=num_clus)

                if np.unique(clusters_iter["poly"], return_counts=False).size == 1:
                    clusters_iter["poly"] = rng.integers(
                        0, num_clus-1, size=sample_data.shape[0])

                ch_scores = {}
                for method_name in ['depcon', 'rbf', 'poly', 'pkmeans']:
                    ch_scores[method_name] = metrics.calinski_harabasz_score(
                        sample_data, clusters_iter[method_name])

                    if ch_scores[method_name] > best_scores[method_name]:
                        best_scores[method_name] = ch_scores[method_name]
                        cluster_predict[method_name] = num_clus
                        cluster_result[method_name] = clusters_iter[method_name]
            for method_name in ['rbf', 'depcon', 'poly', 'pkmeans']:
                predictions[method_name] += [cluster_predict[method_name]]
        else:
            for method_name, algo in [('depcon', kernel_k_means_depcon), ('rbf', kkm_rbf),
                                      ('poly', kernel_k_means_poly), ('pkmeans', plain_kmeans)]:
                cluster_result[method_name] = algo(sample_data, num_clus=K)

        print(true_ue_labels)
        print(true_dag_labels)

        for method_name in ['depcon', 'rbf', 'poly', 'pkmeans']:
            adj_rand[method_name] += [metrics.adjusted_rand_score(
                true_dag_labels, cluster_result[method_name])]
            v_measure[method_name] += [metrics.v_measure_score(
                true_dag_labels, cluster_result[method_name])]
            calinski[method_name] += [metrics.calinski_harabasz_score(
                sample_data, cluster_result[method_name])]

    print("---", runs, "RUNS ---")
    print("(RBF)Mean Adj Rand Index: ", np.mean(adj_rand["rbf"]))
    print("(RBF)StdDev Adj Rand Index: ", np.std(adj_rand["rbf"]))
    print("---")
    print("(POLY 2D)Mean Adj Rand Index: ", np.mean(adj_rand["poly"]))
    print("(POLY 2D)StdDev Adj Rand Index: ", np.std(adj_rand["poly"]))
    print("---")
    print("(DEP-CON)Mean Adj Rand Index: ", np.mean(adj_rand["depcon"]))
    print("(DEP-CON)StdDev Adj Rand Index: ", np.std(adj_rand["depcon"]))
    print("---")
    print("(K-MEANS)Mean Adj Rand Index: ", np.mean(adj_rand["pkmeans"]))
    print("(K-MEANS)StdDev Adj Rand Index: ", np.std(adj_rand["pkmeans"]))
    print("---")
    print("---")
    print("(RBF)Mean V-Measure: ", np.mean(v_measure["rbf"]))
    print("(RBF)StdDev V-Measure: ", np.std(v_measure["rbf"]))
    print("---")
    print("(POLY 2D)Mean V-Measure: ", np.mean(v_measure["poly"]))
    print("(POLY 2D)StdDev V-Measure: ", np.std(v_measure["poly"]))
    print("---")
    print("(DEP-CON)Mean V-Measure: ", np.mean(v_measure["depcon"]))
    print("(DEP-CON)StdDev V-Measure: ", np.std(v_measure["depcon"]))
    print("---")
    print("(K-MEANS)Mean V-Measure: ", np.mean(v_measure["pkmeans"]))
    print("(K-MEANS)StdDev V-Measure: ", np.std(v_measure["pkmeans"]))
    print("---")

    if CLUSTER_SEARCH:
        print("CLUSTER PREDICTIONS::: TRUE={}".format(K))
        print(predictions)
        print("V-MEASURE EACH RUN::: ")
        print(v_measure)
        print("ADJUSTED RAND INDEX EACH RUN::: ")
        print(adj_rand)

    export_dict = {}
    export_dict["name"] = f"Linear - SAME_TC={SAME_TC} LINCORR_ZERO={LINCORR_ZERO} SAME_MEAN={SAME_MEAN}"
    export_dict["V"] = V
    export_dict["K"] = K
    export_dict["p"] = p
    export_dict["sample_size"] = sample_size
    export_dict["runs"] = runs
    export_dict["LINCORR_ZERO"] = LINCORR_ZERO
    export_dict["SAME_TC"] = SAME_TC
    export_dict["SAME_MEAN"] = SAME_MEAN
    export_dict["cluster_predictions"] = predictions
    export_dict["v_measure_scores"] = v_measure
    export_dict["adj_rand_scores"] = adj_rand
    export_dict["calinski"] = calinski

    if LINCORR_ZERO:
        with open(f'Lincorrzero_{V}_{K}_{p}.json', 'w') as fp:
            json.dump(export_dict, fp)
    else:
        with open(f'Linear_{V}_{K}_{p}_sametc{SAME_TC}_samemean{SAME_MEAN}.json', 'w') as fp:
            json.dump(export_dict, fp)


if __name__ == "__main__":
    # Load seeds
    seed_dict = load_seed()
    np.random.seed(seed_dict["np.random.seed"])
    random.seed(seed_dict["random.seed"])
    rng = default_rng(seed_dict["np.random.default_rng"])
    #

    V = args.V
    K = args.K
    p = args.p
    sample_size = args.S
    runs = args.runs
    sametc = args.SAME_TC
    lincorr_zero = args.LINCORR_ZERO
    same_mean = args.SAME_MEAN
    search_range = range(args.sl, args.su)

    multiple_runs_comparison(
        V=V, K=K, p=p, sample_size=sample_size, runs=runs,
        CLUSTER_SEARCH=True, SAME_TC=sametc, LINCORR_ZERO=lincorr_zero, 
        SAME_MEAN=same_mean, search_range=search_range)
