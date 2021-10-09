import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph

import random
import numpy as np
from numpy.random import default_rng

from utils.seed_handler import load_seed

# Load seeds
seed_dict = load_seed()
np.random.seed(seed_dict.get("np.random.seed"))
random.seed(seed_dict.get("random.seed"))
rng = default_rng(seed_dict.get("np.random.default_rng"))
#


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
    print("SAME_TC")
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
