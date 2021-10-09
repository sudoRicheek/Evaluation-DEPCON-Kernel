import random
import itertools

import networkx as nx

import numpy as np
from numpy.random import default_rng

from utils.seed_handler import load_seed


# Load seeds
seed_dict = load_seed()
np.random.seed(seed_dict.get("np.random.seed"))
random.seed(seed_dict.get("random.seed"))
rng = default_rng(seed_dict.get("np.random.default_rng"))
#


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
