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


def generate_sample_data(dag_list, V, sample_size=50, p_dag=None, SAME_MEAN=False):
    print("LINEAR")
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
    print("LINCORR ZERO")
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

