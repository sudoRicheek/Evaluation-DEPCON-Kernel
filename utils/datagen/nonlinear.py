import numpy as np
import networkx as nx

from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.backend.causality.sem import RandomGPGaussianNoiseSEM

from utils.seed_handler import load_seed

sigma_low = 0.3
sigma_medium = .5
sigma_high = 1
sigma_veryhigh = 1.5
sigma_dict = {"low": sigma_low, "medium": sigma_medium,
              "high": sigma_high, "veryhigh": sigma_veryhigh}

SAMPLE_SEED = load_seed().get("rfi.sample.seed") # Auto sets on import

def gendata_nonlinear_sem(dag_list, V, sample_size=60, noise_dict=dict(), p_dag=None):
    print("NONLINEAR")
    """
    Input:
        daglist <- list of DAGs nx.Digraphs
        sample_size <- sample size from all dags
        noise_dict <- additive gaussian noise to each node
                    provided as a list of ["low","medium","high","veryhigh"]
        p_dag <- fraction of samples from each dag
    """
    K = len(dag_list)  # Number of DAGs in the list of DAGs
    if not p_dag:
        p_dag = [1/K] * K
    
    if noise_dict:
        noise_dict = {f'x{i}': sigma_dict[noise_dict[i]] for i in range(1, V+1)}

    data = []
    for p, dag in zip(p_dag, dag_list):
        num_samps = int(sample_size * p)

        sem = RandomGPGaussianNoiseSEM(
            dag=DirectedAcyclicGraph(
                adjacency_matrix=nx.adjacency_matrix(dag).todense(),
                var_names=[f'x{i}' for i in range(1, V+1)]
            ),
            noise_std_dict=noise_dict,
            default_noise_std_bounds=(0.0, 2.0),
        )
        data.append(sem.sample(size=num_samps, seed=SAMPLE_SEED).numpy())

    return data