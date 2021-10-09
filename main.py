# Tested on:
# ---------------
# python==3.9.4
# networkx==2.5.1
# numpy==1.21.0
# scikit-learn==0.24.2

import json
import random
import argparse

import numpy as np
from numpy.random import default_rng

from sklearn import metrics

from utils.seed_handler import (
    load_seed
)

from utils.datagen.linear import (
    generate_sample_data,
    generate_sample_data_lincorrzero,
)
from utils.datagen.dags import (
    gen_random_weighted_dag,
    gen_rand_dags_same_tc,
)
from utils.datagen.data_op import (
    get_causal_dependencies,
    organise_data,
)
from utils.datagen.nonlinear import (
    gendata_nonlinear_sem
)

from utils.methods.dep_con_kernel_k_means import kernel_k_means_depcon
from utils.methods.poly_kern import kernel_k_means_poly
from utils.methods.rbf_kernel_k_means import kkm_rbf
from utils.methods.kmeans import plain_kmeans

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
parser.add_argument('--NON_LINEAR', dest='NON_LINEAR', action='store_true')
parser.add_argument('--SHUFFLE', dest='SHUFFLE', action='store_true')
args = parser.parse_args()


def multiple_runs_comparison(V=10, K=6, p=0.1, sample_size=600, runs=50, SAME_TC=False, CLUSTER_SEARCH=False, LINCORR_ZERO=False, SAME_MEAN=False, NON_LINEAR=False, search_range=np.arange(3, 13), shuffle=False):
    METHOD_NAMES = ['rbf', 'depcon', 'poly', 'pkmeans']
    METHOD2FUNCS = [('depcon', kernel_k_means_depcon), ('rbf', kkm_rbf),
                    ('poly', kernel_k_means_poly), ('pkmeans', plain_kmeans)]
    METHOD2VERBOSE = {'rbf': "RBF", 'depcon': "DEP-CON", 'poly': "POLY-2D", 'pkmeans': "K-MEANS"}

    adj_rand = {N: [] for N in METHOD_NAMES}
    v_measure = {N: [] for N in METHOD_NAMES}
    calinski = {N: [] for N in METHOD_NAMES}

    predictions = {N: [] for N in METHOD_NAMES}

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

            sample_data, true_dag_labels, _ = organise_data(
                data, dag_list*2, shuffle=shuffle)
        elif NON_LINEAR:
            data = gendata_nonlinear_sem(
                [dag for (dag, _) in dag_list], V, sample_size)
            print(len(data))
            sample_data, true_dag_labels, _ = organise_data(
                data, dag_list, shuffle=shuffle)
        else:
            data = generate_sample_data(
                dag_list, V, sample_size, SAME_MEAN=SAME_MEAN)
            print(len(data))
            sample_data, true_dag_labels, _ = organise_data(
                data, dag_list, shuffle=shuffle)

        cluster_result = {N: 0 for N in METHOD_NAMES}
        if CLUSTER_SEARCH:
            clusters_iter = {N: 0 for N in METHOD_NAMES}
            cluster_predict = {N: -1 for N in METHOD_NAMES}
            best_scores = {N: -1 for N in METHOD_NAMES}
            for num_clus in search_range:
                for method_name, algo in METHOD2FUNCS:
                    clusters_iter[method_name] = algo(
                        sample_data, num_clus=num_clus)

                if np.unique(clusters_iter["poly"], return_counts=False).size == 1:
                    clusters_iter["poly"] = rng.integers(
                        0, num_clus-1, size=sample_data.shape[0])

                ch_scores = {}
                for method_name in METHOD_NAMES:
                    ch_scores[method_name] = metrics.calinski_harabasz_score(
                        sample_data, clusters_iter[method_name])

                    if ch_scores[method_name] > best_scores[method_name]:
                        best_scores[method_name] = ch_scores[method_name]
                        cluster_predict[method_name] = num_clus
                        cluster_result[method_name] = clusters_iter[method_name]
            for method_name in METHOD_NAMES:
                predictions[method_name] += [cluster_predict[method_name]]
        else:
            for method_name, algo in METHOD2FUNCS:
                cluster_result[method_name] = algo(sample_data, num_clus=K)

        # print(true_ue_labels)
        print(true_dag_labels)

        for method_name in METHOD_NAMES:
            adj_rand[method_name] += [metrics.adjusted_rand_score(
                true_dag_labels, cluster_result[method_name])]
            v_measure[method_name] += [metrics.v_measure_score(
                true_dag_labels, cluster_result[method_name])]
            calinski[method_name] += [metrics.calinski_harabasz_score(
                sample_data, cluster_result[method_name])]

    print("---", runs, "RUNS ---")
    for mn in METHOD_NAMES:
        print(f"({METHOD2VERBOSE[mn]})Mean Adj Rand Index: ", np.mean(adj_rand[mn]))
        print(f"({METHOD2VERBOSE[mn]})StdDev Adj Rand Index: ", np.std(adj_rand[mn]))
        print("---")
    print("---")
    for mn in METHOD_NAMES:
        print(f"({METHOD2VERBOSE[mn]})Mean V-Measure: ", np.mean(v_measure[mn]))
        print(f"({METHOD2VERBOSE[mn]})StdDev V-Measure: ", np.std(v_measure[mn]))
        print("---")
    print("---")

    if CLUSTER_SEARCH:
        print("CLUSTER PREDICTIONS::: TRUE={}".format(K))
        print(predictions)
        print("V-MEASURE EACH RUN::: ")
        print(v_measure)
        print("ADJUSTED RAND INDEX EACH RUN::: ")
        print(adj_rand)

    export_dict = {}
    export_dict["name"] = f"NON_LINEAR={NON_LINEAR} - SAME_TC={SAME_TC} LINCORR_ZERO={LINCORR_ZERO} SAME_MEAN={SAME_MEAN}"
    export_dict["V"] = V
    export_dict["K"] = K
    export_dict["p"] = p
    export_dict["sample_size"] = sample_size
    export_dict["runs"] = runs
    export_dict["LINCORR_ZERO"] = LINCORR_ZERO
    export_dict["SAME_TC"] = SAME_TC
    export_dict["SAME_MEAN"] = SAME_MEAN
    export_dict["NON_LINEAR"] = NON_LINEAR
    export_dict["cluster_predictions"] = predictions
    export_dict["v_measure_scores"] = v_measure
    export_dict["adj_rand_scores"] = adj_rand
    export_dict["calinski"] = calinski

    if LINCORR_ZERO:
        with open(f'Lincorrzero_{V}_{K}_{p}.json', 'w') as fp:
            json.dump(export_dict, fp)
    elif NON_LINEAR:
        with open(f'Nonlinear_{V}_{K}_{p}.json', 'w') as fp:
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
    non_linear = args.NON_LINEAR
    shuffle = args.SHUFFLE
    search_range = range(args.sl, args.su)

    multiple_runs_comparison(
        V=V, K=K, p=p, sample_size=sample_size, runs=runs, CLUSTER_SEARCH=True,
        SAME_TC=sametc, LINCORR_ZERO=lincorr_zero, NON_LINEAR=non_linear,
        SAME_MEAN=same_mean, search_range=search_range, shuffle=shuffle)
