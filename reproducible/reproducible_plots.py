import os
import sys

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), ".."))

import json
import argparse
import numpy as np
from sklearn import metrics
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({
    "text.usetex": True,
    "hatch.linewidth": 1.5,
})

from utils.seed_handler import (
    load_seed, 
    save_seed
)


parser = argparse.ArgumentParser(description='generate plots')
parser.add_argument('-t', type=int, help='Type of plot')
parser.add_argument('--SAVE', dest="SAVE", action="store_true", help='save the generated plot')
args = parser.parse_args()

sl=3
su=10

FILENAME_LINEAR = "reproducible_data/Linear_10_6_0.2_100_sametcFalse_samemeanFalse"
FILENAME_LINCORRZERO = "reproducible_data/Lincorrzero_10_6_0.2_100_sametcFalse_samemeanFalse"

method_names = ["depcon", "rbf", "poly", "pkmeans"]
method_names_verbose = ["DEP-CON", "RBF", "POLY 2D", "K-MEANS"]
method2verbose = {"rbf": "RBF", "depcon": "DEP-CON", "poly": "POLY-2D", "pkmeans": "K-MEANS"}


def case1and2(SAVE_PLOTS=False):
    store_seed = load_seed()
    save_seed({"random.seed": 42, "np.random.seed": 42, "np.random.default_rng": 2021, "rfi.sample.seed": 42})

    from utils.methods import (
        kkm_depcon_w_alpha,
    )
    from utils.datagen import (
        gen_random_weighted_dag,
        organise_data,
        generate_sample_data
    )
    from utils.methods.dep_con_kkm import (
        dep_contrib_kernel
    )

    V=10
    K=6
    p=0.2
    sample_size=360

    dag_list = [gen_random_weighted_dag(V, p) for _ in range(K)]
    data = generate_sample_data(dag_list, V, sample_size)
    sample_data, true_dag_labels, _ = organise_data(
            data, dag_list, shuffle=True)
    
    best_alpha, _ = kkm_depcon_w_alpha(sample_data, num_clus=K)

    X = sample_data
    alpha = best_alpha
    sim_ind = dep_contrib_kernel(X, alpha)
    transformer_2d = KernelPCA(n_components=2, kernel='precomputed')
    Xpca_2d = transformer_2d.fit_transform(sim_ind)

    # restore seed
    save_seed(store_seed)

    x = Xpca_2d[:,0]
    y = Xpca_2d[:,1]
    labels = true_dag_labels
    colormap = np.array(['r', 'g', 'b', 'c', 'm', 'y'])

    plt.scatter(x, y, c=colormap[labels], alpha=0.8, marker='o', edgecolors='black', linewidths=0.75)
    plt.title(r"KPCA of the "f"{V}"r" dimensional data set with DEP-CON Kernel", fontsize=18)
    plt.xlabel(r'$\longleftarrow$ \textbf{X-component} $\longrightarrow$', fontsize=17)
    plt.ylabel(r'$\longleftarrow$ \textbf{Y-component} $\longrightarrow$', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if SAVE_PLOTS:
        plt.savefig("depcon_pca.jpeg", dpi=1000)
    plt.show()

    X = sample_data
    linear_pca2d = PCA(n_components=2)
    linpca_2d = linear_pca2d.fit_transform(X)

    x1 = linpca_2d[:,0]
    y1 = linpca_2d[:,1]

    plt.scatter(x1, y1, c=colormap[labels], alpha=0.8, marker='o', edgecolors='black', linewidths=0.75)
    plt.title(r'Linear PCA of the {} dimensional data set'.format(V), fontsize=20)
    plt.xlabel(r'$\longleftarrow$ \textbf{X-component} $\longrightarrow$', fontsize=18)
    plt.ylabel(r'$\longleftarrow$ \textbf{Y-component} $\longrightarrow$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if SAVE_PLOTS:
        plt.savefig("lin_pca.jpeg", dpi=1000)
    plt.show()


"""
Plot 3:
Cluster Search over a single dataset.
"""
def case3(SAVE_PLOTS=False):
    store_seed = load_seed()
    save_seed({"random.seed": 42, "np.random.seed": 42, "np.random.default_rng": 2021, "rfi.sample.seed": 42})

    from utils.methods import (
        kkm_depcon,
        plain_kmeans,
        kkm_poly,
        kkm_rbf
    )
    from utils.datagen import (
        gen_random_weighted_dag,
        organise_data,
        generate_sample_data
    )

    V = 10
    K = 6
    p = 0.2
    sample_size = 360
    search_range = range(2,13) ## fixed for now
    method2funcs = [("depcon", kkm_depcon), ("rbf", kkm_rbf), ("poly", kkm_poly), ("pkmeans", plain_kmeans)]

    dag_list = [gen_random_weighted_dag(V, p) for _ in range(K)]
    data = generate_sample_data(dag_list, V, sample_size)
    sample_data, _, _ = organise_data(data, dag_list, shuffle=True)

    calinski_scores = {x: [] for x in method_names}
    best_clusters = {x: [] for x in method_names}
    for num_clus in search_range:
        for method, algo in method2funcs:
            best_clusters[method] = algo(sample_data, num_clus=num_clus)
            calinski_scores[method] += [metrics.calinski_harabasz_score(sample_data, best_clusters[method])]

    print(calinski_scores)
    save_seed(store_seed)

    plt.figure(figsize=[8, 6])
    plt.plot([str(i) for i in search_range], calinski_scores["depcon"], marker='o', linestyle=":", linewidth=3, color="r", alpha=0.6, label=r'DEP-CON')
    plt.plot([str(i) for i in search_range], calinski_scores["poly"], marker='.', linestyle=":", linewidth=3, color="g", alpha=0.6, label=r'POLY-2D')
    plt.plot([str(i) for i in search_range], calinski_scores["rbf"], marker='p', linestyle=":", linewidth=3, color="b", alpha=0.6, label=r'Radial Basis Function')
    plt.plot([str(i) for i in search_range], calinski_scores["pkmeans"], marker='8', linestyle=":", linewidth=3, color="orchid", alpha=0.6, label=r'Plain K-Means')
    
    
    plt.title(r'Unsupervised Kernels for \# cluster search', fontsize=22)
    plt.xlabel(r'\textbf{Number of Clusters} $\longrightarrow$', fontsize=18)
    plt.ylabel(r'\textbf{Calinski-Harabasz Scores} $\longrightarrow$', fontsize=18)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)

    if SAVE_PLOTS:
        plt.savefig("clustersearch.jpeg", dpi=700)
    plt.show()

"""
Plot 4:
Generates plot for the Linear Case.
4 Histograms in a single figure with 
# cluster counts
"""
def case4(F1, F2, SAVE_PLOTS=False):
    with open(os.path.join(__location__, '{}.json'.format(F1))) as f:
        data1 = json.load(f)
    with open(os.path.join(__location__, '{}.json'.format(F2))) as f:
        data2 = json.load(f)

    print(data1["name"])
    print("==="*20)

    number_of_runs = data1["runs"]
    cluster_pred_count_lin = {method: {i: 0 for i in range(sl, su)} for method in method_names}
    cluster_pred_count_lincorr0 = {method: {i: 0 for i in range(sl, su)} for method in method_names}

    for method_name in method_names:
        scores = np.array(data1["cluster_predictions"][method_name])
        for clus in range(sl,su):
            cluster_pred_count_lin[method_name][clus] = np.count_nonzero(scores==clus)
        
        scores = np.array(data2["cluster_predictions"][method_name])
        for clus in range(sl,su):
            cluster_pred_count_lincorr0[method_name][clus] = np.count_nonzero(scores==clus)


    print("--- {}RUNS ---".format(number_of_runs))
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Number of correct cluster predictions LINEAR: ".format(verbose_name), cluster_pred_count_lin[method_name])
        print("---")
    print("---")
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Number of correct cluster predictions LINEAR CORR ZERO: ".format(verbose_name), cluster_pred_count_lincorr0[method_name])
        print("---")
    print("---")

    
    # Number of correct cluster predictions
    bars1 = dict()
    for method_name in method_names:
        clus_temp = []
        for i in range(sl,su):
            clus_temp+=[cluster_pred_count_lin[method_name][i]]
        bars1[method_name] = clus_temp
    
    bars2 = dict()
    for method_name in method_names:
        clus_temp = []
        for i in range(sl,su):
            clus_temp+=[cluster_pred_count_lincorr0[method_name][i]]
        bars2[method_name] = clus_temp


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,8))

    color=['r','b','g','darkorchid']
    barWidth = 0.3
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.bar(np.arange(sl,su)-barWidth/2, bars1[method_names[2*i+j]], width=barWidth, color=color[2*i+j], alpha=0.7, edgecolor='black', linewidth=1.5, align='center', label=method_names_verbose[2*i+j])
            col.legend(loc='upper right', fontsize=14)
            col.set_ylim(top=42)
            col.set_xticks(range(sl,su))
            col.tick_params(axis='both', which='major', labelsize=16)

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.bar(np.arange(sl,su)+barWidth/2, bars2[method_names[2*i+j]], width=barWidth, color=color[2*i+j], alpha=0.7, edgecolor='black', linewidth=1.5, hatch="/", align='center', label=method_names_verbose[2*i+j])
            col.set_ylim(top=42)
            col.set_xticks(range(sl,su))
            col.tick_params(axis='both', which='major', labelsize=16)

    fig.suptitle(r"Correct \# cluster predictions over 100 randomized runs.""\n"r" \textbf{True cluster count=6}", fontsize=22)
    fig.supxlabel(t=r"\textbf{Cluster Search Range} $\longrightarrow$", fontsize=18, y=0.07)
    fig.supylabel(r"\textbf{Number of times \# cluster predicted} $\longrightarrow$", fontsize=18)

    solid_patch = mpatches.Patch(color='black', label=r"\textbf{Linear data set}")
    striped_patch = mpatches.Patch(facecolor='white', edgecolor="black", linewidth=1.5, hatch="/", label=r"\textbf{Nonlinear data set}")
    fig.legend(handles=[solid_patch, striped_patch], loc="lower center", fontsize=15, bbox_to_anchor=(0.47, -0.0), fancybox=True, ncol=5)
    
    fig.subplots_adjust(top=0.89, bottom=0.142, left=0.093, right=0.953, hspace=0.152, wspace=0.123)

    if SAVE_PLOTS:
        plt.savefig("numclus_preds{}.jpeg".format(number_of_runs), dpi=1000)
    plt.show()

"""
Plot 5:
Adjusted Rand Indices plot for standard 
linear case and linear correlation zero case. 
"""
def case5(F1, F2, SAVE_PLOTS=False):
    with open(os.path.join(__location__, '{}.json'.format(F1))) as f:
        data1 = json.load(f)
    with open(os.path.join(__location__, '{}.json'.format(F2))) as f:
        data2 = json.load(f)

    number_of_runs = data1["runs"]

    adj_rand_means1 = dict.fromkeys(method_names)
    adj_rand_stddevs1 = dict.fromkeys(method_names)
    adj_rand_means2 = dict.fromkeys(method_names)
    adj_rand_stddevs2 = dict.fromkeys(method_names)
    
    for method_name in method_names:
        scores = np.array(data1["adj_rand_scores"][method_name])
        adj_rand_means1[method_name] = np.mean(scores)
        adj_rand_stddevs1[method_name] = np.std(scores)

        scores = np.array(data2["adj_rand_scores"][method_name])
        adj_rand_means2[method_name] = np.mean(scores)
        adj_rand_stddevs2[method_name] = np.std(scores)
    
    # 95% CI, t-distribution
    # assume 60 datapoints here, t-score ~ 2.001
    confidence_bound = 1.980/np.sqrt(100) # automate this

    error_bounds_adjrand1 = dict.fromkeys(method_names)
    error_bounds_adjrand2 = dict.fromkeys(method_names)
    for method_name in method_names:
        error_bounds_adjrand1[method_name] = confidence_bound * adj_rand_stddevs1[method_name]
        error_bounds_adjrand2[method_name] = confidence_bound * adj_rand_stddevs2[method_name]

    # Plotting
    x = np.arange(len(method_names_verbose))  # the label locations
    barWidth = 0.25
    plt.figure(figsize=[8, 6])
    color=['r','b','g','darkorchid']

    bar = []
    yerr = []
    for method_name in method_names:
        bar += [adj_rand_means1[method_name]]
        yerr += [error_bounds_adjrand1[method_name]]
    plt.bar(x-barWidth/2, bar, width=barWidth, color=color, alpha=0.7, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=7, align='center', label=r"\textbf{Linear data set}")

    bar = []
    yerr = []
    for method_name in method_names:
        bar += [adj_rand_means2[method_name]]
        yerr += [error_bounds_adjrand2[method_name]]
    plt.bar(x+barWidth/2, bar, width=barWidth, color=color, alpha=0.7, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=7, align='center', hatch="/", label=r"\textbf{Nonlinear data set}")

    plt.xticks(range(len(method_names)), method_names_verbose, fontsize=15)
    plt.title(r"Adjusted Rand Indices over {} randomized runs".format(number_of_runs), fontsize=22)
    plt.xlabel(r"\textbf{Clustering Algorithms} $\longrightarrow$", fontsize=18)
    plt.ylabel(r"\textbf{Adjusted Rand Indices} $\longrightarrow$", fontsize=18)
    plt.yticks(fontsize=16)

    solid_patch = mpatches.Patch(color='black', label=r"\textbf{Linear data set}")
    striped_patch = mpatches.Patch(facecolor='white', edgecolor="black", linewidth=1.5, hatch="/", label=r"\textbf{Nonlinear data set}")
    plt.legend(handles=[solid_patch, striped_patch], loc="lower center", fontsize=15, bbox_to_anchor=(0.47, -0.25), fancybox=True, ncol=5)
    
    plt.subplots_adjust(top=0.92, bottom=0.185, left=0.105, right=0.93, hspace=0.26, wspace=0.19)

    if SAVE_PLOTS:
        plt.savefig("adjrand_indices{}.jpeg".format(number_of_runs), dpi=1000)
    plt.show()

if __name__=="__main__":
    t = args.t
    SAVE = args.SAVE

    if t==1 or t==2:
        case1and2(SAVE)
    elif t==3:
        case3(SAVE)
    elif t==4:
        case4(FILENAME_LINEAR, FILENAME_LINCORRZERO, SAVE)
    elif t==5:
        case5(FILENAME_LINEAR, FILENAME_LINCORRZERO, SAVE)

    