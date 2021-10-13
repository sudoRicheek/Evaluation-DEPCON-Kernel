import json
import numpy as np
import matplotlib.pyplot as plt


def plotJSON(fp):
    # with open('{}.json'.format(FILENAME)) as f:
    data = json.load(fp)

    print(data["name"])
    print("==="*20)

    number_of_runs = data["runs"]
    clusters = data["K"]

    # sorry for this mess
    fileext = ""
    fileext += "_lincorrzero" if data["LINCORR_ZERO"] else ""
    fileext += "_samemean" if data["SAME_MEAN"] else ""
    fileext += "_nonlinear" if data["NON_LINEAR"] else "_linear"

    method_names = ['depcon', 'rbf', 'poly', 'pkmeans']
    method_names_verbose = ["DEP-CON", "RBF", "POLY 2D", "K-MEANS"]

    # IMPORTANT stuff
    v_measure_means = dict.fromkeys(method_names)
    v_measure_stddevs = dict.fromkeys(method_names)

    adj_rand_means = dict.fromkeys(method_names)
    adj_rand_stddevs = dict.fromkeys(method_names)

    calinski_means = dict.fromkeys(method_names)
    calinski_stddevs = dict.fromkeys(method_names)

    cluster_pred_count = dict.fromkeys(method_names)
    ###

    for method_name in method_names:
        scores = np.array(data["v_measure_scores"][method_name])
        v_measure_means[method_name] = np.mean(scores)
        v_measure_stddevs[method_name] = np.std(scores)

        scores = np.array(data["adj_rand_scores"][method_name])
        adj_rand_means[method_name] = np.mean(scores)
        adj_rand_stddevs[method_name] = np.std(scores)

        scores = np.array(data["calinski"][method_name])
        calinski_means[method_name] = np.mean(scores)
        calinski_stddevs[method_name] = np.std(scores)

        scores = np.array(data["cluster_predictions"][method_name])
        cluster_pred_count[method_name] = np.count_nonzero(scores == 6)

    print("--- {}RUNS ---".format(number_of_runs))
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Mean Adj Rand Index: ".format(
            verbose_name), adj_rand_means[method_name])
        print("({})StdDev Adj Rand Index: ".format(
            verbose_name), adj_rand_stddevs[method_name])
        print("---")
    print("---")
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Mean V-Measure: ".format(verbose_name),
              v_measure_means[method_name])
        print("({})StdDev V-Measure: ".format(verbose_name),
              v_measure_stddevs[method_name])
        print("---")
    print("---")
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Mean Calinski-Harabasz Score: ".format(verbose_name),
              calinski_means[method_name])
        print("({})StdDev Calinski-Harabasz Score: ".format(verbose_name),
              calinski_stddevs[method_name])
        print("---")
    print("---")
    for method_name, verbose_name in zip(method_names, method_names_verbose):
        print("({})Number of correct cluster predictions: ".format(
            verbose_name), cluster_pred_count[method_name])
        print("---")
    print("---")

    # 95% CI, t-distribution
    # assume 60 datapoints here, t-score ~ 2.001
    confidence_bound = 2.001/np.sqrt(60)  # automate this

    error_bounds_adjrand = dict.fromkeys(method_names)
    error_bounds_vmeasure = dict.fromkeys(method_names)
    error_bounds_calinski = dict.fromkeys(method_names)
    for method_name in method_names:
        error_bounds_adjrand[method_name] = confidence_bound * \
            adj_rand_stddevs[method_name]
        error_bounds_vmeasure[method_name] = confidence_bound * \
            v_measure_stddevs[method_name]
        error_bounds_calinski[method_name] = confidence_bound * \
            calinski_stddevs[method_name]

    # Number of correct cluster predictions
    bar = []
    yerr = []
    for method_name in method_names:
        bar += [cluster_pred_count[method_name]]

    barWidth = 0.3
    plt.figure(figsize=[8, 6])
    plt.bar(range(len(method_names)), bar, width=barWidth, color=[
            'r', 'b', 'g', 'darkorchid'], alpha=0.7, edgecolor='black', linewidth=1.5, align='center')
    plt.xticks(range(len(method_names)), method_names_verbose, fontsize=15)
    plt.title(
        "Correct # cluster predictions over 100 randomized runs.\n True cluster count=6", fontsize=22)
    plt.xlabel("Clustering Algorithms", fontsize=18)
    plt.ylabel("Number of correct # cluster predictions", fontsize=18)
    plt.yticks(fontsize=15)

    # plt.savefig("numclus_preds{}.svg".format(number_of_runs))
    plt.savefig(f"plots/numclus_preds{number_of_runs}{fileext}.jpeg", dpi=1000)
    plt.show()

    # V-Measure case
    bar = []
    yerr = []
    for method_name in method_names:
        bar += [v_measure_means[method_name]]
        yerr += [error_bounds_vmeasure[method_name]]

    barWidth = 0.3
    plt.figure(figsize=[8, 6])
    plt.bar(range(len(method_names)), bar, width=barWidth, color=[
            'r', 'b', 'g', 'darkorchid'], alpha=0.7, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=7, align='center')
    plt.xticks(range(len(method_names)), method_names_verbose, fontsize=15)
    plt.title(
        f"V-Measure Scores over {number_of_runs} randomized runs", fontsize=22)
    plt.xlabel("Clustering Algorithms", fontsize=18)
    plt.ylabel("V-Measure Scores", fontsize=18)
    plt.ylim([0, 1])
    plt.yticks(fontsize=15)

    # plt.savefig("v_measure_linearcase{}.svg".format(number_of_runs))
    plt.savefig(f"plots/v_measure{number_of_runs}{fileext}.jpeg", dpi=1000)
    plt.show()

    # Adj-Rand case
    bar = []
    yerr = []
    for method_name in method_names:
        bar += [adj_rand_means[method_name]]
        yerr += [error_bounds_adjrand[method_name]]

    barWidth = 0.3
    plt.figure(figsize=[8, 6])
    plt.bar(range(len(method_names)), bar, width=barWidth, color=[
            'r', 'b', 'g', 'darkorchid'], alpha=0.7, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=7, align='center')
    plt.xticks(range(len(method_names)), method_names_verbose, fontsize=13)
    plt.title(
        f"Adjusted Rand Indices over {number_of_runs} randomized runs", fontsize=18)
    plt.xlabel("Clustering Algorithms", fontsize=16)
    plt.ylabel("Adjusted Rand Index Scores", fontsize=16)
    plt.ylim([0, 1])
    plt.yticks(fontsize=13)

    # plt.savefig("adj_rand_linearcase{}.svg".format(number_of_runs))
    plt.savefig(f"plots/adj_rand{number_of_runs}{fileext}.jpeg", dpi=1000)
    plt.show()

    # Calinski-Harabasz case
    # bar = []
    # yerr = []
    # for method_name in method_names:
    #     bar += [calinski_means[method_name]]
    #     yerr += [error_bounds_calinski[method_name]]

    # barWidth = 0.3
    # plt.figure(figsize=[8, 6])
    # plt.bar(range(len(method_names)), bar, width=barWidth, color=['r','b','g','orchid'], alpha=0.7, edgecolor='black', linewidth=1.5, yerr=yerr, capsize=7, align='center')
    # plt.xticks(range(len(method_names)), method_names_verbose, fontsize=13)
    # plt.title(r"Calinski-Harabasz Score over {} randomized runs".format(number_of_runs), fontsize=18)
    # plt.xlabel(r"\textbf{Clustering Algorithms}", fontsize=16)
    # plt.ylabel(r"\textbf{Calinski-Harabasz Score}", fontsize=16)
    # plt.yticks(fontsize=13)

    # plt.savefig("calinski_linearcase{}.svg".format(number_of_runs))
    # plt.savefig("calinski_linearcase{}.jpeg".format(number_of_runs), dpi=1000)
    # plt.show()
