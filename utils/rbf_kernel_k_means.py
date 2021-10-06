import numpy as np
from numpy.random import default_rng

from sklearn import metrics

from utils.seed_handler import load_seed

rng = None


def param_search_gamma(X, sr_range=np.float_power([2], np.arange(-10, 3)), num_clus=6, metric=metrics.calinski_harabasz_score):
    scores = []
    best_score = -1
    best_gamma = -1
    best_cluster = None
    for gamma in sr_range:
        result = kkm_rbf_algo(X, num_clus=num_clus, max_iter=300, gamma=gamma)
        if np.unique(result, return_counts=False).size == 1:
            score = -1
        else:
            score = metric(X, result)
        scores += [score]
        if score > best_score:
            best_score = score
            best_gamma = gamma
            best_cluster = result
    print(best_score)
    return best_gamma, best_cluster


def kkm_rbf_algo(X, num_clus=6, max_iter=300, gamma=0.1):
    N = np.shape(X)[0]
    y = rng.integers(low=0, high=num_clus, size=N)

    kernel = kernel = lambda X: metrics.pairwise.rbf_kernel(X, gamma=gamma)
    K = kernel(X)

    for _ in range(max_iter):
        obj = np.tile(np.diag(K).reshape((-1, 1)), num_clus)
        N_c = np.bincount(y, minlength=num_clus)
        for c in range(num_clus):
            obj[:, c] -= 2 * np.sum((K)[:, y == c], axis=1) / N_c[c]
            obj[:, c] += np.sum((K)[y == c][:, y == c]) / (N_c[c] ** 2)
        y = np.argmin(obj, axis=1)
    return y


def kkm_rbf(X, num_clus=6, max_iter=300, gamma=0.1):
    # Load seed
    global rng
    rng = default_rng(load_seed()["np.random.default_rng"])
    #

    best_gamma, best_cluster = param_search_gamma(
        X, num_clus=num_clus, metric=metrics.calinski_harabasz_score)
    print(best_gamma)
    return best_cluster
