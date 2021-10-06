# Copyleft 2021, Alex Markham, see https://medil.causal.dev/license.html
# Tested with versions:
# python: 3.9.5
# numpy: 1.20.3
# scipy: 1.6.3
import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA

from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2

from sklearn import metrics

from utils.seed_handler import load_seed

rng = None


def dep_contrib_kernel(X, alpha=0.05):
    num_samps, num_feats = X.shape
    thresh = np.eye(num_feats)
    if alpha is not None:
        thresh[thresh == 0] = (
            chi2(1).ppf(1 - alpha) / num_samps
        )  # critical value corresponding to alpha
        thresh[thresh == 1] = 0
    Z = np.zeros((num_feats, num_samps, num_samps))
    for j in range(num_feats):
        n = num_samps
        t = np.tile
        D = squareform(pdist(X[:, j].reshape(-1, 1), "cityblock"))
        D_bar = D.mean()
        D -= (
            t(D.mean(0), (n, 1)) + t(D.mean(1), (n, 1)).T - t(D_bar, (n, n))
        )  # doubly centered
        Z[j] = D / (D_bar)  # standardized
    F = Z.reshape(num_feats * num_samps, num_samps)
    left = np.tensordot(Z, thresh, axes=([0], [0]))
    left_right = np.tensordot(left, Z, axes=([2, 1], [0, 1]))
    gamma = (F.T @ F) ** 2 - 2 * (left_right) + \
        LA.norm(thresh)  # helper kernel

    diag = np.diag(gamma)
    kappa = gamma / np.sqrt(np.outer(diag, diag))  # cosine similarity
    kappa[kappa > 1] = 1  # correct numerical errors
    return kappa


def kernel_k_means_algo(data, num_clus=6, kernel=dep_contrib_kernel, max_iters=100, alpha=0.05):
    num_samps, num_feats = data.shape
    init = rng.choice(num_samps, num_clus, replace=False)  # use Forgy method
    inner_prods = kernel(data, alpha=alpha)
    left = np.tile(np.diag(inner_prods)[:, np.newaxis], (1, num_clus))
    distances = (
        left
        - 2 * inner_prods[:, init]
        + np.tile(inner_prods[init, init], (num_samps, 1))
    )
    # use law of cosines to get angle instead of Euc dist
    # print(np.min(1 - (distances ** 2 / 2)))
    # print(np.max(1 - (distances ** 2 / 2)))

    arc_distances = np.arccos(np.clip(1 - (distances ** 2 / 2), -1, 1))
    labels = np.argmin(arc_distances, axis=1)
    for itr in range(max_iters):
        # compute Euclidean kernel distance using ||x - mu||^2 = k(x,x) - 2k(x,mu).mean() + k(mu,mu).mean() = left - 2*middle + right
        ip_clus = np.tile(inner_prods, (num_clus, 1, 1))

        m_idx = np.fromiter(
            (j for c in range(num_clus) for i in labels for j in labels == c),
            bool,
            num_clus * num_samps ** 2,
        )
        m_idx = m_idx.reshape(num_clus, num_samps, num_samps)
        counts = np.fromiter(
            ((labels == label).sum()
             for label in range(num_clus)), int, num_clus
        )
        # counts = m_idx[:, 0, :].sum(1)
        ip_clus[~m_idx] = 0
        # sum/ counts, because 0s through off mean
        middle = ip_clus.sum(2).T / counts

        r_idx = np.fromiter(
            (
                (i and j)
                for c in range(num_clus)
                for i in labels == c
                for j in labels == c
            ),
            bool,
            num_clus * num_samps ** 2,
        )
        r_idx = r_idx.reshape(num_clus, num_samps, num_samps)
        ip_clus[~r_idx] = 0
        right = ip_clus.sum((1, 2)) / (counts ** 2)

        distances = left - 2 * middle + right
        # law of cosines
        arc_distances = np.arccos(np.clip(1 - (distances ** 2 / 2), -1, 1))
        new_labels = np.argmin(arc_distances, axis=1)
        if (labels == new_labels).all():
            print("converged")
            break
        print("iteration {} with cluster sizes {}".format(itr, counts))
        labels = new_labels
    return labels


def param_search_alpha(X, sr_range=[0.005, 0.01, 0.02, 0.03, 0.042, 0.043, 0.045, 0.047, 0.05, 0.053, 0.055, 0.057, 0.058, 0.07, 0.08, 0.09, 0.1], num_clus=6, metric=metrics.calinski_harabasz_score):
    scores = []
    best_score = -1
    best_alpha = -1
    best_cluster = None
    for alpha in sr_range:
        result = kernel_k_means_algo(
            X, num_clus=num_clus, max_iters=300, alpha=alpha)
        if np.unique(result, return_counts=False).size == 1:
            score = -1
        else:
            score = metric(X, result)
        scores += [score]
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_cluster = result
    print(best_score)
    return best_alpha, best_cluster


def kernel_k_means(data, num_clus=6, kernel=dep_contrib_kernel, max_iters=100, alpha=0.05):
    # Load seed
    global rng
    rng = default_rng(load_seed()["np.random.default_rng"])
    #

    best_alpha, best_cluster = param_search_alpha(
        data, num_clus=num_clus, metric=metrics.calinski_harabasz_score)
    print(best_alpha)
    return best_cluster
