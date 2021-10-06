import numpy as np
from numpy.random import default_rng

from utils.seed_handler import load_seed

# take L2 norm of data (per sample) first, so that
# polynomial kernel computes linear correlation
def d2_kernel(x, y):
    return np.dot(x, y.T) ** 2


def kernel_k_means_poly(data, num_clus=5, kernel=d2_kernel, max_iters=100):
    # Load seed
    rng = default_rng(load_seed()["np.random.default_rng"])
    #

    num_samps, num_feats = data.shape
    init = rng.choice(num_samps, num_clus, replace=False)
    # Normalise first!
    newdata = (data.T/np.linalg.norm(data, axis=1)).T
    inner_prods = kernel(newdata, newdata)
    left = np.tile(np.diag(inner_prods)[:, np.newaxis], (1, num_clus))
    distances = (
        left
        - 2 * inner_prods[:, init]
        + np.tile(inner_prods[init, init], (num_samps, 1))
    )
    labels = np.argmin(distances, axis=1)
    for itr in range(max_iters):
        # compute kernel distance using ||x - mu|| = k(x,x) - 2k(x,mu).mean() + k(mu,mu).mean() = left - 2*middle + right
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
        new_labels = np.argmin(distances, axis=1)
        if (labels == new_labels).all():
            print("converged")
            break
        print("iteration {} with cluster sizes {}".format(itr, counts))
        labels = new_labels
    return labels
