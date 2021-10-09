from sklearn.cluster import KMeans


def plain_kmeans(X, num_clus=6):
    kmeans_setup = KMeans(n_clusters=num_clus).fit(X)
    return kmeans_setup.labels_
