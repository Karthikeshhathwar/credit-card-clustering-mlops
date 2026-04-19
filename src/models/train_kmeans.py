from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# 🔹 Train KMeans Model
# =========================================================
def train_kmeans(X, n_clusters, n_init=10, max_iter=300):
    """
    Train KMeans clustering model

    Parameters:
    - X: input data
    - n_clusters: number of clusters
    - n_init: number of initializations
    - max_iter: max iterations

    Returns:
    - trained KMeans model
    """

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=42
    )

    model.fit(X)

    return model


# =========================================================
# 🔹 Find Best K using Silhouette Score
# =========================================================
def find_best_k(X, k_range):
    """
    Find optimal number of clusters using silhouette score

    Parameters:
    - X: input data
    - k_range: tuple/list (start, end) e.g. [2, 10]

    Returns:
    - best_k
    - best_score
    """

    best_k = k_range[0]
    best_score = -1

    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score