from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    results = {}

    # Some models (DBSCAN) may give -1 labels (noise)
    if len(set(labels)) > 1:
        results["silhouette"] = silhouette_score(X, labels)
        results["davies_bouldin"] = davies_bouldin_score(X, labels)
        results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    else:
        results["silhouette"] = -1
        results["davies_bouldin"] = -1
        results["calinski_harabasz"] = -1

    return results