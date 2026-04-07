from sklearn.cluster import AgglomerativeClustering

def train_hierarchical(X, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels