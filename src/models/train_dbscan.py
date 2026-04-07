from sklearn.cluster import DBSCAN

def train_dbscan(X, eps=1.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels