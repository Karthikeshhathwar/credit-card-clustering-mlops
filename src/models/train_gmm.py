from sklearn.mixture import GaussianMixture

def train_gmm(X, n_components=4):
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(X)
    labels = model.predict(X)
    return model, labels