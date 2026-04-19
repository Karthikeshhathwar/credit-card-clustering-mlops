from sklearn.mixture import GaussianMixture


def train_gmm(X, n_components, covariance_type="full", reg_covar=1e-3):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=42
    )

    model.fit(X)
    labels = model.predict(X)

    return model, labels