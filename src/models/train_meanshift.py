from sklearn.cluster import MeanShift


def train_meanshift(X, bandwidth=None):
    model = MeanShift(bandwidth=bandwidth)

    labels = model.fit_predict(X)

    return model, labels