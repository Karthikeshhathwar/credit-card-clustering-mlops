from sklearn.cluster import MeanShift

def train_meanshift(X):
    model = MeanShift()
    labels = model.fit_predict(X)
    return model, labels