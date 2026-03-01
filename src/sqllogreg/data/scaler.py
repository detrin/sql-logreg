from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self):
        self._scaler = StandardScaler()

    def fit_transform(self, X):
        return self._scaler.fit_transform(X)

    def transform(self, X):
        return self._scaler.transform(X)
