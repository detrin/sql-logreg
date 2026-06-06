import numpy as np
from sqllogreg.data.scaler import Scaler


def test_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = Scaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
