import numpy as np
from sqllogreg.data.loader import DataLoader

def test_load_data():
    loader = DataLoader()
    X, y = loader.load("data/winequality-red.csv")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 11
    assert set(y) == {0, 1}
