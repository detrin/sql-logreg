import numpy as np
from sqllogreg.data.splitter import Splitter

def test_splitter():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    splitter = Splitter()
    X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=0.2)

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1
