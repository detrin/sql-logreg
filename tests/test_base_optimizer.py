import numpy as np
from sqllogreg.optimizers.base import BaseOptimizer


def test_predict():
    X = np.array([[0.5], [-0.5]])
    weights = np.array([1.0])
    bias = 0.0

    pred = BaseOptimizer.predict(X, weights, bias)

    assert pred.shape == (2,)
    assert pred[0] > 0.5
    assert pred[1] < 0.5
