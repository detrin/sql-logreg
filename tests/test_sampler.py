import numpy as np
from sqllogreg.data.sampler import Sampler


def test_sampler():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 0, 1])

    sampler = Sampler()
    X_resampled, y_resampled = sampler.resample(X, y)

    assert np.sum(y_resampled == 0) == np.sum(y_resampled == 1)
    assert len(y_resampled) >= len(y)
