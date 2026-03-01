from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @staticmethod
    def predict(X, weights, bias):
        logits = X @ weights + bias
        return 1 / (1 + np.exp(-logits))
