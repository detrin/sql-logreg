import numpy as np
import time
from sqllogreg.optimizers.base import BaseOptimizer
from sqllogreg.metrics.result import TrainResult


class GradientDescent(BaseOptimizer):
    def __init__(self, lr=0.01, C=0.1, tol=1e-4, max_iter=1000, decay=0.0):
        self.lr = lr
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.decay = decay

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape

        weights = np.zeros(n_features)
        bias = 0.0

        losses = []
        for i in range(self.max_iter):
            lr_current = self.lr / (1 + self.decay * i)

            y_pred = self.predict(X, weights, bias)
            loss = self._compute_loss(y, y_pred, weights)
            losses.append(loss)

            grad_w, grad_b = self._compute_gradients(X, y, y_pred, weights)

            if np.linalg.norm(grad_w) < self.tol:
                break

            weights -= lr_current * grad_w
            bias -= lr_current * grad_b

        train_time = time.time() - start_time

        return TrainResult(
            optimizer_name="GradientDescent",
            weights=weights,
            bias=bias,
            train_time=train_time,
            iterations=i + 1,
            final_loss=losses[-1],
            train_auc=0.0,
            train_f1=0.0,
            test_auc=0.0,
            test_f1=0.0,
            convergence_info={"losses": losses},
        )

    def _compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        bce = -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2_penalty = 0.5 * self.C * np.sum(weights**2)
        return bce + l2_penalty

    def _compute_gradients(self, X, y_true, y_pred, weights):
        n_samples = X.shape[0]
        diff = y_pred - y_true
        grad_w = (X.T @ diff) / n_samples + self.C * weights
        grad_b = np.mean(diff)
        return grad_w, grad_b
