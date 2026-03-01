import numpy as np
import time
from scipy.optimize import minimize
from sqllogreg.optimizers.base import BaseOptimizer
from sqllogreg.metrics.result import TrainResult


class LBFGS(BaseOptimizer):
    def __init__(self, C=0.1, tol=1e-4, max_iter=1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape

        self.X = X
        self.y = y
        self.n_samples = n_samples

        params_init = np.zeros(n_features + 1)

        result = minimize(
            fun=self._objective,
            x0=params_init,
            method="L-BFGS-B",
            jac=self._gradient,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        weights = result.x[:-1]
        bias = result.x[-1]

        train_time = time.time() - start_time

        y_pred = self.predict(X, weights, bias)
        final_loss = self._compute_loss(y, y_pred, weights)

        return TrainResult(
            optimizer_name="LBFGS",
            weights=weights,
            bias=bias,
            train_time=train_time,
            iterations=result.nit,
            final_loss=final_loss,
            train_auc=0.0,
            train_f1=0.0,
            test_auc=0.0,
            test_f1=0.0,
            convergence_info={
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
            },
        )

    def _objective(self, params):
        weights = params[:-1]
        bias = params[-1]

        y_pred = self.predict(self.X, weights, bias)
        return self._compute_loss(self.y, y_pred, weights)

    def _gradient(self, params):
        weights = params[:-1]
        bias = params[-1]

        y_pred = self.predict(self.X, weights, bias)
        diff = y_pred - self.y

        grad_w = (self.X.T @ diff) / self.n_samples + self.C * weights
        grad_b = np.mean(diff)

        return np.concatenate([grad_w, [grad_b]])

    def _compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        bce = -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2_penalty = 0.5 * self.C * np.sum(weights**2)
        return bce + l2_penalty
