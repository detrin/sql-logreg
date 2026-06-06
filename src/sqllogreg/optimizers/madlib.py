import numpy as np
import time
from sqlalchemy import text
from sqllogreg.optimizers.base import BaseOptimizer
from sqllogreg.metrics.result import TrainResult


class MADlib(BaseOptimizer):
    def __init__(self, engine, C=0.1, max_iter=1000, table_name="madlib_logreg"):
        self.engine = engine
        self.C = C
        self.max_iter = max_iter
        self.table_name = table_name

    def fit(self, X, y):
        start_time = time.time()

        self._create_training_table(X, y)
        self._train_model()
        weights, bias = self._extract_coefficients()

        train_time = time.time() - start_time

        y_pred = self.predict(X, weights, bias)
        final_loss = self._compute_loss(y, y_pred, weights)

        return TrainResult(
            optimizer_name="MADlib",
            weights=weights,
            bias=bias,
            train_time=train_time,
            iterations=self.max_iter,
            final_loss=final_loss,
            train_auc=0.0,
            train_f1=0.0,
            test_auc=0.0,
            test_f1=0.0,
            convergence_info={
                "method": "madlib.logregr_train",
                "table": self.table_name,
            },
        )

    def _create_training_table(self, X, y):
        n_samples, n_features = X.shape

        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}"))
            conn.commit()

            create_query = f"""
                CREATE TABLE {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    features FLOAT[],
                    target INT
                )
            """
            conn.execute(text(create_query))
            conn.commit()

            for i in range(n_samples):
                features_str = "{" + ",".join(map(str, X[i])) + "}"
                insert_query = f"""
                    INSERT INTO {self.table_name} (features, target)
                    VALUES ('{features_str}', {int(y[i])})
                """
                conn.execute(text(insert_query))
            conn.commit()

    def _train_model(self):
        lambda_val = self.C

        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}_model"))
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}_model_summary"))
            conn.commit()

            train_query = f"""
                SELECT madlib.logregr_train(
                    '{self.table_name}',
                    '{self.table_name}_model',
                    'target',
                    'features',
                    NULL,
                    {self.max_iter},
                    'irls',
                    1e-4,
                    TRUE,
                    {lambda_val}
                )
            """
            conn.execute(text(train_query))
            conn.commit()

    def _extract_coefficients(self):
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT coef FROM {self.table_name}_model")
            ).fetchone()

            coef_array = np.array(result[0])
            weights = coef_array[:-1]
            bias = coef_array[-1]

            return weights, bias

    def _compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        bce = -np.mean(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
        )
        l2_penalty = 0.5 * self.C * np.sum(weights**2)
        return bce + l2_penalty
