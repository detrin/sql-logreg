import numpy as np
import time
from sqlalchemy import text
from sqllogreg.optimizers.base import BaseOptimizer
from sqllogreg.metrics.result import TrainResult

class LBFGSSQLOptimizer(BaseOptimizer):
    def __init__(self, engine, m=5, max_iter=100, table_prefix="lbfgs_sql_logreg", tol=1e-5):
        self.engine = engine
        self.m = m
        self.max_iter = max_iter
        self.table_prefix = table_prefix
        self.tol = tol
        self.data_table = f"{table_prefix}_data"
        self.state_table = f"{table_prefix}_state"
        self.history_table = f"{table_prefix}_history"

    def fit(self, X, y):
        start_time = time.time()

        n_features = X.shape[1]
        self._create_tables(X, y, n_features)
        self._initialize_state(n_features)

        iterations = 0
        for i in range(self.max_iter):
            grad_norm = self._compute_gradient()

            if grad_norm < self.tol:
                iterations = i + 1
                break

            self._update_position(n_features)
            iterations = i + 1

        weights, bias = self._extract_coefficients()
        train_time = time.time() - start_time

        y_pred = self.predict(X, weights, bias)
        final_loss = self._compute_loss(y, y_pred, weights)

        return TrainResult(
            optimizer_name="LBFGS_SQL",
            weights=weights,
            bias=bias,
            train_time=train_time,
            iterations=iterations,
            final_loss=final_loss,
            train_auc=0.0,
            train_f1=0.0,
            test_auc=0.0,
            test_f1=0.0,
            convergence_info={"method": "sql_lbfgs", "memory": self.m}
        )

    def _create_tables(self, X, y, n_features):
        n_samples = X.shape[0]

        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.data_table}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {self.state_table}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {self.history_table}"))
            conn.commit()

            col_defs = ", ".join([f"x{i} FLOAT" for i in range(n_features)])
            conn.execute(text(f"""
                CREATE TABLE {self.data_table} (
                    id SERIAL PRIMARY KEY,
                    {col_defs},
                    y INT
                )
            """))
            conn.commit()

            for i in range(n_samples):
                values = ", ".join(map(str, X[i])) + f", {int(y[i])}"
                conn.execute(text(f"""
                    INSERT INTO {self.data_table} ({", ".join([f"x{j}" for j in range(n_features)])}, y)
                    VALUES ({values})
                """))
            conn.commit()

            conn.execute(text(f"""
                CREATE TABLE {self.state_table} (
                    id SERIAL PRIMARY KEY,
                    weights FLOAT[],
                    bias FLOAT,
                    gradient FLOAT[],
                    grad_bias FLOAT
                )
            """))
            conn.commit()

            conn.execute(text(f"""
                CREATE TABLE {self.history_table} (
                    iteration INT PRIMARY KEY,
                    s FLOAT[],
                    y FLOAT[],
                    rho FLOAT
                )
            """))
            conn.commit()

    def _initialize_state(self, n_features):
        with self.engine.connect() as conn:
            zeros = "{" + ",".join(["0"] * n_features) + "}"
            conn.execute(text(f"""
                INSERT INTO {self.state_table} (weights, bias, gradient, grad_bias)
                VALUES ('{zeros}', 0, '{zeros}', 0)
            """))
            conn.commit()

    def _compute_gradient(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT array_length(weights, 1) FROM {self.state_table}
                WHERE id = (SELECT MAX(id) FROM {self.state_table})
            """)).fetchone()
            n_features = result[0]

            grad_queries = []
            for i in range(n_features):
                grad_queries.append(f"""
                    (SELECT AVG(
                        (1.0 / (1.0 + EXP(-(
                            {' + '.join([f's.weights[{j+1}] * d.x{j}' for j in range(n_features)])}
                            + s.bias
                        ))) - d.y) * d.x{i}
                    ) FROM {self.data_table} d, {self.state_table} s
                    WHERE s.id = (SELECT MAX(id) FROM {self.state_table}))
                """)

            grad_array = "ARRAY[" + ", ".join(grad_queries) + "]"

            conn.execute(text(f"""
                UPDATE {self.state_table}
                SET gradient = {grad_array},
                    grad_bias = (
                        SELECT AVG(
                            1.0 / (1.0 + EXP(-(
                                {' + '.join([f's.weights[{j+1}] * d.x{j}' for j in range(n_features)])}
                                + s.bias
                            ))) - d.y
                        ) FROM {self.data_table} d, {self.state_table} s
                        WHERE s.id = (SELECT MAX(id) FROM {self.state_table})
                    )
                WHERE id = (SELECT MAX(id) FROM {self.state_table})
            """))
            conn.commit()

            result = conn.execute(text(f"""
                SELECT
                    (SELECT SUM(g*g) FROM unnest(gradient) g) + grad_bias*grad_bias as norm_sq
                FROM {self.state_table}
                WHERE id = (SELECT MAX(id) FROM {self.state_table})
            """)).fetchone()

            return np.sqrt(result[0]) if result[0] else 0.0

    def _update_position(self, n_features):
        with self.engine.connect() as conn:
            alpha = 0.01

            update_expr = "ARRAY[" + ", ".join([
                f"s.weights[{i+1}] - {alpha} * s.gradient[{i+1}]"
                for i in range(n_features)
            ]) + "]"

            conn.execute(text(f"""
                INSERT INTO {self.state_table} (weights, bias, gradient, grad_bias)
                SELECT
                    {update_expr},
                    s.bias - {alpha} * s.grad_bias,
                    s.gradient,
                    s.grad_bias
                FROM {self.state_table} s
                WHERE s.id = (SELECT MAX(id) FROM {self.state_table})
            """))
            conn.commit()

    def _extract_coefficients(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT weights, bias
                FROM {self.state_table}
                WHERE id = (SELECT MAX(id) FROM {self.state_table})
            """)).fetchone()

            weights = np.array(result[0])
            bias = result[1]

            return weights, bias

    def _compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return bce
