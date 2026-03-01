import numpy as np
import time
from sqlalchemy import text
from sqllogreg.optimizers.base import BaseOptimizer
from sqllogreg.metrics.result import TrainResult

class SQLOptimizer(BaseOptimizer):
    def __init__(self, engine, learning_rate=0.01, C=0.1, max_iter=1000,
                 table_prefix="sql_logreg"):
        self.engine = engine
        self.learning_rate = learning_rate
        self.C = C
        self.max_iter = max_iter
        self.table_prefix = table_prefix
        self.data_table = f"{table_prefix}_data"
        self.coef_table = f"{table_prefix}_coef"
        self.conv_table = f"{table_prefix}_convergence"

    def fit(self, X, y):
        start_time = time.time()

        n_features = X.shape[1]
        self._create_tables(X, y, n_features)
        self._initialize_coefficients(n_features)
        iterations = self._train_loop()
        weights, bias = self._extract_coefficients()

        train_time = time.time() - start_time

        y_pred = self.predict(X, weights, bias)
        final_loss = self._compute_loss(y, y_pred, weights)

        return TrainResult(
            optimizer_name="SQL",
            weights=weights,
            bias=bias,
            train_time=train_time,
            iterations=iterations,
            final_loss=final_loss,
            train_auc=0.0,
            train_f1=0.0,
            test_auc=0.0,
            test_f1=0.0,
            convergence_info={
                "method": "raw_sql_gradient_descent",
                "learning_rate": self.learning_rate
            }
        )

    def _create_tables(self, X, y, n_features):
        n_samples = X.shape[0]

        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.data_table}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {self.coef_table}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {self.conv_table}"))
            conn.commit()

            col_defs = ", ".join([f"x{i} FLOAT" for i in range(n_features)])
            create_data = f"""
                CREATE TABLE {self.data_table} (
                    id SERIAL PRIMARY KEY,
                    {col_defs},
                    y INT
                )
            """
            conn.execute(text(create_data))
            conn.commit()

            for i in range(n_samples):
                values = ", ".join(map(str, X[i])) + f", {int(y[i])}"
                insert_query = f"""
                    INSERT INTO {self.data_table} ({", ".join([f"x{j}" for j in range(n_features)])}, y)
                    VALUES ({values})
                """
                conn.execute(text(insert_query))
            conn.commit()

            coef_cols = ", ".join([f"w{i} FLOAT" for i in range(n_features)])
            create_coef = f"""
                CREATE TABLE {self.coef_table} (
                    id SERIAL PRIMARY KEY,
                    {coef_cols},
                    bias FLOAT
                )
            """
            conn.execute(text(create_coef))
            conn.commit()

            conn.execute(text(f"""
                CREATE TABLE {self.conv_table} (
                    iteration INT PRIMARY KEY,
                    loss FLOAT
                )
            """))
            conn.commit()

    def _initialize_coefficients(self, n_features):
        with self.engine.connect() as conn:
            cols = ", ".join([f"w{i}" for i in range(n_features)])
            values = ", ".join(["0"] * n_features)
            conn.execute(text(f"""
                INSERT INTO {self.coef_table} ({cols}, bias)
                VALUES ({values}, 0)
            """))
            conn.commit()

    def _train_loop(self):
        for iteration in range(self.max_iter):
            self._update_coefficients()
            loss = self._compute_current_loss()

            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.conv_table} (iteration, loss)
                    VALUES ({iteration}, {loss})
                """))
                conn.commit()

        return self.max_iter

    def _update_coefficients(self):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT COUNT(*) FROM {self.coef_table}
            """)).fetchone()
            n_features = len([c for c in conn.execute(text(f"SELECT * FROM {self.data_table} LIMIT 1")).keys() if c.startswith('x')])

            weight_cols = ", ".join([f"w{i}" for i in range(n_features)])
            feature_cols = ", ".join([f"x{i}" for i in range(n_features)])

            logit_expr = " + ".join([f"c.w{i} * d.x{i}" for i in range(n_features)])

            grad_updates = []
            for i in range(n_features):
                grad_updates.append(f"""
                    c.w{i} - {self.learning_rate} * (
                        (SELECT AVG((1.0 / (1.0 + EXP(-({logit_expr} + c.bias))) - d.target) * d.x{i})
                         FROM {self.data_table} d, {self.coef_table} c2
                         WHERE c2.id = (SELECT MAX(id) FROM {self.coef_table}))
                    ) AS w{i}
                """)

            grad_updates_str = ", ".join(grad_updates)

            conn.execute(text(f"""
                INSERT INTO {self.coef_table} ({weight_cols}, bias)
                SELECT {grad_updates_str},
                       c.bias - {self.learning_rate} * (
                           SELECT AVG(1.0 / (1.0 + EXP(-({logit_expr} + c.bias))) - d.target)
                           FROM {self.data_table} d
                       )
                FROM {self.coef_table} c
                WHERE c.id = (SELECT MAX(id) FROM {self.coef_table})
            """))
            conn.commit()

    def _compute_current_loss(self):
        with self.engine.connect() as conn:
            n_features = len([c for c in conn.execute(text(f"SELECT * FROM {self.data_table} LIMIT 1")).keys() if c.startswith('x')])
            logit_expr = " + ".join([f"c.w{i} * d.x{i}" for i in range(n_features)])

            result = conn.execute(text(f"""
                SELECT AVG(
                    -1.0 * (
                        d.target * LN(1.0 / (1.0 + EXP(-({logit_expr} + c.bias))) + 1e-9) +
                        (1 - d.target) * LN(1 - 1.0 / (1.0 + EXP(-({logit_expr} + c.bias))) + 1e-9)
                    )
                ) as loss
                FROM {self.data_table} d, {self.coef_table} c
                WHERE c.id = (SELECT MAX(id) FROM {self.coef_table})
            """)).fetchone()

            return float(result[0]) if result[0] is not None else 0.0

    def _extract_coefficients(self):
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {self.coef_table} ORDER BY id DESC LIMIT 1")
            ).fetchone()

            weights = np.array(result[1:-1])
            bias = result[-1]

            return weights, bias

    def _compute_loss(self, y_true, y_pred, weights):
        eps = 1e-9
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        l2_penalty = 0.5 * self.C * np.sum(weights ** 2)
        return bce + l2_penalty
