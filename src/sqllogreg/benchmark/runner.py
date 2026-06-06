from sqllogreg.metrics.result import BenchmarkResult, TrainResult
from sqllogreg.optimizers.base import BaseOptimizer


class BenchmarkRunner:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def run(self, optimizers, X_train, y_train, X_test, y_test):
        results = []

        for optimizer in optimizers:
            try:
                print(f"Running {optimizer.__class__.__name__}...")

                train_result = optimizer.fit(X_train, y_train)

                y_train_prob = BaseOptimizer.predict(
                    X_train, train_result.weights, train_result.bias
                )
                y_train_pred = (y_train_prob >= 0.5).astype(int)
                train_metrics = self.evaluator.calculate_metrics(
                    y_train, y_train_pred, y_train_prob
                )

                y_test_prob = BaseOptimizer.predict(
                    X_test, train_result.weights, train_result.bias
                )
                y_test_pred = (y_test_prob >= 0.5).astype(int)
                test_metrics = self.evaluator.calculate_metrics(
                    y_test, y_test_pred, y_test_prob
                )

                result = TrainResult(
                    optimizer_name=train_result.optimizer_name,
                    weights=train_result.weights,
                    bias=train_result.bias,
                    train_time=train_result.train_time,
                    iterations=train_result.iterations,
                    final_loss=train_result.final_loss,
                    train_auc=train_metrics["auc"],
                    train_f1=train_metrics["f1"],
                    test_auc=test_metrics["auc"],
                    test_f1=test_metrics["f1"],
                    convergence_info=train_result.convergence_info,
                )

                results.append(result)
                print(f"  Completed in {train_result.train_time:.3f}s")

            except Exception as e:
                print(f"  Error: {e}")
                continue

        return BenchmarkResult(results=results)
