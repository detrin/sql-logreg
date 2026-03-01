import numpy as np
from sqllogreg.benchmark.runner import BenchmarkRunner
from sqllogreg.metrics.evaluator import Evaluator
from sqllogreg.optimizers.gradient import GradientDescent

def test_benchmark_runner():
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(20, 5)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    runner = BenchmarkRunner(Evaluator())
    optimizers = [GradientDescent(lr=0.1, max_iter=50)]

    result = runner.run(optimizers, X_train, y_train, X_test, y_test)

    assert len(result.results) == 1
    assert result.results[0].optimizer_name == "GradientDescent"
