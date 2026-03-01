import numpy as np
from sqllogreg.optimizers.gradient import GradientDescent

def test_gradient_descent_converges():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    opt = GradientDescent(lr=0.1, C=0.01, max_iter=100)
    result = opt.fit(X, y)

    assert result.optimizer_name == "GradientDescent"
    assert result.iterations <= 100
    assert result.final_loss < 1.0
    assert len(result.weights) == 5
