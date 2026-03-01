import numpy as np
from sqllogreg.optimizers.lbfgs import LBFGS


def test_lbfgs_converges():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    opt = LBFGS(C=0.01)
    result = opt.fit(X, y)

    assert result.optimizer_name == "LBFGS"
    assert result.final_loss < 1.0
    assert len(result.weights) == 5
