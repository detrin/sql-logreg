import numpy as np
import pytest
from sqlalchemy import create_engine
from sqllogreg.optimizers.lbfgs_sql import LBFGSSQLOptimizer

@pytest.fixture
def db_engine():
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/sqllogreg")
    return engine

def test_lbfgs_sql_converges(db_engine):
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    opt = LBFGSSQLOptimizer(db_engine, max_iter=20)
    result = opt.fit(X, y)

    assert result.optimizer_name == "LBFGS_SQL"
    assert result.iterations <= 20
    assert result.final_loss < 1.0
    assert len(result.weights) == 5

def test_lbfgs_sql_uses_arrays(db_engine):
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    opt = LBFGSSQLOptimizer(db_engine, max_iter=10, table_prefix="test_lbfgs_sql")
    result = opt.fit(X, y)

    assert result.optimizer_name == "LBFGS_SQL"
    assert len(result.weights) == 4
    assert result.convergence_info["method"] == "sql_lbfgs"
