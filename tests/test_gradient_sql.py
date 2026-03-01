import numpy as np
import pytest
from sqlalchemy import create_engine
from sqllogreg.optimizers.gradient_sql import GradientSQLOptimizer

@pytest.fixture
def db_engine():
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/sqllogreg")
    return engine

def test_gradient_sql_converges(db_engine):
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    opt = GradientSQLOptimizer(db_engine, learning_rate=0.01, max_iter=50)
    result = opt.fit(X, y)

    assert result.optimizer_name == "GradientSQL"
    assert result.iterations == 50
    assert result.final_loss < 1.0
    assert len(result.weights) == 5

def test_gradient_sql_creates_tables(db_engine):
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    opt = GradientSQLOptimizer(db_engine, max_iter=10, table_prefix="test_grad_sql")
    result = opt.fit(X, y)

    assert result.optimizer_name == "GradientSQL"
    assert len(result.weights) == 3
