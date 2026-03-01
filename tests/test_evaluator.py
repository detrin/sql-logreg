import numpy as np
from sqllogreg.metrics.evaluator import Evaluator


def test_calculate_metrics():
    evaluator = Evaluator()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0


def test_calculate_metrics_imperfect():
    evaluator = Evaluator()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.8, 0.1, 0.7, 0.9])

    metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)

    assert 0 < metrics["accuracy"] < 1.0
    assert 0 < metrics["f1"] < 1.0
    assert 0 < metrics["auc"] < 1.0
