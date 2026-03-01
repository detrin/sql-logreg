import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class Evaluator:
    def calculate_metrics(self, y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
        }
