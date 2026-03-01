from dataclasses import dataclass
import numpy as np
import pandas as pd
import json

@dataclass
class TrainResult:
    optimizer_name: str
    weights: np.ndarray
    bias: float
    train_time: float
    iterations: int
    final_loss: float
    train_auc: float
    train_f1: float
    test_auc: float
    test_f1: float
    convergence_info: dict

    def to_dict(self):
        return {
            "optimizer_name": self.optimizer_name,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "train_time": self.train_time,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
            "train_auc": self.train_auc,
            "train_f1": self.train_f1,
            "test_auc": self.test_auc,
            "test_f1": self.test_f1,
            "convergence_info": self.convergence_info,
        }

@dataclass
class BenchmarkResult:
    results: list

    def to_dataframe(self):
        data = []
        for r in self.results:
            data.append({
                "optimizer": r.optimizer_name,
                "train_time": r.train_time,
                "iterations": r.iterations,
                "final_loss": r.final_loss,
                "train_auc": r.train_auc,
                "train_f1": r.train_f1,
                "test_auc": r.test_auc,
                "test_f1": r.test_f1,
            })
        return pd.DataFrame(data)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def summary(self):
        df = self.to_dataframe()
        return df.to_string(index=False)
