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
        result = dict(self.__dict__)
        result['weights'] = self.weights.tolist()
        result['bias'] = float(result['bias'])
        return result

@dataclass
class BenchmarkResult:
    results: list[TrainResult]

    def to_dataframe(self):
        data = []
        for r in self.results:
            d = r.to_dict()
            d['optimizer'] = d.pop('optimizer_name')
            d.pop('weights', None)
            d.pop('convergence_info', None)
            data.append(d)
        return pd.DataFrame(data)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def summary(self):
        df = self.to_dataframe()
        return df.to_string(index=False)
