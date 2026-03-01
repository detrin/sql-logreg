import pandas as pd
import numpy as np

class DataLoader:
    def load(self, path):
        data = pd.read_csv(path)
        data["target"] = (data["quality"] >= 6.5).astype(int)
        data = data.drop("quality", axis=1)

        cols_pred = data.columns[:-1].tolist()
        cols_pred = [self._clean_name(col) for col in cols_pred]
        data.columns = cols_pred + ["target"]

        X = data[cols_pred].values
        y = data["target"].values
        return X, y

    def _clean_name(self, name):
        return (name.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                    .replace("-", "_")
                    .lower())
