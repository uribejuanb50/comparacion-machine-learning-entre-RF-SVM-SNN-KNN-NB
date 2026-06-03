import json
from src.models.base import model
from xgboost import XGBClassifier

class RFmodel(model) :
    def __init__(self, **kwargs) :
        self.hiperparametros = kwargs
        self.model = XGBClassifier(**self.hiperparametros)

    def fit(self, x_train, y_train, x_val = None, y_val = None) :

        print("[RFmodel] Empezando entrenamiento...")
        self.model.fit(
            X = x_train,
            y = y_train,
            eval_set = [(x_val, y_val)],
            verbose = True,
        )

        resultados = self.model.evals_result()
        print(f"[RFmodel] resultados:\n{json.dumps(resultados, indent = 2, default = str)}")

        print("\n[RFmodel] Terminó el entrenamiento")

        return
    
    def predict(self, x_train):
        pass

    def predict_proba(self, x_train):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


