import json
import time
import joblib
from src.models.base import model
from xgboost import XGBClassifier

class RFmodel(model) :
    def __init__(self, **kwargs) :
        super().__init__()

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

        self.entrenado = True
        print("\n[RFmodel] Terminó el entrenamiento")

        return
    
    def predict(self, x_train):
        super().verificar_entrenado()

        inicio_prediccion = time.perf_counter()

        prediccion = self.model.predict(x_train)

        fin_prediccion = time.perf_counter()
        self.tiempo_prediccion = fin_prediccion - inicio_prediccion

        return prediccion

    def predict_proba(self, x_train):
        super().verificar_entrenado()

        return self.model.predict_proba(x_train)

    def save(self, path):

        joblib.dump(self.model, path)
        print("[RFmodel] Modelo guardado exitosamente")

        return

    def load(self, path):

        self.model = joblib.load(path)
        print("[RFmodel] Modelo cargado exitosamente")
        
        return

    