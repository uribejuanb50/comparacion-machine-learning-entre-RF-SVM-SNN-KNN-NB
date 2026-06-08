from src.models.base import model

import joblib
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class NBmodel(model) : 

    def __init__(self, priors, dict) :
        super().__init__()
        self.priors = priors
        self.param_grid = dict
        self.model = GaussianNB(priors=self.priors)

    def fit(self, x_train, y_train, x_validar = None, y_validar = None) :
        print("[NBmodel] entrenamiento incido")

        x_train_denso = x_train.toarray()
        grid = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, scoring="f1", verbose=3)
        grid.fit(x_train_denso, y_train)

        self.model = grid.best_estimator_

        self.entrenado = True

        print(f"[NBmodel] los mejores parametros {grid.best_params_}")
        print(f"[NBmodel] el score f1 en cv fue {grid.best_score_:.4f}")
        print("[NBmodel] Finalizando entrenamiento")

        return
    
    def predict(self, x_test):
        super().verificar_entrenado()
        x_test_denso = x_test.toarray()
        tiempo_inicio = time.perf_counter()

        prediccion = self.model.predict(x_test_denso)

        tiempo_fin = time.perf_counter()
        self.tiempo_prediccion = tiempo_fin - tiempo_inicio

        return prediccion
    
    def predict_proba(self, x_test):
        super().verificar_entrenado()
        x_test_denso = x_test.toarray()
        return self.model.predict_proba(x_test_denso)
    
    def save(self, path) :
        joblib.dump(self.model, path)
        print(f"[NBmodel] modelo guardado en {path}")

    def load(self, path) :
        self.model = joblib.load(path)
        print(f"[NBmodel] modelo cargado desde {path}")



