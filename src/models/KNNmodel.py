import joblib
import time
import pandas as pd

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from src.models.base import model
from src.data.splitter import submuestreo_estratificado


class KNNmodel(model):

    def __init__(self, algorithm, n_jobs, semilla, n_muestras, dict) :
        super().__init__()

        self.model = KNeighborsClassifier(algorithm = algorithm, n_jobs = n_jobs)
        self.n_jobs = n_jobs
        self.param_grid = dict
        self.semilla = semilla
        self.n_muestras = n_muestras
    
    def fit(self, x_train, y_train, x_validar = None, y_validar = None) :
        print("[KNNmodel] Empezando entrenamiento...")

        x_train_copia, y_train_copia = submuestreo_estratificado(x_train, y_train, self.n_muestras, self.semilla)

        grid = GridSearchCV(estimator = self.model,
                            param_grid = self.param_grid,
                            cv = 5, #investigar por qué los folds
                            scoring = "f1",
                            n_jobs = self.n_jobs,
                            verbose = 3
                            )
        
        grid.fit(x_train_copia, y_train_copia)

        self.model = grid.best_estimator_
        self.entrenado = True

        print(f"[KNNmodel] Mejores params: {grid.best_params_}")
        print(f"[KNNmodel] Puntaje f1 en Cross Validation (CV): {grid.best_score_:.4f}")
        print("[KNNmodel] Entrenamiento terminado")
        
        return

    def predict(self, x_test):
        super().verificar_entrenado()

        inicio_predict = time.perf_counter()

        prediccion = self.model.predict(x_test)

        fin_predict = time.perf_counter()
        self.tiempo_prediccion = fin_predict - inicio_predict

        return prediccion

    def predict_proba(self, x_test):
        super().verificar_entrenado()

        return self.model.predict_proba(x_test)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"[KNNmodel] Modelo guardado en {path}")

        return

    def load(self, path):
        self.model = joblib.load(path)
        print(f"[KNNmodel] Modelo cargado desde {path}")

        return
        