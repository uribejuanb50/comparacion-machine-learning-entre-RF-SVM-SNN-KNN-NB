import joblib
import pandas as pd

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from src.models.base import model
from src.data.splitter import submuestreo_estratificado


class KNNmodel(model):

    def __init__(self, n_jobs, semilla, n_muestras, dict) :
        self.entrenado = False

        self.model = KNeighborsClassifier(n_jobs = n_jobs)
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
                            n_jobs = self.n_jobs
                            )
        
        grid.fit(x_train_copia, y_train_copia)

        self.modelo = grid.best_estimator_
        self.entrenado = True

        print(f"[KNNmodel] Mejores params: {grid.best_params_}")
        print(f"[KNNmodel] Puntaje f1 en Cross Validation (CV): {grid.best_score_:.4f}")
        print("[KNNmodel] Entrenamiento terminado")
        return

    def predict(self, x_test):
        pass

    def predict_proba(self, x_test):
        return 

    def save(self, path):
        pass

    def load(self, path):
        return
        