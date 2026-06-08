from src.data.splitter import submuestreo_estratificado
from src.models.base import model

import joblib
import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMmodel(model):

    def __init__(self, n_muestras, probabilidad, semilla, dict, dict2) :
        super().__init__()
        self.n_muestras = n_muestras
        self.semilla = semilla
        self.param_grid = [dict, dict2]
        self.model = SVC(probability=probabilidad, random_state=semilla)

    def fit(self, x_train, y_train, x_validar, y_validar) :
        print("[SVMmodel] empezando entrenamiento")
        x_train_copia, y_train_copia = submuestreo_estratificado(x_train, y_train, self.n_muestras, self.semilla)

        grid = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=5, scoring="f1", verbose=3)
        grid.fit(x_train_copia, y_train_copia)

        self.model = grid.best_estimator_

        self.entrenado = True

        print(f"[SVMmodel] mejores params: {grid.best_params_}")
        print(f"[SVMmodel] puntaje f1 en cv: {grid.best_score_:.4f}")
        print("[SVMmodel] termino el entreno")

    def predict(self, x_test):
        super().verificar_entrenado()
        tiempo_inicio = time.perf_counter()

        prediccion = self.model.predict(x_test)

        tiempo_fin = time.perf_counter()
        self.tiempo_prediccion = tiempo_fin - tiempo_inicio

        return prediccion
    
    def predict_proba(self, x_test) :
        super().verificar_entrenado()
        return self.model.predict_proba(x_test)
    
    def save(self, path) :
        joblib.dump(self.model, path)
        print(f"[SVMmodel] modelo guardado en {path}")
        return 
    
    def load(self, path):
        self.model = joblib.load(path)
        print(f"[SVMmodel] cargado desde {path}")
        return




