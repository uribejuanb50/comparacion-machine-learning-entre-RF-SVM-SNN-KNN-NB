import joblib
import pandas as pd

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

from src.models.base import model
from src.data.splitter import submuestreo_estratificado


class KNNmodel(model):

    def __init__(self, n_neighbors, weights, metric, p, algorithm, n_jobs, semilla, n_muestras) :
        self.entrenado = False

        self.model = KNeighborsClassifier(n_neighbors = n_neighbors,
                                          weights = weights,
                                          metric = metric,
                                          p = p,
                                          algorithm = algorithm,
                                          n_jobs = n_jobs)
        self.semilla = semilla
        self.n_muestras = n_muestras
    
    def fit(self, x_train, y_train, x_validar = None, y_validar = None) :

        x_train_copia, y_train_copia = submuestreo_estratificado(x_train, y_train, self.n_muestras, self.semilla)


        pass

    def predict(self, x_test):
        pass

    def predict_proba(self, x_test):
        return 

    def save(self, path):
        pass

    def load(self, path):
        return
        