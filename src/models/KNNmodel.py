import joblib

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

from src.models.base import model


class KNNmodel(model):

    def __init__(self, n_neighbors, weights, metric, p, algorithm, n_jobs) :
        self.entrenado = False
        
        self.model = KNeighborsClassifier(n_neighbors = n_neighbors,
                                          weights = weights,
                                          metric = metric,
                                          p = p,
                                          algorithm = algorithm,
                                          n_jobs = n_jobs)
    
    def fit(self, x_train, y_train, x_validar = None, y_validar = None) :

        pass

    def predict():
        pass

    def save(self, path):
        pass

    def load(self, path):
        return
        