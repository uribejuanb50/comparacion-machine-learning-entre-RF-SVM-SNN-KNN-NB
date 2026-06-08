from src.models.base import model as Modelo_base
from src.models.RFmodel import RFmodel
from src.models.ANNmodel import ANNmodel
from src.models.KNNmodel import KNNmodel
from src.models.SVMmodel import SVMmodel

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def entrenar_modelos(semilla, valores, hp_rf, hp_ann, hp_knn, hp_svm, hp_nb = None) :
    x_train = valores["x_train"]
    y_train = valores["y_train"]
    x_validar = valores["x_validar"]
    y_validar = valores["y_validar"]
    x_test = valores["x_test"]
    y_test = valores["y_test"]
    features = valores["features"]

    #dummy, el piso, si no se supera el dummy implica que nuestro problema no es de ml
    #logreg, implica que es el modelo más básico de aprendizaje, si no se supera hay que evaluar la complejidad de nuestro modelo
    '''
    dummy = DummyClassifier(strategy = "most_frequent", random_state = semilla)
    dummy.fit(x_train, y_train)

    logreg = LogisticRegression(max_iter = 1000, random_state = semilla)
    logreg.fit(x_train, y_train)

    rf_model : Modelo_base  = RFmodel(**hp_rf)
    rf_model.fit(x_train, y_train, x_validar, y_validar)

    ann_model : Modelo_base = ANNmodel(**hp_ann)
    ann_model.fit(x_train, y_train, x_validar, y_validar)

    knn_model : Modelo_base = KNNmodel(**hp_knn)
    knn_model.fit(x_train, y_train, x_validar, y_validar)
    '''
    svm_model : Modelo_base = SVMmodel(**hp_svm)
    svm_model.fit(x_train, y_train, x_validar, y_validar)

    return {
        '''
        "Dummy" : dummy,
        "LogReg" : logreg,
        "RF" : rf_model,
        "ANN" : ann_model,
        "KNN" : knn_model,
        ''' : 0,
        "SVM" : svm_model
    }
