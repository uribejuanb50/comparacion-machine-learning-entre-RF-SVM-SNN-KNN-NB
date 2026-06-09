from src.models.base import model as Modelo_base
from src.models.RFmodel import RFmodel
from src.models.ANNmodel import ANNmodel
from src.models.KNNmodel import KNNmodel
from src.models.SVMmodel import SVMmodel
from src.models.NBmodel import NBmodel


def entrenar_modelos(semilla, valores, hp_rf, hp_ann, hp_knn, hp_svm, hp_nb) :
    x_train = valores["x_train"]
    y_train = valores["y_train"]
    x_validar = valores["x_validar"]
    y_validar = valores["y_validar"]
    x_test = valores["x_test"]
    y_test = valores["y_test"]
    features = valores["features"]

    rf_model : Modelo_base  = RFmodel(**hp_rf)
    rf_model.fit(x_train, y_train, x_validar, y_validar)

    ann_model : Modelo_base = ANNmodel(**hp_ann)
    ann_model.fit(x_train, y_train, x_validar, y_validar)

    knn_model : Modelo_base = KNNmodel(**hp_knn)
    knn_model.fit(x_train, y_train, x_validar, y_validar)
    
    svm_model : Modelo_base = SVMmodel(**hp_svm)
    svm_model.fit(x_train, y_train, x_validar, y_validar)

    nb_model : Modelo_base = NBmodel(**hp_nb)
    nb_model.fit(x_train, y_train, x_validar, y_validar)
    
    return {
        "RF" : rf_model,
        "ANN" : ann_model,
        "KNN" : knn_model,
        "SVM" : svm_model,
        "NB" : nb_model
    }
