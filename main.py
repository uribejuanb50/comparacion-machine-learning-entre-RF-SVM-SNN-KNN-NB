import src.data.loader as loader
import src.experiments.preparation as preparation
import src.experiments.training as training

from sklearn.metrics import confusion_matrix, classification_report

from src.config import (DATASET,
                        ESPERADOS,
                        TRAD_NAN,
                        INGRESO,
                        CARACTERISTICAS,
                        COLS_A_DROPEAR,
                        OBJETIVO,
                        SEMILLA,
                        CATEGORICAS,
                        NUMERICAS,
                        LOG_TRANSFORM,
                        HIPERPARAMETROS_RF,
                        HIPERPARAMETROS_ANN,
                        HIPERPARAMETROS_KNN)

from src.models.RFmodel import RFmodel
from src.models.ANNmodel import ANNmodel
from src.models.KNNmodel import KNNmodel


def main():

    dataframe = loader.leer_archivo(DATASET, ESPERADOS)

    valores = preparation.preparacion(dataframe= dataframe,
                                      col_objetivo= OBJETIVO[0],
                                      diccionario= INGRESO,
                                      caracteristicas= CARACTERISTICAS,
                                      columna_codif= "income",
                                      semilla= SEMILLA,
                                      categoricas= CATEGORICAS,
                                      numericas= NUMERICAS,
                                      log_transform= LOG_TRANSFORM,
                                      cols_dropear= COLS_A_DROPEAR)
    
    modelos = training.entrenar_modelos(semilla= SEMILLA,
                                        valores= valores,
                                        hp_rf= HIPERPARAMETROS_RF,
                                        hp_ann= HIPERPARAMETROS_ANN,
                                        hp_knn= HIPERPARAMETROS_KNN)

    model = modelos["ANN"]
    
    print(f"[main] predict:\n{model.predict(valores['x_test'])}")
    print(f"[main] predict_proba:\n{model.predict_proba(valores['x_test'])}")

    y_pred = model.predict(valores["x_test"])

    print(confusion_matrix(valores["y_test"], y_pred))
    print(classification_report(valores["y_test"], y_pred))



if __name__ == "__main__" :
    main()