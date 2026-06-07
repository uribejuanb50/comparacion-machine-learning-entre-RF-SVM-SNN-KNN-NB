import src.data.loader as loader
import src.experiments.preparation as preparation

from sklearn.metrics import confusion_matrix, classification_report

from src.config import (DATASET1,
                        DATASET2,
                        ESPERADOS,
                        TRAD_NAN,
                        INGRESO,
                        CARACTERISTICAS,
                        COLS_A_DROPEAR,
                        OBJETIVO,
                        SEMILLA,
                        CATEGORICAS,
                        NUMERICAS,
                        HIPERPARAMETROS_RF,
                        HIPERPARAMETROS_ANN)
from src.models.RFmodel import RFmodel
from src.models.ANNmodel import ANNmodel

def main():

    #dataframe = loader.leer_archivo(DATASET1, ESPERADOS)
    #print(dataframe)
    dataframe = loader.leer_archivo(DATASET2, ESPERADOS)

    dataframe = dataframe.drop(columns = COLS_A_DROPEAR)
    
    valores = preparation.preparacion(dataframe= dataframe,
                                      col_objetivo= OBJETIVO[0],
                                      diccionario= INGRESO,
                                      caracteristicas= CARACTERISTICAS,
                                      columna_codif= "class",
                                      semilla= SEMILLA,
                                      categoricas= CATEGORICAS,
                                      numericas= NUMERICAS,
                                      cols_dropear= COLS_A_DROPEAR)
    x_train = valores["x_train"]

    primera_linea = x_train[0].toarray().flatten().tolist()
    cabeceras = valores["features"]

    for cabecera, valor in zip(cabeceras, primera_linea) :
        print(f"{cabecera} - {valor}")

    #rf_model = RFmodel(**HIPERPARAMETROS_RF)
    #rf_model.fit(valores["x_train"], valores["y_train"], valores["x_validar"], valores["y_validar"])
    ann_model = ANNmodel(**HIPERPARAMETROS_ANN)
    ann_model.fit(valores["x_train"], valores["y_train"], valores["x_validar"], valores["y_validar"])
    print(f"[main] predict:\n{ann_model.predict(valores['x_test'])}")
    print(f"[main] predict_proba:\n{ann_model.predict_proba(valores['x_test'])}")

    y_pred = ann_model.predict(valores["x_test"])

    print(confusion_matrix(valores["y_test"], y_pred))
    print(classification_report(valores["y_test"], y_pred))



if __name__ == "__main__" :
    main()