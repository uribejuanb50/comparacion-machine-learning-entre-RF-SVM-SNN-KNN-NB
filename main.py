import src.data.loader as loader
import src.experiments.preparation as preparation

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
                        HIPERPARAMETROS_ANN)

from src.models.RFmodel import RFmodel
from src.models.ANNmodel import ANNmodel

def main():

    #dataframe = loader.leer_archivo(DATASET1, ESPERADOS)
    #print(dataframe)
    dataframe = loader.leer_archivo(DATASET, ESPERADOS)
    
    print(dataframe["income"].unique())
    print(dataframe["income"].value_counts())

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
    x_train = valores["x_train"]

    primera_linea = x_train[0].toarray().flatten().tolist()
    cabeceras = valores["features"]

    for cabecera, valor in zip(cabeceras, primera_linea) :
        print(f"{cabecera} - {valor}")

    model = RFmodel(**HIPERPARAMETROS_RF)
    model.fit(valores["x_train"], valores["y_train"], valores["x_validar"], valores["y_validar"])
    #model = ANNmodel(**HIPERPARAMETROS_ANN)
    #model.fit(valores["x_train"], valores["y_train"], valores["x_validar"], valores["y_validar"])
    print(f"[main] predict:\n{model.predict(valores['x_test'])}")
    print(f"[main] predict_proba:\n{model.predict_proba(valores['x_test'])}")

    y_pred = model.predict(valores["x_test"])

    print(confusion_matrix(valores["y_test"], y_pred))
    print(classification_report(valores["y_test"], y_pred))



if __name__ == "__main__" :
    main()