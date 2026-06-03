import src.data.loader as loader
import src.experiments.preparation as preparation

from src.config import (DATASET1,
                        DATASET2,
                        ESPERADOS,
                        TRAD_NAN,
                        COMESTIBLES,
                        CARACTERISTICAS,
                        OBJETIVO,
                        SEMILLA,
                        CATEGORICAS,
                        NUMERICAS,
                        HIPERPARAMETROS_RF)
from src.models.RFmodel import RFmodel

def main():

    #dataframe = loader.leer_archivo(DATASET1, ESPERADOS)
    #print(dataframe)
    dataframe = loader.leer_archivo(DATASET2, ESPERADOS)
    
    valores = preparation.preparacion(dataframe= dataframe,
                                      col_objetivo= OBJETIVO[0],
                                      diccionario= COMESTIBLES,
                                      caracteristicas= CARACTERISTICAS,
                                      columna_codif= "class",
                                      semilla= SEMILLA,
                                      categoricas= CATEGORICAS,
                                      numericas= NUMERICAS)
    x_train = valores["x_train"]

    primera_linea = x_train[0].toarray().flatten().tolist()
    cabeceras = valores["features"]

    for cabecera, valor in zip(cabeceras, primera_linea) :
        print(f"{cabecera} - {valor}")

    rf_model = RFmodel(**HIPERPARAMETROS_RF)
    rf_model.fit(valores["x_train"], valores["y_train"], valores["x_validar"], valores["y_validar"])
    print(f"[main] predict:\n{rf_model.predict(valores['x_test'])}")
    print(f"[main] predict_proba:\n{rf_model.predict_proba(valores['x_test'])}")



if __name__ == "__main__" :
    main()