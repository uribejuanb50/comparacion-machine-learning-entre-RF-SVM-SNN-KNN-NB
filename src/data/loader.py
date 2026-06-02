import pandas as pd

def leer_archivo(ruta_archivo, esperados) :

    dataframe = pd.read_csv(ruta_archivo, sep = ";")

    validar_schema(esperados, dataframe)
    validar_errores(dataframe)

    return dataframe

def validar_schema(esperados, dataframe) :

    columnas_actuales = set(dataframe.columns)
    set_esperados = set(esperados)

    if set_esperados != columnas_actuales :

        faltan = set_esperados - columnas_actuales
        sobran = columnas_actuales - set_esperados

        raise ValueError(
            f"[Loader] Los valores no coinciden (los de la columna)\n"
            f"[Loader] Sobran: {sobran}\n"
            f"[Loader] Faltan: {faltan}"
        )
    
    else:
        print(f"[Loader] estas son las columnas del dataset\n{columnas_actuales}")

def validar_errores(dataframe) :
    errores = dataframe.isnull().sum().sum()

    if errores > 0 :
        columnas_con_vacios = dataframe.columns[dataframe.isnull().any()]
        porcentajes_vacios = (dataframe.isnull().sum() / len(dataframe)) * 100

        print(f"[Loader] Hay {errores} en el dataset y las columnas son:\n{columnas_con_vacios}\n"
              f"[Loader] Porcentaje de errores:\n{porcentajes_vacios}")

        