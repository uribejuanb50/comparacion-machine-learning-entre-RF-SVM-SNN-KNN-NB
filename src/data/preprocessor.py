

def codificar_columna(dataframe, columna, diccionario, tipo) :
    dataframe_copia = dataframe.copy()

    dataframe_copia[columna] = dataframe_copia[columna].map(diccionario).astype(tipo)

    return dataframe_copia

