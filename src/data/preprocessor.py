from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def codificar_columna(dataframe, columna, diccionario, tipo) :
    dataframe_copia = dataframe.copy()

    dataframe_copia[columna] = dataframe_copia[columna].map(diccionario).astype(tipo)

    return dataframe_copia

def one_hot_cambio(x_train, x_validar, x_test, categoricas, numericas) :

    print(f"categoricas:\n{categoricas}\nnumericas:\n{numericas}")   
    one_hot = OneHotEncoder(handle_unknown = "infrequent_if_exist", sparse_output= True)

    escalador_numerico = StandardScaler()

    procesador = ColumnTransformer(
        transformers=[
            ("categorico", one_hot, categoricas),
            ("numerico", escalador_numerico, numericas)
        ]
    )

   
    x_train_procesado = procesador.fit_transform(x_train)
    x_validar_procesado = procesador.transform(x_validar)
    x_test_procesado = procesador.transform(x_test)

    nombres_columnas = procesador.get_feature_names_out()

    print(f"[preprocessor] Cantidad de columnas nuevas: {len(nombres_columnas.tolist())} a partir de {len(x_train.columns)} columnas originlas")

    return x_train_procesado, x_validar_procesado, x_test_procesado, nombres_columnas