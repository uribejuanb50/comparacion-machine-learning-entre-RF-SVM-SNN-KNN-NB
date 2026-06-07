import src.data.splitter as splitter
import src.data.preprocessor as preprocessor

def preparacion(dataframe, col_objetivo, diccionario, caracteristicas, columna_codif, semilla, categoricas, numericas, log_transform, cols_dropear) :
    dataframe_copia = dataframe.copy()

    dataframe_copia = dataframe_copia.fillna("none")
    dataframe_copia = dataframe_copia.drop(columns = cols_dropear)
    dataframe_copia = preprocessor.codificar_columna(dataframe_copia, col_objetivo, diccionario, int)

    x_train, y_train, x_validar, y_validar, x_test, y_test = splitter.split_dataframe(dataframe_copia,
                                                                                      caracteristicas,
                                                                                      columna_codif,
                                                                                      semilla)
    
    x_train_calibrado, x_validar_calibrado, x_test_calibrado, nombre_features = preprocessor.one_hot_cambio(x_train,
                                                                                           x_validar,
                                                                                           x_test,
                                                                                           categoricas,
                                                                                           numericas,
                                                                                           log_transform)

    return {
        "x_train" : x_train_calibrado,
        "y_train" : y_train,
        "x_validar" : x_validar_calibrado,
        "y_validar" : y_validar,
        "x_test" : x_test_calibrado,
        "y_test" : y_test,
        "features" : nombre_features
    }