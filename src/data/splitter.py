from sklearn.model_selection import train_test_split

def split_dataframe(dataframe, caracteristicas, objetivo, semilla) :
    dataframe_copia = dataframe.copy()

    df_caracteristicas = dataframe_copia[caracteristicas]
    df_objetivos = dataframe_copia[objetivo]

    x_train, x_test, y_train, y_test = train_test_split(df_caracteristicas, 
                                                        df_objetivos,
                                                        test_size = 0.3,
                                                        random_state = semilla
                                                        )
    
    x_validar, x_test, y_validar, y_test = train_test_split(x_test,
                                                            y_test,
                                                            test_size = 0.5,
                                                            random_state= semilla
                                                            )
    
    return x_train, y_train, x_validar, y_validar, x_test, y_test

