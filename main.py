import src.data.loader as loader
import src.experiments.preparation as preparation
import src.experiments.training as training
import src.experiments.evaluation as evaluation
import src.experiments.compare as compare

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
                        HIPERPARAMETROS_KNN,
                        HIPERPARAMETROS_SVM,
                        HIPERPARAMETROS_NB,
                        FIGURES,
                        METRICS,
                        COLORES_MODELOS,
                        COLORMAPS_MODELOS,
                        NOMBRE_CLASES,
                        ALPHA)

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
                                        hp_knn= HIPERPARAMETROS_KNN,
                                        hp_svm=HIPERPARAMETROS_SVM,
                                        hp_nb=HIPERPARAMETROS_NB)
    
    metricas_todos, predicciones_por_modelo = evaluation.evaluar(valores,
                                                                 modelos,
                                                                 FIGURES, 
                                                                 METRICS, 
                                                                 COLORES_MODELOS, 
                                                                 COLORMAPS_MODELOS,
                                                                 NOMBRE_CLASES)

    resultados = compare.comparar_modelos_estadisticamente(valores= valores,
                                                           predicciones= predicciones_por_modelo,
                                                           alpha= ALPHA,
                                                           path_metricas= METRICS)



if __name__ == "__main__" :
    main()