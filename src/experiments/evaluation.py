import src.evaluation.metrics as metrics
import src.evaluation.visualization as visualization

from sklearn.metrics import f1_score

def evaluar(valores, modelos, path_figures, path_metrics, colores) :

    x_test = valores["x_test"]
    y_test = valores["y_test"]
    features = valores["features"]

    metricas_todos = {}
    probas_por_modelo = {}
    predicciones_por_modelo = {}

    for nombre, modelo in modelos.items():
        print(f"{nombre} ============================")

        objetivo_predict = modelo.predict(x_test)
        objetivo_proba = modelo.predict_proba(x_test)

        probas_por_modelo[nombre] = (y_test, objetivo_proba)
        predicciones_por_modelo[nombre] = objetivo_predict

        metricas = metrics.calcular_metricas(y_test, objetivo_predict, objetivo_proba, features)
        metrics.guardar_metricas(metricas, path_metrics / f"metricas_{nombre}.JSON")
        metricas_todos[nombre] = metricas

        visualization.graficar_matriz_confusion(y_test,
                                                objetivo_predict,
                                                colores,
                                                nombre,
                                                features,
                                                f"Matriz de confusión '{nombre}'",
                                                path_figures / f"matriz_confusion_{nombre}.png")
        visualization.graficar_curva_roc_binaria(y_test,
                                                 objetivo_proba,
                                                 colores,
                                                 nombre,
                                                 f"Curva ROC binaria {nombre}",
                                                 path_figures / f"curva_roc_binaria_{nombre}.png")
    
    visualization.grafica_roc_comparativa(y_test,
                                          probas_por_modelo,
                                          colores,
                                          path_figures / "roc_comparativo_todos.png")
    visualization.graficar_curvas_aprendizaje(modelos["ANN"].history,
                                              path_figures / "curvas_aprendizaje_ANN.png")
    visualization.graficar_curva_aprendizaje_vs_tamano(
        modelos       = modelos,
        x_train       = valores["x_train"],
        y_train       = valores["y_train"],
        x_val         = valores["x_validar"],
        y_val         = valores["y_validar"],
        tamanos       = [100, 500, 1000, 5000, 10000],
        metrica_fn    = lambda yt, yp: f1_score(yt, yp, average="macro"),
        ruta_salida   = path_figures,
        nombre_metrica= "F1-macro",
        nombre_archivo= "learning_curve_vs_tamano.png"
    )
    metrics.imprimir_resumen_corporativo(metricas_todos, 
                                         "f1_macro",
                                         path_metrics / "resumen_corporativo_metricas.csv")