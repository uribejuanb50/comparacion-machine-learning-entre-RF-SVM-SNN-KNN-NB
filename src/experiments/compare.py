import src.evaluation.statistical as statistical

import json

def comparar_modelos_estadisticamente(valores, predicciones, alpha, path_metricas) :

    pares = [
        #("RF", "ANN"),
        #("KNN", "ANN"),
        #("SVM", "ANN"),
        ("NB", "ANN")
    ]

    y_true = valores["y_test"]

    resultados = {}

    for modelo_a, modelo_b in pares :
        resultado = statistical.mcnemar_test(y_true,
                                             predicciones[modelo_a],
                                             predicciones[modelo_b],
                                             modelo_a,
                                             modelo_b,
                                             alpha)
        
        statistical.imprimir_resultado_mcnemar(resultado)

        resultado_json = json.dumps(resultado, indent = 2, default = str)
        clave = f"{modelo_a}_vs_{modelo_b}"
        statistical.guardar_resultado_mcnemar(resultado_json, path_metricas / f"test_mc_nemar_{clave}.json")

        resultados[clave] = resultado

    return resultado