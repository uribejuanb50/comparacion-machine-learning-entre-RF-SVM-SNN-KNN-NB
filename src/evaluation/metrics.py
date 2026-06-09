import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix,
                             cohen_kappa_score,
                             classification_report)

def calcular_metricas(y_true, y_pred, y_proba, nombre_clases) :

    exactitud = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_por_clase = f1_score(y_true, y_pred, average=None)
    precision_por_clase = precision_score(y_true, y_pred, average=None)
    recall_por_clase = recall_score(y_true, y_pred, average=None)

    f1_dict = {clase: valor for clase, valor in zip(nombre_clases, f1_por_clase)}
    precision_por_clase_dict = {clase: valor for clase, valor in zip(nombre_clases, precision_por_clase)}
    recall_por_clase_dict = {clase: valor for clase, valor in zip(nombre_clases, recall_por_clase)}

    roc_auc = roc_auc_score(y_true, y_proba[:,1])
    matriz_confusion = confusion_matrix(y_true, y_pred).tolist()

    report = classification_report(y_true, y_pred, target_names=nombre_clases, output_dict=True)

    return {
        "accuracy" : float(exactitud),
        "kappa" : float(kappa),
        "f1_macro" : float(f1_macro),
        "f1_weighted" : float(f1_weighted),
        "f1_dict" : f1_dict,
        "precision_dict" : precision_por_clase_dict,
        "recall_dict" : recall_por_clase_dict,
        "roc_auc" : roc_auc,
        "mc" : matriz_confusion,
        "classification_report" : report
    }

def guardar_metricas(metricas, path) :

    with open(path, "w", encoding="utf-8") as f :
        json.dump(metricas, f, indent=2, default=str)

def imprimir_resumen_corporativo(metricas, metrica_primaria, path):

    df_metricas = pd.DataFrame(metricas).T
    
    if metrica_primaria in df_metricas.columns :
        df_metricas = df_metricas.sort_values(by=metrica_primaria, ascending=False)

    else:
        raise ValueError(
            "[Metrics] no se encontrola metrica primaria, revisa los diccionarios"
        )
    
    print("[Metrics] resumen comparativo de los modelos " + "="*60)
    print(df_metricas.to_string(float_format = lambda x : f"{x:.4f}"))
    
    df_metricas.to_csv(path, index_label="modelo", float_format="%.4f")
    print(f"[Metrics] resumen guardado en {path}")