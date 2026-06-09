import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import (List, Callable, Optional)
from pathlib import Path

from sklearn.metrics import (confusion_matrix, 
                             roc_curve, 
                             auc, 
                             precision_recall_curve, 
                             average_precision_score)

COLOR_TO_CMAP = {
    "C0": "Blues", "C1": "Oranges", "C2": "Greens",
    "C3": "Reds",  "C4": "Purples",
}

def graficar_matriz_confusion(y_true, y_pred, colores, nombre_modelo, nombre_clases, titulo, ruta_salida, normalizar=False):
    # Reemplaza la línea 15 problemática por esto:
    if normalizar:
        matriz = confusion_matrix(y_true, y_pred, normalize="true")
        fmt_val = ".2f"
        i = 1
    else:
        matriz = confusion_matrix(y_true, y_pred)
        fmt_val = "d"
        i = 0

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    cmap_modelo = colores.get(nombre_modelo, "Blues")

    sns.heatmap(matriz,
                annot=True,
                fmt=fmt_val,
                cmap=cmap_modelo,
                xticklabels=nombre_clases,
                yticklabels=nombre_clases,
                ax=axes[i],
                cbar=True)
    
    axes[i].set_title(f"{titulo}\nConteos absolutos")
    axes[i].set_xlabel("Predicho")
    axes[i].set_ylabel("Verdadero")

    plt.savefig(ruta_salida, dpi = 300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] grafica de {nombre_modelo} guardada en {ruta_salida}")

def graficar_curva_roc_binaria(y_true, y_proba, colores, nombre_modelo, titulo, ruta_salida) :

    fpr, tpr, _= roc_curve(y_true, y_proba[:,1])
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(8, 6))

    color_linea = colores.get(nombre_modelo, "C0")

    plt.plot(fpr, tpr, color= color_linea, lw=2,
             label=f"Curva ROC (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel("tasa de falsos positivos")
    plt.ylabel("tasa de verdaderos positivos")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] grafica curva roc auc de {nombre_modelo} guardada en {ruta_salida}")


def grafica_roc_comparativa(y_true, probas_por_modelo, colores, ruta_salida):

    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    for nombre, (_, y_proba) in probas_por_modelo.items():
        color = colores.get(nombre, "C0")
        fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{nombre} (AUC = {roc_auc:.3f})")

    axes[0].plot([0,1], [0,1], color="gray", lw=2, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    axes[0].set_title("Curva ROC comparativa")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    baseline = np.sum(y_true) / len(y_true)

    for nombre, (_, y_proba) in probas_por_modelo.items():
        color = colores.get(nombre, "C0")
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:,1])
        pr_auc = average_precision_score(y_true, y_proba[:,1])
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{nombre} (AP = {pr_auc:.3f})")

    axes[1].axhline(y=baseline, color="gray", lw=2, linestyle="--", label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Exhaustividad)")
    axes[1].set_ylabel("Precision (Precisión)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.savefig(ruta_salida, dpi = 300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] Grafica ROC compaartiva gardada en {ruta_salida}")

def graficar_curvas_aprendizaje(history, ruta_salida):
    fig = plt.figure(figsize=(10,6))
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epocas = range(1, len(train_loss) + 1)

    plt.plot(epocas, train_loss, "b-", lw=2, label="Pérdida en entrenamiento")
    plt.plot(epocas, val_loss, "r-", lw=2, label="Pérdida en validacion")

    if val_loss:
        mejor_epoca = np.argmin(val_loss) + 1
        mejor_val_loss = val_loss[mejor_epoca - 1]
        plt.axvline(x=mejor_epoca, color="gray", linestyle="--",
                    label=f"Mejor época ({mejor_epoca})")
        
        plt.plot(mejor_epoca, mejor_val_loss, "ko")

    plt.title("Curva de aprendizaje de la red neuronal")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (Loss)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)

def graficar_curva_aprendizaje_vs_tamano(
    modelos: dict,
    x_train,
    y_train,
    x_val,
    y_val,
    tamanos: List[int],
    metrica_fn: Callable,
    ruta_salida: Path,
    nombre_metrica: str = "F1-macro",
    nombre_archivo: str = "learning_curve_vs_tamano.png"
) -> None:
    """
    Para cada tamaño de entrenamiento en `tamanos`, entrena cada modelo sobre un
    subconjunto y evalúa en validación. Plotea métrica vs tamaño para todos los
    modelos en una sola figura.
 
    Esta función revela qué modelos aprenden rápido con poca data (ANN, RF) y
    cuáles son más estables pero lentos (SVM, KNN). Es la 'sección 5' del proyecto.
 
    Parámetros
    ----------
    modelos : dict
        {"NombreModelo": instancia_modelo, ...}
        Cada instancia debe implementar fit(x, y) y predict(x).
        IMPORTANTE: fit() sobrescribe el modelo en cada llamada. Las instancias
        deben soportar llamadas múltiples a fit() — todas las del proyecto lo hacen.
    x_train, y_train : datos de entrenamiento completos (se submuestrea internamente).
    x_val, y_val     : datos de validación fijos para todas las corridas.
    tamanos : list[int]
        Ej. [100, 500, 1000, 5000, 10000]. Se filtra automáticamente para no
        exceder el tamaño real de x_train.
    metrica_fn : callable
        Función (y_true, y_pred) → float. Ej: lambda yt, yp: f1_score(yt, yp, average="macro")
    ruta_salida : Path
        Directorio donde guardar la figura.
    nombre_metrica : str
        Etiqueta del eje Y.
    nombre_archivo : str
        Nombre del archivo PNG de salida.
 
    Notas
    -----
    - SVM y KNN son lentos: con 6 tamaños × 2 modelos puede tardar varios minutos.
      Considera excluirlos o limitar los tamaños máximos para ellos.
    - La ANN crea NnModule internamente en cada fit(), así que cada tamaño parte
      desde cero con pesos aleatorios (semilla fijada en __init__).
    """
    from src.data.splitter import submuestreo_estratificado
 
    n_total  = x_train.shape[0]
    tamanos  = [t for t in sorted(tamanos) if t < n_total] + [n_total]
    colores  = plt.cm.tab10(np.linspace(0, 0.8, len(modelos)))
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    for (nombre, modelo), color in zip(modelos.items(), colores):
        scores = []
        print(f"[Plots] Curva de tamaño — {nombre}...")
 
        for t in tamanos:
            if t < n_total:
                x_sub, y_sub = submuestreo_estratificado(x_train, y_train,
                                                          n_muestras=t, semilla=42)
            else:
                x_sub, y_sub = x_train, y_train
 
            modelo.fit(x_sub, y_sub, x_val, y_val)            # re-entrena desde cero
            y_pred  = modelo.predict(x_val)
            score   = metrica_fn(y_val, y_pred)
            scores.append(score)
            print(f"  tam={t:>6}  {nombre_metrica}={score:.4f}")
 
        ax.plot(tamanos, scores,
                label=nombre, color=color, linewidth=2.0,
                marker="o", markersize=5)
 
    # ── Formato ───────────────────────────────────────────────────────────────
    ax.set_xlabel("Tamaño de entrenamiento", fontsize=11)
    ax.set_ylabel(nombre_metrica, fontsize=11)
    ax.set_title(f"{nombre_metrica} vs tamaño de entrenamiento",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
 
    # Eje x en escala log si el rango es grande
    if tamanos[-1] / tamanos[0] > 50:
        ax.set_xscale("log")
        ax.set_xlabel("Tamaño de entrenamiento (escala log)", fontsize=11)
 
    plt.tight_layout()
    ruta_figura = Path(ruta_salida) / nombre_archivo
    plt.savefig(ruta_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plots] Curva vs tamaño guardada en {ruta_figura}")
 












