import json
import numpy as np
from pathlib import Path
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

def mcnemar_test(
    y_true,
    y_pred_a,
    y_pred_b,
    nombre_a: str,
    nombre_b: str,
    alpha: float = 0.05
) -> dict:
    y_true   = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    # ── 1. Vectores de acierto ────────────────────────────────────────────────
    aciertos_a = (y_pred_a == y_true)
    aciertos_b = (y_pred_b == y_true)

    # ── 2. Tabla de contingencia 2×2 ─────────────────────────────────────────
    #         B acierta   B falla
    # A acierta   n00       n01
    # A falla     n10       n11
    n00 = int(np.sum( aciertos_a &  aciertos_b))
    n01 = int(np.sum( aciertos_a & ~aciertos_b))   # A✓ B✗  → A mejor en este punto
    n10 = int(np.sum(~aciertos_a &  aciertos_b))   # A✗ B✓  → B mejor en este punto
    n11 = int(np.sum(~aciertos_a & ~aciertos_b))

    tabla = [[n00, n01], [n10, n11]]

    # ── 3. Test de McNemar ────────────────────────────────────────────────────
    discordantes = n01 + n10
    exact        = discordantes < 25           # binomial exacto si hay pocos discordantes
    correction   = not exact                   # Yates solo en la aproximación chi²

    resultado_statsmodels = _mcnemar(tabla, exact=exact, correction=correction)

    statistic = (float(resultado_statsmodels.statistic)
                 if not exact else None)       # test exacto no produce estadístico χ²
    pvalue    = float(resultado_statsmodels.pvalue)

    # ── 4. Decisión ───────────────────────────────────────────────────────────
    rechaza_h0 = pvalue < alpha

    # ── 5. Dirección de la diferencia (quién gana en los discordantes) ────────
    if n01 > n10:
        ganador   = nombre_a   # A acierta donde B falla más → A es mejor
        perdedor  = nombre_b
    elif n10 > n01:
        ganador   = nombre_b   # B acierta donde A falla más → B es mejor
        perdedor  = nombre_a
    else:
        ganador   = None       # empate exacto en discordantes

    # ── 6. Texto de interpretación ────────────────────────────────────────────
    metodo = ("binomial exacto" if exact
              else "χ² con corrección de Yates")

    if rechaza_h0:
        if ganador:
            interpretacion = (
                f"Diferencia estadísticamente significativa "
                f"(p={pvalue:.4f} < α={alpha}): {ganador} acierta en casos donde "
                f"{perdedor} falla con mayor frecuencia "
                f"(n01={n01}, n10={n10}). → {ganador} es superior."
            )
        else:
            interpretacion = (
                f"Diferencia estadísticamente significativa (p={pvalue:.4f} < α={alpha}), "
                f"pero los modelos presentan simetría perfecta en los discordantes "
                f"(n01=n10={n01})."
            )
    else:
        interpretacion = (
            f"Sin diferencia estadísticamente significativa "
            f"(p={pvalue:.4f} ≥ α={alpha}): no se puede distinguir "
            f"a {nombre_a} de {nombre_b} con este nivel de confianza."
        )

    return {
        "modelo_a":       nombre_a,
        "modelo_b":       nombre_b,
        "alpha":          alpha,
        "tabla_2x2":      tabla,
        "n00":            n00,
        "n01":            n01,          # A✓ B✗
        "n10":            n10,          # A✗ B✓
        "n11":            n11,
        "discordantes":   discordantes,
        "metodo":         metodo,
        "statistic":      statistic,
        "pvalue":         pvalue,
        "rechaza_h0":     rechaza_h0,
        "ganador":        ganador,
        "interpretacion": interpretacion,
    }


# ─────────────────────────────────────────────────────────────────────────────

def imprimir_resultado_mcnemar(resultado: dict) -> None:
    """
    Imprime el resultado de un test de McNemar de forma legible por consola.
    Muestra la dirección de la diferencia, no solo la significancia.
    """
    a    = resultado["modelo_a"]
    b    = resultado["modelo_b"]
    sep  = "─" * 55

    print(f"\n{sep}")
    print(f"  McNemar: {a}  vs  {b}")
    print(sep)

    # Tabla 2×2
    n00, n01 = resultado["n00"], resultado["n01"]
    n10, n11 = resultado["n10"], resultado["n11"]
    print(f"  {'':20s}  {'B acierta':>10s}  {'B falla':>10s}")
    print(f"  {'A acierta':20s}  {n00:>10d}  {n01:>10d}")
    print(f"  {'A falla':20s}  {n10:>10d}  {n11:>10d}")
    print()

    # Discordantes y dirección
    print(f"  n01 (A✓ B✗) = {n01:<6d}   n10 (A✗ B✓) = {n10}")
    if resultado["ganador"]:
        print(f"  ▶ {resultado['ganador']} gana en los casos discordantes")
    else:
        print(f"  ▶ Empate exacto en discordantes")
    print()

    # Estadísticos
    print(f"  Método      : {resultado['metodo']}")
    if resultado["statistic"] is not None:
        print(f"  Estadístico : {resultado['statistic']:.4f}")
    print(f"  p-value     : {resultado['pvalue']:.6f}")
    print(f"  α           : {resultado['alpha']}")
    print()

    # Decisión
    if resultado["rechaza_h0"]:
        print(f"  ✅ RECHAZA H0 — diferencia estadísticamente significativa")
    else:
        print(f"  ❌ NO rechaza H0 — modelos estadísticamente equivalentes")
    print()

    print(f"  {resultado['interpretacion']}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────

def guardar_resultado_mcnemar(resultado: dict, ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, default=str)

    print(f"[McNemar] resultado guardado en {ruta}")


# ─────────────────────────────────────────────────────────────────────────────

def comparar_todos_pares(
    y_true,
    predicciones: dict,
    alpha: float = 0.05,
    correccion_bonferroni: bool = True,
    ruta_salida: Path = None,
    imprimir: bool = True
) -> dict:
    nombres  = list(predicciones.keys())
    pares    = list(combinations(nombres, 2))
    n_tests  = len(pares)

    alpha_usado = alpha / n_tests if correccion_bonferroni else alpha

    if correccion_bonferroni:
        print(f"[McNemar] Corrección de Bonferroni: α {alpha} / {n_tests} tests "
              f"= α_corr {alpha_usado:.4f}")
    print(f"[McNemar] Ejecutando {n_tests} tests pareados...\n")

    resultados = {}

    for nombre_a, nombre_b in pares:
        clave     = f"{nombre_a}_vs_{nombre_b}"
        resultado = mcnemar_test(
            y_true,
            predicciones[nombre_a],
            predicciones[nombre_b],
            nombre_a, nombre_b,
            alpha=alpha_usado
        )
        resultados[clave] = resultado

        if imprimir:
            imprimir_resultado_mcnemar(resultado)

        if ruta_salida is not None:
            ruta_json = Path(ruta_salida) / f"mcnemar_{clave}.json"
            guardar_resultado_mcnemar(resultado, ruta_json)

    # ── Resumen global ────────────────────────────────────────────────────────
    n_significativos = sum(r["rechaza_h0"] for r in resultados.values())
    print(f"\n[McNemar] Resumen: {n_significativos}/{n_tests} pares "
          f"con diferencia significativa (α={'_corr=' if correccion_bonferroni else ''}"
          f"{alpha_usado:.4f})")

    if ruta_salida is not None:
        resumen = {
            "alpha_original":          alpha,
            "correccion_bonferroni":   correccion_bonferroni,
            "n_tests":                 n_tests,
            "alpha_usado":             alpha_usado,
            "n_significativos":        n_significativos,
            "pares":                   {
                clave: {
                    "pvalue":      r["pvalue"],
                    "rechaza_h0":  r["rechaza_h0"],
                    "ganador":     r["ganador"],
                }
                for clave, r in resultados.items()
            }
        }
        ruta_resumen = Path(ruta_salida) / "mcnemar_resumen.json"
        guardar_resultado_mcnemar(resumen, ruta_resumen)

    return resultados