"""
Construcción de los lotes de la simulación de producción (sección 5 del PSet 4).

No hay datos posteriores al corte (la fuente UCI termina en sep-2005), así que se usa el conjunto
de prueba del PSet 3 (6.000 clientes que ningún modelo vio): se ordena por customer_id y se parte
en N_LOTES bloques consecutivos de igual tamaño, que representan meses sucesivos de producción.
customer_id no tiene significado temporal, por eso los lotes limpios son intercambiables entre sí:
sirven para medir la variación natural entre lotes, no para estudiar estacionalidad.

Horizonte: el target de un lote llega UN lote después (horizonte de 1 mes del PSet 1). La tabla
TARGETS guarda `disponible_en_lote = lote_id + 1` y nadie lo usa antes.

Deriva inducida (lotes >= INICIO_DERIVA), documentada exactamente:
  1. Deriva de datos ("recorte de cupos"): limit_bal se multiplica por FACTOR_CUPO (0,6) y se
     recalculan las columnas que dependen del cupo con las mismas fórmulas de Silver/OBT:
     util_m1..m6 = bill/limit, avg_utilization_6m, max_utilization_6m, n_overlimit_months_6m.
  2. Deriva conceptual ("choque de liquidez"): entre los clientes AL DÍA (pay_status_m1 <= 0) que con
     el nuevo cupo quedan con util_m1 >= UMBRAL_UTIL (0,70) y que sí iban a pagar, una fracción
     PROB_CAMBIO (0,60) pasa a no pagar. Cambia la relación utilización -> impago entre clientes sin
     atrasos, que el modelo consideraba de bajo riesgo porque no tenían historial de mora.
  3. Calidad de datos: en TODOS los lotes, N_CORRUPTOS registros llegan con errores del sistema de
     origen (edad 0, código de educación desconocido o un campo vacío). La API debe rechazarlos
     con 422; así las métricas de sistema tienen errores 4xx reales que medir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.datos import ID, TARGET

N_LOTES = 10
INICIO_DERIVA = 6
FACTOR_CUPO = 0.6
UMBRAL_UTIL = 0.70
PROB_CAMBIO = 0.60
N_CORRUPTOS = 3
SEMILLA_SIM = 2026
MESES = range(1, 7)


def periodo(lote_id: int) -> str:
    """Mes simulado de producción: lote 1 = oct-2005 (el mes siguiente al corte)."""
    return (pd.Period("2005-09", "M") + int(lote_id)).strftime("%Y-%m")


def recortar_cupo(df: pd.DataFrame, factor: float = FACTOR_CUPO) -> pd.DataFrame:
    d = df.copy()
    d["limit_bal"] = d["limit_bal"] * factor
    lim = d["limit_bal"].where(d["limit_bal"] > 0)
    for m in MESES:
        d[f"util_m{m}"] = d[f"bill_amt_m{m}"] / lim
    utils = d[[f"util_m{m}" for m in MESES]]
    d["avg_utilization_6m"] = utils.mean(axis=1)
    d["max_utilization_6m"] = utils.max(axis=1)
    d["n_overlimit_months_6m"] = sum((d[f"bill_amt_m{m}"] > d["limit_bal"]).astype(int) for m in MESES)
    return d


def choque_liquidez(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = df.copy()
    u = rng.random(len(d))
    # clientes al día pero muy endeudados con el nuevo cupo empiezan a no pagar
    caen = (d["util_m1"] >= UMBRAL_UTIL) & (d["pay_status_m1"] <= 0) & (d[TARGET] == 0) & (u < PROB_CAMBIO)
    d.loc[caen, TARGET] = 1
    d["target_alterado"] = caen.astype(int)
    return d


def corromper(df: pd.DataFrame, rng: np.random.Generator, n: int = N_CORRUPTOS) -> pd.DataFrame:
    d = df.copy()
    d["registro_corrupto"] = 0
    idx = rng.choice(len(d), size=min(n, len(d)), replace=False)
    tipos = ["edad_cero", "educacion_desconocida", "campo_vacio"]
    for i, pos in enumerate(idx):
        t = tipos[i % len(tipos)]
        if t == "edad_cero":
            d.iloc[pos, d.columns.get_loc("age")] = 0
        elif t == "educacion_desconocida":
            d.iloc[pos, d.columns.get_loc("education_code")] = 9
        else:
            d.iloc[pos, d.columns.get_loc("pay_status_m1")] = np.nan
        d.iloc[pos, d.columns.get_loc("registro_corrupto")] = 1
    return d


def construir_lotes(test: pd.DataFrame, n_lotes: int = N_LOTES, inicio_deriva: int = INICIO_DERIVA,
                    semilla: int = SEMILLA_SIM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (lotes, targets). `lotes` NO trae el target; `targets` sí, con su lote de llegada."""
    rng = np.random.default_rng(semilla)
    base = test.sort_values(ID).reset_index(drop=True)
    orden = np.random.default_rng(semilla).permutation(len(base))   # asignación aleatoria y fija
    partes = np.array_split(orden, n_lotes)
    salida = []
    for k, pos in enumerate(partes, start=1):
        d = base.iloc[pos].copy()
        d["target_alterado"] = 0
        con_deriva = k >= inicio_deriva
        if con_deriva:
            d = recortar_cupo(d)
            d = choque_liquidez(d, rng)
        d = corromper(d, rng)
        d["lote_id"] = k
        d["periodo"] = periodo(k)
        d["deriva_inducida"] = int(con_deriva)
        salida.append(d)
    todo = pd.concat(salida, ignore_index=True)
    targets = todo[[ID, "lote_id", TARGET, "target_alterado"]].copy()
    targets["disponible_en_lote"] = targets["lote_id"] + 1
    lotes = todo.drop(columns=[TARGET, "target_alterado"])
    return lotes, targets
