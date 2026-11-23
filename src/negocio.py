"""Métrica de negocio del PSet 1: pérdida evitada neta de contactar preventivamente.

    Ahorro = VP * saldo_medio * t_exito - (VP + FP) * c_contacto

VP = moras detectadas entre los contactados; FP = contactados que sí iban a pagar.
Los parámetros son supuestos (PSet 1, Fuente C) y se reportan con análisis de sensibilidad.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score

T_EXITO = 0.30        # tasa de éxito del refinanciamiento (sensibilidad: 0.10 / 0.30 / 0.50)
C_CONTACTO = 500.0    # NT$ por contacto (llamada + gestión de la oferta)
CAPACIDAD = 0.20      # el call center atiende como máximo al 20% de la cartera por mes


def ahorro(y, contactar, saldo_medio, t_exito=T_EXITO, c_contacto=C_CONTACTO) -> float:
    y = np.asarray(y).astype(bool)
    contactar = np.asarray(contactar).astype(bool)
    vp = int((y & contactar).sum())
    fp = int((~y & contactar).sum())
    return vp * saldo_medio * t_exito - (vp + fp) * c_contacto


def contactar_top(score, umbral, capacidad=CAPACIDAD):
    """Contacta a quien supera el umbral, acotado a los `capacidad` de mayor riesgo."""
    score = np.asarray(score, dtype=float)
    k = int(np.floor(capacidad * len(score)))
    corte_cap = np.sort(score)[::-1][k - 1] if k > 0 else np.inf
    return (score >= umbral) & (score >= corte_cap)


def recall_precision_top(y, score, frac=CAPACIDAD):
    score = np.asarray(score, dtype=float)
    k = int(np.floor(frac * len(score)))
    idx = np.argsort(-score, kind="stable")[:k]
    pred = np.zeros(len(score), dtype=int)
    pred[idx] = 1
    return recall_score(y, pred), precision_score(y, pred, zero_division=0)


def elegir_umbral(y, score, saldo_medio, **kw):
    """Umbral que maximiza el ahorro respetando la capacidad (se elige con OOF de entrenamiento)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    vals = [ahorro(y, contactar_top(score, u), saldo_medio, **kw) for u in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i]), grid, np.array(vals)


def contactar_k(score, k):
    """Contacta exactamente a los k clientes de mayor riesgo (comparación a igual volumen)."""
    score = np.asarray(score, dtype=float)
    out = np.zeros(len(score), dtype=bool)
    out[np.argsort(-score, kind="stable")[:k]] = True
    return out


def corte_capacidad(score, capacidad=CAPACIDAD):
    """Probabilidad del último cliente que entra en la capacidad (umbral efectivo)."""
    score = np.asarray(score, dtype=float)
    k = int(np.floor(capacidad * len(score)))
    return float(np.sort(score)[::-1][k - 1])


# ----------------------------------------------------------------------------- PSet 4: decisión
# Umbral de equilibrio (corrección de la observación del PSet 1 sobre el umbral 0,6):
# contactar a un cliente con probabilidad p vale la pena si p * saldo_medio * t_exito > c_contacto,
#   p* = c_contacto / (saldo_medio * t_exito) = 500 / (47.967 * 0,30) ≈ 0,035.
# Por costo conviene llamar a casi todos; lo que limita es la CAPACIDAD (20% de la cartera). Por eso
# el umbral operativo es el corte de capacidad medido con predicciones OOF de entrenamiento
# (corte_efectivo_train_oof ≈ 0,34, parámetro de la corrida del champion), no 0,6.
SALDO_MEDIO = 47966.81   # media de bill_amt_m1 de los morosos de entrenamiento (PSet 3)


def umbral_equilibrio(saldo_medio: float = SALDO_MEDIO, t_exito: float = T_EXITO,
                      c_contacto: float = C_CONTACTO) -> float:
    return c_contacto / (saldo_medio * t_exito)


ACCIONES = {
    "contactar": "Llamada preventiva de cobranza con oferta de refinanciamiento o diferimiento; "
                 "el asesor revisa el caso y elige la oferta",
    "no_contactar": "Sin contacto preventivo; sigue el proceso normal de cobranza",
}

# Respaldo (PSet 3, línea base de negocio): contactar si hay >= 1 mes de atraso en el último estado.
# La "probabilidad" que devuelve es la tasa histórica de impago de cada grupo en entrenamiento.
TASA_REGLA_ATRASADO = 0.504   # pay_status_m1 >= 1  (22,9% de la cartera)
TASA_REGLA_AL_DIA = 0.137     # pay_status_m1 <  1


def decision_regla(pay_status_m1: float) -> tuple[float, str]:
    atrasado = pay_status_m1 >= 1
    return (TASA_REGLA_ATRASADO if atrasado else TASA_REGLA_AL_DIA,
            "contactar" if atrasado else "no_contactar")


def decision_modelo(probabilidad: float, corte: float) -> str:
    return "contactar" if probabilidad >= corte else "no_contactar"
