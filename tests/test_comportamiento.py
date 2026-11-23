"""Comportamiento de negocio que el modelo debe cumplir (dominio: mora de tarjetas de crédito).

Tolerancia de 0,01 en probabilidad: el modelo de árboles no tiene restricciones de monotonía y puede
tener escalones mínimos entre cortes; un cambio mayor en la dirección contraria sí sería un error de
lógica que llegaría a la decisión de cobranza. En el pago se exige además lo que importa al negocio:
que pagar más nunca convierta a un cliente en "contactar".
"""
import numpy as np
import pandas as pd

from api.esquemas import FEATURES
from tests.conftest import cliente_valido

TOL = 0.01


def _p(modelo, X: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(X[FEATURES])[:, 1]


def _con_atraso(X: pd.DataFrame, meses: int) -> pd.DataFrame:
    """Mismo cliente, pero con `meses` de atraso en el último estado (y las columnas que dependen de eso)."""
    Y = X.copy()
    estaba = Y["pay_status_m1"] >= 1
    Y["pay_status_m1"] = meses
    Y["is_delinquent_last_month"] = 1
    Y["max_delay_6m"] = np.maximum(Y["max_delay_6m"], meses)
    Y["n_delinquent_months_3m"] = np.minimum(3, Y["n_delinquent_months_3m"] + (~estaba))
    Y["n_delinquent_months_6m"] = np.minimum(6, Y["n_delinquent_months_6m"] + (~estaba))
    return Y


def test_mas_atraso_no_reduce_la_probabilidad(modelo, muestra):
    """Un cliente al día que pasa a tener 2 o 3 meses de atraso no puede bajar de riesgo, y pasar de
    1 a 2 meses de atraso tampoco. (El paso 0 -> 1 no se exige cliente por cliente: en la fuente UCI
    el estado 1 mezcla atrasos reales con clientes que pagaron el total, hallazgo H4 del PSet 3.)"""
    X = muestra[muestra["pay_status_m1"] <= 0]
    p0 = _p(modelo, X)
    p1 = _p(modelo, _con_atraso(X, 1))
    p2 = _p(modelo, _con_atraso(X, 2))
    p3 = _p(modelo, _con_atraso(X, 3))
    assert np.all(p2 >= p0 - TOL), f"{np.sum(p2 < p0 - TOL)} clientes bajan de riesgo con 2 meses de atraso"
    assert np.all(p3 >= p0 - TOL), f"{np.sum(p3 < p0 - TOL)} clientes bajan de riesgo con 3 meses de atraso"
    assert np.all(p2 >= p1 - TOL), f"{np.sum(p2 < p1 - TOL)} clientes bajan de riesgo al pasar de 1 a 2 meses"
    assert p2.mean() > p0.mean() + 0.10


def test_pagar_el_saldo_no_aumenta_el_riesgo_ni_provoca_una_llamada(modelo, muestra, referencia):
    """Si el cliente paga todo el saldo anterior en el último mes: en promedio su riesgo baja, ningún
    cliente sube más de 0,05 y NINGUNO pasa de "no contactar" a "contactar" por haber pagado más."""
    X = muestra[muestra["bill_amt_m2"] > 0].copy()
    Y = X.copy()
    extra = np.maximum(Y["bill_amt_m2"] - Y["pay_amt_m1"], 0)
    Y["pay_amt_m1"] = Y["pay_amt_m1"] + extra
    Y["total_pay_amt_6m"] = Y["total_pay_amt_6m"] + extra
    p0, p1 = _p(modelo, X), _p(modelo, Y)
    corte = max(referencia["corte_efectivo"], referencia["umbral_decision"])
    assert p1.mean() < p0.mean()
    assert np.all(p1 <= p0 + 0.05), f"{np.sum(p1 > p0 + 0.05)} clientes suben más de 0,05 al pagar más"
    nuevas_llamadas = (p0 < corte) & (p1 >= corte)
    assert not nuevas_llamadas.any(), f"{nuevas_llamadas.sum()} clientes pasarían a ser llamados por pagar más"


def test_clientes_con_atraso_tienen_mas_riesgo_en_promedio(modelo, muestra):
    p = _p(modelo, muestra)
    atrasado = muestra["pay_status_m1"].to_numpy() >= 1
    assert p[atrasado].mean() > 2 * p[~atrasado].mean()


def test_decision_coherente_con_el_umbral(api):
    """La API contacta si y solo si la probabilidad supera el umbral de capacidad del PSet 3."""
    base = cliente_valido()
    for estado in (-1, 0, 1, 2, 3):
        j = api.post("/predecir", json={**base, "pay_status_m1": estado, "pay_status_m2": max(estado, 0)}).json()
        assert (j["decision"] == "contactar") == (j["probabilidad"] >= j["umbral"])
