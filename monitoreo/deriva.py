"""
Deriva de datos y calidad de entrada (sección 6.2).

PSI (Population Stability Index) por feature contra la distribución de ENTRENAMIENTO:
    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)
con 10 intervalos por cuantiles de entrenamiento (variables continuas) o una categoría por valor
(variables con <= 12 valores distintos). Umbrales:
    PSI < 0,10  estable | 0,10-0,25 moderada (aviso) | > 0,25 alta (alerta)
Son los cortes estándar en riesgo de crédito, y además se verifican con el piso de ruido de ESTE
proyecto: piso_ruido_psi() mide el PSI entre entrenamiento y 200 muestras aleatorias de 600 clientes
del propio entrenamiento (donde no hay deriva por construcción); su percentil 99 queda muy por
debajo de 0,10, así que un PSI > 0,25 no es ruido de muestreo.

La decisión de alerta la toma el PSI. KS (numéricas) y chi-cuadrado (categóricas) se calculan como
respaldo: con 600 clientes contra 24.000 detectan diferencias mínimas sin importancia práctica, por
eso solo confirman (p < 0,01) y no disparan nada por sí solas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERADA = 0.10
PSI_ALTA = 0.25
EPS = 1e-4
MAX_CATEGORIAS = 12
CATEGORICAS = {"education_code", "is_education_imputed", "is_marriage_imputed", "is_inactive_6m",
               "is_delinquent_last_month"} | {f"pay_status_m{m}" for m in range(1, 7)}


def es_categorica(ref: pd.Series, nombre: str) -> bool:
    return nombre in CATEGORICAS or ref.nunique(dropna=True) <= MAX_CATEGORIAS


def _proporciones(ref: pd.Series, act: pd.Series, categorica: bool):
    ref, act = ref.dropna(), act.dropna()
    if categorica:
        cats = sorted(set(ref.unique()) | set(act.unique()))
        e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        return e, a
    bordes = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(bordes) < 3:                       # casi constante: se trata como categórica
        return _proporciones(ref, act, True)
    bordes[0], bordes[-1] = -np.inf, np.inf
    e = np.histogram(ref, bordes)[0] / max(len(ref), 1)
    a = np.histogram(act, bordes)[0] / max(len(act), 1)
    return e, a


def psi(ref: pd.Series, act: pd.Series, categorica: bool = False) -> float:
    if act.dropna().empty:
        return float("nan")
    e, a = _proporciones(ref, act, categorica)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def nivel_psi(v: float) -> str:
    if np.isnan(v):
        return "sin_datos"
    return "alta" if v > PSI_ALTA else ("moderada" if v > PSI_MODERADA else "estable")


def prueba(ref: pd.Series, act: pd.Series, categorica: bool) -> tuple[str, float, float]:
    ref, act = ref.dropna(), act.dropna()
    if len(act) < 5:
        return ("ninguna", float("nan"), float("nan"))
    if categorica:
        cats = sorted(set(ref.unique()) | set(act.unique()))
        tabla = np.vstack([ref.value_counts().reindex(cats, fill_value=0),
                           act.value_counts().reindex(cats, fill_value=0)])
        tabla = tabla[:, tabla.sum(axis=0) > 0]
        if tabla.shape[1] < 2:
            return ("chi2", 0.0, 1.0)
        chi, p, _, _ = stats.chi2_contingency(tabla)
        return ("chi2", float(chi), float(p))
    r = stats.ks_2samp(ref, act)
    return ("ks", float(r.statistic), float(r.pvalue))


def deriva_lote(ref: pd.DataFrame, act: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    filas = []
    for f in features:
        cat = es_categorica(ref[f], f)
        v = psi(ref[f], act[f], cat)
        nombre, est, p = prueba(ref[f], act[f], cat)
        filas.append({"feature": f, "tipo_variable": "categorica" if cat else "numerica", "psi": v,
                      "nivel": nivel_psi(v), "prueba": nombre, "estadistico": est, "p_valor": p,
                      "prueba_confirma": bool(p < 0.01) if not np.isnan(p) else False})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def calidad_lote(ref: pd.DataFrame, act: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """% de faltantes, % fuera del rango de entrenamiento y % de categorías no vistas, por feature."""
    filas = []
    for f in features:
        a = act[f]
        cat = es_categorica(ref[f], f)
        lo, hi = ref[f].min(), ref[f].max()
        fuera = ((a < lo) | (a > hi)).mean() if not cat else 0.0
        no_vistas = (~a.dropna().isin(set(ref[f].dropna().unique()))).mean() if cat and a.notna().any() else 0.0
        filas.append({"feature": f, "pct_faltantes": float(a.isna().mean()), "pct_fuera_rango": float(fuera),
                      "pct_categorias_no_vistas": float(no_vistas)})
    return pd.DataFrame(filas)


def piso_ruido_psi(ref: pd.DataFrame, features: list[str], n: int = 600, repeticiones: int = 200,
                   semilla: int = 0) -> pd.DataFrame:
    """PSI entre entrenamiento y submuestras del propio entrenamiento (sin deriva): ruido de muestreo."""
    rng = np.random.default_rng(semilla)
    cats = {f: es_categorica(ref[f], f) for f in features}
    valores = {f: [] for f in features}
    for _ in range(repeticiones):
        m = ref.iloc[rng.choice(len(ref), size=n, replace=False)]
        for f in features:
            valores[f].append(psi(ref[f], m[f], cats[f]))
    return pd.DataFrame({"feature": features,
                         "psi_p50": [np.nanpercentile(valores[f], 50) for f in features],
                         "psi_p99": [np.nanpercentile(valores[f], 99) for f in features]})
