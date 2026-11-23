"""
Preparación de datos del proyecto (importada por todos los notebooks; no se copia).

  construir_pipeline(modelo) = FeatureEngineer -> ColumnTransformer -> modelo

- FeatureEngineer crea las features derivadas. Es "sin estado" (no aprende nada de los datos),
  así que no puede filtrar información del conjunto de prueba.
- El ColumnTransformer imputa, escala y codifica; se ajusta SOLO con los datos con que se llama
  fit() (entrenamiento, o el pliegue de entrenamiento dentro de la CV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MESES = range(1, 7)  # m1 = sep-2005 (último observado) ... m6 = abr-2005

# Montos con cola larga (hallazgo H2 de la EDA): se transforman con log con signo.
MONTOS = (
    ["limit_bal", "total_pay_amt_6m", "avg_bill_amt_6m", "bill_trend_m1_vs_m6"]
    + [f"bill_amt_m{m}" for m in MESES]
    + [f"pay_amt_m{m}" for m in MESES]
)

DERIVADAS = [
    "meses_desde_ultimo_atraso",   # recencia del atraso (H3)
    "tendencia_mora",              # atrasos recientes vs. antiguos
    "cobertura_ultimo_pago",       # pago de sep / saldo de ago
    "tendencia_utilizacion",       # util_m1 - util_m6
    "estado_ultimo_mes",           # pay_status_m1 como categoría (H4: relación no monótona)
]
CATEGORICAS = ["education_code", "estado_ultimo_mes"]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Features derivadas a partir de columnas observadas hasta sep-2005 (sin fuga).

    derivadas=False solo aplica las transformaciones de montos (se usa en la ablación).
    """

    def __init__(self, derivadas: bool = True):
        self.derivadas = derivadas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["education_code"] = X["education_code"].astype(int).astype(str)
        if self.derivadas:
            # Las derivadas se calculan con los montos ORIGINALES (antes del log).
            estados = X[[f"pay_status_m{m}" for m in MESES]].to_numpy()
            atraso = estados >= 1
            # mes más reciente con atraso (1 = sep-2005); 7 si nunca se atrasó en 6 meses
            X["meses_desde_ultimo_atraso"] = np.where(atraso.any(axis=1), atraso.argmax(axis=1) + 1, 7)
            X["tendencia_mora"] = X["n_delinquent_months_3m"] - (
                X["n_delinquent_months_6m"] - X["n_delinquent_months_3m"]
            )
            saldo_prev = X["bill_amt_m2"].where(X["bill_amt_m2"] > 0)
            X["cobertura_ultimo_pago"] = (X["pay_amt_m1"] / saldo_prev).clip(upper=5)
            X["tendencia_utilizacion"] = X["util_m1"] - X["util_m6"]
            X["estado_ultimo_mes"] = X["pay_status_m1"].clip(-2, 3).astype(int).astype(str)
        for c in MONTOS:
            if c in X:
                X[c] = signed_log1p(X[c])
        self.columnas_salida_ = list(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(getattr(self, "columnas_salida_", []), dtype=object)


def _columnas_numericas(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in CATEGORICAS]


def preprocesador(columnas_entrada: list[str], derivadas: bool = True) -> ColumnTransformer:
    ejemplo = FeatureEngineer(derivadas).transform(
        pd.DataFrame(np.zeros((1, len(columnas_entrada))), columns=columnas_entrada)
    )
    numericas = _columnas_numericas(ejemplo)
    categoricas = [c for c in CATEGORICAS if c in ejemplo.columns]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                              ("escalar", StandardScaler())]), numericas),
            ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             categoricas),
        ],
        verbose_feature_names_out=False,
    )


def construir_pipeline(modelo, columnas_entrada: list[str], derivadas: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer(derivadas)),
            ("prep", preprocesador(columnas_entrada, derivadas)),
            ("modelo", modelo),
        ]
    )


class ReglaNegocio(BaseEstimator, ClassifierMixin):
    """Línea base de negocio (PSet 1): riesgo = 1 si hay >= 1 mes de atraso en el último estado."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (np.asarray(X["pay_status_m1"]) >= 1).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def pipeline_regla() -> Pipeline:
    return Pipeline([("modelo", ReglaNegocio())])
