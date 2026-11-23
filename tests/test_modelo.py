"""El modelo exportado reproduce la métrica registrada en MLflow sobre una muestra fija.

Tolerancia: 1e-3 de AUC. Es el mismo archivo de modelo y las mismas versiones de librerías, así que
la diferencia esperada es 0 (el AUC solo depende del orden de las probabilidades); 1e-3 deja margen
para diferencias de punto flotante entre plataformas (Mac ARM vs. Linux x86 en CI) y es 4 veces
menor que la desviación de la CV del champion (~0,004): cualquier cambio real en el modelo o en la
preparación (otra versión, otra columna, otro escalado) la rompe.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from api.esquemas import FEATURES

TARGET = "default_next_month"


def test_reproduce_auc_registrado(modelo, muestra, referencia):
    auc = roc_auc_score(muestra[TARGET], modelo.predict_proba(muestra[FEATURES])[:, 1])
    assert abs(auc - referencia["auc_muestra_fija"]) <= referencia["tolerancia_auc"], (
        f"AUC {auc:.5f} vs registrado {referencia['auc_muestra_fija']:.5f}")


def test_auc_dentro_de_lo_esperado_por_la_cv(modelo, muestra, referencia):
    """Sanidad: con 600 clientes el AUC puede variar; debe quedar a < 4 sigma_lote de la CV."""
    auc = roc_auc_score(muestra[TARGET], modelo.predict_proba(muestra[FEATURES])[:, 1])
    sigma_lote = referencia["cv_roc_auc_std"] * np.sqrt(4800 / len(muestra))
    assert auc > referencia["cv_roc_auc_mean"] - 4 * max(sigma_lote, 0.02)


def test_predicciones_son_probabilidades_y_deterministas(modelo, muestra):
    p1 = modelo.predict_proba(muestra[FEATURES])[:, 1]
    p2 = modelo.predict_proba(muestra[FEATURES].sample(frac=1, random_state=1).sort_index())[:, 1]
    assert np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_allclose(p1, p2)
